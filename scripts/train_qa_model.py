from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import typer
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from app.core.config import get_settings
from app.utils.text import sentences

app = typer.Typer(help="Fine-tune the QA model on the labelled campus FAQ dataset.")

logger = logging.getLogger("tunispeak.train_qa")


@dataclass
class SquadExample:
    """Lightweight container for a SQuAD-style record."""

    id: str
    question: str
    context: str
    answers: dict[str, list]
    language: str
    document_id: str


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_corpus(chunks_path: Path) -> dict[str, str]:
    corpus: dict[str, str] = {}
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            corpus[str(record["document_id"])]=str(record["text"])
    return corpus


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    lowered = stripped.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _token_overlap_score(sentence: str, answers: Sequence[str]) -> float:
    if not sentence or not answers:
        return 0.0
    sentence_tokens = set(_normalize(sentence).split())
    if not sentence_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        answer_tokens = set(_normalize(answer).split())
        if not answer_tokens:
            continue
        shared = sentence_tokens & answer_tokens
        best = max(best, len(shared) / max(len(answer_tokens), 1))
    return best


def select_answer_span(answers: Sequence[str], context: str) -> tuple[str, int] | None:
    if not answers or not context:
        return None
    for answer in answers:
        start = context.find(answer)
        if start != -1:
            return answer, start
    context_sentences = [segment for segment in sentences(context) if segment.strip()]
    if not context_sentences:
        return context, 0
    best_sentence = max(context_sentences, key=lambda sent: _token_overlap_score(sent, answers))
    overlap = _token_overlap_score(best_sentence, answers)
    if overlap == 0.0:
        logger.warning("train_qa.no_overlap", sentence=best_sentence)
    start_index = context.find(best_sentence)
    if start_index == -1:
        start_index = 0
    return best_sentence, start_index


def build_squad_examples(dataset_path: Path, corpus_path: Path) -> list[SquadExample]:
    raw_examples = read_jsonl(dataset_path)
    corpus = load_corpus(corpus_path)
    squad_examples: list[SquadExample] = []
    skipped = 0

    for idx, row in enumerate(raw_examples):
        relevant_docs: Sequence[str] = row.get("relevant_documents") or []
        context_id: str | None = None
        for doc_id in relevant_docs:
            if doc_id in corpus:
                context_id = doc_id
                break
        if not context_id:
            skipped += 1
            logger.warning(
                "train_qa.missing_document", question=row.get("question"), documents=relevant_docs
            )
            continue
        context = corpus[context_id]
        span = select_answer_span(row.get("answers", []), context)
        if not span:
            skipped += 1
            logger.warning(
                "train_qa.missing_span", question=row.get("question"), document=context_id
            )
            continue
        answer_text, start_index = span
        example = SquadExample(
            id=f"train-{idx}",
            question=str(row.get("question", "")),
            context=context,
            answers={"text": [answer_text], "answer_start": [start_index]},
            language=str(row.get("metadata", {}).get("language", "unknown")),
            document_id=context_id,
        )
        squad_examples.append(example)

    if skipped:
        logger.info("train_qa.skipped_examples total=%d", skipped)
    logger.info("train_qa.loaded_examples total=%d", len(squad_examples))
    return squad_examples


def prepare_datasets(
    examples: Sequence[SquadExample],
    tokenizer,
    max_length: int,
    doc_stride: int,
    eval_split: float,
    seed: int,
) -> tuple[Dataset, Dataset | None, Dataset | None]:
    records = [
        {
            "id": example.id,
            "question": example.question,
            "context": example.context,
            "answers": example.answers,
            "language": example.language,
            "document_id": example.document_id,
        }
        for example in examples
    ]
    dataset = Dataset.from_list(records)
    validation_examples = None
    tokenized_validation = None

    if eval_split > 0.0 and len(dataset) >= 2:
        dataset_dict: DatasetDict = dataset.train_test_split(test_size=eval_split, seed=seed)
        train_dataset = dataset_dict["train"]
        validation_examples = dataset_dict["test"]
    else:
        train_dataset = dataset

    def prepare_train_features(batch):
        tokenized = tokenizer(
            batch["question"],
            batch["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized.sequence_ids(i)
            sample_index = sample_mapping[i]

            answer = batch["answers"][sample_index]
            start_char = int(answer["answer_start"][0])
            answer_text = str(answer["text"][0])
            end_char = start_char + len(answer_text)

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(max(token_start_index - 1, 0))

            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(max(token_end_index + 1, 0))

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    tokenized_train = train_dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    if validation_examples is not None:
        def prepare_validation_features(batch):
            tokenized = tokenizer(
                batch["question"],
                batch["context"],
                truncation="only_second",
                max_length=max_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            sample_mapping = tokenized.pop("overflow_to_sample_mapping")
            example_ids = []

            for i in range(len(tokenized["input_ids"])):
                sequence_ids = tokenized.sequence_ids(i)
                offsets = tokenized["offset_mapping"][i]
                sample_index = sample_mapping[i]
                example_ids.append(batch["id"][sample_index])
                tokenized["offset_mapping"][i] = [
                    offset if sequence_ids[idx] == 1 else None for idx, offset in enumerate(offsets)
                ]

            tokenized["example_id"] = example_ids
            return tokenized

        tokenized_validation = validation_examples.map(
            prepare_validation_features,
            batched=True,
            remove_columns=validation_examples.column_names,
        )

    return tokenized_train, tokenized_validation, validation_examples


def postprocess_qa_predictions(
    examples: Dataset,
    features: Dataset,
    raw_predictions: tuple[np.ndarray, np.ndarray],
    tokenizer,
    n_best_size: int,
    max_answer_length: int,
) -> dict[str, str]:
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example: dict[int, list[int]] = defaultdict(list)

    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions: dict[str, str] = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example.get(example_index, [])
        min_null_score = None
        valid_answers: list[tuple[float, int, int, str]] = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offsets)
                        or end_index >= len(offsets)
                        or offsets[start_index] is None
                        or offsets[end_index] is None
                    ):
                        continue
                    if end_index < start_index:
                        continue
                    length = offsets[end_index][1] - offsets[start_index][0]
                    if length > max_answer_length:
                        continue
                    text = example["context"][offsets[start_index][0] : offsets[end_index][1]]
                    score = start_logits[start_index] + end_logits[end_index]
                    valid_answers.append((score, offsets[start_index][0], offsets[end_index][1], text))

        if valid_answers:
            best_answer = max(valid_answers, key=lambda item: item[0])
            predictions[example["id"]] = best_answer[3]
        else:
            predictions[example["id"]] = ""

    return predictions


@app.command()
def train(
    dataset: Path = typer.Option(
        Path("data/faq/mini_faq.jsonl"),
        help="Path to the labelled FAQ dataset (JSONL).",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        Path("models/tunispeak-qa"),
        help="Directory to save the fine-tuned model.",
    ),
    base_model: str = typer.Option(
        None,
        help="Base model name or path. Defaults to settings.qa_model if omitted.",
    ),
    max_length: int = typer.Option(384, help="Maximum total sequence length."),
    doc_stride: int = typer.Option(128, help="Stride when splitting long documents."),
    n_best_size: int = typer.Option(20, help="Number of n-best predictions to consider."),
    max_answer_length: int = typer.Option(64, help="Maximum length of predicted answers."),
    eval_split: float = typer.Option(0.2, help="Proportion of data reserved for evaluation."),
    learning_rate: float = typer.Option(2e-5, help="Learning rate."),
    batch_size: int = typer.Option(8, help="Per-device batch size."),
    weight_decay: float = typer.Option(0.01, help="Weight decay."),
    epochs: float = typer.Option(3.0, help="Number of epochs."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    settings = get_settings()
    chunks_path = Path(settings.processed_data_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        raise typer.BadParameter(f"Corpus chunks file not found at {chunks_path}")

    model_name = base_model or settings.qa_model
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger.info("train_qa.start dataset=%s model=%s", dataset, model_name)

    squad_examples = build_squad_examples(dataset, chunks_path)
    if not squad_examples:
        raise typer.BadParameter("No training examples could be derived from the dataset.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    train_dataset, eval_dataset, eval_examples = prepare_datasets(
        squad_examples,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
        eval_split=eval_split,
        seed=seed,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    total_steps = math.ceil(len(train_dataset) / batch_size * epochs)
    logger.info(
        "train_qa.dataset_sizes train=%d eval=%d",
        len(train_dataset),
        0 if eval_dataset is None else len(eval_dataset),
    )
    logger.info("train_qa.steps total_steps=%d", total_steps)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "artifacts"),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=100 if eval_dataset is not None else None,
        save_strategy="steps" if eval_dataset is not None else "no",
        save_total_limit=1,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        seed=seed,
        logging_steps=25,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    data_collator = DefaultDataCollator()
    metric = load_metric("squad") if eval_dataset is not None else None

    def compute_metrics(eval_prediction):
        if metric is None or eval_examples is None or eval_dataset is None:
            return {}
        predictions = postprocess_qa_predictions(
            examples=eval_examples,
            features=eval_dataset,
            raw_predictions=eval_prediction.predictions,
            tokenizer=tokenizer,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
        )
        references = [
            {"id": example_id, "answers": answers}
            for example_id, answers in zip(eval_examples["id"], eval_examples["answers"])
        ]
        return metric.compute(predictions=[{"id": k, "prediction_text": v} for k, v in predictions.items()], references=references)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if eval_dataset is not None:
        metrics = trainer.evaluate(eval_dataset)
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump({k: float(v) for k, v in metrics.items()}, handle, indent=2, ensure_ascii=False)
        logger.info("train_qa.metrics_saved path=%s", metrics_path)

    logger.info("train_qa.complete model_dir=%s", output_dir)


if __name__ == "__main__":  # pragma: no cover
    app()
