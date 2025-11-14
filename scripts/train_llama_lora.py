from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - accelerate is an optional runtime dependency
    Accelerator = None  # type: ignore

ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful trilingual assistant for Tunisian university student services."
)


def _resolve_reporting_targets() -> List[str]:
    try:
        import tensorboard  # type: ignore  # noqa: F401

        return ["tensorboard"]
    except ImportError:
        return ["none"]


def _patch_accelerate_for_transformers() -> None:
    if Accelerator is None:
        return
    import inspect

    signature = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" in signature.parameters:
        return

    original_unwrap = Accelerator.unwrap_model

    def _shim(self, model, *args, **kwargs):
        kwargs.pop("keep_torch_compile", None)
        return original_unwrap(self, model, *args, **kwargs)

    Accelerator.unwrap_model = _shim  # type: ignore[assignment]


def _ensure_optimizer_train_eval_methods() -> None:
    if not hasattr(torch.optim.Optimizer, "train"):

        def _optimizer_train(self, mode: bool = True):  # noqa: ARG001 - signature parity with nn.Module
            return self

        torch.optim.Optimizer.train = _optimizer_train  # type: ignore[assignment]

    if not hasattr(torch.optim.Optimizer, "eval"):

        def _optimizer_eval(self):
            return self

        torch.optim.Optimizer.eval = _optimizer_eval  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with LoRA adapters")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/faq/dataset_finetune.jsonl"),
        help="Path to the JSONL dataset containing chat-formatted records.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Hugging Face model identifier or a local checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/llama-lora"),
        help="Directory where the trained LoRA adapter will be written.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length used during tokenisation.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.05,
        help="Fraction of samples reserved for evaluation (0 disables validation).",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of full passes over the training set.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Per-device batch size for both training and evaluation.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Number of update steps to accumulate before optimising.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="Rank for the LoRA adaptation matrices.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="Scaling factor for LoRA updates.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout probability applied inside the LoRA layers.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Interval (in steps) for trainer logging.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Interval (in steps) between checkpoint saves.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warm-up ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Optional cap on the total number of training steps (0 = disabled).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    dataset_path = args.dataset_path
    base_model = args.base_model
    output_dir = args.output_dir
    max_seq_length = args.max_seq_length
    eval_ratio = args.eval_ratio
    num_train_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    per_device_batch_size = args.per_device_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    warmup_ratio = args.warmup_ratio
    max_train_steps = args.max_train_steps

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_dict = _build_datasets(dataset_path, eval_ratio)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt_text, full_text = _format_conversation(example["messages"])
        full_tokens = tokenizer(
            full_text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )
        prompt_tokens = tokenizer(
            prompt_text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )
        labels = full_tokens["input_ids"].copy()
        prompt_len = sum(
            int(token != tokenizer.pad_token_id) for token in prompt_tokens["input_ids"]
        )
        labels[:prompt_len] = [-100] * prompt_len
        full_tokens["labels"] = labels
        return full_tokens

    tokenized = datasets_dict.map(
        tokenize_fn,
        remove_columns=datasets_dict["train"].column_names,
        desc="Tokenizing",
    )

    using_gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto" if using_gpu else torch.float32,
        device_map="auto" if using_gpu else None,
        low_cpu_mem_usage=not using_gpu,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    use_fp16 = using_gpu
    use_bf16 = False
    if using_gpu:
        capability = torch.cuda.get_device_capability(0)
        use_bf16 = capability[0] >= 8

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="steps" if "validation" in tokenized else "no",
        eval_steps=save_steps,
    save_total_limit=2,
    max_steps=max_train_steps if max_train_steps > 0 else -1,
        report_to=_resolve_reporting_targets(),
        dataloader_pin_memory=using_gpu,
        use_cpu=not using_gpu,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    _patch_accelerate_for_transformers()
    _ensure_optimizer_train_eval_methods()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if "validation" in tokenized:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "base_model": base_model,
        "dataset_path": str(dataset_path),
        "config": {
            "max_seq_length": max_seq_length,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
    }
    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_datasets(dataset_path: Path, eval_ratio: float) -> DatasetDict:
    raw = load_dataset("json", data_files=str(dataset_path))
    dataset: Dataset = raw["train"]
    filtered = dataset.filter(lambda x: bool(x.get("messages")))
    if eval_ratio > 0.0:
        split = filtered.train_test_split(test_size=eval_ratio, seed=42)
        return DatasetDict(train=split["train"], validation=split["test"])
    return DatasetDict(train=filtered)


def _format_conversation(messages: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    system_prompt, turns = _normalise_messages(messages)
    if not turns:
        raise ValueError("Training sample lacks conversational turns")
    if turns[-1][0] != "assistant":
        raise ValueError("Last message must be from the assistant")

    prompt_segments: List[str] = []
    if system_prompt:
        prompt_segments.append(_render_segment("system", system_prompt))
    else:
        prompt_segments.append(_render_segment("system", DEFAULT_SYSTEM_PROMPT))

    for role, content in turns[:-1]:
        prompt_segments.append(_render_segment(role, content))

    # Include the assistant tag so the model learns to produce the reply.
    prompt_segments.append(_render_segment("assistant", ""))

    prompt_text = "".join(prompt_segments)
    assistant_response = turns[-1][1].strip()
    full_text = f"{prompt_text}{assistant_response}\n<|end|>"
    return prompt_text, full_text


def _normalise_messages(messages: Sequence[Dict[str, Any]]) -> Tuple[str | None, List[Tuple[str, str]]]:
    system_prompt: str | None = None
    turns: List[Tuple[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system" and system_prompt is None:
            system_prompt = content
            continue
        if role in {"user", "assistant"}:
            turns.append((role, content))
    return system_prompt, turns


def _render_segment(role: str, content: str) -> str:
    token = ROLE_TOKENS.get(role, f"<|{role}|>")
    content = content.strip()
    return f"{token}\n{content}\n" if content else f"{token}\n"


if __name__ == "__main__":
    main(parse_args())
