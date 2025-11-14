from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train_qa_model import build_squad_examples, select_answer_span


def test_select_answer_span_handles_exact_match():
    context = "Bonjour tout le monde dans le campus."
    span = select_answer_span(["tout le monde"], context)
    assert span is not None
    answer_text, start_index = span
    assert answer_text == "tout le monde"
    assert start_index == context.index("tout le monde")


def test_select_answer_span_falls_back_to_sentence_for_partial_overlap():
    context = "نسخة من بطاقة التعريف وشهادة تسجيل للسنة الحالية.\nتودع الوثائق لدى المكتب."  # noqa: E501
    span = select_answer_span(["شهادة التسجيل للسنة الحالية"], context)
    assert span is not None
    answer_text, start_index = span
    assert "نسخة من بطاقة التعريف" in answer_text
    assert start_index == context.index(answer_text)


@pytest.fixture()
def tmp_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = tmp_path / "dataset.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"

    chunk_record = {
        "document_id": "doc-test",
        "chunk_id": "doc-test-0000",
        "text": "La carte etudiant est delivree apres inscription finale.\nApportez une photo recente et un justificatif d'identite au bureau des cartes.",
        "language": "fr",
    }
    with chunk_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(chunk_record, ensure_ascii=False))

    dataset_record = {
        "question": "Quels documents dois-je apporter pour retirer la carte?",
        "answers": ["photo recente et justificatif d'identite"],
        "relevant_documents": ["doc-test"],
        "relevance_gains": {"doc-test": 2},
        "metadata": {"language": "fr", "category": "logistique"},
    }
    with dataset_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dataset_record, ensure_ascii=False))

    return dataset_path, chunk_path


def test_build_squad_examples_recovers_sentence(tmp_dataset: tuple[Path, Path]):
    dataset_path, chunk_path = tmp_dataset
    examples = build_squad_examples(dataset_path, chunk_path)
    assert len(examples) == 1
    example = examples[0]
    assert example.answers["text"][0].startswith("Apportez une photo recente")
    assert example.answers["answer_start"][0] >= 0