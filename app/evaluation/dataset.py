from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationExample:
    question: str
    answers: list[str]
    relevant_documents: list[str]
    relevance_gains: dict[str, float]
    metadata: dict[str, Any]


def load_dataset(path: Path) -> list[EvaluationExample]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    examples: list[EvaluationExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(_parse_example(payload))
    if not examples:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return examples


def _parse_example(payload: dict[str, Any]) -> EvaluationExample:
    question = payload.get("question")
    if not question:
        raise ValueError("Example missing 'question'")
    answers = payload.get("answers") or []
    if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
        raise ValueError("Example 'answers' must be a list of strings")
    relevant = payload.get("relevant_documents") or []
    if not isinstance(relevant, list):
        raise ValueError("Example 'relevant_documents' must be a list")
    relevance_gains = payload.get("relevance_gains") or {}
    metadata = payload.get("metadata") or {}
    return EvaluationExample(
        question=question,
        answers=answers,
        relevant_documents=[str(doc) for doc in relevant],
        relevance_gains={str(doc): float(score) for doc, score in relevance_gains.items()},
        metadata=metadata,
    )
