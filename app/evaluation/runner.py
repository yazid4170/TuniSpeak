from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import structlog

from app.core.config import get_settings
from app.evaluation.dataset import EvaluationExample, load_dataset
from app.evaluation.metrics import (
    CalibrationSummary,
    exact_match,
    expected_calibration_error,
    f1_score,
    ndcg_at_k,
    recall_at_k,
)
from app.pipelines.qa import QAPipeline

logger = structlog.get_logger(__name__)


@dataclass
class ExampleResult:
    question: str
    answers: list[str]
    prediction: str
    confidence: float
    abstain: bool
    reason: str | None
    em: float
    f1: float
    recall_at_5: float
    ndcg_at_5: float
    relevant_documents: list[str]
    retrieved_documents: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answers": self.answers,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "abstain": self.abstain,
            "reason": self.reason,
            "em": self.em,
            "f1": self.f1,
            "recall@5": self.recall_at_5,
            "ndcg@5": self.ndcg_at_5,
            "relevant_documents": self.relevant_documents,
            "retrieved_documents": self.retrieved_documents,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationSummary:
    examples: list[ExampleResult]
    aggregate: dict[str, float]
    calibration: CalibrationSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate": self.aggregate,
            "calibration": {
                "ece": self.calibration.ece,
                "brier": self.calibration.brier,
                "bins": self.calibration.bins,
            },
            "examples": [example.to_dict() for example in self.examples],
        }


class EvaluationRunner:
    def __init__(self, pipeline: QAPipeline | None = None) -> None:
        self.pipeline = pipeline or QAPipeline()
        self.settings = get_settings()

    def evaluate(self, dataset_path: Path, limit: int | None = None) -> EvaluationSummary:
        examples = load_dataset(dataset_path)
        if limit:
            examples = examples[:limit]

        results: list[ExampleResult] = []
        confidences: list[float] = []
        correctness: list[float] = []

        for example in examples:
            result = self._evaluate_example(example)
            results.append(result)
            confidences.append(result.confidence)
            correctness.append(result.em)

        aggregate = self._aggregate_metrics(results)
        calibration = expected_calibration_error(confidences, correctness)
        return EvaluationSummary(examples=results, aggregate=aggregate, calibration=calibration)

    def _evaluate_example(self, example: EvaluationExample) -> ExampleResult:
        pipeline_answer = self.pipeline.run(
            question=example.question,
            top_k=self.settings.qa_max_chunks,
        )
        retrieved_ids = [source.document_id for source in pipeline_answer.sources]
        em = exact_match(pipeline_answer.answer, example.answers)
        f1 = f1_score(pipeline_answer.answer, example.answers)
        recall5 = recall_at_k(retrieved_ids, example.relevant_documents, k=5)
        gains = example.relevance_gains or {doc_id: 1.0 for doc_id in example.relevant_documents}
        ndcg5 = ndcg_at_k(retrieved_ids, gains, k=5)
        return ExampleResult(
            question=example.question,
            answers=example.answers,
            prediction=pipeline_answer.answer,
            confidence=pipeline_answer.confidence,
            abstain=pipeline_answer.abstain,
            reason=pipeline_answer.reason,
            em=em,
            f1=f1,
            recall_at_5=recall5,
            ndcg_at_5=ndcg5,
            relevant_documents=example.relevant_documents,
            retrieved_documents=retrieved_ids,
            metadata=example.metadata,
        )

    def _aggregate_metrics(self, results: Iterable[ExampleResult]) -> dict[str, float]:
        results_list = list(results)
        if not results_list:
            return {}
        em_scores = [result.em for result in results_list]
        f1_scores = [result.f1 for result in results_list]
        recall_scores = [result.recall_at_5 for result in results_list]
        ndcg_scores = [result.ndcg_at_5 for result in results_list]
        abstain_rate = np.mean([1.0 if result.abstain else 0.0 for result in results_list])

        return {
            "em": float(np.mean(em_scores)),
            "f1": float(np.mean(f1_scores)),
            "recall@5": float(np.mean(recall_scores)),
            "ndcg@5": float(np.mean(ndcg_scores)),
            "abstention_rate": float(abstain_rate),
        }


def save_summary(summary: EvaluationSummary, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "evaluation_summary.json"
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2, ensure_ascii=False)
    logger.info("evaluation.summary_written", path=str(target_path), examples=len(summary.examples))
    return target_path
