from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match(prediction: str, references: Sequence[str]) -> float:
    if not prediction:
        return 0.0
    normalized_pred = _normalize_text(prediction)
    for reference in references:
        if _normalize_text(reference) == normalized_pred:
            return 1.0
    return 0.0


def f1_score(prediction: str, references: Sequence[str]) -> float:
    if not references:
        return 0.0
    normalized_pred_tokens = _normalize_text(prediction).split()
    if not normalized_pred_tokens:
        return 0.0
    best_f1 = 0.0
    for reference in references:
        ref_tokens = _normalize_text(reference).split()
        if not ref_tokens:
            continue
        common = set(normalized_pred_tokens) & set(ref_tokens)
        if not common:
            continue
        # Count overlaps per token
        pred_counts = {token: normalized_pred_tokens.count(token) for token in common}
        ref_counts = {token: ref_tokens.count(token) for token in common}
        overlap = sum(min(pred_counts[token], ref_counts[token]) for token in common)
        precision = overlap / len(normalized_pred_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def recall_at_k(predictions: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_pred = predictions[:k]
    hits = len(set(top_pred) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(
    predictions: Sequence[str],
    gains: dict[str, float],
    k: int,
) -> float:
    if not predictions or not gains:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(predictions[:k], start=1):
        gain = gains.get(doc_id, 0.0)
        if gain > 0:
            dcg += (2**gain - 1) / math.log2(rank + 1)
    ideal_docs = sorted(gains.items(), key=lambda item: item[1], reverse=True)
    ideal_dcg = 0.0
    for rank, (_, gain) in enumerate(ideal_docs[:k], start=1):
        ideal_dcg += (2**gain - 1) / math.log2(rank + 1)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def brier_score(probabilities: Sequence[float], outcomes: Sequence[float]) -> float:
    if not probabilities:
        return 0.0
    arr_p = np.asarray(probabilities, dtype=float)
    arr_o = np.asarray(outcomes, dtype=float)
    return float(np.mean((arr_p - arr_o) ** 2))


@dataclass
class CalibrationSummary:
    ece: float
    brier: float
    bins: list[dict[str, float]]


def expected_calibration_error(
    confidences: Sequence[float],
    outcomes: Sequence[float],
    n_bins: int = 10,
) -> CalibrationSummary:
    if not confidences:
        return CalibrationSummary(ece=0.0, brier=0.0, bins=[])

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(confidences, bin_edges, right=True)

    ece = 0.0
    bins: list[dict[str, float]] = []
    total = len(confidences)

    for bin_index in range(1, n_bins + 1):
        indices = [i for i, b in enumerate(bin_assignments) if b == bin_index]
        if not indices:
            continue
        bin_confs = [confidences[i] for i in indices]
        bin_outcomes = [outcomes[i] for i in indices]
        avg_conf = float(np.mean(bin_confs))
        avg_outcome = float(np.mean(bin_outcomes))
        weight = len(indices) / total
        ece += weight * abs(avg_conf - avg_outcome)
        bins.append(
            {
                "bin": bin_index,
                "avg_confidence": avg_conf,
                "avg_outcome": avg_outcome,
                "count": len(indices),
            }
        )

    brier = brier_score(confidences, outcomes)
    return CalibrationSummary(ece=float(ece), brier=brier, bins=bins)
