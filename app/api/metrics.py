from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends

from app.evaluation.metrics import expected_calibration_error
from app.models.schemas import CalibrationMetrics
from app.services.feedback import FeedbackRepository
from app.services.telemetry import TelemetryLogger

router = APIRouter(prefix="/metrics", tags=["metrics"])


@lru_cache(maxsize=1)
def _get_feedback_repo() -> FeedbackRepository:
    return FeedbackRepository()


def get_feedback_repo() -> FeedbackRepository:
    return _get_feedback_repo()


@lru_cache(maxsize=1)
def _get_telemetry_logger() -> TelemetryLogger:
    return TelemetryLogger()


def get_telemetry_logger() -> TelemetryLogger:
    return _get_telemetry_logger()


@router.get("/calibration", response_model=CalibrationMetrics)
def calibration_summary(
    telemetry: TelemetryLogger = Depends(get_telemetry_logger),
    feedback: FeedbackRepository = Depends(get_feedback_repo),
) -> CalibrationMetrics:
    telemetry_records = list(telemetry.iter_records())
    feedback_records = {record.interaction_id: record for record in feedback.iter_records()}

    confidences: list[float] = []
    outcomes: list[float] = []
    for record in telemetry_records:
        feedback_match = feedback_records.get(record.interaction_id)
        if feedback_match is None:
            continue
        confidences.append(record.confidence)
        outcomes.append(feedback_match.outcome)

    summary = expected_calibration_error(confidences, outcomes)
    return CalibrationMetrics(
        total_interactions=len(telemetry_records),
        feedback_count=len(confidences),
        ece=summary.ece,
        brier=summary.brier,
        bins=summary.bins,
    )
