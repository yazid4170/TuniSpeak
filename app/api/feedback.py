from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import FeedbackRequest, FeedbackResponse
from app.services.feedback import FeedbackRepository
from app.services.telemetry import TelemetryLogger

router = APIRouter(prefix="/feedback", tags=["feedback"])


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


@router.post("", response_model=FeedbackResponse, status_code=202)
def submit_feedback(
    payload: FeedbackRequest,
    repository: FeedbackRepository = Depends(get_feedback_repo),
    telemetry: TelemetryLogger = Depends(get_telemetry_logger),
) -> FeedbackResponse:
    if not telemetry.has_interaction(payload.interaction_id):
        raise HTTPException(status_code=404, detail="Interaction not found")

    record = repository.submit(
        interaction_id=payload.interaction_id,
        rating=payload.rating,
        comment=payload.comment,
        correction=payload.correction,
    )
    return FeedbackResponse(
        interaction_id=record.interaction_id,
        outcome=record.outcome,
    )
