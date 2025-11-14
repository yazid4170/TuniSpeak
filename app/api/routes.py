from __future__ import annotations

from functools import lru_cache
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import AnswerRequest, AnswerResponse, DocumentChunk
from app.pipelines.qa import QAPipeline
from app.services.language import LanguageDetector, SUPPORTED_LANGS
from app.services.telemetry import TelemetryLogger

router = APIRouter(prefix="/qa", tags=["qa"])


@lru_cache(maxsize=1)
def _get_pipeline() -> QAPipeline:
    return QAPipeline()


def get_pipeline() -> QAPipeline:
    return _get_pipeline()


@lru_cache(maxsize=1)
def _get_language_detector() -> LanguageDetector:
    return LanguageDetector()


def get_language_detector() -> LanguageDetector:
    return _get_language_detector()


@lru_cache(maxsize=1)
def _get_telemetry_logger() -> TelemetryLogger:
    return TelemetryLogger()


def get_telemetry_logger() -> TelemetryLogger:
    return _get_telemetry_logger()


@router.get("/health", summary="Health check")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/answer", response_model=AnswerResponse)
def answer(
    payload: AnswerRequest,
    pipeline: QAPipeline = Depends(get_pipeline),
    detector: LanguageDetector = Depends(get_language_detector),
    telemetry: TelemetryLogger = Depends(get_telemetry_logger),
) -> AnswerResponse:
    detected = detector.detect(payload.question)
    if detected.language not in SUPPORTED_LANGS:
        raise HTTPException(status_code=422, detail="Unsupported language")

    query_text = detected.normalized or payload.question
    answer = pipeline.run(question=query_text, top_k=payload.top_k)
    interaction_id = uuid4().hex

    telemetry.log_answer(
        interaction_id=interaction_id,
        question=payload.question,
        normalized_question=query_text,
        language=detected.language,
        answer=answer.answer,
        confidence=answer.confidence,
        abstain=answer.abstain,
        sources=[source.document_id for source in answer.sources],
    )

    return AnswerResponse(
        question=payload.question,
        answer=answer.answer,
        language=detected.language,
        normalized_query=query_text,
        sources=answer.sources,
        confidence=answer.confidence,
        abstain=answer.abstain,
        reason=answer.reason,
        interaction_id=interaction_id,
    )


@router.get("/documents", response_model=list[DocumentChunk])
def list_documents(
    pipeline: QAPipeline = Depends(get_pipeline),
    limit: int = 10,
) -> list[DocumentChunk]:
    return pipeline.retriever.preview(limit=limit)
