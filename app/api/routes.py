from __future__ import annotations

import base64
from functools import lru_cache
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.models.schemas import (
    AnswerRequest,
    AnswerResponse,
    DocumentChunk,
    SpeechAnswerResponse,
)
from app.pipelines.qa import QAPipeline
from app.services.language import LanguageDetector, SUPPORTED_LANGS
from app.services.speech import SpeechService
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


@lru_cache(maxsize=1)
def _get_speech_service() -> SpeechService:
    return SpeechService()


def get_speech_service() -> SpeechService:
    return _get_speech_service()


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
    # Normalise the question first so retrieval sees consistent Darija/Arabic tokens.
    detected = detector.detect(payload.question)
    if detected.language not in SUPPORTED_LANGS:
        raise HTTPException(status_code=422, detail="Unsupported language")

    query_text = detected.normalized or payload.question
    answer = pipeline.run(question=query_text, top_k=payload.top_k)
    interaction_id = uuid4().hex

    # Persist telemetry for offline evaluation dashboards.
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


@router.post("/voice", response_model=SpeechAnswerResponse)
async def voice_answer(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
    pipeline: QAPipeline = Depends(get_pipeline),
    detector: LanguageDetector = Depends(get_language_detector),
    speech: SpeechService = Depends(get_speech_service),
    telemetry: TelemetryLogger = Depends(get_telemetry_logger),
) -> SpeechAnswerResponse:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    try:
        # Transcribe speech locally before reusing the same QA pipeline.
        transcription = speech.transcribe(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail="Transcription failed") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Transcription failed") from exc

    transcript = transcription.text.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="Unable to transcribe audio input")

    detected = detector.detect(transcript)
    if detected.language not in SUPPORTED_LANGS:
        raise HTTPException(status_code=422, detail="Unsupported language")

    query_text = detected.normalized or transcript
    answer = pipeline.run(question=query_text, top_k=top_k)
    interaction_id = uuid4().hex

    telemetry.log_answer(
        interaction_id=interaction_id,
        question=transcript,
        normalized_question=query_text,
        language=detected.language,
        answer=answer.answer,
        confidence=answer.confidence,
        abstain=answer.abstain,
        sources=[source.document_id for source in answer.sources],
    )

        # gTTS returns raw MP3 bytes; encode in base64 for the HTTP response payload.
    audio_reply = speech.synthesize(answer.answer, language=detected.language)
    encoded_audio = base64.b64encode(audio_reply).decode("ascii") if audio_reply else None

    return SpeechAnswerResponse(
        question=transcript,
        answer=answer.answer,
        language=detected.language,
        normalized_query=query_text,
        sources=answer.sources,
        confidence=answer.confidence,
        abstain=answer.abstain,
        reason=answer.reason,
        interaction_id=interaction_id,
        transcript=transcript,
        transcript_language=transcription.language or detected.language,
        audio_base64=encoded_audio,
    )


@router.get("/documents", response_model=list[DocumentChunk])
def list_documents(
    pipeline: QAPipeline = Depends(get_pipeline),
    limit: int = 10,
) -> list[DocumentChunk]:
    return pipeline.retriever.preview(limit=limit)
