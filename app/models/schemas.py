from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceAttribution(BaseModel):
    document_id: str = Field(..., description="Identifier of the source document")
    title: str | None = Field(default=None, description="Human readable title")
    snippet: str = Field(..., description="Excerpt providing supporting evidence")
    score: float = Field(..., ge=0.0, description="Retrieval score")
    url: str | None = Field(default=None, description="Optional link to the document")


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    document_id: str
    chunk_id: str
    text: str
    language: str
    score: float | None = None
    source_path: str | None = None


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class PipelineAnswer(BaseModel):
    answer: str
    sources: list[SourceAttribution]
    confidence: float = 0.0
    abstain: bool = False
    reason: str | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    language: str
    normalized_query: str
    sources: list[SourceAttribution]
    confidence: float
    abstain: bool
    reason: str | None = None
    interaction_id: str


class SpeechAnswerResponse(AnswerResponse):
    transcript: str
    transcript_language: str | None = None
    audio_base64: str | None = None


class FeedbackRequest(BaseModel):
    interaction_id: str = Field(..., min_length=8)
    rating: Literal["helpful", "unhelpful"]
    comment: str | None = Field(default=None, max_length=2000)
    correction: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    interaction_id: str
    outcome: float
    acknowledged: bool = True


class CalibrationMetrics(BaseModel):
    total_interactions: int
    feedback_count: int
    ece: float
    brier: float
    bins: list[dict[str, float]]


class LanguageDetectionResult(BaseModel):
    language: str
    normalized: str
    confidence: float
