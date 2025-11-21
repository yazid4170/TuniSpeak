from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import difflib
import re
import structlog
import unicodedata
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline

from app.core.config import get_settings
from app.models.schemas import DocumentChunk
from app.services.language import TUNISIAN_DARIJA_CODES
from app.utils.normalization import normalize_darija
from app.utils.text import sentences

_TOKEN_SYNONYMS: dict[str, set[str]] = {
    "\u0631\u0639\u0627\u064a\u0629": {"sante", "sant\u00e9", "medicale", "m\u00e9dicale"},
    "\u0635\u062d\u0629": {"sante", "sant\u00e9", "medicale", "m\u00e9dicale"},
    "\u0635\u062d\u064a": {"sante", "sant\u00e9", "medicale", "m\u00e9dicale"},
    "\u0635\u062d\u064a\u0629": {"sante", "sant\u00e9", "medicale", "m\u00e9dicale"},
    "sante": {"\u0635\u062d\u0629", "\u0635\u062d\u064a\u0629"},
    "sant\u00e9": {"\u0635\u062d\u0629", "\u0635\u062d\u064a\u0629"},
    "medicale": {"\u0631\u0639\u0627\u064a\u0629", "\u0635\u062d\u064a", "\u0635\u062d\u064a\u0629"},
    "m\u00e9dicale": {"\u0631\u0639\u0627\u064a\u0629", "\u0635\u062d\u064a", "\u0635\u062d\u064a\u0629"},
}


@dataclass
class QAEngineResult:
    answer: str
    confidence: float
    chunk_index: int | None
    abstain: bool
    reason: str | None = None


@dataclass
class ExtractiveQASystem:
    model_name: str = field(default_factory=lambda: get_settings().qa_model)
    threshold: float = field(default_factory=lambda: get_settings().qa_confidence_threshold)
    max_chunks: int = field(default_factory=lambda: get_settings().qa_max_chunks)
    abstain_message: str = field(default_factory=lambda: get_settings().qa_abstain_message)
    language_thresholds: dict[str, float] = field(
        default_factory=lambda: get_settings().qa_language_thresholds
    )
    language_abstain_messages: dict[str, str] = field(
        default_factory=lambda: get_settings().qa_abstain_messages
    )
    language_overlap_thresholds: dict[str, float] = field(
        default_factory=lambda: get_settings().qa_language_overlap
    )
    language_similarity_thresholds: dict[str, float] = field(
        default_factory=lambda: get_settings().qa_language_similarity
    )
    language_context_override_thresholds: dict[str, float] = field(
        default_factory=lambda: get_settings().qa_language_context_override
    )

    def __post_init__(self) -> None:
        self._logger = structlog.get_logger(__name__)
        settings = get_settings()
        # Load tokenizer/model once during service boot so subsequent calls stay low-latency.
        tokenizer = self._load_tokenizer()
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        pipeline_device = 0 if device.type == "cuda" else -1
        self._pipeline = QuestionAnsweringPipeline(
            model=model,
            tokenizer=tokenizer,
            max_answer_len=64,
            device=pipeline_device,
        )
        self._curated_prefixes = tuple(settings.retriever_curated_prefixes)
        self._curated_boost = max(0.0, float(settings.retriever_curated_boost))

    def _load_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            self._logger.info(
                "qa.tokenizer_loaded",
                model=self.model_name,
                fast_tokenizer=getattr(tokenizer, "is_fast", False),
            )
            return tokenizer
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._logger.warning(
                "qa.tokenizer_fallback",
                model=self.model_name,
                error_message=str(exc),
                exception_type=exc.__class__.__name__,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            return tokenizer

    def answer(
        self,
        question: str,
        chunks: Sequence[DocumentChunk],
        language: str | None = None,
    ) -> QAEngineResult:
        # No retrieval hit -> immediately abstain with a language-aware message.
        if not chunks:
            message = self._resolve_abstain_message(language)
            return QAEngineResult(
                answer="",
                confidence=0.0,
                chunk_index=None,
                abstain=True,
                reason=message,
            )

        candidate_answers: list[tuple[float, int, str, str]] = []
        last_error: str | None = None

        threshold = self._resolve_threshold(language)

        # Score each candidate chunk using the extractive QA pipeline, tracking context for post-processing.
        for idx, chunk in enumerate(chunks[: self.max_chunks]):
            context = " ".join(chunk.text.splitlines()).strip()
            if not context:
                continue
            try:
                result = self._pipeline(question=question, context=context)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.warning(
                    "qa.inference_failed",
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    error_message=str(exc),
                    exception_type=exc.__class__.__name__,
                    error_repr=repr(exc),
                )
                last_error = str(exc)
                continue
            answer_text = str(result.get("answer", "")).strip()
            score = float(result.get("score", 0.0))
            if answer_text and score > 0.0:
                boosted_score = self._apply_curated_boost(score, chunk)
                candidate_answers.append((boosted_score, idx, answer_text, context))

        chosen = self._select_candidate(candidate_answers, question, language)

        if not chosen:
            if last_error:
                return QAEngineResult(
                    answer="",
                    confidence=0.0,
                    chunk_index=None,
                    abstain=True,
                    reason=f"Erreur du modèle QA: {last_error}",
                )
            message = self._resolve_abstain_message(language)
            return QAEngineResult(
                answer="",
                confidence=0.0,
                chunk_index=None,
                abstain=True,
                reason=message,
            )

        best_score, best_index, best_answer, best_context = chosen

        abstain = best_score < threshold
        reason = None
        if abstain:
            reason = f"Score {best_score:.2f} inférieur au seuil {threshold:.2f}."

        processed_answer = self._post_process_answer(
            best_answer,
            best_context,
            language,
            question,
        )

        overlap_ratio = self._token_overlap_ratio(question, processed_answer)
        context_override_threshold: float | None = None
        context_overlap: float | None = None
        # Overrides: for Arabic/Darija we may override abstention if answer/context strongly matches the question.
        if abstain and processed_answer:
            override_method: str | None = None
            overlap_threshold = self._resolve_overlap_threshold(language)
            if overlap_ratio >= overlap_threshold:
                override_method = "overlap"
            else:
                similarity_threshold = self._resolve_similarity_threshold(language)
                if similarity_threshold is not None:
                    similarity_ratio = self._char_similarity_ratio(
                        question, processed_answer, language
                    )
                    if similarity_ratio >= similarity_threshold:
                        override_method = "similarity"
                if not override_method:
                    context_override_threshold = self._resolve_context_override_threshold(
                        language
                    )
                    if context_override_threshold is not None:
                        context_overlap = self._token_overlap_ratio(
                            question, best_context
                        )
                        if context_overlap >= context_override_threshold:
                            override_method = "context"
            if override_method:
                abstain = False
                reason = None
                best_score = max(best_score, threshold)
                self._logger.info(
                    "qa.abstain_overridden",
                    method=override_method,
                    overlap=overlap_ratio,
                    overlap_threshold=overlap_threshold,
                    context_overlap=context_overlap,
                    context_threshold=context_override_threshold,
                    language=language,
                )

        return QAEngineResult(
            answer=processed_answer,
            confidence=best_score,
            chunk_index=best_index,
            abstain=abstain,
            reason=reason,
        )

    def _resolve_threshold(self, language: str | None) -> float:
        if not language:
            return self.threshold
        key = language.lower()
        if key in TUNISIAN_DARIJA_CODES:
            key = "aeb"
        return self.language_thresholds.get(key, self.threshold)

    def _resolve_abstain_message(self, language: str | None) -> str:
        if not language:
            return self.abstain_message
        key = language.lower()
        if key in TUNISIAN_DARIJA_CODES:
            key = "aeb"
        return self.language_abstain_messages.get(key, self.abstain_message)

    def get_abstain_message(self, language: str | None) -> str:
        return self._resolve_abstain_message(language)

    def _resolve_overlap_threshold(self, language: str | None) -> float:
        default = self.language_overlap_thresholds.get("default", 0.5)
        if not language:
            return default
        key = language.lower()
        if key in TUNISIAN_DARIJA_CODES:
            key = "aeb"
        return self.language_overlap_thresholds.get(key, default)

    def _resolve_similarity_threshold(self, language: str | None) -> float | None:
        if not language:
            return None
        key = language.lower()
        if key in TUNISIAN_DARIJA_CODES:
            key = "aeb"
        return self.language_similarity_thresholds.get(key)

    def _resolve_context_override_threshold(self, language: str | None) -> float | None:
        default = self.language_context_override_thresholds.get("default")
        if not language:
            return default
        key = language.lower()
        if key in TUNISIAN_DARIJA_CODES:
            key = "aeb"
        return self.language_context_override_thresholds.get(key, default)

    def _post_process_answer(
        self,
        answer: str,
        context: str | None,
        language: str | None,
        question: str,
    ) -> str:
        if not context:
            return answer
        if not answer:
            return answer
        tokens = answer.strip().split()
        if len(tokens) >= 7:
            return answer
        if language:
            key = language.lower()
            if key in TUNISIAN_DARIJA_CODES:
                key = "aeb"
        else:
            key = None
        if key not in {"aeb", "ar"}:
            return answer

        try:
            answer_sentence: str | None = None
            best_sentence: str | None = None
            best_overlap = 0.0
            answer_overlap = 0.0
            for sentence in sentences(context):
                cleaned = sentence.strip()
                if not cleaned:
                    continue
                normalized_answer = re.sub(r"[^\w\s]", "", answer.lower())
                normalized_sentence = re.sub(r"[^\w\s]", "", cleaned.lower())
                if normalized_answer and normalized_answer in normalized_sentence and not answer_sentence:
                    answer_sentence = cleaned
                    answer_overlap = self._token_overlap_ratio(question, cleaned)
                overlap = self._token_overlap_ratio(question, cleaned)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = cleaned
            if best_sentence and best_overlap >= answer_overlap:
                return best_sentence
            if answer_sentence:
                return answer_sentence
        except Exception:  # pragma: no cover - defensive fallback
            self._logger.debug("qa.answer_postprocess_failed", answer=answer)
        return answer

    def _select_candidate(
        self,
        candidates: list[tuple[float, int, str, str]],
        question: str,
        language: str | None,
    ) -> tuple[float, int, str, str] | None:
        if not candidates:
            return None
        # Prefer answers whose tokens intersect with the question; otherwise fall back to highest-overlap context.
        significant = {
            token
            for token in re.findall(r"\w+", question.lower())
            if len(token) >= 4
        }
        augmented_significant = self._augment_tokens(significant)
        candidates.sort(key=lambda item: item[0], reverse=True)
        context_threshold = self._resolve_overlap_threshold(language)
        if not significant:
            return candidates[0]
        best_candidate = candidates[0]
        best_overlap = -1.0
        for score, idx, answer, context in candidates:
            answer_tokens = {
                token
                for token in re.findall(r"\w+", answer.lower())
                if len(token) >= 4
            }
            augmented_answer_tokens = self._augment_tokens(answer_tokens)
            if augmented_answer_tokens & augmented_significant:
                return score, idx, answer, context
            context_overlap = self._token_overlap_ratio(question, context)
            if context_overlap >= context_threshold:
                return score, idx, answer, context
            if context_overlap > best_overlap:
                best_candidate = (score, idx, answer, context)
                best_overlap = context_overlap
        return best_candidate

    def _apply_curated_boost(self, score: float, chunk: DocumentChunk) -> float:
        if score <= 0.0 or self._curated_boost <= 0.0:
            return score
        if not self._curated_prefixes:
            return score
        document_id = chunk.document_id or ""
        if document_id.startswith(self._curated_prefixes):
            return score * (1.0 + self._curated_boost)
        return score

    @staticmethod
    def _token_overlap_ratio(question: str, answer: str) -> float:
        if not question or not answer:
            return 0.0
        question_tokens = ExtractiveQASystem._significant_tokens(question)
        if not question_tokens:
            return 0.0
        answer_tokens = ExtractiveQASystem._significant_tokens(answer)
        if not answer_tokens:
            return 0.0
        augmented_question = ExtractiveQASystem._augment_tokens(question_tokens)
        augmented_answer = ExtractiveQASystem._augment_tokens(answer_tokens)
        shared = augmented_question & augmented_answer
        return len(shared) / float(len(question_tokens))

    @staticmethod
    def _significant_tokens(text: str) -> set[str]:
        base_tokens = [token for token in re.findall(r"\w+", text.lower()) if len(token) >= 4]
        expanded: set[str] = set()
        for token in base_tokens:
            expanded.add(token)
            stripped = ExtractiveQASystem._strip_diacritics(token)
            if stripped:
                expanded.add(stripped)
        return expanded

    @staticmethod
    def _strip_diacritics(token: str) -> str:
        if not token:
            return token
        decomposed = unicodedata.normalize("NFD", token)
        stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        return stripped

    @staticmethod
    def _augment_tokens(tokens: set[str]) -> set[str]:
        if not tokens:
            return set()
        augmented = set(tokens)
        for token in list(tokens):
            synonyms = _TOKEN_SYNONYMS.get(token)
            if synonyms:
                augmented.update(synonyms)
        return augmented

    @staticmethod
    def _char_similarity_ratio(question: str, answer: str, language: str | None) -> float:
        if not question or not answer:
            return 0.0
        if language:
            lowered = language.lower()
            if lowered in TUNISIAN_DARIJA_CODES:
                question = normalize_darija(question)
                answer = normalize_darija(answer)
        return difflib.SequenceMatcher(None, question, answer, autojunk=False).ratio()
