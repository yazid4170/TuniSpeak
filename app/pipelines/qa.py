from __future__ import annotations

from dataclasses import dataclass, field

from app.models.schemas import DocumentChunk, PipelineAnswer
from app.retrieval.hybrid import HybridRetriever
from app.services.language import LanguageDetector, TUNISIAN_DARIJA_CODES
from app.services.qa import ExtractiveQASystem
from app.services.synthesis import AnswerSynthesizer


@dataclass
class QAPipeline:
    retriever: HybridRetriever = field(default_factory=HybridRetriever)
    qa_engine: ExtractiveQASystem = field(default_factory=ExtractiveQASystem)
    synthesizer: AnswerSynthesizer = field(default_factory=AnswerSynthesizer)
    language_detector: LanguageDetector = field(default_factory=LanguageDetector)

    def run(self, question: str, top_k: int = 5) -> PipelineAnswer:
        # Retrieval first: hybrid retriever already handles BM25 + dense fusion.
        chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            return PipelineAnswer(
                answer="Aucune réponse trouvée.",
                sources=[],
                confidence=0.0,
                abstain=True,
                reason="Aucun document pertinent trouvé.",
            )

        detection = self.language_detector.detect(question)
        preferred_language = detection.language if detection.language != "unknown" else None
        if preferred_language:
            # Bias retrieval list toward user language without dropping high-scoring alternates.
            chunks = self._prioritise_language(chunks, preferred_language)
            language_filtered = self._filter_by_language(chunks, preferred_language)
            if language_filtered:
                chunks = language_filtered

        qa_result = self.qa_engine.answer(
            question,
            chunks,
            language=preferred_language,
        )

        if qa_result.chunk_index is not None and 0 <= qa_result.chunk_index < len(chunks):
            best_chunk = chunks[qa_result.chunk_index]
            ordered_chunks = [best_chunk] + [
                chunk for idx, chunk in enumerate(chunks) if idx != qa_result.chunk_index
            ]
        else:
            ordered_chunks = list(chunks)

        sources = self.retriever.to_attributions(ordered_chunks)

        if qa_result.abstain:
            # When extraction abstains we still build a language-aware summary to feed the LLM.
            summary_chunks = ordered_chunks
            if preferred_language:
                filtered = [
                    chunk
                    for chunk in ordered_chunks
                    if (chunk.language or "").lower() == preferred_language
                ]
                if filtered:
                    summary_chunks = filtered
            fallback = self.synthesizer.summarise(question, summary_chunks)
            generative = self.synthesizer.generate(
                question=question,
                chunks=summary_chunks,
                draft=fallback,
                language=preferred_language,
            )
            abstain_message = self.qa_engine.get_abstain_message(preferred_language)
            answer_text = generative or fallback or abstain_message
            return PipelineAnswer(
                answer=answer_text,
                sources=sources,
                confidence=qa_result.confidence,
                abstain=True,
                reason=qa_result.reason or abstain_message,
            )

        # Otherwise we let the generator polish the extract while keeping citations aligned.
        refined = self.synthesizer.generate(
            question=question,
            chunks=ordered_chunks,
            draft=qa_result.answer,
            language=preferred_language,
        )
        return PipelineAnswer(
            answer=refined or qa_result.answer,
            sources=sources,
            confidence=qa_result.confidence,
            abstain=False,
        )

    def _prioritise_language(
        self, chunks: list[DocumentChunk], language: str
    ) -> list[DocumentChunk]:
        preferred = self._canonical_language(language)

        def rank(chunk: DocumentChunk) -> tuple[int, float]:
            chunk_lang = self._canonical_language(chunk.language or "unknown")
            if chunk_lang == preferred:
                return (0, -(chunk.score or 0.0))
            if preferred == "ar" and chunk_lang == "aeb":
                return (1, -(chunk.score or 0.0))
            if preferred == "aeb" and chunk_lang == "ar":
                return (1, -(chunk.score or 0.0))
            return (2, -(chunk.score or 0.0))

        return sorted(chunks, key=rank)

    def _filter_by_language(
        self,
        chunks: list[DocumentChunk],
        language: str,
    ) -> list[DocumentChunk]:
        preferred = self._canonical_language(language)
        matched: list[DocumentChunk] = []
        for chunk in chunks:
            chunk_lang = self._canonical_language(chunk.language or "unknown")
            if chunk_lang == preferred:
                matched.append(chunk)
            elif preferred == "ar" and chunk_lang == "aeb":
                matched.append(chunk)
            elif preferred == "aeb" and chunk_lang == "ar":
                matched.append(chunk)
        return matched

    @staticmethod
    def _canonical_language(code: str) -> str:
        lowered = code.lower()
        if lowered in TUNISIAN_DARIJA_CODES:
            return "aeb"
        return lowered
