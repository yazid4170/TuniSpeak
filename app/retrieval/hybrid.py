from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.models.schemas import DocumentChunk, SourceAttribution
from app.services.embedding import get_sentence_transformer
from app.services.reranker import get_reranker
from app.utils.text import tokenize_text

logger = structlog.get_logger(__name__)


@dataclass
class HybridRetriever:
    data_dir: Path = field(default_factory=lambda: Path(get_settings().processed_data_dir))
    index_dir: Path = field(default_factory=lambda: Path(get_settings().hybrid_index_path))

    def __post_init__(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._settings = get_settings()
        # Load all retrieval assets once so every request reuses the same encoder/index.
        self._encoder = get_sentence_transformer(self._settings.dense_model)
        self._chunks = self._load_chunks()
        self._curated_prefixes = tuple(self._settings.retriever_curated_prefixes)
        self._curated_boost = max(0.0, float(self._settings.retriever_curated_boost))
        if not self._chunks:
            logger.warning("retriever.empty_corpus", processed_dir=str(self.data_dir))
            self._bm25 = None
            self._embeddings = None
            self._reranker = None
            return
        self._bm25 = self._load_bm25()
        self._embeddings = self._load_embeddings()
        if self._bm25 is None or self._embeddings is None or self._embeddings.size == 0:
            self._reranker = None
        else:
            self._reranker = get_reranker(self._settings.cross_encoder_model)

    def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        if not self._chunks:
            return []

        # Score with both sparse BM25 and dense embeddings before fusing.
        bm25_scores = self._score_sparse(query)
        dense_scores = self._score_dense(query)

        combined = self._combine_scores(bm25_scores, dense_scores)
        combined = self._apply_curated_bias(combined)
        if combined.size == 0:
            return []

        top_indices = np.argsort(combined)[::-1][:top_k]
        candidate_chunks: list[DocumentChunk] = []
        candidate_scores: list[float] = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            score = float(combined[idx])
            enriched = chunk.model_copy(update={"score": score})
            candidate_chunks.append(enriched)
            candidate_scores.append(score)

        # Optional cross-encoder reranker refines the top candidates when available.
        if self._reranker and candidate_chunks:
            rerank_scores = self._reranker.score(query, candidate_chunks)
            if rerank_scores.size:
                rerank_norm = self._normalize(rerank_scores)
                combined_norm = self._normalize(np.asarray(candidate_scores, dtype=float))
                final_scores = 0.5 * combined_norm + 0.5 * rerank_norm
                ordering = np.argsort(final_scores)[::-1]
                reranked = [
                    candidate_chunks[i].model_copy(
                        update={"score": float(final_scores[i])}
                    )
                    for i in ordering
                ]
                return self._prioritise_curated(reranked)

        return self._prioritise_curated(candidate_chunks)

    def preview(self, limit: int = 10) -> list[DocumentChunk]:
        return self.retrieve(query="preview", top_k=limit)

    def to_attributions(self, chunks: list[DocumentChunk]) -> list[SourceAttribution]:
        return [
            SourceAttribution(
                document_id=chunk.document_id,
                title=f"Document {chunk.document_id}",
                snippet=chunk.text[:200],
                score=chunk.score or 0.0,
                url=chunk.source_path,
            )
            for chunk in chunks
        ]

    def _load_chunks(self) -> list[DocumentChunk]:
        # Ingestion writes all chunk metadata to this JSONL file.
        chunk_path = self.data_dir / "chunks.jsonl"
        if not chunk_path.exists():
            logger.warning("retriever.missing_chunks", path=str(chunk_path))
            return []
        chunks: list[DocumentChunk] = []
        with chunk_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunk = DocumentChunk.model_validate(record)
                chunks.append(chunk)
        logger.info("retriever.chunks_loaded", count=len(chunks), path=str(chunk_path))
        return chunks

    def _load_bm25(self) -> BM25Okapi | None:
        index_path = self.index_dir / "bm25.pkl"
        if not index_path.exists():
            logger.error("retriever.missing_bm25", path=str(index_path))
            return None
        with index_path.open("rb") as handle:
            bm25 = pickle.load(handle)
        if not isinstance(bm25, BM25Okapi):
            logger.error("retriever.invalid_bm25", path=str(index_path))
            return None
        return bm25

    def _load_embeddings(self) -> np.ndarray | None:
        embed_path = self.index_dir / "dense_embeddings.npy"
        texts = [chunk.text for chunk in self._chunks]
        if embed_path.exists():
            embeddings = np.load(embed_path)
            if embeddings.shape[0] == len(texts):
                return embeddings
            logger.warning(
                "retriever.embedding_mismatch",
                expected=len(texts),
                found=embeddings.shape[0],
                path=str(embed_path),
            )
        logger.error("retriever.missing_embeddings", path=str(embed_path))
        return None

    def _score_sparse(self, query: str) -> np.ndarray:
        # BM25 fails gracefully by returning zeros so dense retrieval can still run.
        if not self._bm25:
            return np.zeros(len(self._chunks), dtype=float)
        tokens = tokenize_text(query)
        if not tokens:
            return np.zeros(len(self._chunks), dtype=float)
        return np.array(self._bm25.get_scores(tokens))

    def _score_dense(self, query: str) -> np.ndarray:
        if (
            self._embeddings is None
            or not len(self._chunks)
            or self._embeddings.size == 0
        ):
            return np.zeros(len(self._chunks), dtype=float)
        query_vec = self._encoder.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.dot(self._embeddings, query_vec)

    def _combine_scores(self, sparse: np.ndarray, dense: np.ndarray) -> np.ndarray:
        # Require both signals; if one is missing keep output zero to trigger abstention upstream.
        if sparse.size == 0 or dense.size == 0:
            return np.zeros(len(self._chunks), dtype=float)
        sparse_norm = self._normalize(sparse)
        dense_norm = self._normalize(dense)
        return 0.5 * sparse_norm + 0.5 * dense_norm

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if not scores.size:
            return scores
        max_score = float(scores.max())
        min_score = float(scores.min())
        if max_score == min_score == 0.0:
            return np.zeros_like(scores)
        if max_score == min_score:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def _apply_curated_bias(self, scores: np.ndarray) -> np.ndarray:
        if not scores.size or self._curated_boost <= 0.0:
            return scores
        if not self._curated_prefixes:
            return scores
        adjusted = scores.copy()
        applied = False
        for idx, chunk in enumerate(self._chunks):
            if chunk.document_id.startswith(self._curated_prefixes):
                adjusted[idx] *= 1.0 + self._curated_boost
                applied = True
        return adjusted if applied else scores

    def _prioritise_curated(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        if not chunks or not self._curated_prefixes:
            return chunks
        curated: list[DocumentChunk] = []
        others: list[DocumentChunk] = []
        for chunk in chunks:
            if chunk.document_id.startswith(self._curated_prefixes):
                curated.append(chunk)
            else:
                others.append(chunk)
        if not curated:
            return chunks
        return curated + others
