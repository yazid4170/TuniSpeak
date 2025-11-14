from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from app.models.schemas import DocumentChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> np.ndarray:
        if not chunks:
            return np.array([], dtype=float)
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)
        return np.asarray(scores, dtype=float)


@lru_cache(maxsize=2)
def get_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)
