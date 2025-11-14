from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def get_sentence_transformer(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)
