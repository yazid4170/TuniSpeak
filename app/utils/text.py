from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"[.!؟\?]\s+")


def segment_text(text: str, max_tokens: int = 220) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    segments: list[str] = []
    buffer: list[str] = []
    token_count = 0

    for paragraph in paragraphs:
        tokens = paragraph.split()
        if not tokens:
            continue
        if token_count + len(tokens) > max_tokens and buffer:
            segments.append(" ".join(buffer).strip())
            buffer = [paragraph]
            token_count = len(tokens)
            continue
        buffer.append(paragraph)
        token_count += len(tokens)

    if buffer:
        segments.append(" ".join(buffer).strip())

    cleaned = [segment for segment in segments if segment]
    return cleaned if cleaned else [text.strip()]


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [part.strip() for part in parts if part.strip()]
