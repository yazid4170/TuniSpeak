from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import LangDetectException, detect_langs

from app.models.schemas import LanguageDetectionResult
from app.utils.normalization import normalize_darija

SUPPORTED_LANGS = {"fr": "French", "ar": "Arabic", "aeb": "Tunisian Darija"}
TUNISIAN_DARIJA_CODES = {"aeb", "ary", "ar-dz", "ar-tn"}
ROMANIZED_DARIJA_MARKERS = {
    "kif",
    "kifeh",
    "kifesh",
    "kifach",
    "kifech",
    "chno",
    "shnu",
    "chnou",
    "shnou",
    "barcha",
    "famma",
    "fama",
    "mouch",
    "mawjoud",
    "nsajel",
    "bech",
    "fel",
    "ena",
    "inti",
}
ROMANIZED_DARIJA_STRONG_MARKERS = {
    "kifeh",
    "kifesh",
    "kifach",
    "kifech",
    "chno",
    "shnu",
    "chnou",
    "shnou",
    "barcha",
    "famma",
    "mouch",
    "nsajel",
    "bech",
}


@dataclass
class DetectionCandidate:
    lang: str
    prob: float


class LanguageDetector:
    def detect(self, text: str) -> LanguageDetectionResult:
        cleaned = text.strip()
        if not cleaned:
            return LanguageDetectionResult(language="unknown", normalized="", confidence=0.0)

        looks_like_arabizi = self._looks_like_arabizi(cleaned)
        has_arabic_script = self._has_arabic_script(cleaned)
        has_latin_letters = bool(re.search(r"[A-Za-z]", cleaned))
        has_digits = bool(re.search(r"\d", cleaned))
        looks_like_mixed_darija = has_arabic_script and (has_latin_letters or has_digits)
        looks_like_romanized_darija = self._looks_like_romanized_darija(cleaned)

        try:
            candidates = detect_langs(cleaned)
        except LangDetectException:
            candidates = []

        if looks_like_arabizi and not has_arabic_script:
            lang = "aeb"
            confidence = 0.8
        elif looks_like_mixed_darija:
            lang = "aeb"
            confidence = max(0.7, 0.6 if not candidates else max(c.prob for c in candidates))
        elif looks_like_romanized_darija and not has_arabic_script:
            lang = "aeb"
            confidence = max(0.75, 0.6 if not candidates else max(c.prob for c in candidates))
        elif not candidates:
            return LanguageDetectionResult(language="unknown", normalized=cleaned, confidence=0.0)
        else:
            top = max(candidates, key=lambda c: c.prob)
            lang = top.lang
            confidence = top.prob

        normalized = cleaned

        if lang == "ar":
            normalized = self._normalise_arabic(cleaned)
        if lang in TUNISIAN_DARIJA_CODES:
            lang = "aeb"
            normalized = normalize_darija(cleaned)
        elif looks_like_arabizi or looks_like_romanized_darija:
            lang = "aeb"
            normalized = normalize_darija(cleaned)

        return LanguageDetectionResult(language=lang, normalized=normalized, confidence=confidence)

    def _normalise_arabic(self, text: str) -> str:
        # Basic normalizing of Arabic script to reduce diacritics and Tatweel.
        text = re.sub("[\u064B-\u065F]", "", text)  # Strip diacritics
        text = text.replace("\u0640", "")  # Remove tatweel
        return text

    def _looks_like_arabizi(self, text: str) -> bool:
        if re.search(r"[\u0600-\u06FF]", text):
            return False
        return bool(re.search(r"(\d|3|7|9)", text) and re.search(r"[A-Za-z]", text))

    def _has_arabic_script(self, text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text))

    def _looks_like_romanized_darija(self, text: str) -> bool:
        if re.search(r"[\u0600-\u06FF]", text):
            return False
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        if not tokens:
            return False
        hits = sum(token in ROMANIZED_DARIJA_MARKERS for token in tokens)
        strong_hit = any(token in ROMANIZED_DARIJA_STRONG_MARKERS for token in tokens)
        ratio = hits / max(len(tokens), 1)
        return strong_hit or (hits >= 2 and ratio >= 0.25)
