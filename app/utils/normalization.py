from __future__ import annotations

import re

DARIJA_LATIN_REPLACEMENTS = {
    "ch": "sh",
    "gh": "gh",
    "kh": "kh",
}


def normalize_darija(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    for needle, repl in DARIJA_LATIN_REPLACEMENTS.items():
        lowered = lowered.replace(needle, repl)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()
