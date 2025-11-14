from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.core.config import get_settings


@dataclass
class FeedbackRecord:
    interaction_id: str
    rating: str
    outcome: float
    comment: str | None
    correction: str | None
    timestamp: str


class FeedbackRepository:
    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        resolved_path = Path(path) if path is not None else Path(settings.feedback_log_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = resolved_path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def submit(
        self,
        interaction_id: str,
        rating: str,
        comment: str | None = None,
        correction: str | None = None,
    ) -> FeedbackRecord:
        normalized_rating = rating.lower()
        if normalized_rating not in {"helpful", "unhelpful"}:
            raise ValueError("rating must be 'helpful' or 'unhelpful'")
        outcome = 1.0 if normalized_rating == "helpful" else 0.0
        record = FeedbackRecord(
            interaction_id=interaction_id,
            rating=normalized_rating,
            outcome=outcome,
            comment=comment,
            correction=correction,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        payload = asdict(record)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")
        return record

    def iter_records(self) -> Iterator[FeedbackRecord]:
        if not self._path.exists():
            return iter(())
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                yield FeedbackRecord(
                    interaction_id=data["interaction_id"],
                    rating=data.get("rating", ""),
                    outcome=float(data.get("outcome", 0.0)),
                    comment=data.get("comment"),
                    correction=data.get("correction"),
                    timestamp=data.get("timestamp", ""),
                )


def create_feedback_repository(path: Path | None = None) -> FeedbackRepository:
    return FeedbackRepository(path=path)
