from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from app.core.config import get_settings


@dataclass
class TelemetryRecord:
    interaction_id: str
    question: str
    normalized_question: str
    language: str
    answer: str
    confidence: float
    abstain: bool
    sources: Sequence[str]
    timestamp: str


class TelemetryLogger:
    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        resolved_path = Path(path) if path is not None else Path(settings.telemetry_log_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = resolved_path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def log(self, record: TelemetryRecord) -> None:
        payload = asdict(record)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")

    def log_answer(
        self,
        interaction_id: str,
        question: str,
        normalized_question: str,
        language: str,
        answer: str,
        confidence: float,
        abstain: bool,
        sources: Sequence[str],
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        record = TelemetryRecord(
            interaction_id=interaction_id,
            question=question,
            normalized_question=normalized_question,
            language=language,
            answer=answer,
            confidence=confidence,
            abstain=abstain,
            sources=list(sources),
            timestamp=timestamp,
        )
        self.log(record)

    def iter_records(self) -> Iterator[TelemetryRecord]:
        if not self._path.exists():
            return iter(())
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                yield TelemetryRecord(
                    interaction_id=data["interaction_id"],
                    question=data.get("question", ""),
                    normalized_question=data.get("normalized_question", ""),
                    language=data.get("language", ""),
                    answer=data.get("answer", ""),
                    confidence=float(data.get("confidence", 0.0)),
                    abstain=bool(data.get("abstain", False)),
                    sources=data.get("sources", []) or [],
                    timestamp=data.get("timestamp", ""),
                )

    def has_interaction(self, interaction_id: str) -> bool:
        return any(record.interaction_id == interaction_id for record in self.iter_records())


def create_telemetry_logger(path: Path | None = None) -> TelemetryLogger:
    return TelemetryLogger(path=path)
