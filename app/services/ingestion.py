from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from docx import Document as DocxDocument
from pypdf import PdfReader

from app.core.config import Settings
from app.services.language import LanguageDetector, TUNISIAN_DARIJA_CODES
from app.utils.text import segment_text

logger = structlog.get_logger(__name__)


@dataclass
class DocumentIngestionService:
    settings: Settings
    buffer: list[dict[str, Any]] = field(default_factory=list)
    detector: LanguageDetector = field(default_factory=LanguageDetector)

    def ingest(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            self._ingest_jsonl(path)
            return

        # Non-JSONL files get normalised into text then segmented before staging.
        text = self._read_text(path)
        if not text.strip():
            logger.warning("document.empty", document=str(path))
            return

        chunks = segment_text(text)
        for index, chunk_text in enumerate(chunks):
            self._stage_chunk(path, index, chunk_text)
        logger.info("document.staged", document=str(path), chunks=len(chunks))

    def flush(self) -> None:
        if not self.buffer:
            logger.info("ingestion.flush", status="noop")
            return

        target_dir = Path(self.settings.processed_data_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = target_dir / "chunks.jsonl"
        existing: dict[tuple[str, str], dict[str, Any]] = {}
        # We keep previously ingested chunks to allow idempotent updates per chunk id.
        if chunk_path.exists():
            with chunk_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = (record["document_id"], record["chunk_id"])
                    existing[key] = record

        for record in self.buffer:
            self._canonicalise_record(record)
            key = (record["document_id"], record["chunk_id"])
            existing[key] = record

        with chunk_path.open("w", encoding="utf-8") as handle:
            for record in existing.values():
                self._canonicalise_record(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "ingestion.flush",
            chunks_written=len(self.buffer),
            total=len(existing),
            output=str(chunk_path.resolve()),
        )
        self.buffer.clear()

    def _ingest_jsonl(self, path: Path) -> None:
        count = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                        logger.warning(
                            "document.jsonl_invalid",
                            document=str(path),
                            line=index,
                            error=str(exc),
                        )
                        continue

                    chunk_text = self._render_jsonl_entry(data)
                    if not chunk_text:
                        continue
                    self._stage_chunk(path, index, chunk_text)
                    count += 1
        except OSError as exc:  # pragma: no cover - defensive
            logger.error("document.read_failed", document=str(path), error=str(exc))
            return

        if count == 0:
            logger.warning("document.empty", document=str(path))
            return

        logger.info("document.staged", document=str(path), chunks=count)

    def _read_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                return self._read_pdf(path)
            if suffix == ".docx":
                return self._read_docx(path)
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("document.read_failed", document=str(path), error=str(exc))
            return ""

    def _read_pdf(self, path: Path) -> str:
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted:
                parts.append(extracted)
        return "\n".join(parts)

    def _read_docx(self, path: Path) -> str:
        document = DocxDocument(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    @staticmethod
    def _canonicalise_record(record: dict[str, Any]) -> None:
        language = str(record.get("language", "unknown")).lower()
        if language in TUNISIAN_DARIJA_CODES:
            record["language"] = "aeb"

    def _stage_chunk(self, path: Path, index: int, chunk_text: str) -> None:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            return
        # Detect language per chunk so retrieval can prioritise user-preferred dialects later.
        detection = self.detector.detect(chunk_text)
        record = {
            "document_id": path.stem,
            "chunk_id": f"{path.stem}-{index:04d}",
            "text": chunk_text,
            "language": detection.language,
            "normalized_text": detection.normalized,
            "source_path": str(path.resolve()),
        }
        self.buffer.append(record)

    @staticmethod
    def _render_jsonl_entry(data: Any) -> str:
        if isinstance(data, dict) and "messages" in data:
            lines: list[str] = []
            for message in data.get("messages", []):
                if not isinstance(message, dict):
                    continue
                role = message.get("role", "").strip().lower()
                content = message.get("content", "").strip()
                if not content:
                    continue
                prefix = "Assistant" if role == "assistant" else "Question" if role == "user" else role.title() or "Message"
                lines.append(f"{prefix}: {content}")
            return "\n".join(lines).strip()
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False)
        if isinstance(data, (list, tuple)):
            return "\n".join(str(item) for item in data)
        return str(data)
