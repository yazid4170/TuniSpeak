from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from app.core.config import get_settings
from app.services.language import TUNISIAN_DARIJA_CODES
from app.utils.normalization import normalize_darija


def parse_args() -> tuple[Path | None, bool]:
    parser = ArgumentParser(description="Ensure corpus language codes use canonical values")
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="Path to chunks.jsonl file; defaults to processed corpus",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing",
    )
    ns = parser.parse_args()
    return ns.file, ns.dry_run


def main() -> None:
    chunk_file, dry_run = parse_args()
    settings = get_settings()
    target = chunk_file or Path(settings.processed_data_dir) / "chunks.jsonl"
    if not target.exists():
        print(f"Corpus file not found at {target}")
        raise SystemExit(1)

    print(f"Loading corpus from {target}")
    records = _load_records(target)
    changed = 0

    for record in records:
        changed += _canonicalise(record)

    print(f"Canonicalised {changed} of {len(records)} records")

    if dry_run:
        print("Dry run requested; no changes written")
        return

    if changed == 0:
        print("Corpus already canonicalised; no write needed")
        return

    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("Corpus file updated")


def _load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _canonicalise(record: dict[str, object]) -> int:
    raw_lang = str(record.get("language", "unknown")).lower()
    if raw_lang in TUNISIAN_DARIJA_CODES:
        if record.get("language") != "aeb":
            record["language"] = "aeb"
            normalised = record.get("normalized_text")
            if isinstance(normalised, str) and normalised.strip():
                record["normalized_text"] = normalize_darija(normalised)
            else:
                text = record.get("text", "")
                if isinstance(text, str):
                    record["normalized_text"] = normalize_darija(text)
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
