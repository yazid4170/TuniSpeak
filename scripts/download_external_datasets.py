from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

from datasets import load_dataset

logger = logging.getLogger("tunispeak.external_data")

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "external"


class DatasetRecord(dict):
    """JSON-serialisable QA record with provenance metadata."""


def _write_jsonl(records: Iterable[DatasetRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _slice_arg(split: str, limit: int | None) -> str:
    if limit is None:
        return split
    return f"{split}[:{limit}]"


def _mlqa_records(limit: int | None) -> Iterator[DatasetRecord]:
    split = _slice_arg("validation", limit)
    dataset = load_dataset(
        "mlqa",
        "mlqa.ar.ar",
        split=split,
        trust_remote_code=True,
    )
    for example in dataset:
        yield DatasetRecord(
            id=str(example["id"]),
            question=example["question"],
            context=example["context"],
            answers=example["answers"]["text"],
            answer_starts=example["answers"]["answer_start"],
            metadata={
                "language": "ar",
                "source_dataset": "mlqa.ar.ar",
                "split": "validation",
                "license": "CC BY-SA 3.0",
                "source_url": "https://github.com/facebookresearch/MLQA",
            },
        )


def _piaf_records(limit: int | None) -> Iterator[DatasetRecord]:
    split = _slice_arg("train", limit)
    dataset = load_dataset(
        "etalab-ia/piaf",
        "plain_text",
        split=split,
        trust_remote_code=True,
    )
    for example in dataset:
        yield DatasetRecord(
            id=str(example.get("id")),
            question=example["question"],
            context=example["context"],
            answers=example["answers"]["text"],
            answer_starts=example["answers"]["answer_start"],
            metadata={
                "language": "fr",
                "source_dataset": "etalab-ia/piaf",
                "split": "train",
                "license": "Etalab Open Licence 2.0",
                "source_url": "https://huggingface.co/datasets/etalab-ia/piaf",
            },
        )


def _tydiqa_records(limit: int | None) -> Iterator[DatasetRecord]:
    dataset = load_dataset(
        "tydiqa",
        "secondary_task",
        split="validation",
        trust_remote_code=True,
    )
    count = 0
    for example in dataset:
        if not str(example.get("id", "")).startswith("arabic-"):
            continue
        yield DatasetRecord(
            id=str(example["id"]),
            question=example["question"],
            context=example["context"],
            answers=example["answers"]["text"],
            answer_starts=example["answers"]["answer_start"],
            metadata={
                "language": "ar",
                "source_dataset": "tydiqa.secondary_task",
                "split": "validation",
                "license": "CC BY 4.0",
                "source_url": "https://huggingface.co/datasets/tydiqa",
            },
        )
        count += 1
        if limit is not None and count >= limit:
            break


DATASET_LOADERS = {
    "mlqa": ("mlqa_ar_validation.jsonl", _mlqa_records),
    "piaf": ("piaf_train.jsonl", _piaf_records),
    "tydiqa": ("tydiqa_ar_validation.jsonl", _tydiqa_records),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download curated external QA datasets for augmentation and benchmarking.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="append",
        choices=sorted(DATASET_LOADERS.keys()),
        help="Dataset key to download (default: all). Repeat flag to select multiple.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT,
        help="Destination directory for the exported JSONL files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples per dataset (omit to download the full split).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")

    selected = args.dataset or list(DATASET_LOADERS.keys())
    for name in selected:
        filename, loader = DATASET_LOADERS[name]
        print(f"Downloading {name} dataset...", flush=True)
        records = list(loader(args.limit))
        if not records:
            print(f"[warn] no records materialised for {name}; skipping write", flush=True)
            continue
        destination = args.output_dir / name / filename
        _write_jsonl(records, destination)
        print(f"[done] wrote {len(records)} records to {destination}", flush=True)


if __name__ == "__main__":
    main()
