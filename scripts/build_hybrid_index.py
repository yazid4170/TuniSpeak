from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import typer
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.services.embedding import get_sentence_transformer
from app.utils.text import tokenize_text

app = typer.Typer(help="Build hybrid retrieval index")


def execute(force: bool = False) -> None:
    settings = get_settings()
    processed_dir = Path(settings.processed_data_dir)
    chunk_path = processed_dir / "chunks.jsonl"
    if not chunk_path.exists():
        typer.echo("No processed chunks found. Run the ingestion pipeline first.")
        raise typer.Exit(code=1)

    records = _load_records(chunk_path)
    if not records:
        typer.echo("Chunk file is empty. Nothing to index.")
        raise typer.Exit(code=1)

    texts = [record["text"] for record in records]

    hybrid_dir = Path(settings.hybrid_index_path)
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    embed_path = hybrid_dir / "dense_embeddings.npy"

    if embed_path.exists() and not force:
        stored = np.load(embed_path)
        if stored.shape[0] == len(texts):
            typer.echo(f"Existing embeddings match corpus size ({len(texts)} chunks). Skipping.")
            return

    typer.echo(f"Building BM25 index over {len(texts)} chunks ...")
    tokenized = [tokenize_text(text) for text in texts]
    bm25 = BM25Okapi(tokenized)
    with (hybrid_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(bm25, handle)

    typer.echo(f"Encoding {len(texts)} chunks with {settings.dense_model} ...")
    model = get_sentence_transformer(settings.dense_model)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(embed_path, embeddings)

    metadata = {
        "chunk_count": len(texts),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_model": settings.dense_model,
        "chunk_path": str(chunk_path.resolve()),
        "bm25_doc_count": len(texts),
    }
    (hybrid_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    typer.echo(f"Hybrid index updated at {hybrid_dir}")


@app.command()
def run(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Recompute embeddings even if cached",
        is_flag=True,
    )
) -> None:
    execute(force=force)


def _load_records(chunk_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with chunk_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


if __name__ == "__main__":  # pragma: no cover
    app()
