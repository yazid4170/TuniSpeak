from __future__ import annotations

from pathlib import Path
from typing import Iterable

import typer

from app.core.config import get_settings
from app.services.ingestion import DocumentIngestionService

app = typer.Typer(help="Document ingestion utilities")


@app.command()
def run(directory: Path = typer.Argument(Path("data/raw"), exists=True)) -> None:
    """Ingest documents from DIRECTORY into the processed corpus."""
    settings = get_settings()
    service = DocumentIngestionService(settings=settings)
    files = list(_iter_documents(directory))
    typer.echo(f"Found {len(files)} documents to ingest")
    for path in files:
        service.ingest(path)
    service.flush()
    typer.echo("Ingestion complete")


def _iter_documents(directory: Path) -> Iterable[Path]:
    exts = {".pdf", ".docx", ".txt", ".md"}
    for path in directory.rglob("*"):
        if path.suffix.lower() in exts:
            yield path


if __name__ == "__main__":  # pragma: no cover
    app()
