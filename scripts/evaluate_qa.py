from __future__ import annotations

from pathlib import Path

import typer

from app.core.config import get_settings
from app.evaluation.runner import EvaluationRunner, save_summary

app = typer.Typer(help="Evaluate QA pipeline against labelled dataset")


def execute(dataset: Path, limit: int | None = None) -> Path:
    settings = get_settings()
    runner = EvaluationRunner()
    summary = runner.evaluate(dataset_path=dataset, limit=limit)
    return save_summary(summary, Path(settings.eval_output_dir))


@app.command()
def run(
    dataset: Path = typer.Option(
        Path("data/faq/mini_faq.jsonl"),
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to JSONL dataset with questions, answers, and relevance metadata.",
    ),
    limit: int | None = typer.Option(
        None, help="Optional limit on number of examples to evaluate"
    ),
) -> None:
    output_path = execute(dataset=dataset, limit=limit)
    typer.echo(f"Evaluation complete. Summary saved to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
