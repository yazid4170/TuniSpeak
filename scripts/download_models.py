from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import typer
# FIX: older huggingface_hub doesn't have HfHubError → use universal fallback
from huggingface_hub import snapshot_download


DEFAULT_TINY_LLAMA_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

app = typer.Typer(help="Download the model artifacts required to run TuniSpeak.")


@dataclass
class ModelSpec:
    name: str
    repo_id: str
    target_dir: Path
    allow_patterns: tuple[str, ...]


def _download(spec: ModelSpec, force: bool) -> bool:
    target = spec.target_dir
    if target.exists():
        if force:
            shutil.rmtree(target)
        else:
            typer.secho(
                f"{spec.name} already present at {target}", fg=typer.colors.GREEN
            )
            return True

    typer.echo(f"Downloading {spec.name} from {spec.repo_id} ...")
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            allow_patterns=list(spec.allow_patterns),
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:  # FIX: catch all errors
        typer.secho(
            f"Failed to fetch {spec.name}: {exc}", fg=typer.colors.RED, err=True
        )
        return False

    typer.secho(f"Downloaded {spec.name} → {target}", fg=typer.colors.GREEN)
    return True


def _collect_specs() -> Iterable[ModelSpec]:
    yield ModelSpec(
        name="TinyLlama base",
        repo_id=DEFAULT_TINY_LLAMA_REPO,
        target_dir=Path("models/base/tinyllama-1.1b-chat"),
        allow_patterns=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "*.md",
        ),
    )


@app.command()
def main(
    force: bool = typer.Option(
        False,
        "--force",
        help="Redownload the TinyLlama base even if the target folder already exists.",
    ),
) -> None:
    successes = 0
    for spec in _collect_specs():
        if _download(spec, force=force):
            successes += 1

    if successes == 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
