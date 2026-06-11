"""Create and check reimplementation kata branches.

``start`` copies a kata skeleton over its production module on a fresh
``kata/<name>`` branch so the diff against ``main`` is exactly the code to
rebuild. ``check`` runs the kata's oracle tests plus ruff. See
``katas/README.md`` for rules.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Kata:
    skeleton: Path
    target: Path
    oracle: tuple[str, ...]


KATAS: dict[str, Kata] = {
    "rope": Kata(
        skeleton=Path("katas/rope/skeleton_rope.py"),
        target=Path("src/gpt2_rope/rope.py"),
        oracle=("tests/test_model.py",),
    ),
    "kv-cache": Kata(
        skeleton=Path("katas/kv-cache/skeleton_generation.py"),
        target=Path("src/gpt2_rope/generation.py"),
        oracle=("tests/test_lora_generation.py", "tests/test_serving.py"),
    ),
    "gqa": Kata(
        skeleton=Path("katas/gqa/skeleton_model.py"),
        target=Path("src/gpt2_rope/model.py"),
        oracle=("tests/test_model.py", "tests/test_lora_generation.py"),
    ),
    "checkpoint": Kata(
        skeleton=Path("katas/checkpoint/skeleton_checkpoint.py"),
        target=Path("src/gpt2_rope/checkpoint.py"),
        oracle=(
            "tests/test_checkpoint.py",
            "tests/test_training.py::test_resume_reproduces_uninterrupted_training",
        ),
    ),
    "bpe": Kata(
        skeleton=Path("katas/bpe/skeleton_tokenizer.py"),
        target=Path("src/gpt2_rope/tokenizer.py"),
        oracle=("tests/test_tokenizer.py", "tests/test_cli.py"),
    ),
    "dpo-loss": Kata(
        skeleton=Path("katas/dpo-loss/skeleton_dpo.py"),
        target=Path("src/gpt2_rope/dpo.py"),
        oracle=("tests/test_dpo.py",),
    ),
}

app = typer.Typer(help="Manage reimplementation kata branches (see katas/README.md).")


def _git(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=capture,
        text=True,
    )
    return result.stdout if capture else ""


def _resolve(name: str) -> Kata:
    kata = KATAS.get(name)
    if kata is None:
        known = ", ".join(sorted(KATAS))
        typer.echo(f"unknown kata {name!r}; available: {known}", err=True)
        raise typer.Exit(code=2)
    return kata


def _oracle_command(kata: Kata) -> str:
    return "UV_CACHE_DIR=.uv-cache uv run pytest " + " ".join(kata.oracle)


@app.command()
def start(name: Annotated[str, typer.Argument(help="Kata name, e.g. rope.")]) -> None:
    """Create branch kata/<name> with the skeleton replacing the module."""
    kata = _resolve(name)
    branch = f"kata/{name}"
    if _git("status", "--porcelain", "--untracked-files=no", capture=True).strip():
        typer.echo("tracked files have changes; commit or stash them first", err=True)
        raise typer.Exit(code=1)
    branches = _git("branch", "--list", branch, capture=True).strip()
    if branches:
        typer.echo(f"branch {branch} already exists; delete or rename it first", err=True)
        raise typer.Exit(code=1)
    _git("checkout", "-b", branch)
    shutil.copyfile(REPO_ROOT / kata.skeleton, REPO_ROOT / kata.target)
    _git("add", str(kata.target))
    _git("commit", "-m", f"kata({name}): gut {kata.target} for reimplementation")
    typer.echo(f"branch {branch} ready; {kata.target} is yours to rebuild.")
    typer.echo(f"assignment: katas/{name}/README.md")
    typer.echo(f"oracle:     {_oracle_command(kata)}")


@app.command()
def check(name: Annotated[str, typer.Argument(help="Kata name, e.g. rope.")]) -> None:
    """Run the kata's oracle tests and ruff; exit non-zero while red."""
    kata = _resolve(name)
    failures: list[str] = []
    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", *kata.oracle],
        cwd=REPO_ROOT,
    )
    if pytest_result.returncode != 0:
        failures.append("pytest")
    ruff_result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(kata.target)],
        cwd=REPO_ROOT,
    )
    if ruff_result.returncode != 0:
        failures.append("ruff")
    if failures:
        typer.echo(f"RED: {', '.join(failures)} failing for kata {name}")
        raise typer.Exit(code=1)
    typer.echo(f"GREEN: oracle and lint pass for kata {name}.")
    typer.echo("Run the full gates (pytest, mypy, ruff) and then review with:")
    typer.echo(f"  git diff main -- {kata.target}")


if __name__ == "__main__":
    app()
