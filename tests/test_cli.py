from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gpt2_rope.cli import app


def test_tokenizer_cli_workflow(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello grouped query attention\nhello rotary embeddings\n", encoding="utf-8")
    tokenizer_dir = tmp_path / "tokenizer"
    runner = CliRunner()

    trained = runner.invoke(
        app,
        [
            "tokenizer",
            "train",
            str(corpus),
            str(tokenizer_dir),
            "--vocab-size",
            "280",
        ],
    )
    assert trained.exit_code == 0, trained.output

    inspected = runner.invoke(
        app,
        ["tokenizer", "inspect", str(tokenizer_dir), "--text", "hello"],
    )
    assert inspected.exit_code == 0, inspected.output
    assert '"round_trip": "hello"' in inspected.output


def test_cli_help_exposes_all_workflows() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "tokenizer",
        "data",
        "pretrain",
        "finetune",
        "evaluate",
        "generate",
        "profile",
        "checkpoint",
    ):
        assert command in result.output

