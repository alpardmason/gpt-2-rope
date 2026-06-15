from __future__ import annotations

from pathlib import Path

import torch
import yaml
from typer.testing import CliRunner

from gpt2_rope.cli import app
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT


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
        "benchmark",
        "checkpoint",
    ):
        assert command in result.output


def test_inference_benchmark_cli_writes_json_report(tmp_path: Path) -> None:
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "vocab_size": 97,
                    "context_length": 16,
                    "d_model": 16,
                    "num_layers": 1,
                    "num_heads": 2,
                    "num_kv_heads": 1,
                },
                "data": {
                    "train_path": str(tmp_path / "train.bin"),
                    "tokenizer_dir": str(tmp_path / "tokenizer"),
                    "sequence_length": 8,
                },
                "training": {"device": "cpu"},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save(
        GPT(
            ModelConfig(
                vocab_size=97,
                context_length=16,
                d_model=16,
                num_layers=1,
                num_heads=2,
                num_kv_heads=1,
            )
        ).state_dict(),
        checkpoint / "model.pt",
    )
    output = tmp_path / "benchmark.json"

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "inference",
            str(config_path),
            str(checkpoint),
            "--output",
            str(output),
            "--prompt-tokens",
            "4",
            "--generated-tokens",
            "3",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert '"kv_cache_reduction_factor_vs_mha": 2.0' in output.read_text(encoding="utf-8")
