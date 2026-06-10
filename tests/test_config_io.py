from __future__ import annotations

from pathlib import Path

import pytest

from gpt2_rope.config_io import load_experiment_config

MINIMAL_YAML = """
model:
  vocab_size: 300
  context_length: 16
  d_model: 16
  num_layers: 1
  num_heads: 2
  num_kv_heads: 1

data:
  train_path: data/train.bin
  tokenizer_dir: tokenizer
"""


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(MINIMAL_YAML, encoding="utf-8")
    return path


def test_load_without_overrides(tmp_path: Path) -> None:
    config = load_experiment_config(_write_config(tmp_path))
    assert config.model.vocab_size == 300
    assert config.training.max_steps == 100_000


def test_override_parses_json_values(tmp_path: Path) -> None:
    config = load_experiment_config(
        _write_config(tmp_path),
        [
            "training.max_steps=20",
            "training.learning_rate=0.001",
            "model.gradient_checkpointing=true",
            'training.device="cpu"',
        ],
    )
    assert config.training.max_steps == 20
    assert config.training.learning_rate == pytest.approx(0.001)
    assert config.model.gradient_checkpointing is True
    assert config.training.device == "cpu"


def test_override_falls_back_to_string(tmp_path: Path) -> None:
    config = load_experiment_config(
        _write_config(tmp_path),
        ["training.output_dir=runs/custom"],
    )
    assert config.training.output_dir == Path("runs/custom")


def test_override_creates_nested_sections(tmp_path: Path) -> None:
    config = load_experiment_config(
        _write_config(tmp_path),
        ["monitoring.log_every=5"],
    )
    assert config.monitoring.log_every == 5


def test_override_requires_key_value_form(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="key=value"):
        load_experiment_config(_write_config(tmp_path), ["training.max_steps"])


def test_override_through_scalar_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-mapping"):
        load_experiment_config(
            _write_config(tmp_path),
            ["model.vocab_size.nested=1"],
        )


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_experiment_config(_write_config(tmp_path), ["model.head_count=4"])
