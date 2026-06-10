from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpt2_rope.config import ExperimentConfig
from gpt2_rope.sweeps import (
    SweepConfig,
    assignment_overrides,
    enumerate_assignments,
    load_sweep_config,
    read_objective,
    run_sweep,
)

BASE_YAML = """
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


def _sweep(tmp_path: Path, **overrides: Any) -> SweepConfig:
    base = tmp_path / "base.yaml"
    base.write_text(BASE_YAML, encoding="utf-8")
    values: dict[str, Any] = {
        "base_config": base,
        "output_dir": tmp_path / "sweep",
        "parameters": {
            "training.learning_rate": [0.001, 0.0005],
            "training.warmup_steps": [1, 2],
        },
    }
    values.update(overrides)
    return SweepConfig.model_validate(values)


def test_grid_enumeration_is_exhaustive_and_sorted(tmp_path: Path) -> None:
    assignments = enumerate_assignments(_sweep(tmp_path))
    assert len(assignments) == 4
    assert {tuple(sorted(a.items())) for a in assignments} == {
        (("training.learning_rate", 0.001), ("training.warmup_steps", 1)),
        (("training.learning_rate", 0.001), ("training.warmup_steps", 2)),
        (("training.learning_rate", 0.0005), ("training.warmup_steps", 1)),
        (("training.learning_rate", 0.0005), ("training.warmup_steps", 2)),
    }


def test_random_search_is_seeded_and_sized(tmp_path: Path) -> None:
    sweep = _sweep(tmp_path, method="random", trials=3, seed=11)
    first = enumerate_assignments(sweep)
    second = enumerate_assignments(sweep)
    assert first == second
    assert len(first) == 3

    with pytest.raises(ValueError, match="trials"):
        _sweep(tmp_path, method="random")


def test_assignment_overrides_round_trip_json() -> None:
    overrides = assignment_overrides(
        {"training.learning_rate": 0.0005, "training.device": "cpu"}
    )
    assert overrides == [
        'training.device="cpu"',
        "training.learning_rate=0.0005",
    ]


def test_read_objective_returns_last_value(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "validation/loss": 3.0}),
                json.dumps({"step": 2, "train/loss": 2.0}),
                json.dumps({"step": 3, "validation/loss": 1.5}),
            ]
        ),
        encoding="utf-8",
    )
    assert read_objective(metrics, "validation/loss") == 1.5
    assert read_objective(metrics, "missing/metric") is None
    assert read_objective(tmp_path / "absent.jsonl", "validation/loss") is None


def test_run_sweep_writes_results_and_ranked_summary(tmp_path: Path) -> None:
    sweep = _sweep(tmp_path)

    def fake_train(config: ExperimentConfig) -> Path:
        run_dir = config.training.output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        # Deterministic fake objective: lower learning rate scores better.
        loss = config.training.learning_rate * 1000 + config.training.warmup_steps
        (run_dir / "metrics.jsonl").write_text(
            json.dumps({"step": 1, "validation/loss": loss}) + "\n",
            encoding="utf-8",
        )
        return run_dir

    results = run_sweep(sweep, train_fn=fake_train)
    assert len(results) == 4
    assert all(result.objective is not None for result in results)

    lines = (sweep.output_dir / "sweep_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4

    summary = json.loads((sweep.output_dir / "sweep_summary.json").read_text(encoding="utf-8"))
    assert summary["trials"] == 4
    best = summary["best"]
    assert best["assignment"]["training.learning_rate"] == 0.0005
    assert best["assignment"]["training.warmup_steps"] == 1
    ranked_objectives = [entry["objective"] for entry in summary["ranked"]]
    assert ranked_objectives == sorted(ranked_objectives)


def test_load_sweep_config_validates_root(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_sweep_config(bad)
