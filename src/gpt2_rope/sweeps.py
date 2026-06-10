"""Local-first hyperparameter sweeps over dotted configuration overrides.

Trials run sequentially in per-trial run directories; results aggregate into
``sweep_results.jsonl`` and ``sweep_summary.json``. The same search spec maps
directly onto W&B sweeps or Optuna when a hosted/parallel optimizer is needed.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gpt2_rope.config_io import load_experiment_config
from gpt2_rope.training import train_pretraining

LOGGER = logging.getLogger("gpt2_rope")


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_config: Path
    output_dir: Path
    method: Literal["grid", "random"] = "grid"
    trials: int | None = Field(default=None, ge=1)
    seed: int = 0
    parameters: dict[str, list[Any]] = Field(min_length=1)
    objective: str = "validation/loss"
    minimize: bool = True

    @model_validator(mode="after")
    def validate_search(self) -> SweepConfig:
        if any(not values for values in self.parameters.values()):
            raise ValueError("every swept parameter needs at least one value")
        if self.method == "random" and self.trials is None:
            raise ValueError("random search requires trials")
        return self


def load_sweep_config(path: Path) -> SweepConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sweep specification root must be a mapping")
    return SweepConfig.model_validate(raw)


def enumerate_assignments(config: SweepConfig) -> list[dict[str, Any]]:
    """Expand the search space: full Cartesian product or seeded random draws."""
    keys = sorted(config.parameters)
    if config.method == "grid":
        product = itertools.product(*(config.parameters[key] for key in keys))
        assignments = [dict(zip(keys, combo, strict=True)) for combo in product]
        return assignments[: config.trials] if config.trials else assignments
    generator = random.Random(config.seed)
    assert config.trials is not None
    return [
        {key: generator.choice(config.parameters[key]) for key in keys}
        for _ in range(config.trials)
    ]


def assignment_overrides(assignment: dict[str, Any]) -> list[str]:
    """Render one assignment as ``--set``-style dotted overrides."""
    return [f"{key}={json.dumps(value)}" for key, value in sorted(assignment.items())]


def read_objective(metrics_path: Path, objective: str) -> float | None:
    """Return the last logged value of the objective metric, if any."""
    if not metrics_path.exists():
        return None
    value: float | None = None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if objective in record:
            value = float(record[objective])
    return value


@dataclass(slots=True)
class TrialResult:
    index: int
    assignment: dict[str, Any]
    run_dir: Path
    objective: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial": self.index,
            "assignment": self.assignment,
            "run_dir": str(self.run_dir),
            "objective": self.objective,
        }


def run_sweep(
    config: SweepConfig,
    train_fn: Callable[[Any], Path] | None = None,
) -> list[TrialResult]:
    """Run every trial sequentially and aggregate ranked results."""
    train = train_fn or train_pretraining
    assignments = enumerate_assignments(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.output_dir / "sweep_results.jsonl"
    results: list[TrialResult] = []
    with results_path.open("a", encoding="utf-8") as results_file:
        for index, assignment in enumerate(assignments):
            trial_dir = config.output_dir / f"trial-{index:04d}"
            overrides = [
                *assignment_overrides(assignment),
                f"training.output_dir={json.dumps(str(trial_dir))}",
            ]
            experiment = load_experiment_config(config.base_config, overrides)
            LOGGER.info("sweep trial %d/%d: %s", index + 1, len(assignments), assignment)
            run_dir = train(experiment)
            objective = read_objective(run_dir / "metrics.jsonl", config.objective)
            result = TrialResult(index, assignment, run_dir, objective)
            results_file.write(json.dumps(result.as_dict(), sort_keys=True) + "\n")
            results_file.flush()
            results.append(result)

    scored = [result for result in results if result.objective is not None]
    scored.sort(key=lambda result: result.objective or 0.0, reverse=not config.minimize)
    summary = {
        "objective": config.objective,
        "minimize": config.minimize,
        "trials": len(results),
        "ranked": [result.as_dict() for result in scored],
        "best": scored[0].as_dict() if scored else None,
    }
    (config.output_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
