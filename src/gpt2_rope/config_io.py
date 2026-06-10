"""YAML loading and explicit dotted-key experiment overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from gpt2_rope.config import ExperimentConfig


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def load_experiment_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override {override!r} must have key=value form")
        dotted_key, value = override.split("=", 1)
        cursor: dict[str, Any] = raw
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            nested = cursor.setdefault(part, {})
            if not isinstance(nested, dict):
                raise ValueError(f"cannot override through non-mapping key {part!r}")
            cursor = nested
        cursor[parts[-1]] = _parse_value(value)
    return ExperimentConfig.model_validate(raw)

