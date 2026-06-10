"""Resilient local metrics with optional hosted experiment tracking."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger("gpt2_rope")


class MetricLogger:
    """Write every metric locally; optional integrations are best-effort."""

    def __init__(
        self,
        run_dir: Path,
        *,
        tensorboard: bool = True,
        wandb_project: str | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.run_dir = run_dir
        self.writer: SummaryWriter | None = None
        self.wandb_run: Any | None = None
        if not enabled:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        if tensorboard:
            self.writer = SummaryWriter(run_dir / "tb")
        if wandb_project:
            try:
                wandb = importlib.import_module("wandb")
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    dir=run_dir,
                    config=config,
                )
            except Exception:
                LOGGER.exception("W&B initialization failed; continuing with local metrics")

    def log(self, step: int, metrics: Mapping[str, float | int | str]) -> None:
        if not self.enabled:
            return
        record = {"step": step, **metrics}
        self.metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.metrics_file.flush()
        message = " ".join(f"{key}={value}" for key, value in metrics.items())
        LOGGER.info("step=%d %s", step, message)
        numeric = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        if self.writer is not None:
            for key, value in numeric.items():
                self.writer.add_scalar(key, value, step)
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(numeric, step=step)
            except Exception:
                LOGGER.exception("W&B logging failed; local metrics remain available")

    def close(self) -> None:
        if not self.enabled:
            return
        self.metrics_file.close()
        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                LOGGER.exception("W&B shutdown failed")

    def __enter__(self) -> MetricLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
