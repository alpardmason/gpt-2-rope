from __future__ import annotations

import json
from pathlib import Path

from gpt2_rope.monitoring import MetricLogger


def test_metric_logger_writes_jsonl_records(tmp_path: Path) -> None:
    with MetricLogger(tmp_path, tensorboard=False) as logger:
        logger.log(1, {"train/loss": 2.5, "note": "warmup"})
        logger.log(2, {"train/loss": 2.25})

    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert records[0] == {"step": 1, "train/loss": 2.5, "note": "warmup"}
    assert records[1] == {"step": 2, "train/loss": 2.25}


def test_metric_logger_appends_across_sessions(tmp_path: Path) -> None:
    with MetricLogger(tmp_path, tensorboard=False) as logger:
        logger.log(1, {"train/loss": 1.0})
    with MetricLogger(tmp_path, tensorboard=False) as logger:
        logger.log(2, {"train/loss": 0.5})

    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_disabled_logger_writes_nothing(tmp_path: Path) -> None:
    with MetricLogger(tmp_path / "rank1", tensorboard=True, enabled=False) as logger:
        logger.log(1, {"train/loss": 1.0})
    assert not (tmp_path / "rank1").exists()


def test_tensorboard_directory_only_created_when_enabled(tmp_path: Path) -> None:
    with MetricLogger(tmp_path, tensorboard=True) as logger:
        logger.log(1, {"train/loss": 1.0})
    assert (tmp_path / "tb").is_dir()
