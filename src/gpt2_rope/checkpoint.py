"""Atomic, exact-resume checkpoints and inference-only exports."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch.amp.grad_scaler import GradScaler


@dataclass(slots=True)
class CheckpointState:
    progress: dict[str, int]
    config: dict[str, Any]
    tokenizer_identity: dict[str, Any]


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: GradScaler | None,
    progress: dict[str, int],
    config: dict[str, Any],
    tokenizer_identity: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        torch.save(model.state_dict(), temporary / "model.pt")
        training_state = {
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "rng": _rng_state(),
        }
        torch.save(training_state, temporary / "training.pt")
        metadata = {
            "version": 1,
            "progress": progress,
            "config": config,
            "tokenizer_identity": tokenizer_identity,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        if path.exists():
            shutil.rmtree(path)
        temporary.rename(path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: GradScaler | None,
    restore_rng: bool = True,
) -> CheckpointState:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    training = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
    if optimizer is not None and training["optimizer"] is not None:
        optimizer.load_state_dict(training["optimizer"])
    if scheduler is not None and training["scheduler"] is not None:
        scheduler.load_state_dict(training["scheduler"])
    if scaler is not None and training["scaler"] is not None:
        scaler.load_state_dict(training["scaler"])
    if restore_rng:
        _restore_rng_state(training["rng"])
    return CheckpointState(
        progress=metadata["progress"],
        config=metadata["config"],
        tokenizer_identity=metadata["tokenizer_identity"],
    )


def export_safetensors(model: nn.Module, path: Path) -> None:
    state = {
        # safetensors rejects shared storage. GPT-2 intentionally ties these
        # weights in memory, so export independent tensors and retie on load.
        name: tensor.detach().cpu().clone().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(path))
