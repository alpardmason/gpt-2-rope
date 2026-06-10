"""Minimal native LoRA adapters with mergeable linear layers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.merged = False

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.base(inputs)
        if not self.merged:
            low_rank = functional.linear(self.dropout(inputs), self.lora_a)
            update = functional.linear(low_rank, self.lora_b)
            result = result + update * self.scale
        return cast(Tensor, result)

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight.add_((self.lora_b @ self.lora_a) * self.scale)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.sub_((self.lora_b @ self.lora_a) * self.scale)
            self.merged = False


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
) -> int:
    targets = set(target_modules)
    replaced = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha, dropout))
                replaced += 1
    if replaced == 0:
        raise ValueError(f"no linear modules matched targets {sorted(targets)}")
    return replaced


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    }


def save_lora(model: nn.Module, path: Path) -> None:
    save_file(lora_state_dict(model), str(path))


def load_lora(model: nn.Module, path: Path) -> None:
    state = load_file(str(path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_non_lora = [name for name in missing if name.endswith(("lora_a", "lora_b"))]
    if missing_non_lora or unexpected:
        raise ValueError(f"adapter mismatch: missing={missing_non_lora}, unexpected={unexpected}")


def set_lora_merged(model: nn.Module, merged: bool) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge() if merged else module.unmerge()
