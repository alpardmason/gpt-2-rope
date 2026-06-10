"""Post-training weight-only INT8 quantization for inference.

Per-output-channel symmetric quantization of ``nn.Linear`` weights. Weights are
stored as INT8 plus one FP32 scale per output channel and dequantized on the
fly in ``forward``; activations stay in floating point. This is the simplest
member of the family that includes GPTQ, AWQ, and FP8 serving formats.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

DEFAULT_SKIP_MODULES = ("lm_head",)


class QuantizedLinear(nn.Module):
    """INT8 weight storage with per-channel scales, dequantized per forward."""

    def __init__(self, weight_int8: Tensor, scales: Tensor, bias: Tensor | None) -> None:
        super().__init__()
        if weight_int8.dtype != torch.int8:
            raise ValueError("weight_int8 must be int8")
        if scales.shape != (weight_int8.size(0),):
            raise ValueError("scales must have one entry per output channel")
        self.weight_int8: Tensor
        self.scales: Tensor
        self.bias: Tensor | None
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> QuantizedLinear:
        weight = linear.weight.detach().float()
        scales = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        quantized = torch.clamp((weight / scales.unsqueeze(1)).round(), -127, 127).to(torch.int8)
        bias = None if linear.bias is None else linear.bias.detach().clone()
        return cls(quantized, scales, bias)

    def forward(self, inputs: Tensor) -> Tensor:
        weight = self.weight_int8.to(inputs.dtype) * self.scales.unsqueeze(1).to(inputs.dtype)
        return functional.linear(inputs, weight, self.bias)

    def quantized_bytes(self) -> int:
        total = self.weight_int8.numel() + self.scales.numel() * 4
        if self.bias is not None:
            total += self.bias.numel() * 4
        return total


def quantize_model(
    model: nn.Module,
    skip_modules: tuple[str, ...] = DEFAULT_SKIP_MODULES,
) -> int:
    """Replace every eligible ``nn.Linear`` in place; returns the count.

    ``lm_head`` is skipped by default: it shares storage with the token
    embedding, which must stay floating point for the embedding lookup.
    """
    skip = set(skip_modules)
    replaced = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name in skip or not isinstance(child, nn.Linear):
                continue
            setattr(module, name, QuantizedLinear.from_linear(child))
            replaced += 1
    if replaced == 0:
        raise ValueError("no linear modules were quantized")
    return replaced


def quantization_report(model: nn.Module) -> dict[str, float | int]:
    """Byte accounting comparing quantized storage to its FP32 equivalent."""
    quantized_bytes = 0
    float_equivalent_bytes = 0
    modules = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            modules += 1
            quantized_bytes += module.quantized_bytes()
            float_equivalent_bytes += module.weight_int8.numel() * 4
            if module.bias is not None:
                float_equivalent_bytes += module.bias.numel() * 4
    return {
        "quantized_modules": modules,
        "quantized_bytes": quantized_bytes,
        "float_equivalent_bytes": float_equivalent_bytes,
        "compression_ratio": (
            float_equivalent_bytes / quantized_bytes if quantized_bytes else 0.0
        ),
    }


def save_quantized(model: nn.Module, path: Path) -> None:
    """Persist the quantized state dict as safetensors plus a sidecar report."""
    state = {
        # Clone to break shared storage (tied embeddings) for safetensors.
        name: tensor.detach().cpu().clone().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(path))
    report = quantization_report(model)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_quantized(
    model: nn.Module,
    path: Path,
    skip_modules: tuple[str, ...] = DEFAULT_SKIP_MODULES,
) -> nn.Module:
    """Restructure ``model`` with quantized layers, then load saved tensors."""
    quantize_model(model, skip_modules)
    state = load_file(str(path))
    # load_state_dict copies in place, so tied tensors stay tied.
    model.load_state_dict(state)
    return model.eval()
