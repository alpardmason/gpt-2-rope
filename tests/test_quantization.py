from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT
from gpt2_rope.quantization import (
    QuantizedLinear,
    load_quantized,
    quantization_report,
    quantize_model,
    save_quantized,
)


def tiny_model() -> GPT:
    torch.manual_seed(0)
    return GPT(
        ModelConfig(
            vocab_size=300,
            context_length=16,
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            dropout=0.0,
        )
    ).eval()


def test_quantized_linear_matches_float_within_tolerance() -> None:
    torch.manual_seed(1)
    linear = nn.Linear(64, 32)
    quantized = QuantizedLinear.from_linear(linear)
    x = torch.randn(8, 64)
    with torch.no_grad():
        expected = linear(x)
        actual = quantized(x)
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
    assert quantized.weight_int8.dtype == torch.int8
    assert int(quantized.weight_int8.abs().max()) <= 127


def test_quantize_model_skips_tied_head_and_reports_compression() -> None:
    model = tiny_model()
    replaced = quantize_model(model)
    # 2 layers x (q, k, v, out, fc, proj) = 12 linears; lm_head skipped.
    assert replaced == 12
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()

    report = quantization_report(model)
    assert report["quantized_modules"] == 12
    assert isinstance(report["compression_ratio"], float)
    assert report["compression_ratio"] > 3.0


def test_quantized_model_logits_stay_close_to_float() -> None:
    model = tiny_model()
    tokens = torch.randint(0, 300, (2, 8))
    with torch.no_grad():
        full_precision = model(tokens, use_cache=False).logits
    quantize_model(model)
    with torch.no_grad():
        quantized = model(tokens, use_cache=False).logits
    torch.testing.assert_close(quantized, full_precision, atol=0.2, rtol=0.2)


def test_save_and_load_quantized_round_trip(tmp_path: Path) -> None:
    model = tiny_model()
    quantize_model(model)
    tokens = torch.randint(0, 300, (1, 6))
    with torch.no_grad():
        expected = model(tokens, use_cache=False).logits
    path = tmp_path / "model-int8.safetensors"
    save_quantized(model, path)
    report = json.loads(path.with_suffix(".safetensors.json").read_text(encoding="utf-8"))
    assert report["quantized_modules"] == 12

    restored = tiny_model()
    load_quantized(restored, path)
    with torch.no_grad():
        actual = restored(tokens, use_cache=False).logits
    torch.testing.assert_close(actual, expected)


def test_quantize_model_requires_a_target() -> None:
    with pytest.raises(ValueError, match="no linear"):
        quantize_model(nn.Sequential(nn.ReLU()))
