from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from gpt2_rope.config import GenerationConfig, ModelConfig
from gpt2_rope.generation import generate
from gpt2_rope.lora import (
    LoRALinear,
    apply_lora,
    load_lora,
    lora_state_dict,
    save_lora,
    set_lora_merged,
)
from gpt2_rope.model import GPT


def test_lora_merge_is_numerically_equivalent() -> None:
    torch.manual_seed(3)
    layer = LoRALinear(nn.Linear(8, 6), rank=2, alpha=4.0)
    nn.init.normal_(layer.lora_b)
    x = torch.randn(4, 8)
    expected = layer(x)
    layer.merge()
    torch.testing.assert_close(layer(x), expected)
    layer.unmerge()
    torch.testing.assert_close(layer(x), expected)


def test_apply_lora_and_adapter_state() -> None:
    model = GPT(
        ModelConfig(
            vocab_size=64,
            context_length=16,
            d_model=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            dropout=0.0,
        )
    )
    replaced = apply_lora(model, rank=2, alpha=4.0, target_modules=("q_proj", "v_proj"))
    assert replaced == 2
    assert set(lora_state_dict(model)) == {
        "blocks.0.attention.q_proj.lora_a",
        "blocks.0.attention.q_proj.lora_b",
        "blocks.0.attention.v_proj.lora_a",
        "blocks.0.attention.v_proj.lora_b",
    }


def _tiny_model() -> GPT:
    return GPT(
        ModelConfig(
            vocab_size=64,
            context_length=16,
            d_model=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            dropout=0.0,
        )
    )


def test_save_load_lora_round_trip_and_merge(tmp_path: Path) -> None:
    torch.manual_seed(5)
    source = _tiny_model()
    apply_lora(source, rank=2, alpha=4.0, target_modules=("q_proj", "v_proj"))
    for name, parameter in source.named_parameters():
        if name.endswith(("lora_a", "lora_b")):
            nn.init.normal_(parameter)
    adapter = tmp_path / "adapter.safetensors"
    save_lora(source, adapter)

    target = _tiny_model()
    apply_lora(target, rank=2, alpha=4.0, target_modules=("q_proj", "v_proj"))
    load_lora(target, adapter)
    for name, parameter in target.named_parameters():
        if name.endswith(("lora_a", "lora_b")):
            torch.testing.assert_close(parameter, dict(source.named_parameters())[name])

    x = torch.randint(0, 64, (1, 6))
    target.eval()
    with torch.no_grad():
        unmerged = target(x).logits
        set_lora_merged(target, True)
        merged = target(x).logits
        set_lora_merged(target, False)
        restored = target(x).logits
    torch.testing.assert_close(merged, unmerged, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(restored, unmerged, atol=2e-5, rtol=2e-5)


def test_load_lora_rejects_mismatched_adapter(tmp_path: Path) -> None:
    source = _tiny_model()
    apply_lora(source, rank=2, alpha=4.0, target_modules=("q_proj", "v_proj"))
    adapter = tmp_path / "adapter.safetensors"
    save_lora(source, adapter)

    target = _tiny_model()
    apply_lora(target, rank=2, alpha=4.0, target_modules=("fc",))
    with pytest.raises(ValueError, match="adapter mismatch"):
        load_lora(target, adapter)


def test_seeded_generation_is_deterministic() -> None:
    config = ModelConfig(
        vocab_size=64,
        context_length=24,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        dropout=0.0,
    )
    model = GPT(config).eval()
    prompt = torch.tensor([[1, 2, 3]])
    generation = GenerationConfig(max_new_tokens=5, temperature=0.8, top_k=10, seed=11)
    first = generate(model, prompt, generation)
    second = generate(model, prompt, generation)
    torch.testing.assert_close(first, second)

