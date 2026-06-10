from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpt2_rope.config import ModelConfig, model_preset


def test_model_config_validates_gqa_geometry() -> None:
    config = ModelConfig(
        vocab_size=128,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=8,
        num_kv_heads=2,
    )
    assert config.head_dim == 8
    assert config.query_groups == 4

    with pytest.raises(ValidationError):
        ModelConfig(d_model=63, num_heads=8)

    with pytest.raises(ValidationError):
        ModelConfig(d_model=64, num_heads=8, num_kv_heads=3)


def test_presets_include_tiny_and_gpt2_family() -> None:
    assert model_preset("tiny").d_model == 128
    assert model_preset("gpt2-small").num_layers == 12
    assert model_preset("gpt2-xl").d_model == 1600

