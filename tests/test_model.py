from __future__ import annotations

import pytest
import torch

from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT
from gpt2_rope.rope import RotaryEmbedding


def tiny_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 97,
        "context_length": 32,
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_rope_preserves_vector_norm() -> None:
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=32)
    q = torch.randn(2, 4, 7, 8)
    k = torch.randn(2, 2, 7, 8)
    q_rot, k_rot = rope(q, k)
    torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1))
    torch.testing.assert_close(k_rot.norm(dim=-1), k.norm(dim=-1))


def test_model_shapes_tying_and_loss() -> None:
    model = GPT(tiny_config())
    input_ids = torch.randint(0, 97, (2, 9))
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 9, 97)
    assert output.loss is not None and output.loss.ndim == 0
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()

    key, value = output.past_key_values[0]
    assert key.shape == (2, 2, 9, 8)
    assert value.shape == (2, 2, 9, 8)


def test_cached_logits_match_full_forward() -> None:
    torch.manual_seed(7)
    model = GPT(tiny_config()).eval()
    tokens = torch.randint(0, 97, (2, 10))

    with torch.no_grad():
        full = model(tokens, use_cache=False).logits
        prefix = model(tokens[:, :6], use_cache=True)
        cached = model(
            tokens[:, 6:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits

    torch.testing.assert_close(cached, full[:, 6:], atol=2e-5, rtol=2e-5)


def test_mha_is_gqa_special_case() -> None:
    config = tiny_config(num_kv_heads=4)
    model = GPT(config).eval()
    x = torch.randint(0, config.vocab_size, (1, 5))
    output = model(x)
    assert output.logits.shape == (1, 5, config.vocab_size)


ABLATION_VARIANTS: dict[str, dict[str, object]] = {
    "default-rope-prenorm-gqa": {},
    "learned-pe": {"position_encoding": "learned"},
    "postnorm": {"norm_placement": "post"},
    "mha": {"num_kv_heads": 4},
    "mqa": {"num_kv_heads": 1},
    "learned-pe-postnorm": {"position_encoding": "learned", "norm_placement": "post"},
}


@pytest.mark.parametrize("overrides", ABLATION_VARIANTS.values(), ids=ABLATION_VARIANTS.keys())
def test_ablation_variants_cached_match_full_forward(overrides: dict[str, object]) -> None:
    torch.manual_seed(11)
    model = GPT(tiny_config(**overrides)).eval()
    tokens = torch.randint(0, 97, (2, 10))

    with torch.no_grad():
        full = model(tokens, use_cache=False).logits
        prefix = model(tokens[:, :6], use_cache=True)
        cached = model(
            tokens[:, 6:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits

    torch.testing.assert_close(cached, full[:, 6:], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("overrides", ABLATION_VARIANTS.values(), ids=ABLATION_VARIANTS.keys())
def test_ablation_variants_train_step(overrides: dict[str, object]) -> None:
    torch.manual_seed(11)
    model = GPT(tiny_config(**overrides)).train()
    tokens = torch.randint(0, 97, (2, 8))
    output = model(tokens, labels=tokens, use_cache=False)
    assert output.loss is not None
    output.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in gradients)


def test_learned_position_encoding_builds_table_and_rope_does_not() -> None:
    learned = GPT(tiny_config(position_encoding="learned"))
    assert learned.position_embedding is not None
    assert learned.position_embedding.weight.shape == (32, 32)
    assert all(block.attention.rope is None for block in learned.blocks)

    rope = GPT(tiny_config())
    assert rope.position_embedding is None
    assert all(block.attention.rope is not None for block in rope.blocks)

