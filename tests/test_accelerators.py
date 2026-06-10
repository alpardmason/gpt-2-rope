from __future__ import annotations

import pytest
import torch

from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT


def _smoke_forward(device: torch.device) -> None:
    model = GPT(
        ModelConfig(
            vocab_size=64,
            context_length=8,
            d_model=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            dropout=0.0,
        )
    ).to(device)
    tokens = torch.randint(0, 64, (2, 8), device=device)
    output = model(tokens, labels=tokens, use_cache=False)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_training_smoke() -> None:
    _smoke_forward(torch.device("mps"))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_training_smoke() -> None:
    _smoke_forward(torch.device("cuda"))
