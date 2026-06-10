from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from gpt2_rope.checkpoint import export_safetensors, load_checkpoint, save_checkpoint
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT


def test_checkpoint_restores_model_optimizer_and_progress(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        dropout=0.0,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randint(0, 32, (2, 5)), labels=torch.randint(0, 32, (2, 5))).loss
    assert loss is not None
    loss.backward()
    optimizer.step()

    checkpoint = save_checkpoint(
        tmp_path / "step-1",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        progress={"step": 1, "tokens": 10, "data_position": 2},
        config={"model": config.model_dump()},
        tokenizer_identity={"sha256": "abc"},
    )

    restored = GPT(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=None,
        scaler=None,
    )
    assert state.progress["step"] == 1
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_safetensors_export_handles_tied_embeddings(tmp_path: Path) -> None:
    model = GPT(
        ModelConfig(
            vocab_size=32,
            context_length=8,
            d_model=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
        )
    )
    path = tmp_path / "model.safetensors"
    export_safetensors(model, path)
    state = load_file(path)
    torch.testing.assert_close(state["token_embedding.weight"], state["lm_head.weight"])
