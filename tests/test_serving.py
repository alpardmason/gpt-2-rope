from __future__ import annotations

import asyncio
import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from gpt2_rope.config import GenerationConfig, ModelConfig
from gpt2_rope.model import GPT
from gpt2_rope.serving import (
    GenerateRequest,
    InferenceService,
    _Pending,
    create_app,
    group_compatible,
)
from gpt2_rope.tokenizer import ByteBPETokenizer


def tiny_tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(
        ["alpha beta gamma delta epsilon zeta"],
        vocab_size=280,
        special_tokens=["<|endoftext|>"],
    )


def tiny_model() -> GPT:
    torch.manual_seed(0)
    return GPT(
        ModelConfig(
            vocab_size=300,
            context_length=64,
            d_model=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            dropout=0.0,
        )
    )


def make_service(metrics_dir: Path | None = None) -> InferenceService:
    return InferenceService(
        tiny_model(),
        tiny_tokenizer(),
        torch.device("cpu"),
        max_batch_size=4,
        batch_window_ms=5.0,
        metrics_dir=metrics_dir,
    )


def test_group_compatible_batches_same_shape_and_settings() -> None:
    loop = asyncio.new_event_loop()
    try:

        def pending(prompt_tokens: int, **settings: object) -> _Pending:
            values: dict[str, object] = {"max_new_tokens": 4, "temperature": 0.0}
            values.update(settings)
            return _Pending(
                token_ids=list(range(1, prompt_tokens + 1)),
                settings=GenerationConfig.model_validate(values),
                enqueued_at=0.0,
                future=loop.create_future(),
            )

        a = pending(3)
        b = pending(3)
        c = pending(5)  # different prompt length
        d = pending(3, temperature=1.0)  # different sampling settings
        groups = group_compatible([a, b, c, d])
        assert sorted(len(group) for group in groups) == [1, 1, 2]
        paired = next(group for group in groups if len(group) == 2)
        assert {id(item) for item in paired} == {id(a), id(b)}
    finally:
        loop.close()


def test_healthz_and_greedy_generation_is_deterministic(tmp_path: Path) -> None:
    service = make_service(metrics_dir=tmp_path)
    with TestClient(create_app(service)) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["parameters"] > 0

        payload = {"prompt": "alpha beta", "max_new_tokens": 4, "temperature": 0.0}
        first = client.post("/generate", json=payload)
        second = client.post("/generate", json=payload)
        assert first.status_code == second.status_code == 200
        assert first.json()["completion"] == second.json()["completion"]
        assert first.json()["completion_tokens"] >= 1
        assert first.json()["prompt_tokens"] >= 1

    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any("serve/latency_ms" in record for record in metrics)
    assert any("serve/batch_size" in record for record in metrics)


def test_generate_rejects_invalid_requests() -> None:
    service = make_service()
    with TestClient(create_app(service)) as client:
        empty = client.post("/generate", json={"prompt": ""})
        assert empty.status_code == 422

        too_long = client.post(
            "/generate",
            json={"prompt": "alpha beta " * 50, "max_new_tokens": 60},
        )
        assert too_long.status_code == 422

        oversized_budget = client.post(
            "/generate",
            json={"prompt": "alpha", "max_new_tokens": 100},
        )
        assert oversized_budget.status_code == 422


def test_validate_returns_token_ids() -> None:
    service = make_service()
    token_ids = service.validate(GenerateRequest(prompt="alpha beta", max_new_tokens=4))
    assert token_ids == service.tokenizer.encode("alpha beta")
