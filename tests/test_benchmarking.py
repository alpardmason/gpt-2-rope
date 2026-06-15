from __future__ import annotations

import pytest
import torch

from gpt2_rope.benchmarking import benchmark_inference
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT


def tiny_model() -> GPT:
    torch.manual_seed(0)
    return GPT(
        ModelConfig(
            vocab_size=97,
            context_length=32,
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            dropout=0.0,
        )
    )


def test_inference_benchmark_reports_latency_throughput_and_cache_bytes() -> None:
    report = benchmark_inference(
        tiny_model(),
        torch.device("cpu"),
        batch_size=2,
        prompt_tokens=4,
        generated_tokens=3,
        warmup_runs=0,
        measured_runs=2,
    )

    assert report.device == "cpu"
    assert report.precision == "fp32"
    assert report.seed == 1337
    assert report.num_heads == 4
    assert report.num_kv_heads == 2
    assert report.time_to_first_token_ms.minimum > 0
    assert report.decode_total_ms.p95 >= report.decode_total_ms.minimum
    assert report.decode_tokens_per_second > 0
    assert report.output_tokens_per_second > 0
    assert report.final_kv_cache_tokens == 6
    expected_cache_bytes = 2 * 2 * 2 * 2 * 6 * 8 * 4
    assert report.kv_cache_bytes == expected_cache_bytes
    assert report.mha_equivalent_kv_cache_bytes == expected_cache_bytes * 2
    assert report.kv_cache_reduction_factor_vs_mha == 2.0
    assert report.peak_accelerator_memory_bytes is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"generated_tokens": 1}, "at least 2"),
        ({"prompt_tokens": 31, "generated_tokens": 2}, "exceed model context"),
        ({"measured_runs": 0}, "must be positive"),
    ],
)
def test_inference_benchmark_rejects_invalid_workloads(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark_inference(tiny_model(), torch.device("cpu"), **kwargs)
