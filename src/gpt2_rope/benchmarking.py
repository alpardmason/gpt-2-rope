"""Reproducible prefill and decode benchmarks for KV-cached inference."""

from __future__ import annotations

import contextlib
import math
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import torch
from torch import Tensor

from gpt2_rope.model import GPT, KVCache

Precision = Literal["auto", "fp32", "bf16", "fp16"]


@dataclass(frozen=True, slots=True)
class LatencySummary:
    minimum: float
    mean: float
    p50: float
    p95: float


@dataclass(frozen=True, slots=True)
class InferenceBenchmarkReport:
    schema_version: int
    generated_at_utc: str
    torch_version: str
    device: str
    device_name: str
    precision: str
    seed: int
    parameter_count: int
    parameter_bytes: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    warmup_runs: int
    measured_runs: int
    time_to_first_token_ms: LatencySummary
    decode_total_ms: LatencySummary
    end_to_end_ms: LatencySummary
    decode_tokens_per_second: float
    output_tokens_per_second: float
    final_kv_cache_tokens: int
    kv_cache_bytes: int
    mha_equivalent_kv_cache_bytes: int
    kv_cache_reduction_factor_vs_mha: float
    peak_accelerator_memory_bytes: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _RunMeasurement:
    time_to_first_token_ms: float
    decode_total_ms: float
    end_to_end_ms: float
    kv_cache_bytes: int
    final_kv_cache_tokens: int


def _effective_precision(device: torch.device, requested: Precision) -> Precision:
    if device.type == "cpu":
        return "fp32"
    if requested != "auto":
        return requested
    if device.type == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if device.type == "mps":
        return "fp16"
    return "fp32"


def _autocast_context(
    device: torch.device,
    precision: Precision,
) -> contextlib.AbstractContextManager[Any]:
    if precision == "fp32" or device.type == "cpu":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _cache_bytes(cache: tuple[KVCache, ...]) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for key_value in cache
        for tensor in key_value
    )


def _summary(values: list[float]) -> LatencySummary:
    ordered = sorted(values)

    def percentile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    return LatencySummary(
        minimum=ordered[0],
        mean=sum(ordered) / len(ordered),
        p50=percentile(0.50),
        p95=percentile(0.95),
    )


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine() or "CPU"


def _measure_once(
    model: GPT,
    input_ids: Tensor,
    generated_tokens: int,
    device: torch.device,
    precision: Precision,
) -> _RunMeasurement:
    with torch.inference_mode(), _autocast_context(device, precision):
        _synchronize(device)
        prefill_started = time.perf_counter()
        output = model(input_ids, use_cache=True)
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        _synchronize(device)
        prefill_finished = time.perf_counter()

        cache = output.past_key_values
        decode_started = time.perf_counter()
        for _ in range(generated_tokens - 1):
            output = model(next_token, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        _synchronize(device)
        decode_finished = time.perf_counter()

    return _RunMeasurement(
        time_to_first_token_ms=(prefill_finished - prefill_started) * 1_000.0,
        decode_total_ms=(decode_finished - decode_started) * 1_000.0,
        end_to_end_ms=(decode_finished - prefill_started) * 1_000.0,
        kv_cache_bytes=_cache_bytes(cache),
        final_kv_cache_tokens=cache[0][0].size(-2),
    )


def benchmark_inference(
    model: GPT,
    device: torch.device,
    *,
    batch_size: int = 1,
    prompt_tokens: int = 128,
    generated_tokens: int = 32,
    warmup_runs: int = 2,
    measured_runs: int = 5,
    precision: Precision = "auto",
    seed: int = 1337,
) -> InferenceBenchmarkReport:
    """Measure prefill, cached decode, cache size, and accelerator memory."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if prompt_tokens < 1:
        raise ValueError("prompt_tokens must be positive")
    if generated_tokens < 2:
        raise ValueError("generated_tokens must be at least 2 to measure decode")
    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative")
    if measured_runs < 1:
        raise ValueError("measured_runs must be positive")
    if prompt_tokens + generated_tokens > model.config.context_length:
        raise ValueError("prompt plus generated tokens exceed model context")

    model = model.to(device).eval()
    effective_precision = _effective_precision(device, precision)
    generator = torch.Generator()
    generator.manual_seed(seed)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size, prompt_tokens),
        generator=generator,
    ).to(device)

    for _ in range(warmup_runs):
        _measure_once(model, input_ids, generated_tokens, device, effective_precision)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measurements = [
        _measure_once(model, input_ids, generated_tokens, device, effective_precision)
        for _ in range(measured_runs)
    ]
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    ttft = _summary([measurement.time_to_first_token_ms for measurement in measurements])
    decode = _summary([measurement.decode_total_ms for measurement in measurements])
    end_to_end = _summary([measurement.end_to_end_ms for measurement in measurements])
    decode_token_count = batch_size * (generated_tokens - 1)
    output_token_count = batch_size * generated_tokens
    kv_cache_bytes = measurements[0].kv_cache_bytes
    query_groups = model.config.query_groups

    return InferenceBenchmarkReport(
        schema_version=1,
        generated_at_utc=datetime.now(UTC).isoformat(),
        torch_version=str(torch.__version__),
        device=str(device),
        device_name=_device_name(device),
        precision=effective_precision,
        seed=seed,
        parameter_count=model.parameter_count(),
        parameter_bytes=sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
        context_length=model.config.context_length,
        d_model=model.config.d_model,
        num_layers=model.config.num_layers,
        num_heads=model.config.num_heads,
        num_kv_heads=model.config.num_kv_heads,
        head_dim=model.config.head_dim,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        time_to_first_token_ms=ttft,
        decode_total_ms=decode,
        end_to_end_ms=end_to_end,
        decode_tokens_per_second=decode_token_count / max(decode.mean / 1_000.0, 1e-9),
        output_tokens_per_second=output_token_count
        / max(end_to_end.mean / 1_000.0, 1e-9),
        final_kv_cache_tokens=measurements[0].final_kv_cache_tokens,
        kv_cache_bytes=kv_cache_bytes,
        mha_equivalent_kv_cache_bytes=kv_cache_bytes * query_groups,
        kv_cache_reduction_factor_vs_mha=float(query_groups),
        peak_accelerator_memory_bytes=peak_memory,
    )
