# 24: Inference Serving and Deployment

## Objectives and Prerequisites

Put the model behind a validated HTTP interface with queueing, micro-
batching, and latency observability -- and understand precisely which
problems vLLM-class servers solve that this one does not. Prerequisite: 09,
14, 23. Install the extra: `uv sync --extra serving`.

**Practice companion:** [24-practice.md](practice/24-practice.md).

**Source map:** [`serving.py`](../../src/gpt2_rope/serving.py)
`GenerateRequest`, `InferenceService`, `group_compatible`, `create_app`;
[`benchmarking.py`](../../src/gpt2_rope/benchmarking.py)
`benchmark_inference`; [`cli.py`](../../src/gpt2_rope/cli.py) `serve`,
`benchmark inference`; [`test_serving.py`](../../tests/test_serving.py);
[`test_benchmarking.py`](../../tests/test_benchmarking.py).

## Service Contract

```text
POST /generate  {prompt, max_new_tokens, temperature, top_k, top_p,
                 repetition_penalty, seed}
             -> {completion, prompt_tokens, completion_tokens,
                 latency_ms, batch_size}
GET  /healthz -> {status, device, parameters, context_length}
```

Architecture: every request is validated at the boundary (non-empty prompt,
token budget vs `context_length`), then enqueued. A single async worker
drains the queue with a batching window (`batch_window_ms`), groups requests
that can share one forward pass -- same prompt token length and identical
sampling settings -- and runs each group as one batched `generate` call in a
thread. Results resolve per-request futures; every batch logs `serve/*`
metrics to `metrics.jsonl`.

Invariants:

- Request validation happens before queueing: a malformed request must never
  consume model time.
- The model is owned by one worker; there is no concurrent forward, so no
  lock and no cache corruption.
- Batched sampling shares one RNG stream per batch: per-request determinism
  holds for greedy decoding; sampled outputs depend on batch composition.

## Why Same-Length Grouping

This model's `generate` has no padding-aware attention mask, so a batch must
share a prompt length (chapter 09). The service therefore groups instead of
pads -- an honest restriction that demonstrates why production engines work
at a different granularity:

| Design | Batches across | Wasted compute | Complexity |
|---|---|---|---|
| Same-shape grouping (here) | identical lengths/settings | none, but few merges | low |
| Padded static batching | any lengths | pad tokens | masking |
| Continuous batching (vLLM) | requests join/leave per step | none | scheduler + paged cache |

**Recommendation:** for a single small model, queue + micro-batch + metrics
is the right amount of machinery; adopt a serving engine rather than growing
this into one. **Rationale:** continuous batching and paged KV caches are
multi-engineer-year systems with stable open implementations.
**Alternatives:** vLLM or SGLang for GPU serving; llama.cpp for CPU/edge;
TorchServe-style wrappers when latency targets are loose.

## Observability Is the Service

`serve/batch_size`, `serve/latency_ms`, and
`serve/completion_tokens_per_second` stream to the same JSONL contract as
training metrics (chapter 14). A serving system without per-request latency
accounting cannot be capacity-planned or debugged; the metrics file is the
SLO evidence.

## Reproducible Inference Benchmarks

The HTTP metrics answer what happened under request load. The benchmark
command isolates model execution and writes a comparable JSON artifact:

```bash
UV_CACHE_DIR=.uv-cache uv run gpt2-rope benchmark inference configs/tiny.yaml \
  runs/tiny/checkpoints/step-00010000 \
  --batch-size 1 --prompt-tokens 64 --generated-tokens 32 \
  --output runs/benchmarks/tiny.json
```

The report separates time to first token (prefill) from cached decode,
records output/decode tokens per second, accounts for live KV-cache bytes and
the equivalent MHA cache, and captures peak CUDA allocation. It uses a
deterministic random-token workload and records the seed plus model geometry.
Warmup runs are excluded, and CUDA/MPS are explicitly synchronized around
timing boundaries; without synchronization, host-side launch time would be
mislabeled as model latency. Results are evidence only for the recorded
device, precision, batch, prompt length, and generation length.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Event loop stalls under load | forward run on the loop thread | latency spikes on /healthz | `asyncio.to_thread` (in place) | async-blocking review |
| One bad request kills batch | exception not isolated per group | error correlation in logs | per-group exception fan-out (in place) | fault-injection test |
| Sampled outputs vary per run | batch composition changes RNG | compare greedy vs sampled | document; per-request generator if needed | determinism contract docs |
| 200 OK but garbage text | prompt silently truncated | token count vs budget | 422 on over-budget (in place) | boundary validation tests |
| Throughput flat as load grows | batches never merge | `serve/batch_size` histogram | longer window, length bucketing | batching metrics review |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra serving
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_serving.py -q
```

With a trained checkpoint (chapters 04 and 11 labs):

```bash
UV_CACHE_DIR=.uv-cache uv run gpt2-rope serve configs/tiny.yaml \
  runs/tiny/checkpoints/step-00010000 --port 8000 --metrics-dir runs/serving &
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt": "The pass key is", "max_new_tokens": 16, "temperature": 0.0}'
kill %1
cat runs/serving/metrics.jsonl
```

Expected: health JSON with parameter count; a completion with latency and
batch size; one `serve/*` record per processed batch.

## Exercises

1. Why is request validation placed before the queue rather than inside the
   batch worker?
2. Two greedy requests with the same prompt arrive together and are batched.
   Are their completions identical to the unbatched case? What about
   `temperature=0.8`?
3. Estimate the maximum merge rate of same-shape grouping if prompt token
   lengths are uniform over 100 values and the window holds 8 requests.

## Solutions

1. Failing fast returns a 422 in microseconds without occupying queue slots
   or worker time, and it keeps the worker's failure domain small: anything
   dequeued is known-runnable.
2. Greedy: yes -- argmax is independent of batch composition and RNG.
   Sampled: each row draws from the shared generator in batch order, so a
   request's tokens depend on its batch neighbors; per-request reproducibility
   would need one generator per row.
3. With uniform lengths, the chance two of 8 requests share a length is
   small (expected matches ~ 8*7/2 / 100 ~= 0.28 pairs per window), so most
   batches are singletons -- which is exactly why padded and continuous
   batching exist.

## Modern LLM Systems Delta

Production inference adds: continuous batching with per-step scheduling,
paged KV caches with prefix sharing, chunked prefill, speculative decoding,
quantized weights and caches (chapter 23), streaming token responses,
multi-replica routing with load-aware dispatch, and autoscaling on
tokens-per-second SLOs. The interface contract -- validate, queue, batch,
measure -- is the part that transfers unchanged.

## Professional Takeaways

Serving is a systems discipline wearing an ML costume: the model is a
function; the product is the queue, the batcher, the failure domains, and
the latency histogram. Always know whether your regime is memory- or
compute-bound before optimizing anything.

## Further Exploration

- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
