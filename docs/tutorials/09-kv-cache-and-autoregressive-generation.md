# 09: KV Cache and Autoregressive Generation

## Objectives and Prerequisites

Understand prefill/decode state, cache correctness, sampling transforms, EOS
state, and deterministic random generators. Prerequisite: 07-08.

**Practice companion:** [09-practice.md](practice/09-practice.md).

**Source map:** [`generation.py`](../../src/gpt2_rope/generation.py)
`_apply_repetition_penalty`, `sample_next_token`, `generate`;
[`model.py`](../../src/gpt2_rope/model.py) cache paths; and
[`test_lora_generation.py`](../../tests/test_lora_generation.py).

## State Machine

Generation has two computational phases:

```text
prefill: model(full prompt) -> last logits + cache for every layer
decode:  sample one token -> model(one token, prior cache) -> append cache
```

The output tensor grows by concatenation; each layer's K/V cache also grows.
The model rejects `prompt_length + max_new_tokens > context_length` before work.

Cache correctness contract:

```text
full_model(tokens)[:, suffix]
~= model(suffix, cache=model(prefix))
```

Approximate equality accounts for floating-point kernel order, not semantic
differences. The regression test includes a multi-token suffix because that
exposes offset-mask bugs hidden by one-token decoding.

## Sampling Pipeline

Applied in order:

1. Repetition penalty over token IDs already present.
2. Temperature; zero selects greedy argmax.
3. Top-k threshold.
4. Top-p sorted cumulative-mass filter.
5. Softmax and multinomial sampling.

For negative repeated logits, multiplying by penalty makes them less likely;
for positive logits, division makes them less likely. Applying one formula to
both signs would be wrong.

A device-local `torch.Generator` is seeded from `GenerationConfig`. This makes
sampling reproducible without resetting global training RNG.

## Batched EOS

`finished: [B]` tracks each sequence. Finished rows emit EOS while unfinished
rows continue. The loop stops when all rows finish. Compute is still spent on
finished rows until the batch completes.

**Recommendation:** keep sampling pure and separately testable.
**Rationale:** policy changes should not threaten cache/model correctness.

| Decode design | Simplicity | Throughput | Memory control |
|---|---:|---:|---:|
| Current tensor concatenation | High | Low/medium | Fixed context |
| Preallocated cache/output | Medium | Higher | Better |
| Paged continuous batching | Low | Highest | Best at serving scale |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Cached text differs | RoPE/mask/cache bug | Full vs cached logits | Correct offset semantics | Equivalence test |
| Same seed differs | Global RNG/kernel nondeterminism | Isolate generator/backend | Local generator/policy | Seeded test |
| Repeated tokens favored | Sign handling wrong | Negative-logit example | Piecewise penalty | Unit test |
| Batch never stops early | EOS state wrong | Trace `finished` | Per-row mask + all | Mixed-EOS test |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import torch
from gpt2_rope.config import GenerationConfig, ModelConfig
from gpt2_rope.generation import generate
from gpt2_rope.model import GPT

torch.manual_seed(1)
c = ModelConfig(vocab_size=32, context_length=16, d_model=16,
                num_layers=1, num_heads=2, num_kv_heads=1, dropout=0)
m = GPT(c).eval()
p = torch.tensor([[1, 2, 3]])
g = GenerationConfig(max_new_tokens=4, top_k=5, seed=9)
print(generate(m, p, g).tolist())
print(generate(m, p, g).tolist())
PY
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_lora_generation.py -q
```

Expected: identical generated rows. Debug prompt: change temperature to zero
and explain why the generator becomes irrelevant.

## Exercises

1. Why sample from only `logits[:, -1]`?
2. Why does top-p use cumulative probability after sorting?
3. Name two scaling problems with repeated `torch.cat`.

## Solutions

1. Only the final prompt/decoded position predicts the next token.
2. Nucleus sampling keeps the smallest highest-probability set whose mass
   reaches the threshold.
3. Reallocation/copy of growing output and cache tensors; memory fragmentation
   and quadratic cumulative copy traffic.

## Modern LLM Systems Delta

Serving engines add request schedulers, continuous batching, paged attention,
prefix caching, speculative decoding, stop strings, streaming, grammar
constraints, quantization, and latency/service-level metrics.

## Professional Takeaways

Treat decoding as a state machine with separate model, cache, sampling, and
batch-lifecycle contracts. Correct logits are necessary but not sufficient for
a reliable serving path.

## Reimplementation Kata

Tier 2: rebuild `generation.py` -- prefill, single-token decode steps,
sampling controls, and EOS handling -- against the determinism and serving
tests. Start with
`UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start kv-cache`
and follow [katas/kv-cache/README.md](../../katas/kv-cache/README.md).

## Further Exploration

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)

