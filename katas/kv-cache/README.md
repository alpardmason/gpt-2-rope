# Kata: kv-cache

Reimplement KV-cached decoding and sampling from a gutted
[`src/gpt2_rope/generation.py`](../../src/gpt2_rope/generation.py).
Tutorial: [09](../../docs/tutorials/09-kv-cache-and-autoregressive-generation.md).
Estimated effort: one evening (2-4 hours).

## Objective

Build the inference loop that production serving depends on: prefill once,
then decode one token at a time against the growing cache, with greedy and
stochastic sampling controls that are exactly reproducible under a seed.

## Contract

You must satisfy, without editing any other file:

- `_apply_repetition_penalty(logits, tokens, penalty)` divides positive and
  multiplies negative logits of already-generated tokens (the GPT-2/CTRL
  convention) and is the identity at `penalty == 1.0`.
- `sample_next_token(logits, tokens, config, generator)` applies, in order:
  repetition penalty, greedy shortcut at `temperature == 0`, temperature,
  top-k, then top-p, and samples with the provided generator so results are
  deterministic per seed. Returns shape `[B, 1]`.
- `generate(model, input_ids, config)` validates input rank and context
  budget, runs one full prefill forward with `use_cache=True`, then feeds
  only the newest token plus the cache per step; honors `eos_token_id` by
  freezing finished rows and stopping early when all rows finish; runs under
  inference mode; returns prompt plus generated tokens.

The skeleton's signatures, docstrings, and `# KATA:` comments state the rest.

## Oracle

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_lora_generation.py::test_seeded_generation_is_deterministic -q
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_lora_generation.py tests/test_serving.py -q
UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py check kv-cache
```

The serving tests exercise your `generate` through the micro-batching
service and assert greedy decoding is deterministic across calls.

## Workflow

Red-green loop on the seeded-determinism test first, then the serving tests,
then the full suite plus `uv run mypy` and `uv run ruff check .`. When green,
`git diff main -- src/gpt2_rope/generation.py` and record the review notes
required by [katas/README.md](../README.md).

## Hint ladder (open one rung at a time)

1. The cache invariant: after prefill, each step's model call receives a
   tensor of shape `[B, 1]` and the previous step's `past_key_values`. If
   you pass the whole sequence every step, results may look right while the
   compute is O(T^2) per token - check what you feed the model.
2. Determinism failures are almost always generator plumbing: every
   `torch.multinomial` call must use the explicit `torch.Generator` seeded
   from `config.seed`, never global RNG.
3. Top-p subtlety: compute cumulative probabilities over sorted logits and
   keep the smallest set whose mass exceeds `top_p`, always keeping the
   top-1 token. The original shifts the cumsum by one position so the first
   token above the threshold survives; scatter the filtered values back to
   original vocabulary order before softmax.
