# Kata: gqa

Reimplement `GroupedQueryAttention.forward` from a gutted
[`src/gpt2_rope/model.py`](../../src/gpt2_rope/model.py).
Tutorial: [08](../../docs/tutorials/08-grouped-query-attention-and-sdpa.md).
Estimated effort: one long evening (3-5 hours). Hardest kata; the rest of
`model.py` (including `__init__` and `_shape`) is intact - only the forward
pass is yours.

## Objective

Write the attention forward pass that every other subsystem sits on: GQA
projections, RoPE offsets, cache append, backend-aware head expansion, and
the prefix-aware causal mask that this repository's #1 documented pitfall
lives in.

## Contract

You must satisfy, without editing anything outside the gutted method:

- Project hidden states to Q with `num_heads` and K/V with `num_kv_heads`
  heads via the existing `_shape` helper; apply RoPE (when configured) with
  `offset = past_length` so cached tokens keep absolute positions.
- Append the incoming `past_key_value` along the sequence axis BEFORE
  attention; return the compact `[B, H_kv, T, D]` cache as `present`.
- Expand K/V across query groups only where required: SDPA's `enable_gqa`
  on CUDA, explicit `repeat_interleave` elsewhere, no expansion for MHA.
- Masking: with no prefix, rely on `is_causal` (only when `T > 1`); with a
  non-empty prefix, build a boolean mask from absolute query and key
  positions - upper-left-aligned `is_causal=True` is exactly the documented
  cached-logits bug.
- Dropout only in training mode; output is reshaped to `[B, T, d_model]`,
  passed through `out_proj` and residual dropout. When `use_cache` is
  false, return zero-length cache tensors instead of `present`.

The skeleton's `# KATA:` comments restate this in place.

## Oracle

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py -q
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_lora_generation.py -q
UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py check gqa
```

`test_cached_logits_match_full_forward` and the ablation matrix
(`mha`, `mqa`, learned positions, post-norm) are the real bar: every variant
must keep cached decoding equal to the full forward pass.

## Workflow

Get `test_model_shapes_tying_and_loss` green first (shapes and cache
layout), then the cached/full equivalence tests, then the ablation matrix,
then mypy/ruff and the full suite. When green,
`git diff main -- src/gpt2_rope/model.py` and record the review notes
required by [katas/README.md](../README.md).

## Hint ladder (open one rung at a time)

1. Compute `past_length` from the cache before anything else; both the RoPE
   offset and the mask need it. The cache shape test pins K/V to
   `[B, num_kv_heads, T, head_dim]` - decide where you concatenate and what
   you store before worrying about the mask.
2. If only cached tests fail: for a non-empty prefix, query row `i` sits at
   absolute position `past_length + i` and may attend to key column `j`
   iff `j <= past_length + i`. Build that `[1, 1, T_q, T_k]` boolean mask
   and pass `is_causal=False`.
3. If MQA/MHA variants fail but the default passes: check
   `config.query_groups`. MHA needs no expansion and no `enable_gqa`;
   CPU/MPS GQA needs `repeat_interleave` on K and V (dim 1) so SDPA sees
   matching head counts; the compact cache must stay unexpanded.
