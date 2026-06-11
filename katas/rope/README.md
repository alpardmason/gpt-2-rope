# Kata: rope

Reimplement rotary position embeddings from a gutted
[`src/gpt2_rope/rope.py`](../../src/gpt2_rope/rope.py).
Tutorial: [07](../../docs/tutorials/07-rotary-position-embeddings-in-pytorch.md).
Estimated effort: one evening (2-4 hours).

## Objective

Turn the RoPE equations into a production module: correct pairwise rotation,
precomputed trig tables with the right state-management policy, and offset
handling that keeps cached decoding equal to a full forward pass.

## Contract

You must satisfy, without editing any other file:

- `rotate_half(x)` rotates adjacent feature pairs `(x0, x1) -> (-x1, x0)`
  along the last dimension.
- `RotaryEmbedding(head_dim, max_position_embeddings, base)` rejects odd
  `head_dim`, precomputes FP32 cos/sin tables for all positions, and keeps
  them out of `state_dict()` (the equivalence tests construct fresh modules
  from config alone).
- `forward(query, key, offset)` applies positions `offset:offset+T` to
  `[B, H, T, D]` tensors where Q and K may have different head counts,
  raises when the requested positions exceed the configured maximum, and
  matches the activation dtype/device.

The skeleton's signatures, docstrings, and `# KATA:` comments state the rest.
Imports used only by the gutted bodies were removed; re-add what you need.

## Oracle

```bash
# fast feedback while iterating
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py::test_rope_preserves_vector_norm -q
# the real bar: every model test, especially cached/full equivalence
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py -q
# done means done
UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py check rope
```

`GPT` constructs `RotaryEmbedding` in every attention layer, so the whole
model suite is downstream of your code. Expect everything to fail until
`__init__` works; that is the point.

## Workflow

Red-green loop: run the norm test, implement, re-run. Then the equivalence
tests (`test_cached_logits_match_full_forward` and the ablation variants),
which exercise `offset` through real cached decoding. Then `uv run mypy` and
`uv run ruff check .`. When green, `git diff main -- src/gpt2_rope/rope.py`
and record the review notes required by [katas/README.md](../README.md).

## Hint ladder (open one rung at a time)

1. The table build is four lines of shape bookkeeping:
   `[D/2] -> outer with [L] -> [L, D/2] -> repeat each angle twice -> [L, D]`.
   `test_rope_preserves_vector_norm` only needs `rotate_half` and the
   rotation formula to be self-consistent.
2. If norms pass but cached equivalence fails, your `offset` slicing or your
   broadcast view is wrong. The sliced tables must be `[1, 1, T, D]` so one
   table serves both the 4-head query and the 2-head key.
3. If `state_dict()` is non-empty, you registered the tables as persistent
   buffers or parameters. Tutorial 07's "PyTorch Engineering" section states
   the policy and why.
