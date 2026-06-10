# 07: Rotary Position Embeddings in PyTorch

## Objectives and Prerequisites

Connect RoPE's pairwise rotations to buffers, dtype/device policy, cached
decoding offsets, and testable invariants. Prerequisite: RoPE theory and 06.

**Source map:** [`rope.py`](../../src/gpt2_rope/rope.py) `rotate_half`,
`RotaryEmbedding`; its call in
[`GroupedQueryAttention.forward`](../../src/gpt2_rope/model.py); and
`test_rope_preserves_vector_norm` in
[`test_model.py`](../../tests/test_model.py).

## Contracts and Shapes

Input query/key shapes are `[B,H,T,D]`, with potentially different head counts
but equal `T,D`. `D` must be even. Adjacent features form 2D vectors:

```text
(x0, x1) -> (-x1, x0)
rot(x, theta) = x*cos(theta) + rotate_half(x)*sin(theta)
```

An orthogonal rotation preserves the last-dimension norm. Applying position
rotations to Q and K makes their dot product depend on relative displacement.

Contrast with GPT-2's original learned absolute table (`wpe`): a
`context_length x d_model` parameter added to embeddings once, encoding
absolute index in the residual stream rather than relative displacement in
the attention operator. That variant remains available as
`ModelConfig.position_encoding = "learned"`; chapter 17 runs the ablation
and tabulates the parameter, extrapolation, and cache-interaction deltas.

Frequency tables:

```text
inverse_frequency: [D/2]
positions:         [context]
outer product:     [context,D/2]
repeated angles:   [context,D]
```

At forward time, `[T,D]` slices become `[1,1,T,D]` and broadcast across batch
and heads.

## PyTorch Engineering

Tables are computed in FP32, registered as non-persistent buffers, and cast to
the activation dtype/device when used.

- **Buffer:** follows `.to(device)` and module lifecycle.
- **Non-persistent:** omitted from checkpoints because config deterministically
  regenerates it.
- **FP32 source:** avoids building frequencies directly at reduced precision.

`offset=past_length` is essential. New tokens must receive absolute positions
after the cached prefix, not restart at zero.

**Recommendation:** test mathematical invariants and integration equivalence.
**Rationale:** a norm test can pass while offset logic is still wrong.

| Table policy | Benefit | Cost |
|---|---|---|
| Precompute full context | Simple, fast lookup | Memory grows with context |
| Grow cache dynamically | Flexible | Mutation/compile complexity |
| Compute per call | Minimal persistent memory | Repeated compute |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Odd feature crash | No adjacent pair | Check `head_dim` | Change geometry | Config validator |
| Cached logits differ | Offset reset | Compare full/cached | Use past length | Integration test |
| Long prompt rejected | Fixed table exhausted | Check context | Scale/extend deliberately | Boundary test |
| Checkpoint bloat | Tables persistent | Inspect state dict | `persistent=False` | State test |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import torch
from gpt2_rope.rope import RotaryEmbedding

r = RotaryEmbedding(8, 32)
q, k = torch.randn(2, 4, 5, 8), torch.randn(2, 2, 5, 8)
qr, kr = r(q, k, offset=3)
print(torch.max(torch.abs(q.norm(dim=-1) - qr.norm(dim=-1))).item())
print("buffers", [n for n, _ in r.named_buffers()])
print("state", list(r.state_dict()))
PY
```

Expected: norm error near floating-point noise, two named buffers, empty state
dict. Debug prompt: explain why `.to(dtype=..., device=...)` is still present
although buffers normally follow module device.

## Exercises

1. Derive the memory cost of two FP32 tables for context `L` and head dim `D`.
2. Why rotate Q and K but not V?
3. Does norm preservation prove relative-position correctness?

## Solutions

1. `2 * L * D * 4` bytes.
2. Attention routing depends on Q/K dot products; V carries retrieved content.
3. No. It proves orthogonality, not frequency, indexing, offset, or composition.

## Modern LLM Systems Delta

Long-context models use partial rotary dimensions, changed bases, NTK-aware or
YaRN-style scaling, dynamic frequencies, and specialized kernels. These alter
extrapolation behavior and require evaluation, not just a larger table.

## Professional Takeaways

Separate mathematical invariants, state-management policy, numerical policy,
and integration behavior. A paper equation is not yet a production component.

## Further Exploration

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [YaRN](https://arxiv.org/abs/2309.00071)
- [PyTorch buffers](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)

