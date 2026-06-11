# 08: Grouped-Query Attention and SDPA

## Objectives and Prerequisites

Implement GQA as projection geometry plus backend-aware kernel dispatch.
Prerequisite: attention theory, 06-07, and tensor stride basics.

**Practice companion:** [08-practice.md](practice/08-practice.md).

**Source map:** [`model.py`](../../src/gpt2_rope/model.py)
`GroupedQueryAttention`; [`config.py`](../../src/gpt2_rope/config.py)
`query_groups`; [`test_model.py`](../../tests/test_model.py)
`test_mha_is_gqa_special_case` and cache equivalence.

## Tensor Contract

Let `C=d_model`, `D=C/Hq`, and `G=Hq/Hkv`:

```text
hidden: [B,T,C]
Q projection: [B,T,C]       -> [B,Hq,T,D]
K/V projection: [B,T,Hkv*D] -> [B,Hkv,T,D]
attention output: [B,Hq,T,D] -> [B,T,C]
```

Each KV head serves `G` query heads. MHA is the special case `Hq == Hkv`;
MQA is `Hkv == 1`. The next step on the same axis is MLA (multi-head latent
attention, DeepSeek-V2/V3): K/V are compressed into one low-rank latent per
token and reconstructed at compute time, shrinking the cache below what any
`Hkv` choice can reach -- at the cost of extra projections and a decoupled
RoPE path. Chapter 18 places MLA against GQA quantitatively.

The persistent cache remains `[B,Hkv,T,D]`. On non-CUDA backends, K/V are
expanded with `repeat_interleave` only at the SDPA boundary. CUDA requests
SDPA's native GQA path through `enable_gqa=True`.

## Kernel and Mask Semantics

`scaled_dot_product_attention` selects among available implementations. Kernel
choice can alter performance and small floating-point details.

No prefix:

```text
attention_mask = None
is_causal = query_length > 1
```

Cached prefix:

```text
query positions = past_length + [0..Tq-1]
key positions   = [0..Tk-1]
allow key <= query
```

The explicit `[1,1,Tq,Tk]` Boolean mask is required because built-in causal
alignment is wrong for a non-empty prefix when `Tq != Tk`.

**Recommendation:** optimize cache representation before kernel-boundary
compatibility. **Rationale:** cache persists across tokens; expansion is
temporary.

| Strategy | Cache bytes | Backend portability | Temporary cost |
|---|---:|---:|---:|
| Compact GQA + native kernel | Low | CUDA-dependent | Low |
| Compact + repeat at SDPA | Low | High | Expanded K/V |
| Store expanded cache | High | High | Low per step |

Cache K/V bytes per layer are approximately:

```text
2 * B * Hkv * T * D * bytes_per_element
```

relative to MHA, the ratio is `Hkv/Hq`.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Head mismatch | Invalid divisibility | Print shapes | Validate geometry | Config tests |
| Future-token leakage | Wrong mask | Synthetic prefix | Absolute-position mask | Cache equivalence |
| CPU shape error | Native GQA assumed | Run CPU test | Repeat K/V | Backend smoke |
| Unexpected slowdown | Fallback kernel/copies | Profile dispatch | Adjust layout/backend | Profiler evidence |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import torch
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GroupedQueryAttention

c = ModelConfig(vocab_size=32, context_length=16, d_model=32,
                num_layers=1, num_heads=4, num_kv_heads=2)
a = GroupedQueryAttention(c).eval()
y, (k, v) = a(torch.randn(2, 6, 32))
print(y.shape, k.shape, v.shape, "groups", c.query_groups)
print("cache ratio vs MHA", c.num_kv_heads / c.num_heads)
PY
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py -q
```

Expected: output `[2,6,32]`, cache `[2,2,6,8]`, ratio `0.5`.

## Exercises

1. Compute per-layer BF16 cache bytes for `B=8,Hkv=8,T=4096,D=128`.
2. Why is `query_length == 1` causal without a mask when no prefix exists?
3. Why can passing tests still miss a backend performance regression?

## Solutions

1. `2*8*8*4096*128*2 = 134,217,728` bytes, 128 MiB.
2. There is only one key position, so no future key exists.
3. Correctness tests do not assert selected kernel, copies, memory, or latency.

## Modern LLM Systems Delta

Production inference uses paged KV caches, continuous batching, quantized
caches, fused RoPE/attention kernels, FlashAttention variants, and tensor
parallel head partitioning. Kernel constraints become architecture constraints.

## Professional Takeaways

Explain GQA in three layers: parameter geometry, persistent memory economics,
and backend dispatch. Never claim speed from FLOPs alone; profile.

## Reimplementation Kata

Tier 2: rebuild `GroupedQueryAttention.forward` -- projections, RoPE offset,
cache append, expansion policy, and the prefix-aware mask -- against the
full ablation matrix. Start with
`UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start gqa` and
follow [katas/gqa/README.md](../../katas/gqa/README.md).

## Further Exploration

- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [PyTorch SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)

