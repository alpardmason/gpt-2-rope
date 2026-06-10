# 18: GPT-2 vs Modern SOTA Architectures

## Objectives and Prerequisites

Place this repository's GPT-2 + RoPE + GQA stack precisely on the map of
modern decoder families, and explain each delta as an engineering decision
rather than fashion. Prerequisite: 06-09, 17. This chapter changes no code:
the project deliberately stays GPT-2-shaped so every modern delta is visible
against a fixed reference.

**Source map:** [`model.py`](../../src/gpt2_rope/model.py) as the reference
implementation; [`architecture.md`](../architecture.md) for the summary.

## The Reference Point

This codebase is GPT-2 (2019) with exactly two modernizations: RoPE replaces
the learned position table, and GQA replaces full MHA. Everything else --
LayerNorm with bias, GELU MLP at 4x, dropout, tied embeddings, depth-scaled
init -- is faithful GPT-2.

## Component-by-Component Comparison

| Component | This repo (GPT-2+RoPE+GQA) | Llama 3 | Qwen 2.5 | DeepSeek-V3 |
|---|---|---|---|---|
| Norm | LayerNorm + bias, pre | RMSNorm, pre | RMSNorm, pre | RMSNorm, pre |
| MLP | GELU, 4x, biases | SwiGLU, ~2.7x, no bias | SwiGLU, no bias | SwiGLU + MoE experts |
| Position | RoPE | RoPE (scaled, long context) | RoPE (YaRN long context) | RoPE (decoupled, on MLA) |
| Attention | GQA | GQA | GQA (+QKV bias) | MLA (latent compression) |
| Vocabulary | 50,257 BPE | 128K tiktoken-style | 152K BPE | 129K BPE |
| Embeddings | tied | untied at large sizes | tied at small, untied large | untied |
| Dropout | yes | no (one epoch, no reuse) | no | no |
| Routing | dense | dense | dense (MoE variants exist) | MoE: ~37B of 671B active |

Three structural observations:

1. **The skeleton is unchanged.** Token embedding, N pre-norm residual
   blocks, final norm, head: every model above is still GPT-2's diagram.
2. **The deltas are throughput/scale-driven.** RMSNorm removes a mean
   subtraction; SwiGLU buys quality per FLOP; bias removal saves memory
   traffic; dropout dies when data is never repeated; MoE decouples
   parameter count from per-token compute.
3. **Attention deltas are cache-driven.** MHA -> GQA -> MLA is one story:
   shrink KV bytes per token without losing quality. GQA shares KV heads
   (`Hkv/Hq` ratio, chapter 08). MLA compresses K/V into a low-rank latent
   (e.g. 512-dim vs 16K) and reconstructs at compute time, cutting cache by
   another order of magnitude at the cost of extra projections and a
   decoupled RoPE path -- RoPE cannot be applied inside the shared latent, so
   a small per-head rotary component is carried separately.

## What Did Not Change and Why

Causal masking, residual connections, softmax attention scaling, AdamW-family
optimization, and the pre-norm decision (chapter 17) survived every
generation since GPT-2. When a component survives a 1000x scale-up, treat it
as a load-bearing wall.

**Recommendation:** learn modern architectures as diffs against GPT-2, not as
new species. **Rationale:** every component above is independently swappable
and independently testable -- exactly how this repository treats RoPE and
GQA. **Alternatives:** studying each model in isolation; slower and obscures
which choices are correlated.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| "RMSNorm is faster" claimed blindly | benchmark from different stack | profile both on target | measure, then claim | accelerator-labeled evidence |
| SwiGLU port has wrong FFN size | 4x ratio assumed | parameter count check | ~2/3 * 4x with two up-projections | parameter-count test |
| MLA treated as free GQA upgrade | decoupled RoPE path missed | logits mismatch vs reference | implement rope/nope split | cached-equivalence test |
| MoE quality regressions | load balancing ignored | expert utilization stats | aux-loss/bias balancing | routing entropy metrics |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from gpt2_rope.config import ModelConfig

# Cache bytes per token per layer, BF16: 2 (K and V) * Hkv * D * 2 bytes.
def kv_bytes(hkv: int, d: int) -> int:
    return 2 * hkv * d * 2

gpt2_small = ModelConfig()  # 12 heads, 4 kv heads by this repo's default
mha = kv_bytes(12, 64)
gqa = kv_bytes(gpt2_small.num_kv_heads, gpt2_small.head_dim)
mla_latent = 512 * 2  # one shared latent vector replaces K and V
print(f"MHA  {mha} B/token/layer")
print(f"GQA  {gqa} B/token/layer  ({gqa/mha:.2f}x)")
print(f"MLA  {mla_latent} B/token/layer ({mla_latent/mha:.2f}x, latent=512)")
PY
```

Expected: GQA cuts cache to `Hkv/Hq` of MHA; an MLA-style 512-dim latent cuts
it by roughly another 3x at GPT-2-small geometry (the advantage grows with
model width).

## Exercises

1. Why does removing biases from linear layers matter more at 70B than 124M?
2. A SwiGLU MLP with hidden ratio `r` has three matrices instead of two.
   What ratio keeps parameters equal to a 4x GELU MLP?
3. Why can DeepSeek-V3 not apply standard RoPE inside MLA's compressed
   latent, and what is the workaround?

## Solutions

1. Bias FLOPs are negligible but bias tensors add kernel launches and memory
   traffic per layer; at depth 80+ and tensor-parallel width the cost and
   the synchronization complexity compound, while quality impact is nil.
2. Two-matrix MLP: `2 * 4 * d^2 = 8d^2`. SwiGLU: `3 * r * d^2`. Equal at
   `r = 8/3 ~= 2.67`, the ratio Llama-family models use.
3. RoPE is a per-position rotation of K; MLA reconstructs K from a shared
   latent, so position-dependent rotation would have to be re-applied per
   cached position, destroying the compression. The workaround is a split
   head: a small rotary sub-dimension cached separately, concatenated with
   the position-free reconstructed part.

## Modern LLM Systems Delta

This chapter is the delta. The remaining gaps between this repo and frontier
practice are scale infrastructure (FSDP/tensor/pipeline/expert parallelism),
data scale (chapter 19 at petabyte size), and post-training depth (chapter 22
plus RLHF/verifier pipelines).

## Professional Takeaways

Interviewers ask "what would you change to make this Llama-class?" The
strong answer is an ordered diff with costs: RMSNorm (cheap, safe), SwiGLU
(retrain), untie embeddings (memory), drop dropout (data-dependent), GQA->MLA
(serious engineering, cache-bound regimes only), MoE (systems project, not an
architecture tweak).

## Further Exploration

- [Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
