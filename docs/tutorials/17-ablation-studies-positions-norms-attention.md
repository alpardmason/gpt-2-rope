# 17: Ablation Studies: Position Encodings, Norm Placement, Attention Geometry

## Objectives and Prerequisites

Run controlled architecture ablations with config switches instead of code
forks, and defend each default with evidence. Prerequisite: 06-11, plus the
RoPE, pre-norm, and GQA papers.

**Source map:** [`config.py`](../../src/gpt2_rope/config.py)
`position_encoding`, `norm_placement`, `num_kv_heads`;
[`model.py`](../../src/gpt2_rope/model.py) `GPT.position_embedding`,
`TransformerBlock.forward`; [`configs/ablations/`](../../configs/ablations/);
[`test_model.py`](../../tests/test_model.py) `ABLATION_VARIANTS`
parameterization.

## Ablation Contract

An ablation is one switched variable against the reference architecture
(RoPE + pre-norm + GQA), with identical seed, data, schedule, and budget:

```text
position_encoding: rope | learned     (default rope)
norm_placement:    pre  | post        (default pre)
num_kv_heads:      Hq (MHA) ... 1 (MQA)   (default GQA)
```

Every variant must pass the same cached-vs-full equivalence and gradient-flow
tests; an ablation that cannot decode correctly is not a comparable system.

## Absolute Learned PE vs RoPE

| Dimension | Learned absolute (GPT-2 `wpe`) | RoPE |
|---|---|---|
| Parameters | `context_length * d_model` table | none |
| What is encoded | absolute index, added to embeddings | relative displacement, rotated into Q/K |
| Beyond trained length | undefined rows; hard failure | degrades; scalable (PI/YaRN) |
| KV-cache interaction | position fixed at embedding time | rotation applied at write time with `offset` |
| Extrapolation evidence | poor | strong with scaling |
| Where it lives | embedding sum, once | every attention layer, every forward |

The deciding question is interface, not accuracy at training length: absolute
PE makes position a property of the residual stream; RoPE makes it a property
of the attention operator. Long-context engineering (chapter 25's capstone)
is only tractable in the second framing.

## PreNorm vs PostNorm

```text
pre:  x = x + Attn(LN(x));  x = x + MLP(LN(x))
post: x = LN(x + Attn(x));  x = LN(x + MLP(x))
```

| Dimension | Pre-norm (GPT-2+) | Post-norm (original Transformer/GPT-1) |
|---|---|---|
| Identity gradient path | yes, unnormalized residual | no, every path crosses LN |
| Deep-stack stability | trains without warmup tricks | needs warmup/smaller LR as depth grows |
| Representation scale | residual norm grows with depth | normalized after every block |
| Final LN | required | redundant but harmless |
| Modern usage | universal (with RMSNorm) | rare; hybrid variants exist |

**Recommendation:** keep pre-norm as the default and treat post-norm purely
as a teaching ablation. **Rationale:** pre-norm's unnormalized residual path
is why deep decoders train reliably. **Alternatives:** post-norm with careful
warmup; hybrid schemes (e.g. "peri-norm"/double-norm) used by some recent
models.

## MHA vs GQA vs MQA

`num_kv_heads` already spans the family (chapter 08): cache bytes scale by
`Hkv/Hq`, quality cost grows as KV heads shrink. The ablation configs pin
everything else so the trade-off curve is measurable, not rhetorical.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Variant beats baseline wildly | second variable changed | diff `resolved_config.json` | re-run single-switch | config diff review |
| Post-norm diverges | LR/warmup tuned for pre-norm | gradient-norm curves | longer warmup, lower LR | log `train/gradient_norm` |
| Learned-PE crash past 128 | position table exhausted | index error trace | respect `context_length` | context validation test |
| Conclusions flip across seeds | run-to-run noise | repeat with 3 seeds | report mean and spread | seeded sweep (chapter 21) |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py -q
UV_CACHE_DIR=.uv-cache uv run gpt2-rope pretrain configs/tiny.yaml \
  --set training.max_steps=200 --set training.output_dir='"runs/ablations/baseline"'
UV_CACHE_DIR=.uv-cache uv run gpt2-rope pretrain configs/ablations/tiny_learned_pe.yaml \
  --set training.max_steps=200
UV_CACHE_DIR=.uv-cache uv run gpt2-rope pretrain configs/ablations/tiny_postnorm.yaml \
  --set training.max_steps=200
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import json, pathlib
for run in ["baseline", "learned-pe", "postnorm"]:
    path = pathlib.Path(f"runs/ablations/{run}/metrics.jsonl")
    losses = [json.loads(l)["train/loss"] for l in path.read_text().splitlines()
              if "train/loss" in l]
    print(f"{run:12s} first={losses[0]:.3f} last={losses[-1]:.3f}")
PY
```

Expected: all three runs train; compare final losses and gradient-norm
behavior. At 200 tiny-scale steps differences are suggestive, not conclusive.

## Exercises

1. Why must the learned-PE variant pass the cached-vs-full test even though
   it has no RoPE `offset` logic?
2. Post-norm both passes its tests and trains at 4 layers. What evidence
   would justify still forbidding it at 48 layers?
3. Count the parameters the learned-PE ablation adds for the tiny config
   (`context_length=128`, `d_model=128`).

## Solutions

1. With cache, position indices continue from `past_length`; the embedding
   lookup must use `past_length + arange(T)`, the same absolute-position bug
   class as RoPE offsets. The test guards the contract, not the mechanism.
2. Gradient-norm growth with depth, divergence without long warmup, and the
   literature's depth-scaling results; a 4-layer pass does not transfer.
3. `128 * 128 = 16,384` parameters.

## Modern LLM Systems Delta

Frontier labs run ablation matrices at small scale and extrapolate with
scaling laws before committing compute. The discipline is identical: one
switch, fixed budget, seeded repeats, decisions recorded with evidence links.
RoPE + pre-norm(RMS) + GQA/MLA is the current consensus stack precisely
because these ablations were run at scale.

## Professional Takeaways

Architecture defaults are claims about evidence. Maintain the ability to
re-litigate them cheaply: switches in config, invariants in tests, budgets in
sweeps. An ablation without a fixed budget and seed protocol is an anecdote.

## Further Exploration

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
