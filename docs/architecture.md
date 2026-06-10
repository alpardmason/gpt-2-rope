# Architecture and Training Notes

## GPT-2 Backbone

Each decoder block is pre-normalized:

1. `x = x + GQA(LayerNorm(x))`
2. `x = x + MLP(LayerNorm(x))`

The MLP expands channels by four, applies GPT-2's tanh-approximated GELU, and
projects back to the residual width. Input embeddings and the language-model
head share one parameter matrix. Residual output projections use standard
deviation `0.02 / sqrt(2 * layers)` to control depth-wise variance.

## Ablation Switches

Two `ModelConfig` fields exist purely for comparison experiments; defaults
preserve the reference architecture (RoPE + pre-norm + GQA):

- `position_encoding: rope | learned` -- `learned` restores GPT-2's original
  absolute position table (`wpe`) and disables RoPE.
- `norm_placement: pre | post` -- `post` restores original-Transformer/GPT-1
  residual ordering (`x = LN(x + sublayer(x))`).

Attention geometry ablations (MHA/MQA) need no switch: set `num_kv_heads`.
Every variant passes the same cached/full equivalence tests; ready-made
configs live in `configs/ablations/`.

## Grouped-Query Attention

Queries use `Hq` heads while keys and values use `Hkv` heads:

```text
Q: [batch, Hq, sequence, head_dim]
K: [batch, Hkv, sequence, head_dim]
V: [batch, Hkv, sequence, head_dim]
```

`Hq` must be divisible by `Hkv`. CUDA dispatches PyTorch SDPA's native GQA path.
CPU and MPS expand K/V only at the SDPA call; the persistent inference cache
remains compact. Compared with MHA, cache bytes fall by `Hkv / Hq`.

Cached multi-token decoding uses an explicit offset-aware causal mask. PyTorch's
plain `is_causal=True` mask is upper-left aligned and is not sufficient when
query length differs from a non-empty key/value prefix.

## RoPE

RoPE treats each adjacent feature pair as a 2D vector and rotates it by a
position-dependent angle. Applying the same orthogonal rotation preserves each
query/key norm. Dot products depend on relative displacement because rotations
compose by angle subtraction.

RoPE is computed in FP32 and cast to the activation dtype. Tables are registered
as non-persistent buffers because they are deterministic from configuration.
This release intentionally omits long-context scaling.

## Causal Objective

The model accepts labels aligned with input IDs and shifts them internally:
logit `t` predicts label `t+1`. SFT labels use `-100` over prompt and padding
positions, so only response tokens contribute to cross entropy.

## Checkpoints and Reproducibility

An exact-resume checkpoint contains:

- model, optimizer, scheduler, and AMP scaler state;
- Python, NumPy, CPU, and CUDA random generator state;
- optimizer step, tokens processed, epoch, and data iterator position;
- resolved configuration and tokenizer identity.

Checkpoint directories are assembled under a temporary sibling and renamed
only after all files are durable enough for normal filesystem semantics.
Inference-only exports use `safetensors`.

## Observability

Metrics are written to JSONL and optionally TensorBoard/W&B. Core signals are
loss, perplexity, learning rate, gradient norm, supervised or total
tokens/second, peak CUDA allocation, and cumulative tokens. PyTorch profiler
traces should be sampled over short windows because shape and stack capture
can materially distort throughput; `monitoring.profile_every` captures one
optimizer step every N steps during pretraining.

## Lifecycle Subsystems

Beyond the pretraining core, the package covers the remaining LLM workflow
stages, all local-first and natively implemented:

- `data_quality.py`: exact (SHA-256) and near (MinHash) deduplication,
  heuristic quality filters with per-reason rejection accounting, and
  content-addressed document shards.
- `evaluation.py`: windowed perplexity suites, length-normalized
  multiple-choice logprob scoring, and synthetic passkey retrieval probes.
- `sweeps.py`: grid/random hyperparameter search over dotted config
  overrides with per-trial run directories and ranked summaries.
- `dpo.py`: Direct Preference Optimization against a frozen reference model
  on prompt/chosen/rejected JSONL, full-parameter or LoRA.
- `quantization.py`: post-training weight-only per-channel INT8 with
  dequantize-on-forward and compression reporting.
- `serving.py` (optional `serving` extra): FastAPI service with request
  validation, same-shape micro-batching, and latency/throughput metrics.
