# GPT-2 with GQA and RoPE

A production-oriented, educational implementation of GPT-2 that replaces
multi-head attention with grouped-query attention (GQA) and learned positional
embeddings with rotary position embeddings (RoPE).

For a source-mapped, 26-part engineering course covering every subsystem, see
the [GPT-2-RoPE Engineering Curriculum](docs/tutorials/README.md).

The project covers the full LLM workflow: a native GPT-2 byte-level BPE
tokenizer, corpus engineering (dedup, quality filters, content-addressed
shards), deterministic memmap preprocessing, pretraining and supervised
fine-tuning infrastructure, LoRA adapters, DPO preference tuning, an
evaluation harness (perplexity suites, logprob tasks, passkey probes),
hyperparameter sweeps, KV-cached generation, post-training INT8
quantization, FastAPI serving with micro-batching, exact-resume checkpoints,
local and optional W&B monitoring, PyTorch profiling, typed YAML
configuration, config-switchable architecture ablations, and CI.

## Architecture

The non-positional architecture follows GPT-2: token embeddings, pre-norm
decoder blocks, GELU MLPs, residual connections, dropout, learned biases, tied
input/output embeddings, and depth-scaled residual initialization.

For hidden states `[B, T, C]`, queries are shaped `[B, Hq, T, D]` while keys and
values are `[B, Hkv, T, D]`. Each KV head serves `Hq / Hkv` query heads. The KV
cache therefore consumes approximately `Hkv / Hq` of the memory used by
ordinary multi-head attention. RoPE rotates adjacent query/key feature pairs at
each position, encoding relative displacement directly in attention dot
products without a learned position table.

Stock GPT-2 checkpoints are not numerically compatible because both positional
representation and key/value projection geometry differ.

For comparison labs, `ModelConfig` exposes ablation switches that restore
GPT-2's learned position table (`position_encoding: learned`) or
original-Transformer post-norm ordering (`norm_placement: post`); MHA and MQA
are reachable through `num_kv_heads`. See `configs/ablations/` and tutorial 17.

## Quick Start

```bash
uv sync
uv run pytest
uv run gpt2-rope --help
```

Train a tokenizer and prepare a corpus:

```bash
uv run gpt2-rope tokenizer train corpus.txt tokenizer/ --vocab-size 50257
uv run gpt2-rope data prepare corpus.txt data/processed tokenizer/
```

Run a tiny experiment:

```bash
uv run gpt2-rope pretrain configs/tiny.yaml
```

Override checked YAML values without editing the file:

```bash
uv run gpt2-rope pretrain configs/tiny.yaml \
  --set training.max_steps=20 --set training.device='"mps"'
```

Downstream workflow stages, each with its own tutorial chapter:

```bash
uv run gpt2-rope data filter raw.txt filtered.txt        # quality gates
uv run gpt2-rope data dedup filtered.txt clean.txt       # exact + MinHash
uv run gpt2-rope data shard clean.txt shards/            # hashed shards
uv run gpt2-rope eval suite configs/tiny.yaml runs/tiny/checkpoints/step-XXXXXXXX \
  --perplexity-file heldout.txt --passkey-samples 8      # evaluation
uv run gpt2-rope sweep run configs/sweeps/lr.yaml        # hyperparameter search
uv run gpt2-rope dpo configs/dpo.yaml                    # preference tuning
uv run gpt2-rope checkpoint quantize configs/tiny.yaml \
  runs/tiny/checkpoints/step-XXXXXXXX model-int8.safetensors
uv sync --extra serving && uv run gpt2-rope serve configs/tiny.yaml \
  runs/tiny/checkpoints/step-XXXXXXXX                    # HTTP inference
```

## Memory Guidance

Parameter memory alone is approximately `parameters * bytes_per_element`.
Training additionally holds gradients, optimizer moments, activations, and
temporary attention buffers. AdamW mixed-precision training commonly needs
roughly 12–20 bytes per parameter before activations. Gradient checkpointing
trades recomputation for lower activation memory. GQA reduces KV-cache memory,
which matters most during long-context batched generation.

## Operational Guidance

- Prefer BF16 on supported CUDA hardware; use FP16 plus loss scaling otherwise.
- MPS is useful for tiny smoke runs but does not match CUDA kernel coverage.
- Use `torchrun --standalone --nproc-per-node=N -m gpt2_rope.cli pretrain ...`
  for single-node DDP. On macOS systems with hostname/IPv6 resolution issues,
  pass `--master-addr=127.0.0.1 --master-port=29500` instead of `--standalone`.
- Checkpoints contain model, optimizer, scheduler, scaler, random generators,
  data progress, and resolved configuration. Resume only against the same data
  and tokenizer identity.
- JSONL and TensorBoard metrics are always local. W&B is optional and failures
  do not stop training.
- Profiler traces are expensive; enable them for short diagnostic windows.

## Development

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

Tests cover tensor geometry, RoPE invariants, cache equivalence across all
ablation variants, tokenizer round-trips and determinism, packed data, SFT
masking, LoRA merging, sampling, exact checkpoint restoration, end-to-end
training smoke and resume reproducibility, dedup/filter/shard accounting,
evaluation scoring, sweep mechanics, DPO loss and training, INT8 parity, and
the serving API.
