# GPT-2-RoPE Engineering Curriculum

This is not a “build a Transformer in 30 minutes” course. It is a guided code
review of a small but production-oriented language-model system. You are
expected to inspect source and tests while reading, run the labs, and explain
each invariant without relying on memorized framework recipes.

## Entry Contract

You should already understand decoder-only Transformers, causal language
modeling, backpropagation, and basic PyTorch modules. Learn LLM theory from
the papers linked in each chapter; this course teaches the system around the
math. By the end, you should be able to turn a paper into a tested subsystem,
defend architecture trade-offs with ablation evidence, debug training
failures, run the full pretrain -> evaluate -> align -> quantize -> serve
workflow, and discuss this repository in a research-engineering interview.

Run every command from the repository root:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen
UV_CACHE_DIR=.uv-cache uv run pytest
```

CPU labs are mandatory. CUDA and MPS sections are explicitly marked; do not
infer accelerator behavior from a CPU result.

## Sequence

| # | Tutorial | Primary system | Difficulty |
|---:|---|---|---|
| 00 | [Course map and engineering workflow](00-course-map-and-engineering-workflow.md) | Repository | 2/5 |
| 01 | [Reproducible Python and CI](01-reproducible-python-and-ci-toolchain.md) | Toolchain | 3/5 |
| 02 | [Configuration as a contract](02-configuration-as-an-experiment-contract.md) | Pydantic/YAML | 3/5 |
| 03 | [Byte-level BPE](03-byte-level-bpe-from-files-to-training.md) | Tokenizer | 4/5 |
| 04 | [Pretraining data and memmap](04-pretraining-data-pipelines-and-memmap.md) | Data | 4/5 |
| 05 | [SFT data and loss masking](05-supervised-data-and-loss-masking.md) | Data/objective | 4/5 |
| 06 | [Model contracts and initialization](06-model-contracts-initialization-and-weight-tying.md) | GPT backbone | 4/5 |
| 07 | [RoPE in PyTorch](07-rotary-position-embeddings-in-pytorch.md) | Position encoding | 4/5 |
| 08 | [GQA and SDPA](08-grouped-query-attention-and-sdpa.md) | Attention | 5/5 |
| 09 | [KV cache and generation](09-kv-cache-and-autoregressive-generation.md) | Inference | 5/5 |
| 10 | [Optimizer, schedule, and precision](10-optimizer-schedules-and-mixed-precision.md) | Optimization | 4/5 |
| 11 | [A correct pretraining loop](11-building-a-correct-pretraining-loop.md) | Training | 5/5 |
| 12 | [Distributed data parallel](12-distributed-data-parallel-training.md) | Scaling | 5/5 |
| 13 | [Exact-resume checkpoints](13-exact-resume-checkpoint-engineering.md) | Reliability | 5/5 |
| 14 | [Observability and profiling](14-observability-evaluation-and-profiling.md) | AIOps | 4/5 |
| 15 | [LoRA and SFT](15-lora-and-supervised-fine-tuning.md) | Adaptation | 4/5 |
| 16 | [PyTorch, JAX, and MLX](16-framework-philosophy-pytorch-jax-and-mlx.md) | Framework design | 4/5 |
| 17 | [Ablations: positions, norms, attention](17-ablation-studies-positions-norms-attention.md) | Architecture evidence | 4/5 |
| 18 | [GPT-2 vs modern SOTA architectures](18-gpt2-vs-modern-sota-architectures.md) | Architecture literacy | 3/5 |
| 19 | [Corpus engineering](19-corpus-engineering-dedup-filtering-sharding.md) | Data quality | 4/5 |
| 20 | [Evaluation harnesses and benchmarks](20-evaluation-harnesses-and-benchmarks.md) | Evaluation | 4/5 |
| 21 | [Sweeps and experiment management](21-hyperparameter-sweeps-and-experiment-management.md) | Experimentation | 3/5 |
| 22 | [Preference optimization with DPO](22-preference-optimization-with-dpo.md) | Alignment | 5/5 |
| 23 | [Post-training quantization](23-post-training-quantization.md) | Compression | 4/5 |
| 24 | [Inference serving and deployment](24-inference-serving-and-deployment.md) | Serving | 4/5 |
| 25 | [Paper-to-production capstone](25-capstone-paper-to-production-engineering.md) | Architecture practice | 5/5 |

The dependency spine is:

```text
toolchain -> contracts -> tokenizer -> data
         -> model -> RoPE -> GQA -> cache
         -> optimization -> training -> DDP
         -> checkpoints -> observability -> LoRA
         -> framework judgment
         -> ablations -> SOTA literacy -> corpus quality
         -> evaluation -> sweeps -> DPO
         -> quantization -> serving -> capstone
```

## Source Index

| Component | Tutorials |
|---|---|
| `config.py`, `config_io.py`, YAML | 02, 17, 21, 25 |
| `tokenizer.py`, `assets.py` | 03 |
| `data.py` | 04, 05, 19 |
| `data_quality.py` | 19 |
| `model.py`, `rope.py` | 06-09, 17, 18 |
| `generation.py` | 09, 24 |
| `training.py` | 10-12, 14, 15 |
| `checkpoint.py` | 13, 23 |
| `monitoring.py` | 14, 21, 24 |
| `lora.py` | 15, 22 |
| `evaluation.py` | 20, 23, 25 |
| `sweeps.py` | 21 |
| `dpo.py` | 22 |
| `quantization.py` | 23 |
| `serving.py` | 24 |
| `cli.py`, CI, packaging | 00-02, 14, 19-24, 25 |

## Study Tracks

- **Model internals:** 00, 02, 06-10, 13, 17, 18.
- **Training systems:** 00-05, 10-15, 19, 21.
- **Post-training and inference:** 05, 09, 15, 20, 22-24.
- **Research engineer interview:** all chapters; present the capstone aloud.
- **Apple-silicon practitioner:** all chapters, with special attention to
  CPU/MPS fallback behavior in 08, precision in 10, and MLX in 16.

## AIOps Tooling Stance

The project is local-first by design: JSONL metrics, TensorBoard, and
optional W&B (14); sweep artifacts as files (21); content hashes as data
versioning (19); a transparent eval harness (20). Each chapter names the
industrial substitute (MLflow, DVC, lm-evaluation-harness, W&B sweeps/
Optuna, vLLM) and states when graduating to it is justified.

## How To Study

For each chapter: read its contracts, open every linked symbol, predict each
test before running it, complete the lab, then answer the exercises without
looking at the solution. A passing test proves a stated behavior for tested
inputs; it does not prove numerical stability, scale, performance, or complete
correctness.
