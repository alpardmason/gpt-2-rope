# Project Working Notes

## Tech Stack and Environment

- Python 3.12 managed exclusively with `uv`.
- PyTorch 2.x for model, SDPA, AMP, DDP, and profiling.
- Pydantic v2 configuration, Typer CLI, native byte-level BPE.
- Ruff, mypy strict mode, and pytest are independent quality gates.
- Standard commands: `uv run ruff check .`, `uv run mypy`, `uv run pytest`.

## Common Errors and Pitfalls

### Cached logits differ from full forward

- Symptom: logits after a multi-token cached suffix do not match a full pass.
- Root cause: using upper-left `is_causal=True` masking with a non-empty prefix.
- Fix: construct a mask from absolute query and key positions.
- Prevention: retain the cached/full equivalence regression test.

### Tokenizer cannot encode arbitrary text

- Symptom: an unknown-token lookup occurs during byte-level BPE encoding.
- Root cause: the vocabulary omitted one or more of the 256 byte symbols.
- Fix: initialize every tokenizer vocabulary with GPT-2's reversible byte map.
- Prevention: validate minimum vocabulary size and test Unicode round trips.

### Resume is not reproducible

- Symptom: resumed loss diverges immediately from an uninterrupted run.
- Root cause: missing RNG, scheduler, scaler, sampler epoch, or iterator state.
- Fix: restore the complete checkpoint before constructing/advancing training.
- Prevention: checkpoint all state and retain restoration/component tests.

### `uv` cache permission failure in the managed workspace

- Symptom: `uv` cannot initialize `$HOME/.cache/uv`.
- Root cause: the execution sandbox cannot write to the default user cache.
- Fix: run commands with `UV_CACHE_DIR=.uv-cache`.
- Prevention: keep `.uv-cache/` ignored and use the local cache in automation.

### `torchrun --standalone` stalls on macOS

- Symptom: rendezvous repeatedly logs malformed IPv6 reverse-DNS warnings.
- Root cause: local hostname resolution selects an unusable IPv6 address.
- Fix: launch with `--master-addr=127.0.0.1 --master-port=<free-port>`.
- Prevention: prefer explicit IPv4 rendezvous settings for local macOS DDP.

### Accelerator inference latency is implausibly low

- Symptom: CUDA or MPS latency resembles Python dispatch time and varies wildly.
- Root cause: accelerator kernels execute asynchronously past the host timer.
- Fix: synchronize the target device before and after each timed phase.
- Prevention: use the benchmark harness and retain its synchronization tests.

## Key Technical Decisions

### GPT-2 fidelity with deliberate GQA/RoPE incompatibility

- Decision: preserve GPT-2's remaining decoder architecture and initialization.
- Context: GQA changes K/V projections and RoPE removes learned positions.
- Alternatives: approximate stock-weight conversion or Hugging Face subclassing.
- Rationale: explicit from-scratch behavior is easier to validate and maintain.

### Native tokenizer

- Decision: own byte mapping, BPE training, encoding, decoding, and persistence.
- Context: transparency was prioritized over Rust-backed tokenizer throughput.
- Alternatives: Hugging Face `tokenizers` or a hybrid reference/fast pair.
- Rationale: keeps the learning surface self-contained; deterministic tests
  mitigate implementation risk.

### Single-node scaling

- Decision: CPU, MPS, CUDA, and single-node DDP for pretraining.
- Context: multi-node and FSDP were explicitly outside v1.
- Alternatives: single-device only or distributed checkpoint/FSDP from day one.
- Rationale: supports realistic local and workstation training without cluster
  complexity.

### Exact-resume directory checkpoints

- Decision: save all training/RNG/progress state atomically and export separate
  inference-only `safetensors`.
- Context: model-only files cannot reproduce interrupted training.
- Alternatives: weights-only checkpoints or sharded distributed checkpoints.
- Rationale: correctness and operability outweigh modest storage overhead.

### Config-switchable architecture ablations

- Decision: expose `position_encoding: rope|learned` and
  `norm_placement: pre|post` on `ModelConfig` (defaults preserve
  RoPE + pre-norm); MHA/MQA remain reachable through `num_kv_heads`.
- Context: the curriculum needs runnable A/B comparison labs without forking
  the model or weakening the default architecture.
- Alternatives: documentation-only comparisons, separate model classes per
  variant, or a strategy-object abstraction.
- Rationale: flags keep one tested code path per concern; every variant must
  pass the same cached/full equivalence and gradient tests, so ablations are
  comparable systems rather than prose claims.

### Native lifecycle subsystems (data quality, eval, sweeps, DPO, INT8, serving)

- Decision: implement the remaining workflow stages as small native modules --
  `data_quality.py` (SHA-256/MinHash dedup, reasoned filters, hashed shards),
  `evaluation.py` (perplexity suites, length-normalized choice logprobs,
  passkey probes), `sweeps.py` (grid/random over dotted overrides),
  `dpo.py` (frozen-reference DPO, full or LoRA), `quantization.py`
  (per-channel weight-only INT8), and `serving.py` (FastAPI, same-shape
  micro-batching; optional `serving` extra) -- each with CLI commands, unit
  plus component tests, and a dedicated tutorial chapter (19-24).
- Context: the project teaches the whole pretrain -> evaluate -> align ->
  compress -> serve workflow, not just pretraining.
- Alternatives: integrating lm-eval-harness/TRL/Optuna/vLLM directly, or
  documenting the stages without code.
- Rationale: transparent reference implementations carry the teaching value;
  each tutorial names the industrial tool and the graduation criterion.

### Local-first AIOps stance

- Decision: keep JSONL + TensorBoard + optional W&B as the only integrated
  tracking; sweeps, eval reports, and serving metrics reuse the same local
  JSONL/JSON artifact contract; document MLflow, DVC, lm-eval-harness, W&B
  sweeps/Optuna, and vLLM as the industrial substitutes per stage.
- Context: chosen over deeper tool integration when the curriculum was
  expanded (user decision).
- Alternatives: integrating a tracking server or hosted sweep scheduler.
- Rationale: file-based evidence is failure-proof, reviewable, and testable;
  hosted tools change the scheduler, not the artifact discipline being taught.

### Phase-separated inference benchmarks

- Decision: benchmark prefill and cached decode separately, with accelerator
  synchronization, warmup exclusion, KV-cache byte accounting, peak CUDA
  allocation, and a versioned JSON report.
- Context: serving metrics alone mix queueing, batching, tokenization, and model
  execution, making optimization claims difficult to reproduce.
- Alternatives: report only HTTP latency, use profiler traces as benchmark
  output, or adopt an external load generator before defining model baselines.
- Rationale: a small native harness produces portable evidence now and provides
  the baseline needed for later vLLM, Triton, quantization, and GPU comparisons.

### Tiered practice loop (companions, notes contract, katas)

- Decision: pair every tutorial with a practice companion
 (`docs/tutorials/practice/NN-practice.md`: tracing, prediction, tool
 walkthrough, persisted deliverables in `notes/chapters/NN.md`), and gut six
 core modules (`bpe`, `rope`, `gqa`, `kv-cache`, `checkpoint`, `dpo-loss`)
 for reimplementation on `kata/<name>` branches generated by
 `scripts/make_kata.py` from full-file skeletons in `katas/`.
- Context: reading plus labs is recognition-level study; retention requires
 retrieval, generation, and feedback against a hard oracle. The existing
 test suite and quality gates already are that oracle.
- Alternatives: scratch toy reimplementations (loses the production
 contract), pre-created long-lived kata branches (drift from `main`), or
 spaced-repetition notes alone (no generation step).
- Rationale: kata branches reuse the unchanged production tests, mypy, and
 ruff, so reimplemented code is production-grade by construction; skeletons
 on `main` plus on-demand branch generation keep `main` green and the diff
 reviewable (`git diff main -- <module>` is the built-in answer key).
 Skeletons are verbatim module copies with gutted bodies; regenerate their
 untouched parts when the source module changes.

### Source-mapped tutorial curriculum

- Decision: maintain the learning series in `docs/tutorials/` as numbered
  Markdown chapters with embedded, runnable labs and separated solutions.
- Context: the project is intended to bridge toy GPT knowledge and professional
  research-engineering practice without turning production modules into
  heavily annotated teaching files.
- Alternatives: notebooks, executable companion labs, or expanded source
  comments.
- Rationale: Markdown keeps code navigation, review, and maintenance simple;
  production code remains concise while tutorials can explain contracts,
  trade-offs, failure modes, tests, and modern-system differences in depth.

## Documentation Conventions

- `docs/tutorials/README.md` is the authoritative course order and source index.
- Tutorial filenames use a zero-padded numeric prefix and must remain sequential.
- Every tutorial includes objectives, source/test maps, contracts, engineering
  trade-offs, failure analysis, a `uv` lab, exercises and solutions, modern LLM
  differences, professional takeaways, and further reading.
- Repository links are relative and labs run from the repository root with
  `UV_CACHE_DIR=.uv-cache`.
- Accelerator-specific behavior is labeled and is not claimed from CPU-only
  evidence.
