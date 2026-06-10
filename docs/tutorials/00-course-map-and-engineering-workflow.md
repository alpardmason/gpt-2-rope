# 00: Course Map and Engineering Workflow

## Objectives and Prerequisites

Learn to read this repository as a set of contracts rather than a pile of
files. Prerequisite: a toy GPT-2 implementation and basic `pytest`.

**Source map:** [README](../../README.md), [architecture](../architecture.md),
[package API](../../src/gpt2_rope/__init__.py), [CLI](../../src/gpt2_rope/cli.py),
[tests](../../tests/), and [CI](../../.github/workflows/ci.yml).

## System Contracts

1. Configuration is validated before expensive work.
2. Text becomes versioned token IDs before training.
3. `GPT.forward` owns tensor geometry and causal-loss semantics.
4. Training owns state transitions: data, gradients, optimizer, metrics, and
   checkpoints.
5. The CLI composes library functions; it must not contain model mathematics.
6. Tests are executable design claims. CI enforces lint, typing, and behavior as
   independent gates.

Data flow:

```text
raw text -> filter/dedup/shard -> tokenizer -> uint16 streams
  -> DataLoader -> GPT -> loss -> backward -> AdamW/schedule
  -> metrics + exact-resume checkpoint
checkpoint -> eval suite (perplexity, tasks, passkey)
checkpoint -> SFT/LoRA -> DPO -> INT8 quantization -> HTTP serving
checkpoint + prompt -> KV-cached generation -> decoded text
```

## Read Code Like an Engineer

Start at a public boundary and follow data, shape, state, and failure:

```python
config = load_experiment_config(path)
run_dir = train_pretraining(config)
```

Do not immediately descend into every helper. First write the input contract,
output contract, mutable state, resource ownership, and expected failures. Then
read tests to see which claims are enforced.

**Recommendation:** read in vertical slices: CLI -> config -> subsystem -> test.
**Rationale:** this preserves user intent and makes hidden coupling visible.

| Alternative | Benefit | Cost |
|---|---|---|
| File-by-file order | Simple | Loses end-to-end behavior |
| Model-first order | Familiar | Hides data and operational correctness |
| Test-first order | Reveals contracts | Can miss untested intent |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| “Correct” function breaks CLI | Boundary contract ignored | Trace call site | Preserve interface | Read vertical slice |
| Green tests, bad training | Missing system invariant | Inspect metrics/data | Add targeted test | Separate proof from evidence |
| Hard-to-review patch | Mixed concerns | Inspect diff ownership | Split change | One subsystem per patch |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run gpt2-rope --help
UV_CACHE_DIR=.uv-cache uv run pytest --collect-only -q
rg -n "^(class|def) " src/gpt2_rope tests
```

Expected: a dozen CLI workflows spanning data preparation, training,
alignment, evaluation, compression, and serving; tests organized by
subsystem; and library entry points that are smaller than their
implementations.

Debug prompt: which production behaviors have no direct test? Name the risk,
not merely the uncovered line.

## Exercises

1. Trace `pretrain` from CLI argument to model loss.
2. Why are lint, type checking, and tests separate CI jobs?
3. Identify one boundary where a manifest or identity prevents silent misuse.

## Solutions

1. `cli.pretrain` -> `load_experiment_config` -> `train_pretraining` ->
   `MemmapTokenDataset`/`DataLoader` -> `GPT.forward(labels=input_ids)` ->
   shifted cross entropy.
2. They answer different questions and produce precise failure attribution.
3. `prepare_corpus` records `tokenizer.identity()`; checkpoints record the same
   identity so operators can detect tokenizer/model-data mismatch.

## Modern LLM Systems Delta

Large systems add dataset registries, distributed checkpointing, orchestration,
FSDP/tensor parallelism, kernel catalogs, evaluation services, and deployment
artifacts. The architectural lesson remains: explicit boundaries and durable
metadata matter more as scale increases.

## Professional Takeaways

In interviews, explain the repository as a lifecycle, then zoom into one
invariant. Strong engineers distinguish API contract, implementation, test
evidence, and operational evidence.

## Further Exploration

- [The Twelve-Factor App](https://12factor.net/)
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/)
- [pytest documentation](https://docs.pytest.org/)

