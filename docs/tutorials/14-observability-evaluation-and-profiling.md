# 14: Observability, Evaluation, and Profiling

## Objectives and Prerequisites

Design evidence for training health, model behavior, and systems performance
without allowing optional telemetry to kill the run. Prerequisite: 10-13.

**Source map:** [`monitoring.py`](../../src/gpt2_rope/monitoring.py)
`MetricLogger`; [`training.py`](../../src/gpt2_rope/training.py) `evaluate`,
`run_profiler`; profile/evaluate commands in
[`cli.py`](../../src/gpt2_rope/cli.py); and `MonitoringConfig`.

## Metrics Contract

Every primary-rank record is appended and flushed to `metrics.jsonl`.
TensorBoard is optional local visualization. W&B is dynamically imported and
best-effort: initialization/logging/shutdown failures are logged while local
metrics remain authoritative.

The same local-first contract has industrial equivalents this project
deliberately documents instead of integrating: MLflow (tracking server with
runs, params, and a model registry), DVC (content-addressed data and
pipeline versioning -- the `source_sha256` manifest in chapter 19 is its
seed), and lm-evaluation-harness for benchmark breadth (chapter 20). Adopt
them when multiple people or machines must share evidence; the JSONL file
remains the failure-proof substrate either way.

Core signals:

```text
model: loss, perplexity, token accuracy
optimization: learning rate, gradient norm
system: tokens/s, peak CUDA allocation
progress: optimizer step, cumulative tokens
```

Loss is useful but insufficient. Throughput can collapse while loss looks
normal; gradients can explode before loss becomes non-finite; validation can
diverge while training improves.

Evaluation weights loss by predicted token count before averaging. Token
accuracy is interpretable but weak for open-ended language modeling. Timing
contains transfer and model execution and is a coarse benchmark.

## Profiler Semantics

Profiler schedule:

```text
wait -> warmup -> active capture, repeated N times
```

`profiler.step()` advances the schedule. Shape recording, memory profiling, and
stack capture add overhead; traces are diagnostic samples, not normal training
mode. CPU is always captured; CUDA activity is included only when requested.

Two entry points exist: the one-shot `gpt2-rope profile` command, and
`MonitoringConfig.profile_every`, which captures one full optimizer step
every N steps during pretraining and writes the trace to
`run_dir/profiler/` -- the "periodic" rung of the ladder below.

CUDA memory snapshots are attempted at checkpoint boundaries and failures do
not stop training. “Allocated” memory differs from reserved memory and total
device use.

**Recommendation:** define an observability ladder from cheap always-on metrics
to expensive sampled traces. **Rationale:** measurement changes the system.

| Layer | Cost | Example |
|---|---:|---|
| Always-on | Low | JSONL loss/LR/tokens |
| Periodic | Medium | Validation, memory snapshot |
| Diagnostic | High | Profiler shapes/stacks |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Run dies when W&B is down | Telemetry coupled to training | Simulate import/network failure | Best-effort integration | Local source of truth |
| Misleading throughput | Different token/count/timing boundary | Define numerator/window | Standardize metric | Metric docs |
| Trace shows slow run | Profiler overhead | Compare unprofiled baseline | Short capture | Sampling policy |
| Perplexity overflows | Exponentiating huge loss | Inspect finite loss | Cap display/repair model | Numerical guard |

## Lab

```bash
tmp="$(mktemp -d)"
UV_CACHE_DIR=.uv-cache uv run python - <<PY
from pathlib import Path
from gpt2_rope.monitoring import MetricLogger

with MetricLogger(Path("$tmp"), tensorboard=False) as log:
    log.log(1, {"train/loss": 3.2, "train/tokens_per_second": 1234})
print(Path("$tmp/metrics.jsonl").read_text())
PY
```

Expected: one flushed JSON object containing step and metrics.

For a configured tiny model, `gpt2-rope profile` writes TensorBoard traces; it
performs real optimization work and should be run only with a suitable config.

## Exercises

1. Why flush JSONL on every log?
2. Distinguish token accuracy from perplexity.
3. What alert would combine gradient norm and loss?

## Solutions

1. It improves crash visibility at modest I/O cost.
2. Accuracy measures argmax matches; perplexity is exponentiated average
   negative log likelihood and uses the full predicted distribution.
3. Alert on non-finite values or a sustained norm spike followed by loss
   increase; thresholds should be baseline-derived, not universal.

## Modern LLM Systems Delta

Production AIOps adds centralized telemetry, GPU fabric metrics, data-quality
signals, run comparison, evaluation harnesses, canary jobs, alert routing,
cost accounting, and model/data lineage.

## Professional Takeaways

For every metric define owner, unit, aggregation, synchronization, frequency,
failure policy, and action. A dashboard without decisions is decoration.

## Further Exploration

- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Weights & Biases experiment tracking](https://docs.wandb.ai/)

