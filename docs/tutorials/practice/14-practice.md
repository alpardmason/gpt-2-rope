# Practice 14: Observability, Evaluation, and Profiling

Companion to [14-observability-evaluation-and-profiling.md](../14-observability-evaluation-and-profiling.md).
Persist all deliverables to `notes/chapters/14.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `profile` CLI to the trace handler

Follow one profiling session from the command line into the scheduler math.
Start at `profile_model` in [`cli.py`](../../../src/gpt2_rope/cli.py) and
trace:

`profile_model` -> the `work()` closure (zero_grad, forward with labels,
backward, step) -> `run_profiler` in
[`training.py`](../../../src/gpt2_rope/training.py) -> `torch.profiler`'s
`schedule(wait, warmup, active, repeat)` -> `tensorboard_trace_handler`.

Record at each hop:

- The shape and dtype of the synthetic `input_ids` that `profile_model`
  builds from the config, and why profiling uses random tokens rather than
  the real dataset.
- With `ProfilingConfig` defaults (`wait=1`, `warmup=1`, `active=3`,
  `repeat=1`), how many times does the loop call `work()`, and how many of
  those iterations are actually captured?
- Which `ProfilerActivity` entries are used on CPU, who advances the
  schedule (`profiler.step()`), and where the trace files land on disk.
- The second profiling entry point: where in `train_pretraining` does
  `_maybe_profile_step` consult `MonitoringConfig.profile_every`, and which
  rank is allowed to profile?

### Trace B: `MetricLogger` lifecycle inside the training loop

Trace `MetricLogger` in
[`monitoring.py`](../../../src/gpt2_rope/monitoring.py) as
`train_pretraining` uses it: construction with `enabled=is_primary`, `log`,
and `close` in the `finally` block. Record: the open mode of
`metrics.jsonl` and what it implies for restarted runs; the exact record
shape (`{"step": step, **metrics}`, sorted keys, flushed per call); which
metric values reach TensorBoard (numeric only) versus the JSONL (all); and
the two places W&B failures are swallowed so the run survives.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_metric_logger_writes_jsonl_records` in
   [`test_monitoring.py`](../../../tests/test_monitoring.py) to make,
   including the exact dicts it expects to parse back from the file. Then
   read it and diff against your guess.
2. **Lab output prediction.** Predict the single JSON line the chapter lab
   prints, character for character - mind `sort_keys=True` and where the
   `step` key lands in the ordering.
3. **Mutation prediction.** Remove the `self.metrics_file.flush()` line in
   `MetricLogger.log`. Predict: does any test in
   `tests/test_monitoring.py` fail, and if not, which real-world failure
   mode (crash mid-run) was the flush protecting? Verify with
   `uv run pytest tests/test_monitoring.py -q`, then revert
   (`git checkout -- src/gpt2_rope/monitoring.py`).
4. **Boundary prediction.** `evaluate` caps perplexity via
   `math.exp(min(mean_loss, 20))`. Predict the reported
   `validation/perplexity` for a mean loss of 25.0 and for exactly 20.0,
   then verify the two numbers in a REPL. State what would happen to the
   JSONL record without the cap.

## 3. Tool walkthrough: TensorBoard and a chrome trace

- **Why this tool.** JSONL answers "what happened"; the profiler trace
  answers "where did the time go". Reading a trace timeline - which op,
  which thread, how long - is a core skill for diagnosing slow steps, and
  TensorBoard is the local-first viewer this project standardizes on.
- **How.** Generate traces with the one-shot CLI (real optimization work on
  the tiny config; expect it to take a minute on CPU), then point the
  viewers at them:

```bash
UV_CACHE_DIR=.uv-cache uv run gpt2-rope profile configs/tiny.yaml runs/practice-14/profiler
ls runs/practice-14/profiler
UV_CACHE_DIR=.uv-cache uv run tensorboard --logdir runs/practice-14/profiler
```

  TensorBoard serves on a local port; the profiler tab needs the
  `torch-tb-profiler` plugin to render. Independently of TensorBoard, each
  `*.pt.trace.json` file is a standard chrome trace: load it in a Chromium
  browser at `chrome://tracing` or at the Perfetto UI. This walkthrough
  asks you to open and describe the timeline; do not record performance
  claims from a casually loaded CPU box.
- **Play.**
  1. Without any GUI, measure the trace from the JSON itself: load the
     `*.pt.trace.json` with `json.load` in a REPL, count the
     `traceEvents`, and find the single longest event by `dur` - name the
     op and relate it to a line of `model.py`.
  2. Re-run the profile command and compare wall time against the same
     five `work()` iterations run bare (time them with
     `time.perf_counter` in a REPL) - record the measured profiler
     overhead and why traces are samples, not normal training mode.
  3. Break the input on purpose: point `tensorboard --logdir` at an empty
     directory and record the diagnostic the page shows; this is the
     symptom signature of a wrong `--logdir` in real incidents.

## 4. Deliverables

Append to `notes/chapters/14.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the profile and TensorBoard commands you will reuse, plus the
  longest trace event you found and the measured profiler overhead.
- 3-5 why-cards. Seed examples: "Why flush the JSONL on every log call?",
  "Why are W&B failures swallowed while JSONL writes are not?", "What
  breaks if `profiler.step()` is never called inside the loop?"
- Feynman summary: explain to a colleague the observability ladder -
  always-on JSONL, periodic validation, sampled profiler traces - and why
  each rung exists at a different cost.
