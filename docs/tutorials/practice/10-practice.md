# Practice 10: Optimizer Schedules and Mixed Precision

Companion to [10-optimizer-schedules-and-mixed-precision.md](../10-optimizer-schedules-and-mixed-precision.md).
Persist all deliverables to `notes/chapters/10.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `pretrain` CLI to one optimizer update

Follow the optimization machinery from the command line into one step. Start
at `pretrain` in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`pretrain` -> `train_pretraining` in
[`training.py`](../../../src/gpt2_rope/training.py) ->
`_effective_precision` -> `GPT.configure_optimizer` in
[`model.py`](../../../src/gpt2_rope/model.py) -> the `LambdaLR` wrapper
around `cosine_learning_rate` -> the `GradScaler` construction -> the update
sequence inside the step loop.

Record at each hop:

- Which parameters land in the decay group versus the no-decay group in
  `configure_optimizer`? State the single rule (`parameter.ndim >= 2`) and
  name two concrete parameters that fall on each side for the tiny config.
- The `LambdaLR` lambda divides `cosine_learning_rate(...)` by
  `config.training.learning_rate`. What multiplicative factor does it return
  at step 0, and what would `scheduler.get_last_lr()` show before the first
  `scheduler.step()`?
- What does `_effective_precision` return on a CPU device with
  `precision="auto"`, and what context does `_autocast_context` produce in
  that case? Is the `GradScaler` enabled?
- Write out the exact order of the six calls from
  `scaler.scale(loss).backward()` to `optimizer.zero_grad(set_to_none=True)`
  and mark where `clip_grad_norm_` sees unscaled gradients.

### Trace B: `cosine_learning_rate` region by region

Trace `cosine_learning_rate` in
[`training.py`](../../../src/gpt2_rope/training.py) line by line. Record the
condition and return expression for each of the three regions (warmup,
cosine, floor), the value of `ratio` and `coefficient` at the first cosine
step, and why the warmup branch uses `(step + 1)` rather than `step`. State
in one sentence what `max(1, max_steps - warmup_steps)` protects against.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_cosine_schedule_warmup_and_floor` in
   [`test_training.py`](../../../tests/test_training.py) to make, including
   the warmup/max-step settings it constructs and the exact values it checks
   at the boundaries. Then read it and diff against your guess.
2. **Lab output prediction.** Predict all six printed lines of the chapter
   lab (steps 0, 1, 3, 4, 7, 10 with `warmup_steps=4`, `max_steps=10`,
   max LR `1e-3`, min LR `1e-4`) to the printed precision before running it.
3. **Mutation prediction.** If the warmup branch returned
   `max_learning_rate * step / warmup_steps` instead of `(step + 1)`,
   predict: which assertion of `test_cosine_schedule_warmup_and_floor` fails
   first, and with what observed value? Verify by temporarily editing
   `training.py`, running
   `uv run pytest tests/test_training.py::test_cosine_schedule_warmup_and_floor`,
   and reverting (`git checkout -- src/gpt2_rope/training.py`).
4. **Boundary prediction.** Predict the return value of
   `cosine_learning_rate(10, warmup_steps=10, max_steps=10, max_learning_rate=1.0, min_learning_rate=0.1)`
   and state which of the three branches fires. Verify in a REPL.

## 3. Tool walkthrough: REPL schedule plotting as a text table

- **Why this tool.** Schedule bugs are off-by-one bugs, and they are
  invisible in a loss curve until thousands of steps are wasted. Printing
  the entire schedule as a text table before launching a run is a one-minute
  check professionals do for every new schedule; it needs no plotting stack
  and works over SSH.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from gpt2_rope.training import cosine_learning_rate

settings = dict(warmup_steps=4, max_steps=20,
                max_learning_rate=1e-3, min_learning_rate=1e-4)
for step in range(24):
    lr = cosine_learning_rate(step, **settings)
    bar = "#" * round(lr / settings["max_learning_rate"] * 40)
    print(f"{step:3d} {lr:.6f} {bar}")
PY
```

- **Play.**
  1. Set `warmup_steps=0` and observe which branch handles step 0 now;
     record the first three values and explain them from the cosine
     expression.
  2. Set `min_learning_rate` equal to `max_learning_rate` and confirm the
     table goes flat after warmup; explain why from the
     `coefficient * (max - min)` term.
  3. Inspect scaler state: run
     `python -c "from torch.amp.grad_scaler import GradScaler; print(GradScaler('cuda', enabled=False).state_dict())"`
     via `uv run` and record what a disabled scaler serializes - this is
     exactly what `save_checkpoint` stores for CPU runs.

## 4. Deliverables

Append to `notes/chapters/10.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the schedule-table command you will reuse, plus the disabled
  `GradScaler` state you observed.
- 3-5 why-cards. Seed examples: "Why must `scaler.unscale_` run before
  `clip_grad_norm_`?", "What breaks if LayerNorm scales receive weight
  decay?", "Why does the LambdaLR factor divide by the configured max LR?"
- Feynman summary: explain to a colleague why the optimizer update is an
  ordered transaction, naming each of the six calls and what state it
  mutates.
