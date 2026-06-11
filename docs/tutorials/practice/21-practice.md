# Practice 21: Hyperparameter Sweeps and Experiment Management

Companion to [21-hyperparameter-sweeps-and-experiment-management.md](../21-hyperparameter-sweeps-and-experiment-management.md).
Persist all deliverables to `notes/chapters/21.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `sweep run` CLI to one trial's objective

Follow one sweep specification from the command line to a ranked summary.
Start at `sweep_run` in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`sweep_run` -> `load_sweep_config` in
[`sweeps.py`](../../../src/gpt2_rope/sweeps.py) -> `run_sweep` ->
`enumerate_assignments` -> `assignment_overrides` ->
`load_experiment_config` in
[`config_io.py`](../../../src/gpt2_rope/config_io.py) -> the train function
-> `read_objective`.

Record at each hop:

- Where parameter keys are sorted in `enumerate_assignments`, and what one
  grid assignment looks like (a plain dict of dotted keys to values).
- The exact override strings `assignment_overrides` renders (JSON-encoded
  values), plus the extra `training.output_dir` override `run_sweep`
  injects so trials cannot collide.
- Who validates each trial's config: confirm the assignment flows through
  the same `load_experiment_config` path as a manual `--set` run.
- When `sweep_results.jsonl` is written and flushed relative to each trial
  finishing, and why that makes the artifact crash-safe.

### Trace B: ranking with an injected trainer

Trace `test_run_sweep_writes_results_and_ranked_summary` in
[`test_sweeps.py`](../../../tests/test_sweeps.py): how `fake_train` replaces
`train_pretraining` through the `train_fn` parameter. Record:

- What `read_objective` returns when the objective key appears several
  times, once, or never in `metrics.jsonl`.
- How `run_sweep` sorts `scored` (`reverse=not config.minimize`) and what
  lands in `sweep_summary.json` under `ranked` and `best` when some
  objectives are `None`.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_grid_enumeration_is_exhaustive_and_sorted` in
   [`test_sweeps.py`](../../../tests/test_sweeps.py) to make for a
   two-parameter spec (2 learning rates x 2 warmup values): how many
   assignments and which combinations. Then read it and diff against your
   guess.
2. **Override rendering prediction.** Predict the exact list of strings
   `assignment_overrides({"training.learning_rate": 0.0005,
   "training.device": "cpu"})` returns (ordering and quoting included),
   then verify in a REPL and against
   `test_assignment_overrides_round_trip_json`.
3. **Mutation prediction.** If `read_objective` returned the FIRST match
   instead of the last (add a `break` after the first hit), predict which
   test fails and with what expected/actual values, and whether
   `test_run_sweep_writes_results_and_ranked_summary` survives (its fake
   trainer logs a single line). Verify by temporarily editing `sweeps.py`,
   running `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_sweeps.py`, and
   reverting (`git checkout -- src/gpt2_rope/sweeps.py`).
4. **Boundary prediction.** Predict the exact validation error when
   `SweepConfig.model_validate` receives `method="random"` without
   `trials`, and when a swept parameter maps to an empty list. Verify both
   in a REPL.

## 3. Tool walkthrough: `jq` and `sort` to re-derive the ranked summary

- **Why this tool.** `sweep_summary.json` is a claim; `sweep_results.jsonl`
  is the evidence. Re-deriving the ranking from the raw per-trial records is
  how you audit any experiment tracker, hosted or local, and it is the habit
  that catches "best run" mistakes before they reach a paper or a launch.
- **How.** Generate a real artifact pair with an injected fake trainer, then
  re-rank it by hand:

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import json
from pathlib import Path
from gpt2_rope.sweeps import SweepConfig, run_sweep

base = Path("runs/sweeps/play-base.yaml")
base.parent.mkdir(parents=True, exist_ok=True)
base.write_text(
    "model:\n  vocab_size: 300\n  context_length: 16\n  d_model: 16\n"
    "  num_layers: 1\n  num_heads: 2\n  num_kv_heads: 1\n"
    "data:\n  train_path: data/train.bin\n  tokenizer_dir: tokenizer\n",
    encoding="utf-8",
)
sweep = SweepConfig.model_validate({
    "base_config": base,
    "output_dir": Path("runs/sweeps/play"),
    "parameters": {"training.learning_rate": [0.001, 0.0005],
                   "training.warmup_steps": [1, 2]},
})

def fake_train(config):
    run_dir = config.training.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    loss = config.training.learning_rate * 1000 + config.training.warmup_steps
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "validation/loss": loss}) + "\n")
    return run_dir

run_sweep(sweep, train_fn=fake_train)
PY
jq -s 'sort_by(.objective) | .[] | {trial, objective}' \
  runs/sweeps/play/sweep_results.jsonl
jq '.ranked[] | {trial, objective}' runs/sweeps/play/sweep_summary.json
```

  The two listings must agree line for line.
- **Play.**
  1. Diff your `jq -s 'sort_by(...)'` ranking against `ranked` in the
     summary, then flip `minimize` to `false` in the spec dict, re-run, and
     confirm the order inverts.
  2. Point `objective` at a key the fake trainer never logs
     (`"validation/perplexity"`), re-run, and record what `sweep_results.jsonl`
     and `best` contain - this is the "objective is null" failure row.
  3. Edit the spec to `method: random` without `trials` and record the
     Pydantic diagnostic from `load_sweep_config`/`SweepConfig`. Clean up
     with `rm -rf runs/sweeps/play runs/sweeps/play-base.yaml`.

## 4. Deliverables

Append to `notes/chapters/21.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the `jq` re-ranking query you will reuse, plus the null-objective
  observation from the play exercise.
- 3-5 why-cards. Seed examples: "Why do sweep parameters reuse dotted config
  overrides instead of a free-form dict?", "Why is appending to
  `sweep_results.jsonl` per trial crash-safe while writing the summary at
  the end is not?", "What breaks if two trials share an output directory?"
- Feynman summary: explain to a colleague why a sweep is an experiment
  design rather than a for-loop, and what artifact must exist before "we
  chose lr=6e-4" is a defensible sentence.
