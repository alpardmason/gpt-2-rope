# 21: Hyperparameter Sweeps and Experiment Management

## Objectives and Prerequisites

Turn "try a few learning rates" into a reproducible search artifact: a
declarative space, seeded trials, ranked results, and a defensible best.
Prerequisite: 02, 11, 14.

**Source map:** [`sweeps.py`](../../src/gpt2_rope/sweeps.py) `SweepConfig`,
`enumerate_assignments`, `run_sweep`, `read_objective`;
[`config_io.py`](../../src/gpt2_rope/config_io.py) dotted overrides;
[`cli.py`](../../src/gpt2_rope/cli.py) `sweep run`;
[`test_sweeps.py`](../../tests/test_sweeps.py).

## Sweep Contract

A sweep specification is YAML:

```yaml
base_config: configs/tiny.yaml
output_dir: runs/sweeps/lr
method: grid            # or random (requires trials)
seed: 1337
objective: validation/loss
minimize: true
parameters:
  training.learning_rate: [0.0003, 0.0006, 0.001]
  training.warmup_steps: [100, 200]
```

Invariants:

- Parameters are dotted config keys, so the sweep layer adds no new
  configuration language -- each assignment becomes `--set`-style overrides
  validated by the same Pydantic models as manual runs.
- Each trial runs in `output_dir/trial-NNNN` with its own
  `resolved_config.json` and `metrics.jsonl`; the objective is the last
  logged value of the objective key.
- `sweep_results.jsonl` appends one record per finished trial (crash-safe);
  `sweep_summary.json` holds the ranked list and best assignment.
- Random search draws each parameter independently from its list with a
  seeded generator: same spec, same trials.

## Grid vs Random vs Bayesian

| Method | When | Cost behavior | Failure mode |
|---|---|---|---|
| Grid (here) | <=2-3 params, few values | multiplicative | curse of dimensionality |
| Random (here) | >3 params, unknown importance | fixed budget | misses narrow optima |
| Bayesian/ASHA (Optuna, W&B) | expensive trials | adaptive | overhead, complexity |

Random search beats grid at equal budget when only a subset of parameters
matters, because grid spends its budget repeating values of unimportant
axes. Learning rate is almost always the parameter that matters.

**Recommendation:** sequential local trials with this module while runs take
minutes; move to W&B sweeps or Optuna when trials are hours long or need
parallel workers and early stopping. **Rationale:** the artifact contract
(spec in, ranked results out) is identical; only the scheduler changes.
**Alternatives:** MLflow for tracking-first stacks; Ray Tune for cluster
scheduling.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Best trial unreproducible | objective read from noisy last step | plot metric curve | average tail or eval at fixed step | objective definition review |
| Trials overwrite each other | shared output_dir | inspect run dirs | per-trial dirs (automatic here) | path uniqueness test |
| Sweep conclusions flip with seed | single-seed trials | repeat best 3 seeds | seed as swept parameter | seed-repeat protocol |
| Objective is null | wrong metric key or no validation set | check metrics.jsonl keys | fix key/eval cadence | result null check |

## Lab

```bash
mkdir -p sweeps
cat > sweeps/demo.yaml <<'YAML'
base_config: configs/tiny.yaml
output_dir: runs/sweeps/demo
method: grid
objective: train/loss
minimize: true
parameters:
  training.learning_rate: [0.0006, 0.001]
  training.max_steps: [50]
YAML
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_sweeps.py -q
```

The unit suite exercises the full mechanics with an injected fake trainer.
For a real (short) run, ensure `data/processed` exists (chapter 04 lab),
then:

```bash
UV_CACHE_DIR=.uv-cache uv run gpt2-rope sweep run sweeps/demo.yaml
cat runs/sweeps/demo/sweep_summary.json
```

Expected: two trials, ranked summary, best assignment with the lower final
training loss.

## Exercises

1. Why does the sweep module reuse dotted overrides instead of accepting a
   dict of arbitrary config mutations?
2. You have budget for 16 trials over 5 parameters. Grid or random, and
   what grid would you build instead if forced?
3. Why is "last logged value" a fragile objective, and what would a robust
   one look like for `validation/loss`?

## Solutions

1. Overrides pass through `load_experiment_config`, so every trial config is
   validated by the same strict Pydantic contract as a manual run; an
   unvalidated mutation path would let sweeps create configs no human could
   reproduce from the CLI.
2. Random: a 5-dimensional grid needs at least 32 cells for two values each.
   If forced to grid, sweep only learning rate (3-4 values) and warmup (2),
   pinning the rest.
3. A single step's validation loss is one noisy draw; robust variants
   average the last k evaluations or evaluate once at a fixed final step
   with a fixed batch count -- both defined before the sweep runs.

## Modern LLM Systems Delta

At frontier scale, sweeps are run on small proxies and extrapolated with
scaling laws (Chinchilla-style compute-optimal fits, muP transfer for
width-invariant learning rates) because a single trial costs GPU-months.
Hosted trackers add parallel agents, early stopping (ASHA/Hyperband), and
experiment registries; the discipline of declarative spaces and ranked,
reproducible artifacts is unchanged.

## Professional Takeaways

A sweep is an experiment design, not a for-loop: declare the space, fix the
objective and budget before launching, and archive the summary next to the
checkpoints it justifies. "We chose lr=6e-4" should always have a
`sweep_summary.json` behind it.

## Further Exploration

- [Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Tensor Programs V: Tuning Large Models via muP](https://arxiv.org/abs/2203.03466)
