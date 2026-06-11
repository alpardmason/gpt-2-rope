# Practice 02: Configuration as an Experiment Contract

Companion to [02-configuration-as-an-experiment-contract.md](../02-configuration-as-an-experiment-contract.md).
Persist all deliverables to `notes/chapters/02.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `--set` to a validated model

Follow one dotted override from the command line into a frozen object.
Start at `pretrain` in [`cli.py`](../../../src/gpt2_rope/cli.py) (the
`--set` option) and trace:

`pretrain` -> `load_experiment_config` in
[`config_io.py`](../../../src/gpt2_rope/config_io.py) -> `_parse_value` ->
`ExperimentConfig.model_validate` -> `validate_geometry` and
`validate_schedule` in [`config.py`](../../../src/gpt2_rope/config.py).

Record at each hop:

- How the dotted key `training.max_steps=20` is walked: what the `cursor`
  variable points at on each loop iteration, and what `setdefault` does
  when a section such as `monitoring` is absent from the YAML.
- What Python type `_parse_value` returns for `"20"`, `"true"`, `"True"`,
  and `"runs/custom"`, and which of those reach Pydantic unchanged.
- Where `extra="forbid"` and `frozen=True` live (`StrictConfig` in
  `config.py`) and which classes inherit them. Who would catch a misspelled
  `model.head_count=4` - the override walker or the validator?

### Trace B: geometry validation order

Trace `ModelConfig.validate_geometry` line by line with
`d_model=64, num_heads=8, num_kv_heads=3` and record which of the three
checks fires first. Then record how the derived properties `head_dim`,
`query_groups`, and `mlp_hidden_size` are computed and why they are
properties rather than stored fields.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_override_parses_json_values` in
   [`test_config_io.py`](../../../tests/test_config_io.py) to make,
   including the override strings it passes and the types it asserts. Then
   read it and diff against your guess.
2. **Lab output prediction.** Predict the chapter lab's printed lines: the
   three values on the first line (`max_steps`, `gradient_checkpointing`,
   `head_dim` for [`tiny.yaml`](../../../configs/tiny.yaml) with its
   overrides) and the exception class name printed when assigning to the
   frozen instance. Then run it.
3. **Mutation prediction.** Delete the
   `num_heads % num_kv_heads` check from `validate_geometry` in
   [`config.py`](../../../src/gpt2_rope/config.py). Predict which assertion
   of `test_model_config_validates_gqa_geometry` in
   [`test_config.py`](../../../tests/test_config.py) fails and with what
   pytest wording (a `DID NOT RAISE` failure). Verify with
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_config.py`, then revert
   (`git checkout -- src/gpt2_rope/config.py`).
4. **Boundary prediction.** Predict the exact exception type and message of
   `load_experiment_config(path, ["training.max_steps"])` (no equals sign)
   and of `load_experiment_config(path, ["model.vocab_size.nested=1"])`
   (override through a scalar). Verify both in a REPL against
   `configs/tiny.yaml`.

## 3. Tool walkthrough: `uv run python -i` against the Pydantic contract

- **Why this tool.** A REPL is the fastest oracle for "what does the
  contract actually accept?" Reading validators tells you intent; calling
  them tells you behavior, including error wording your users will see.
  Professionals interrogate config schemas interactively before wiring
  them into launch scripts.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run python -i -c "
from pathlib import Path
from gpt2_rope.config import ModelConfig, model_preset
from gpt2_rope.config_io import load_experiment_config, _parse_value
c = load_experiment_config(Path('configs/tiny.yaml'))
"
```

  Inside the session, useful probes: `c.model.model_dump()`,
  `ModelConfig.model_json_schema()['properties'].keys()`,
  `_parse_value('null')`, `model_preset('gpt2-medium').query_groups`.
- **Play.**
  1. Construct `ModelConfig(d_model=63, num_heads=8)` and read the full
     `ValidationError`: record which validator produced it and how many
     errors are reported.
  2. Call `model_preset('gpt2-nano')` and record the error message listing
     valid choices; then compute `head_dim` and `query_groups` for every
     real preset and note which presets share a KV-head count.
  3. Attempt `c.model = c.model` and
     `ModelConfig.model_validate({'d_model': 64, 'num_heads': 8, 'head_count': 4})`.
     Record how the frozen and extra-forbid failures differ in wording.

## 4. Deliverables

Append to `notes/chapters/02.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the REPL probe you will reuse, plus the `ValidationError`
  structure you observed.
- 3-5 why-cards. Seed examples: "Why does `vocab_size <= 65_535` belong in
  the config rather than the data pipeline?", "What breaks if overrides
  were applied after `model_validate` instead of before?", "Why is the
  resolved config dump, not the input YAML, the provenance artifact?"
- Feynman summary: explain to a colleague why configuration is an API with
  validation, immutability, and provenance requirements, using the
  `--set` flow as the worked example.
