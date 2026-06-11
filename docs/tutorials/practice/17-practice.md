# Practice 17: Ablation Studies: Position Encodings, Norm Placement, Attention Geometry

Companion to [17-ablation-studies-positions-norms-attention.md](../17-ablation-studies-positions-norms-attention.md).
Persist all deliverables to `notes/chapters/17.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from a `--set` override to a different module tree

Follow one ablation switch from the command line into the constructed
model. Start at `pretrain` in [`cli.py`](../../../src/gpt2_rope/cli.py)
with `--set model.position_encoding=learned` and trace:

`pretrain` -> `load_experiment_config` in
[`config_io.py`](../../../src/gpt2_rope/config_io.py) (dotted-key walk,
`_parse_value` JSON fallback) -> `ModelConfig` validation in
[`config.py`](../../../src/gpt2_rope/config.py) -> `GPT.__init__` and
`GroupedQueryAttention.__init__` in
[`model.py`](../../../src/gpt2_rope/model.py).

Record at each hop:

- How `_parse_value` decides between JSON and raw string for
  `learned`, `200`, and `"runs/ablations/baseline"` - and why the chapter
  lab quotes the output dir as `'"runs/ablations/baseline"'`.
- Which two module attributes differ between the RoPE and learned-PE
  variants: `GPT.position_embedding` (an `nn.Embedding` or `None`) and
  `GroupedQueryAttention.rope` (a `RotaryEmbedding` or `None`). Which
  config field gates each one?
- In `GPT.forward` for the learned variant, the lookup index is
  `past_length + torch.arange(...)`. Which RoPE concept does `past_length`
  correspond to here, and what shape does the added embedding broadcast
  over?
- In `TransformerBlock.forward`, which exact lines differ between the
  `pre` and `post` branches - what gets normalized, and what stays an
  identity path?

### Trace B: the ablation matrix in the test suite

Trace the `ABLATION_VARIANTS` dict in
[`test_model.py`](../../../tests/test_model.py) and the two parameterized
tests that consume it (`test_ablation_variants_cached_match_full_forward`,
`test_ablation_variants_train_step`). Record: the six variant ids; which
config overrides each id applies; and the 6/4 prefix/suffix token split the
cached test uses. State in one sentence why every variant must pass the
cached-vs-full test before its training curve is worth comparing.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the body you expect
   `test_ablation_variants_cached_match_full_forward` in
   [`test_model.py`](../../../tests/test_model.py) to have: how it splits
   the sequence, what it compares, and with what tolerances. Then read it
   and diff against your guess.
2. **Lab output prediction.** Predict the total test count reported by
   `uv run pytest tests/test_model.py -q` - count the standalone tests and
   expand the two parameterizations over the six variants first. Then run
   it.
3. **Mutation prediction.** In `GPT.forward`, change the learned-PE lookup
   from `past_length + torch.arange(...)` to plain `torch.arange(...)`.
   Predict exactly which test ids fail (which of the six variants, in
   which of the two parameterized tests) and why the train-step test
   cannot catch it. Verify with `uv run pytest tests/test_model.py -q`,
   then revert (`git checkout -- src/gpt2_rope/model.py`).
4. **Boundary prediction.** `ModelConfig(d_model=18, num_heads=2, ...)`
   gives an odd `head_dim` of 9. Predict which variant survives: the
   default `position_encoding="rope"` or
   `position_encoding="learned"` - and the exact validation message for
   the one that fails (see `validate_geometry`). Verify both constructions
   in a REPL.

## 3. Tool walkthrough: dotted `--set` overrides on `gpt2-rope pretrain`

- **Why this tool.** An ablation is only trustworthy if exactly one
  variable moved, and dotted overrides are how you move it without forking
  YAML files. Knowing precisely how override values parse - and how loudly
  bad ones fail - is what keeps "single-switch" claims honest.
- **How.** First verify the parsing contract without needing data:

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from pathlib import Path

from gpt2_rope.config_io import load_experiment_config

config = load_experiment_config(
    Path("configs/tiny.yaml"),
    ["model.position_encoding=learned",
     "model.norm_placement=post",
     "model.num_kv_heads=4",
     "training.max_steps=200"],
)
print(config.model.position_encoding, config.model.norm_placement,
      config.model.num_kv_heads, config.training.max_steps)
PY
```

  The real experiment (requires the prepared corpus from chapter 04) is
  the chapter lab's command family:

```bash
UV_CACHE_DIR=.uv-cache uv run gpt2-rope pretrain configs/tiny.yaml \
  --set training.max_steps=200 --set training.output_dir='"runs/ablations/baseline"'
UV_CACHE_DIR=.uv-cache uv run gpt2-rope pretrain configs/ablations/tiny_learned_pe.yaml \
  --set training.max_steps=200
```

- **Play.**
  1. Misspell a key (`--set model.positon_encoding=learned` via the REPL
     loader) and record the diagnostic - `StrictConfig`'s
     `extra="forbid"` is what turns a typo into a hard failure instead of
     a silently ignored setting.
  2. Set `model.num_kv_heads=3` against `num_heads=4` and record which
     validator in `config.py` rejects it and with what message.
  3. Pass an override with no `=` (`["training.max_steps"]`) and record
     the `ValueError` from `load_experiment_config` - compare against
     `test_override_requires_key_value_form` in
     [`test_config_io.py`](../../../tests/test_config_io.py).

## 4. Deliverables

Append to `notes/chapters/17.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the override commands you will reuse, plus the three failure
  diagnostics from the play exercises.
- 3-5 why-cards. Seed examples: "Why must the learned-PE variant pass the
  cached-vs-full test despite having no `offset` logic?", "What breaks if
  an ablation run changes two config fields at once?", "Why is post-norm
  kept as a config switch instead of deleted?"
- Feynman summary: explain to a colleague why an architecture default is a
  claim about evidence, and how config switches plus shared invariant
  tests make that claim cheap to re-litigate.
