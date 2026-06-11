# Practice 25: Capstone - Paper to Production Engineering

Companion to [25-capstone-paper-to-production-engineering.md](../25-capstone-paper-to-production-engineering.md).
Persist all deliverables to `notes/chapters/25.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: the equivalence your design must preserve

The capstone's release blocker is cached/full logit equivalence. Trace the
path the baseline lab exercises, from
`test_cached_logits_match_full_forward` in
[`test_model.py`](../../../tests/test_model.py) through `GPT.forward` in
[`model.py`](../../../src/gpt2_rope/model.py) ->
`GroupedQueryAttention.forward` -> `RotaryEmbedding.forward` in
[`rope.py`](../../../src/gpt2_rope/rope.py).

Record at each hop:

- Where the absolute position offset originates on the cached suffix pass
  and how it reaches `RotaryEmbedding.forward`.
- Which tensors `RotaryEmbedding.__init__` precomputes
  (`inverse_frequency`, `positions`, `frequencies`, the
  `repeat_interleave` into `angles`), their shapes for the tiny config,
  and why the cos/sin tables are non-persistent buffers.
- Every line a RoPE scaling feature would have to touch on this path, and
  the one place the tutorial says the scaling computation must be owned.

### Trace B: the configuration contract a new field must pass

Trace how an architecture flag travels from YAML to the model:
`load_experiment_config` in
[`config_io.py`](../../../src/gpt2_rope/config_io.py) -> `ModelConfig` in
[`config.py`](../../../src/gpt2_rope/config.py) (note `StrictConfig`'s
`extra="forbid"` and the `validate_geometry` model validator) ->
`GPT.__init__`. Record:

- Where a misspelled or unknown field is rejected, and with what error
  class.
- How the existing `position_encoding: Literal["rope", "learned"]` flag is
  consumed, as the template for a `rope_scaling_type` field - including
  which existing tests
  (`test_ablation_variants_cached_match_full_forward`,
  `test_learned_position_encoding_builds_table_and_rope_does_not` in
  [`test_model.py`](../../../tests/test_model.py)) every config-switchable
  variant must pass.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the structure you expect
   `test_cached_logits_match_full_forward` in
   [`test_model.py`](../../../tests/test_model.py) to have: the
   prefix/suffix token split it uses and the tolerance it asserts. Then
   read it and diff against your guess.
2. **Lab output prediction.** Predict the magnitude of the single number
   the capstone baseline lab prints (max abs difference between full and
   cached suffix logits) relative to the repository's stated test
   tolerance, before running it.
3. **Mutation prediction.** Simulate an unconditional position-
   interpolation bug: halve `inverse_frequency` in
   `RotaryEmbedding.__init__` (multiply the expression by `0.5`). Predict
   which tests in `tests/test_model.py` fail - work through
   `test_rope_preserves_vector_norm` and
   `test_cached_logits_match_full_forward` from first principles before
   answering. Verify with
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py`, revert
   (`git checkout -- src/gpt2_rope/rope.py`), and record what the result
   implies about the golden-table differential test the capstone requires.
4. **Boundary prediction.** Predict the exact exception type and message
   for `ModelConfig(rope_scaling_type="linear")` (a field that does not
   exist yet) and for a config whose `d_model`/`num_heads` produce an odd
   head dimension under RoPE. Verify both in a REPL.

## 3. Tool walkthrough: the quality-gate chain on a feature branch

- **Why this tool.** The capstone is judged like a real change: lint, type
  check, and test gates plus a reviewable diff. Professionals run the full
  chain locally before review, and `git diff --stat` discipline - knowing
  exactly which files a design touches - is what keeps a paper-to-code
  change auditable.
- **How.** Work on a scratch branch and run the three independent gates:

```bash
git checkout -b capstone-design
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run mypy
UV_CACHE_DIR=.uv-cache uv run pytest
git diff --stat main
```

- **Play.**
  1. Add an unused import to `src/gpt2_rope/rope.py`, run `ruff check .`
     and record the rule code, then fix it with
     `git checkout -- src/gpt2_rope/rope.py`. Note which CI step would
     have caught it.
  2. Remove one return-type annotation in `rope.py`, run `uv run mypy`, and
     record the strict-mode diagnostic; revert again. Strict typing is part
     of the contract your new `ModelConfig` field must satisfy.
  3. Time `uv run pytest` end to end and record it: that number is the
     budget your capstone's added unit, component, and smoke tests must
     respect. Finish by confirming `git diff --stat main` is empty and
     returning to your main branch (`git checkout -`).

## 4. Deliverables

Append to `notes/chapters/25.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the three gate commands, the recorded test-suite runtime, and
  the `git diff --stat` reading from the play exercise.
- 3-5 why-cards. Seed examples: "Why is cached/full equivalence a release
  blocker for any position-encoding change?", "Why does the scaling
  interface belong in `ModelConfig` while the math belongs in
  `RotaryEmbedding`?", "What breaks if a long-context feature ships
  without a `'none'`-path differential test?"
- A capstone design note in `notes/chapters/25.md`: one page following the
  tutorial's Phase 1-2 structure - the chosen paper's contract, the exact
  `ModelConfig` fields with validation rules, the symbol-by-symbol mapping
  onto `RotaryEmbedding`, and the first three tests you would write before
  any implementation.
- Feynman summary: explain to a colleague why "the model generated text at
  a longer context" is not evidence of long-context quality, and what
  predeclared evaluation would be.
