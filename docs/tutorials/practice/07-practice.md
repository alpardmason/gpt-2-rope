# Practice 07: Rotary Position Embeddings in PyTorch

Companion to [07-rotary-position-embeddings-in-pytorch.md](../07-rotary-position-embeddings-in-pytorch.md).
Persist all deliverables to `notes/chapters/07.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `generate` CLI to the rotation

Follow one decode step from the command line into the math. Start at
`generate_text` in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`generate_text` -> `_load_model` -> `generate` in
[`generation.py`](../../../src/gpt2_rope/generation.py) ->
`GPT.forward` in [`model.py`](../../../src/gpt2_rope/model.py) ->
`GroupedQueryAttention.forward` -> `RotaryEmbedding.forward` in
[`rope.py`](../../../src/gpt2_rope/rope.py).

Record at each hop:

- Where is `offset` born, and what value does it carry on (a) the prompt
  forward pass and (b) the second generated token?
- At the `RotaryEmbedding.forward` call site, what are the exact shapes of
  `query` and `key` for the tiny config (`num_heads=4`, `num_kv_heads=2`,
  `head_dim=8`)? They differ in one dimension - which, and why is that safe
  for the shared `cos`/`sin` tables?
- Who owns the trig tables (module buffer, parameter, or local)? Which
  object's `state_dict()` would they appear in, and do they?

### Trace B: construction-time table build

Trace `RotaryEmbedding.__init__` line by line and record the shape after
each statement: `inverse_frequency`, `positions`, `frequencies` (the
`torch.outer`), and `angles` (the `repeat_interleave`). State in one
sentence why `repeat_interleave(..., 2, dim=-1)` rather than `torch.cat`
matches the `rotate_half` pairing of even/odd features.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertion you expect
   `test_rope_preserves_vector_norm` in
   [`test_model.py`](../../../tests/test_model.py) to make, including the
   tensor shapes it constructs. Then read it and diff against your guess.
2. **Lab output prediction.** Predict the three printed lines of the chapter
   lab (norm error magnitude, buffer names, state-dict contents) before
   running it.
3. **Mutation prediction.** If `forward` ignored its `offset` argument and
   always sliced `[0:sequence_length]`, predict: which test fails first,
   `test_rope_preserves_vector_norm` or
   `test_cached_logits_match_full_forward`, and why does the other still
   pass? Verify by temporarily editing `rope.py`, running
   `uv run pytest tests/test_model.py`, and reverting
   (`git checkout -- src/gpt2_rope/rope.py`).
4. **Boundary prediction.** Predict the exact exception type and message of
   `RotaryEmbedding(8, 16)(torch.randn(1, 1, 12, 8), torch.randn(1, 1, 12, 8), offset=5)`.
   Verify in a REPL.

## 3. Tool walkthrough: `pdb` breakpoints inside a test run

- **Why this tool.** Print-debugging a broadcast bug means re-running the
  whole test per question. A debugger lets you interrogate live tensors at
  the failure site once. Professionals drop into `pdb` inside pytest daily;
  it works over SSH where GUI debuggers do not.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py::test_cached_logits_match_full_forward --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb 2>/dev/null \
  || UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py::test_cached_logits_match_full_forward --pdb
```

  That only triggers on failure. To stop unconditionally, add
  `breakpoint()` on the first line of `RotaryEmbedding.forward` and run
  with `-s`. Useful commands: `p query.shape`, `p offset`, `u`/`d` to walk
  the stack, `ll` to list the current function, `c` to continue.
- **Play.**
  1. Set `breakpoint()` in `RotaryEmbedding.forward`, run the cached/full
     test with `-s`, and `c` through the stops. Count how many times the
     module is entered and record the `offset` sequence you observe -
     explain the sequence from the test's 6/4 token split and layer count.
  2. At one stop, evaluate `cos.shape` before and after the `view` call and
     confirm the broadcast against both the 4-head query and 2-head key.
  3. Remove the breakpoint, then re-run with `-x -q` and confirm green.

## 4. Deliverables

Append to `notes/chapters/07.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the one `pdb` command you will reuse, plus the observed
  `offset` sequence from the play exercise.
- 3-5 why-cards. Seed examples: "Why are the trig tables non-persistent
  buffers instead of parameters?", "What breaks if `head_dim` is odd?",
  "Why is `offset` essential for cached decoding but irrelevant for
  training?"
- Feynman summary: explain to a colleague why rotating Q and K (but not V)
  makes attention scores depend on relative displacement.

Tier 2: this chapter has a kata. After the deliverables above, run
`UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start rope` and
follow [katas/rope/README.md](../../../katas/rope/README.md).
