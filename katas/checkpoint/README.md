# Kata: checkpoint

Reimplement exact-resume checkpointing from a gutted
[`src/gpt2_rope/checkpoint.py`](../../src/gpt2_rope/checkpoint.py).
Tutorial: [13](../../docs/tutorials/13-exact-resume-checkpoint-engineering.md).
Estimated effort: one evening (2-4 hours).

## Objective

Build the reliability layer: a checkpoint directory that is written
atomically, captures every piece of state needed to make a resumed run
bit-identical to an uninterrupted one, and a separate inference-only export
that survives safetensors' shared-storage rules.

## Contract

You must satisfy, without editing any other file (`CheckpointState` is kept
for you):

- `_rng_state()` / `_restore_rng_state(state)` round-trip Python, NumPy,
  and torch RNG (CUDA states only when available).
- `save_checkpoint(...)` writes `model.pt`, `training.pt` (optimizer,
  scheduler, scaler, RNG - each `None`-tolerant), and `metadata.json`
  (version, progress, config, tokenizer identity) into a temporary
  directory that is renamed into place; a failure mid-save must leave no
  partial checkpoint behind, and an existing checkpoint at the target path
  is replaced. Returns the final path.
- `load_checkpoint(...)` restores model weights (CPU map location,
  weights-only load), each optional training component only when both the
  caller passed it and the file recorded it, RNG when `restore_rng`, and
  returns a `CheckpointState` built from the metadata.
- `export_safetensors(model, path)` exports a weights-only file; tied
  weights (GPT-2's `lm_head`/`token_embedding` share storage) must be
  materialized as independent tensors because safetensors rejects shared
  storage.

The skeleton's `# KATA:` comments restate this in place. Imports used only
by the gutted bodies were removed; re-add what you need.

## Oracle

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_checkpoint.py -q
# the real bar: resumed training is bit-identical to uninterrupted training
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_training.py::test_resume_reproduces_uninterrupted_training -q
UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py check checkpoint
```

## Workflow

Round-trip test first, then the resume-reproducibility test (it will expose
any state you forgot to capture - that is the entire lesson of chapter 13),
then mypy/ruff and the full suite. When green,
`git diff main -- src/gpt2_rope/checkpoint.py` and record the review notes
required by [katas/README.md](../README.md).

## Hint ladder (open one rung at a time)

1. Atomicity is a rename: write everything into
   `.<name>.<unique>.tmp` next to the target, then `rename` to the final
   path. Wrap the writes so any exception removes the temporary directory
   and re-raises.
2. If the resume test diverges: enumerate what training consumes besides
   weights - optimizer moments, scheduler step, scaler scale, and all three
   RNG streams. The repository's `AGENTS.md` pitfall entry "Resume is not
   reproducible" lists exactly these.
3. If safetensors export raises about shared memory: detach, move to CPU,
   `clone()`, and make contiguous each tensor so tied parameters become
   independent storage; the loader role is documented to retie them.
