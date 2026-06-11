# Practice 13: Exact-Resume Checkpoint Engineering

Companion to [13-exact-resume-checkpoint-engineering.md](../13-exact-resume-checkpoint-engineering.md).
Persist all deliverables to `notes/chapters/13.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `checkpoint export` CLI to cloned tensors

Follow an inference export from the command line into the storage rules.
Start at `checkpoint_export` in [`cli.py`](../../../src/gpt2_rope/cli.py)
and trace:

`checkpoint_export` -> `_load_model` -> `export_safetensors` in
[`checkpoint.py`](../../../src/gpt2_rope/checkpoint.py).

Record at each hop:

- Which file inside the checkpoint directory `_load_model` reads, with what
  `map_location` and `weights_only` settings, and why `weights_only=True` is
  safe here but would fail for `training.pt`.
- In `export_safetensors`, why every tensor passes through
  `.detach().cpu().clone().contiguous()` - which two state-dict entries
  share storage in `GPT`, and what does safetensors do with shared storage?
- Which keys the exported file contains for the tied pair
  (`token_embedding.weight`, `lm_head.weight`) - one entry or two - and what
  a loader must re-establish that the file cannot encode.

### Trace B: the save/load transaction and the resume path

Trace `save_checkpoint` and `load_checkpoint` in
[`checkpoint.py`](../../../src/gpt2_rope/checkpoint.py) line by line, then
the resume block in `train_pretraining` in
[`training.py`](../../../src/gpt2_rope/training.py).

Record:

- The naming scheme of the temporary sibling directory, the order of the
  three file writes (`model.pt`, `training.pt`, `metadata.json`), and which
  single call publishes the checkpoint atomically. What happens to the
  temporary directory on failure?
- Everything `_rng_state` captures (Python, NumPy, torch CPU, and CUDA
  generators when available) and which `load_checkpoint` flag controls
  restoration.
- In `train_pretraining`: `load_checkpoint` runs before `torch.compile` and
  the DDP wrap, and before the stream is advanced by
  `progress["data_position"]` calls to `next(stream)`. Record why each of
  those three orderings is load-bearing for
  `test_resume_reproduces_uninterrupted_training`.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_checkpoint_restores_model_optimizer_and_progress` in
   [`test_checkpoint.py`](../../../tests/test_checkpoint.py) to make - what
   it does to the source model before saving, and which two things it
   verifies after loading into a fresh model. Then read it and diff against
   your guess.
2. **Test prediction.** Do the same for
   `test_safetensors_export_handles_tied_embeddings`: predict which two
   tensors it compares in the loaded file and what comparison it uses. Then
   read it. Also predict, before running the chapter lab, the order of
   magnitude of the printed file size from the tiny config's parameter
   count (vocab 32, `d_model` 16, one layer, 4 bytes per value).
3. **Mutation prediction.** Remove `.clone()` from the dict comprehension in
   `export_safetensors`. Predict the exact failure: which test fails, and
   does the error come from this repository's code or from safetensors'
   shared-storage rejection? Verify with
   `uv run pytest tests/test_checkpoint.py -q`, then revert
   (`git checkout -- src/gpt2_rope/checkpoint.py`).
4. **Boundary prediction.** Predict the exception type raised by
   `load_checkpoint(Path(d), model=..., optimizer=None, scheduler=None, scaler=None)`
   when `d` is an empty directory (which file is touched first?). Verify in
   a REPL with a `TemporaryDirectory` and a tiny `GPT`.

## 3. Tool walkthrough: checkpoint directory anatomy

- **Why this tool.** A checkpoint you cannot read with `ls` and ten lines of
  Python is a checkpoint you cannot debug at 3 a.m. Production incidents
  routinely come down to "what is actually inside this directory, and does
  the metadata match the code?" - so practice answering that with nothing
  but a shell and a REPL.
- **How.** Create a real checkpoint, then dissect it:

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from pathlib import Path

import torch

from gpt2_rope.checkpoint import save_checkpoint
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT

config = ModelConfig(vocab_size=32, context_length=8, d_model=16,
                     num_layers=1, num_heads=2, num_kv_heads=1, dropout=0.0)
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss = model(torch.randint(0, 32, (2, 5)),
             labels=torch.randint(0, 32, (2, 5))).loss
loss.backward()
optimizer.step()
save_checkpoint(
    Path("runs/practice-13/step-1"), model=model, optimizer=optimizer,
    scheduler=None, scaler=None,
    progress={"step": 1, "tokens": 10, "data_position": 2, "epoch": 0},
    config={"model": config.model_dump()},
    tokenizer_identity={"sha256": "practice"})
PY
ls -la runs/practice-13/step-1
UV_CACHE_DIR=.uv-cache uv run gpt2-rope checkpoint inspect runs/practice-13/step-1
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import json
from pathlib import Path

import torch

root = Path("runs/practice-13/step-1")
metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
print("metadata keys:", sorted(metadata))
training = torch.load(root / "training.pt", map_location="cpu",
                      weights_only=False)
print("training keys:", sorted(training))
print("rng captures:", sorted(training["rng"]))
PY
```

- **Play.**
  1. Try `torch.load(root / "training.pt", weights_only=True)` and record
     the failure - then explain why the trust models of `model.pt` and
     `training.pt` differ (tensors versus pickled RNG/optimizer state).
  2. Compare the byte sizes of `model.pt` and `training.pt` from `ls -la`
     and explain the ratio from AdamW's two moment tensors per parameter.
  3. Re-run the save script and confirm the directory is replaced cleanly:
     no `.step-1.*.tmp` residue next to it, and
     `gpt2-rope checkpoint inspect` still parses. Then explain what an
     interrupted write would leave behind instead.

## 4. Deliverables

Append to `notes/chapters/13.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the inspect commands you will reuse, plus the observed
  `model.pt`/`training.pt` size ratio and its explanation.
- 3-5 why-cards. Seed examples: "Why is the temporary directory a sibling
  of the destination rather than `/tmp`?", "What breaks if optimizer state
  is restored before the optimizer's parameter groups exist?", "Why does
  exact resume need RNG state even though the model weights are bit-exact?"
- Feynman summary: explain to a colleague why "resume" is a claim about the
  next data batch, the next RNG draw, and the next learning rate - not just
  about reloading weights - and how the directory layout encodes that.

Tier 2: this chapter has a kata. After the deliverables above, run
`UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start checkpoint`
and follow [katas/checkpoint/README.md](../../../katas/checkpoint/README.md).
