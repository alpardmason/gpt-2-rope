# 13: Exact-Resume Checkpoint Engineering

## Objectives and Prerequisites

Distinguish inference weights from a training transaction and reason about
atomic publication, RNG, iterator progress, and artifact compatibility.
Prerequisite: 11-12.

**Practice companion:** [13-practice.md](practice/13-practice.md).

**Source map:** [`checkpoint.py`](../../src/gpt2_rope/checkpoint.py)
`CheckpointState`, `_rng_state`, `save_checkpoint`, `load_checkpoint`,
`export_safetensors`; [`test_checkpoint.py`](../../tests/test_checkpoint.py);
and resume logic in [`training.py`](../../src/gpt2_rope/training.py).

## Checkpoint Contract

Directory contents:

```text
model.pt       model state dict
training.pt    optimizer + scheduler + scaler + RNG states
metadata.json  version + progress + config + tokenizer identity
```

RNG state includes Python, NumPy, CPU torch, and all CUDA generators when
available. Progress records optimizer step, tokens, data position, and epoch.
The training loop restores state before advancing the stream.

Exact resume additionally assumes unchanged code, data ordering/content,
hardware/backend behavior, worker behavior, and deterministic kernels. The
checkpoint captures necessary state, not the entire universe.

## Atomic Publication

Files are written into a unique temporary sibling directory. Only after all
writes succeed is the destination replaced and the temporary directory renamed.
On failure, temporary state is removed.

Rename on one filesystem gives readers an all-or-nothing directory name under
normal local semantics. Durability against power loss would require explicit
file and directory `fsync`; object stores require a different commit protocol.

**Recommendation:** separate resume artifacts from inference artifacts.
**Rationale:** executable optimizer/RNG state has different size, trust, and
compatibility requirements than immutable tensor weights.

## Tied Weights and Safetensors

Safetensors rejects shared storage. Export clones each state tensor so tied
embedding/head values appear independently. A loader should reconstruct the
model and retie according to architecture; the export alone does not encode
Python object identity.

| Artifact | Purpose | Trust model | Contents |
|---|---|---|---|
| Directory checkpoint | Resume training | Trusted run | Python/torch state |
| Safetensors | Inference/transfer | Safer tensor format | Weights only |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Immediate loss divergence | Missing state/order | Compare uninterrupted run | Restore all before iteration | Resume regression |
| Partial checkpoint | Direct writes interrupted | Inspect missing files | Temp + rename | Atomic protocol |
| Wrong text behavior | Tokenizer mismatch | Compare identity | Use matching tokenizer | Metadata validation |
| Safetensors export error | Shared storage | Inspect tied params | Clone contiguous tensors | Export test |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_checkpoint.py -q
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from gpt2_rope.checkpoint import export_safetensors
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT

with TemporaryDirectory() as d:
    p = Path(d) / "model.safetensors"
    export_safetensors(GPT(ModelConfig(vocab_size=32, context_length=8,
        d_model=16, num_layers=1, num_heads=2, num_kv_heads=1)), p)
    print(p.stat().st_size)
PY
```

Expected: tests restore equal parameters/progress and export tied values.

## Exercises

1. Why restore optimizer state after constructing the optimizer?
2. What iterator state is approximated by `data_position`?
3. What metadata validation is recorded but not automatically enforced?

## Solutions

1. Optimizer state maps onto parameter objects/groups that must already exist.
2. The number of loader batches consumed from a deterministically seeded
   infinite stream.
3. Resolved config and tokenizer identity are returned/recorded, but the loader
   does not compare them against current inputs.

## Modern LLM Systems Delta

Large systems shard checkpoints across ranks, save asynchronously, use
distributed checkpoint APIs, upload to object storage, maintain retention
policies, and support topology changes. Commit manifests and checksums replace
single-filesystem rename assumptions.

## Professional Takeaways

Define “resume” precisely: next data, next RNG draws, next LR, optimizer
moments, scaler, and progress must agree. Test interrupted versus uninterrupted
trajectories, not only parameter loading.

## Reimplementation Kata

Tier 2: rebuild atomic save, complete restore, RNG capture, and the
safetensors export against the resume-reproducibility test. Start with
`UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start checkpoint`
and follow [katas/checkpoint/README.md](../../katas/checkpoint/README.md).

## Further Exploration

- [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
- [safetensors](https://github.com/huggingface/safetensors)
- [PyTorch distributed checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)

