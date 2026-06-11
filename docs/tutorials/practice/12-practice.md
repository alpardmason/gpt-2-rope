# Practice 12: Distributed Data Parallel Training

Companion to [12-distributed-data-parallel-training.md](../12-distributed-data-parallel-training.md).
Persist all deliverables to `notes/chapters/12.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`. All
work in this companion is CPU/Gloo only; do not infer NCCL behavior from it.

## 1. Tracing tasks

### Trace A: through the two-process Gloo test

The distributed entry point with a hard oracle is the test, not a CLI run.
Start at `test_two_process_cpu_gloo_step` in
[`test_distributed.py`](../../../tests/test_distributed.py) and trace:

`test_two_process_cpu_gloo_step` -> `multiprocessing.spawn` ->
`_ddp_worker` -> `distributed.init_process_group("gloo")` ->
`DistributedDataParallel(GPT(...))` -> one backward/step ->
`distributed.all_gather` of parameter checksums.

Record at each hop:

- Which environment variables `_ddp_worker` sets itself
  (`MASTER_ADDR`, `MASTER_PORT`) versus which `torchrun` would normally
  supply (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`), and where `_free_port`
  decides to skip the test entirely.
- Both ranks call `torch.manual_seed(5)` before constructing the model.
  Name the second, independent mechanism that guarantees identical initial
  parameters across ranks even without that seed (what does DDP broadcast
  at construction?).
- What is the shape and dtype of `checksum`, and how many elements does
  `gathered` hold for `world_size=2`? What exact property does the final
  `assert_close` loop prove - identical gradients, identical parameters, or
  identical losses?

### Trace B: distributed branches inside `train_pretraining`

Trace the distributed code paths in
[`training.py`](../../../src/gpt2_rope/training.py): `initialize_distributed`
(reading `WORLD_SIZE`/`RANK`/`LOCAL_RANK`, choosing `nccl` versus `gloo`),
`seed_everything(seed, rank)`, the `DistributedSampler` construction, the
`model.no_sync()` accumulation branch, and the `is_primary` guards.

Record:

- The exact tuple `initialize_distributed` returns when `WORLD_SIZE` is
  unset, and whether a process group exists afterward.
- Why each rank seeds with `base_seed + rank`, yet the sampler is built with
  the shared `seed=config.training.seed` - which stream must differ across
  ranks and which must agree?
- Which artifacts only rank zero writes (resolved config, metrics,
  checkpoints, evaluation), and where `step_tokens` multiplies by
  `world_size` - what assumption from `drop_last=True` makes that
  multiplication honest?

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertion you expect
   `test_two_process_cpu_gloo_step` in
   [`test_distributed.py`](../../../tests/test_distributed.py) to make and
   the collective it uses to gather evidence across ranks. Then read it and
   diff against your guess.
2. **Lab output prediction.** Predict the output of the chapter lab
   `uv run pytest tests/test_distributed.py -q` on your machine: pass, or
   skip (re-read `_free_port` for the skip condition) - and roughly how
   long it takes given process spawn cost.
3. **Mutation prediction.** Change `torch.manual_seed(5)` in `_ddp_worker`
   to `torch.manual_seed(5 + rank)`. Predict: does the test still pass, and
   which DDP construction-time behavior decides the answer? Verify with
   `uv run pytest tests/test_distributed.py -q`, then revert
   (`git checkout -- tests/test_distributed.py`).
4. **Boundary prediction.** Predict what
   `initialize_distributed()` returns in a fresh REPL with `WORLD_SIZE`
   unset, and what `torch.distributed.is_initialized()` reports afterward.
   Verify:
   `uv run python -c "from gpt2_rope.training import initialize_distributed; import torch.distributed as d; print(initialize_distributed(), d.is_initialized())"`.

## 3. Tool walkthrough: `torchrun` with explicit IPv4 rendezvous

- **Why this tool.** `torchrun` is the production launcher: it injects the
  rank topology as environment variables and restarts failed workers.
  Knowing what it injects - and how rendezvous fails - is the difference
  between debugging a hang in minutes versus hours. On macOS the documented
  pitfall (see `AGENTS.md`) is hostname resolution selecting an unusable
  IPv6 address; explicit `--master-addr=127.0.0.1` avoids it.
- **How.** Probe the injected environment without needing training data:

```bash
cat > /tmp/ddp_probe.py <<'PY'
import os

import torch.distributed as distributed

distributed.init_process_group("gloo")
print(f"rank={os.environ['RANK']} local_rank={os.environ['LOCAL_RANK']} "
      f"world_size={os.environ['WORLD_SIZE']}")
distributed.destroy_process_group()
PY
UV_CACHE_DIR=.uv-cache uv run torchrun \
  --master-addr=127.0.0.1 --master-port=29501 --nproc-per-node=2 \
  /tmp/ddp_probe.py
```

  The same flags prefix a real run, as in the chapter:
  `torchrun --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=2 -m gpt2_rope.cli pretrain configs/tiny.yaml`
  (requires prepared data; do not run it as part of this walkthrough).
- **Play.**
  1. Re-run the probe with `--standalone` instead of the explicit
     address/port flags and watch for the malformed IPv6 reverse-DNS
     warnings or a rendezvous stall (interrupt with Ctrl-C if it hangs);
     record the first diagnostic line you see.
  2. Run with `--nproc-per-node=1` and confirm the probe prints
     `world_size=1` - then explain why `initialize_distributed` would skip
     process-group creation entirely for that value.
  3. Launch the probe twice concurrently on the same `--master-port` and
     record the bind/rendezvous error text - this is the signature to
     recognize when a previous run left a port occupied.

## 4. Deliverables

Append to `notes/chapters/12.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the `torchrun` incantation you will reuse on macOS, plus the
  diagnostic you observed in each play exercise.
- 3-5 why-cards. Seed examples: "Why must per-rank RNG streams differ while
  the sampler seed agrees?", "What breaks if one rank skips a backward that
  the others execute?", "Why does `no_sync` on all but the last micro-batch
  preserve gradient correctness?"
- Feynman summary: explain to a colleague what DDP synchronizes (gradients,
  at backward time) and what it deliberately does not (filesystem writes,
  data partitioning, RNG), and who owns each of those instead.
