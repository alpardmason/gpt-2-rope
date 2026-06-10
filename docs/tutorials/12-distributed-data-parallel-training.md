# 12: Distributed Data Parallel Training

## Objectives and Prerequisites

Understand DDP process topology, data partitioning, gradient synchronization,
accumulation, rank ownership, and reproducibility boundaries. Prerequisite: 11.

**Source map:** [`training.py`](../../src/gpt2_rope/training.py)
`initialize_distributed`, `seed_everything`, training wrappers;
[`test_distributed.py`](../../tests/test_distributed.py); and operational notes
in the root [README](../../README.md).

## Process Contract

`torchrun` supplies `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`.

```text
WORLD_SIZE=1 -> rank 0, no process group
multi-GPU -> NCCL, set CUDA device(local_rank)
multi-CPU -> Gloo
```

`rank` identifies a global process; `local_rank` selects a device on one node.
This project supports one node only.

Each rank constructs the same model and optimizer. `DistributedSampler`
partitions/shuffles dataset indices with a shared seed; `set_epoch(epoch)`
changes the deterministic shuffle each pass. Each rank receives seed
`base_seed + rank`, preventing identical stochastic streams.

## Gradient Synchronization

DDP installs autograd hooks that all-reduce gradients. During accumulation:

```text
micro 0..A-2: model.no_sync() -> local accumulation
micro A-1: normal backward -> all-reduce accumulated gradients
```

Synchronizing every micro-step is usually correct but wastes communication.
`step_tokens` multiplies local token count by `world_size`, assuming equal
batch sizes (`drop_last=True` supports that).

Only rank zero writes config, logs, evaluates, and checkpoints. A barrier is
used in fine-tuning after directory creation; pretraining relies on rank-zero
creation before later writes and does not barrier immediately.

**Recommendation:** distinguish mathematical synchronization from filesystem
coordination. **Rationale:** DDP solves gradients, not concurrent artifact I/O.

| Scaling method | Replicated model | Main use |
|---|---:|---|
| DDP | Yes | Model fits each device |
| FSDP/ZeRO | No/partly | Shard model/optimizer |
| Tensor parallel | No | Layer too large for device |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Hang | Rank missed collective | Stack traces/log ranks | Align control flow | Multi-process smoke |
| Duplicate samples | Sampler absent/wrong | Log indices | DistributedSampler | Data test |
| Slow accumulation | All-reduce every micro | Profile communication | `no_sync` | Profiler |
| macOS rendezvous warnings/hang | Bad IPv6 hostname | Inspect torchrun logs | Explicit IPv4 address/port | Launch recipe |
| Corrupt artifacts | Multiple writers | Inspect rank guards | Rank-zero ownership | I/O contract |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_distributed.py -q
```

The test spawns two CPU/Gloo workers, performs one update, all-gathers parameter
checksums, and asserts equality. It may skip where local sockets are forbidden.

macOS local launch pattern:

```bash
UV_CACHE_DIR=.uv-cache uv run torchrun \
  --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=2 \
  -m gpt2_rope.cli pretrain configs/tiny.yaml
```

Do not run this full training command as a lab unless data exists.

## Exercises

1. Why does DDP not reduce model memory per GPU?
2. Why must all ranks take compatible backward paths?
3. Does `base_seed + rank` guarantee exact resume?

## Solutions

1. Every process stores a full model and usually a full optimizer.
2. Gradient collectives must be entered in matching order; divergence hangs or
   errors.
3. No. RNG states, sampler epoch, data position, model/optimizer/scheduler/
   scaler, and backend determinism also matter.

## Modern LLM Systems Delta

Large training combines data, tensor, pipeline, context, and expert parallelism;
uses topology-aware collectives, distributed checkpoints, elastic recovery,
and communication/computation overlap.

## Professional Takeaways

Explain distributed systems using topology, ownership, synchronization, and
failure domains. “Use DDP” is not an architecture explanation.

## Further Exploration

- [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [ZeRO](https://arxiv.org/abs/1910.02054)

