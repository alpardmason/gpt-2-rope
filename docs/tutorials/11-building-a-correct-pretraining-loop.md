# 11: Building a Correct Pretraining Loop

## Objectives and Prerequisites

Audit a training loop as a state machine with explicit ownership of gradients,
data position, mode, metrics, and failure cleanup. Prerequisite: 04, 06, 10.

**Source map:** [`training.py`](../../src/gpt2_rope/training.py)
`evaluate`, `_infinite_loader`, `train_pretraining`;
[`monitoring.py`](../../src/gpt2_rope/monitoring.py); and
[`tiny.yaml`](../../configs/tiny.yaml).

## Initialization Order

The loop resolves distributed state/device, seeds RNGs, writes resolved config,
builds dataset/sampler/loaders, constructs model/tokenizer/optimizer/scheduler/
scaler, restores checkpoint state, then optionally compiles and wraps DDP.

Order is correctness:

- Load raw-model state before wrappers obscure names/state.
- Restore optimizer/scheduler/scaler before updates.
- Restore progress before advancing the infinite data stream.
- Only rank zero writes shared artifacts.

## Gradient Accumulation

For `A` micro-batches:

```python
loss = output.loss / A
loss.backward()
```

Dividing makes accumulated gradients equivalent to the mean over the effective
batch, assuming equal micro-batch sizes. One optimizer/scheduler step follows.
Effective sequences per optimizer step are:

```text
micro_batch_size * accumulation_steps * world_size
```

The current logged `accumulated_loss` sums already-divided micro-losses, so it
is their mean. Token count includes all input tokens across ranks.

## Evaluation and Mode Ownership

`evaluate` uses `@torch.inference_mode()`, calls `model.eval()`, computes
loss/perplexity/token accuracy/throughput, and the caller restores
`model.train()`. Perplexity exponent is capped at loss 20 to avoid overflow in
monitoring.

The training loop uses `try/finally` to close the logger and destroy the
process group. This protects resources on exceptions; it does not automatically
save an emergency checkpoint.

**Recommendation:** make progress state serializable and update it at one clear
commit point per optimizer step. **Rationale:** resume semantics depend on what
“completed step” means.

| Loop style | Clarity | Flexibility | Hidden state |
|---|---:|---:|---:|
| Explicit Python loop | High | High | Visible |
| Trainer framework | Medium | High | More callbacks/framework state |
| Compiled whole step | Lower initially | Performance | Capture constraints |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Effective LR changes with A | Loss not divided | Compare grad norms | Normalize loss | Equivalence test |
| Dropout stays disabled | Train mode not restored | Check `model.training` | Restore after eval | Mode test |
| Resume repeats/skips data | Position accounting wrong | Trace each `next` | Persist/advance consistently | Resume test |
| OOM after eval | Graph retained | Inspect grad mode | Inference mode | Eval contract |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import torch
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT

torch.manual_seed(0)
c = ModelConfig(vocab_size=32, context_length=8, d_model=16,
                num_layers=1, num_heads=2, num_kv_heads=1, dropout=0)
m = GPT(c)
opt = m.configure_optimizer(1e-3, 0)
opt.zero_grad(set_to_none=True)
for _ in range(2):
    x = torch.randint(0, 32, (2, 8))
    (m(x, labels=x, use_cache=False).loss / 2).backward()
print(float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)))
opt.step()
PY
```

Expected: one finite global norm and one optimizer update after two backwards.

## Exercises

1. Why call `zero_grad(set_to_none=True)`?
2. Why evaluate `raw_model` rather than the DDP wrapper on rank zero?
3. What makes exact accumulation equivalence fail?

## Solutions

1. It avoids filling existing gradient buffers with zeros and lets PyTorch
   distinguish absent gradients.
2. Only rank zero evaluates and no cross-rank gradient synchronization is
   needed; the raw model holds identical parameters.
3. Dropout randomness, batch-dependent operations, unequal micro-batches,
   floating-point order, clipping boundaries, and optimizer steps between
   micros.

## Modern LLM Systems Delta

Production loops add asynchronous data loading, fused steps, distributed
validation, fault preemption hooks, elastic jobs, token-budget schedules,
automatic anomaly policies, and evaluation suites beyond next-token metrics.

## Professional Takeaways

Draw the loop as a transaction: consume data, build gradients, validate/update,
commit progress, emit evidence, checkpoint. Name every mutable state.

## Further Exploration

- [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)
- [PyTorch inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)

