# 10: Optimizer Schedules and Mixed Precision

## Objectives and Prerequisites

Understand optimizer parameter policy, learning-rate time, autocast, scaling,
and clipping as one ordered system. Prerequisite: 06 and backpropagation.

**Practice companion:** [10-practice.md](practice/10-practice.md).

**Source map:** [`model.py`](../../src/gpt2_rope/model.py)
`GPT.configure_optimizer`; [`training.py`](../../src/gpt2_rope/training.py)
`cosine_learning_rate`, `_effective_precision`, `_autocast_context`; and
training fields in [`config.py`](../../src/gpt2_rope/config.py).

## Optimizer Contract

Only trainable parameters enter AdamW. Matrices/tensors with `ndim >= 2`
receive weight decay; scalar/vector parameters do not. Betas default to
`(0.9,0.95)`, common for decoder pretraining rather than Adam's older `0.999`
second moment.

The schedule has three regions:

```text
warmup: lr = max_lr * (step + 1) / warmup_steps
cosine: interpolate max_lr -> min_lr
after max_steps: min_lr
```

`LambdaLR` expects a multiplicative factor, so the function output is divided
by configured maximum LR. Be precise about whether “step” means before or after
an optimizer update; scheduler call order determines the observed sequence.

## Precision Policy

```text
CPU auto -> FP32
CUDA auto -> BF16 if supported, otherwise FP16
MPS auto -> FP16
```

Autocast selects lower precision per operation; it does not convert all model
parameters. FP16 CUDA uses `GradScaler`. BF16 usually does not need loss
scaling because it has FP32-like exponent range.

Update order:

```text
scaled backward -> scaler.unscale_(optimizer)
-> clip global norm -> scaler.step -> scaler.update
-> scheduler.step -> zero gradients
```

Clipping scaled gradients would clip the artificial scale, not the true
gradient.

**Recommendation:** select precision from measured hardware capability and
numerical evidence. **Rationale:** “lower precision” is not one behavior.

| Precision | Range | Precision | Typical use |
|---|---:|---:|---|
| FP32 | High | High | CPU/debug/reference |
| BF16 | High | Lower mantissa | Modern CUDA training |
| FP16 | Low | Higher mantissa | Older CUDA/MPS, scaling needed on CUDA |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| NaN/Inf in FP16 | Under/overflow | Check scaler/grad stats | Scale or BF16 | Precision smoke |
| Tiny effective LR | Scheduler offset/order | Log first steps | Define step semantics | Schedule unit tests |
| Clipping ineffective | Clip before unscale | Inspect order | Unscale first | Loop review |
| Norm/bias over-regularized | Decay all params | Inspect groups | Separate vectors | Group test |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from gpt2_rope.training import cosine_learning_rate

for step in [0, 1, 3, 4, 7, 10]:
    lr = cosine_learning_rate(step, warmup_steps=4, max_steps=10,
                              max_learning_rate=1e-3,
                              min_learning_rate=1e-4)
    print(step, f"{lr:.7f}")
PY
```

Expected: linear warmup, smooth decay, and the minimum at/after step 10.
Debug prompt: sketch the sequence if `scheduler.step()` is called before the
first optimizer update.

## Exercises

1. Why not decay LayerNorm scales?
2. What does gradient clipping protect, and what does it not fix?
3. Why is accelerator precision behavior not fully testable on CPU?

## Solutions

1. Regularizing scale/bias vectors is usually undesirable and can distort
   normalization; the project follows a common matrix-only policy.
2. It bounds update-driving gradient norm; it does not fix bad data, invalid
   operations, chronic instability, or optimizer-state corruption.
3. Operator support, accumulation types, fused kernels, and overflow behavior
   differ by backend.

## Modern LLM Systems Delta

Large runs use FP8, master weights, fused optimizers, sharded optimizer states,
per-parameter scaling, loss/gradient anomaly detection, and token-based rather
than step-based schedules.

## Professional Takeaways

Optimization bugs are often ordering bugs. State the exact update sequence and
log LR, gradient norm, scaler behavior, and skipped steps.

## Further Exploration

- [AdamW](https://arxiv.org/abs/1711.05101)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch optimizer step ordering](https://docs.pytorch.org/docs/stable/optim.html)

