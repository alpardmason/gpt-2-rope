# 16: Framework Philosophy - PyTorch, JAX, and MLX

## Objectives and Prerequisites

Compare frameworks by state, execution, transformation, device, and debugging
models rather than syntax. Prerequisite: 06-15. This chapter is conceptual; the
repository does not claim tested JAX or MLX ports.

**Practice companion:** [16-practice.md](practice/16-practice.md).

**Source map:** [`model.py`](../../src/gpt2_rope/model.py),
[`training.py`](../../src/gpt2_rope/training.py),
[`checkpoint.py`](../../src/gpt2_rope/checkpoint.py), and
[`test_model.py`](../../tests/test_model.py).

## Five Questions for Any Framework

1. Where do parameters and optimizer state live?
2. When does an operation execute?
3. How are gradients/vectorization/compilation expressed?
4. How do arrays move across devices?
5. Which Python side effects survive transformation?

## PyTorch: Stateful Modules, Eager First

This repository uses `nn.Module` objects whose parameters, buffers, training
mode, and submodules are mutable state. Ordinary Python executes eagerly;
autograd records tensor operations dynamically. `torch.compile` can capture and
optimize regions while graph breaks preserve Python flexibility.

```python
model.train()
output = model(input_ids, labels=input_ids)
output.loss.backward()
optimizer.step()
```

This is debuggable and close to imperative software engineering. The cost is
that state mutation, dynamic control flow, and graph capture require discipline.

## JAX: Pure Functions and Transformations

JAX treats nested parameter/optimizer containers as pytrees and centers pure
array functions. `grad`, `jit`, `vmap`, and parallel transformations compose
around functions. Randomness is explicit key data rather than hidden global
generator state.

Conceptual translation, not repository code:

```python
loss, grads = jax.value_and_grad(loss_fn)(params, batch, rng_key)
params, opt_state = update(params, grads, opt_state)
```

The benefit is transformation-friendly semantics. The cost is tracing rules,
static/dynamic shape constraints, explicit state threading, and asynchronous
execution surprises for newcomers.

## MLX: Apple-Silicon-Centered Lazy Arrays

MLX combines NumPy-like APIs, function transformations, lazy computation, and
unified memory on Apple silicon. Operations build a graph until evaluation;
`mx.eval(loss, model.parameters())` is a natural step boundary. CPU/GPU arrays
share memory rather than requiring the same explicit copy model as PyTorch.

The benefit is an elegant Apple-silicon path. The trade-off is a younger,
smaller ecosystem and different execution/performance instincts. Printing or
materializing values can force evaluation.

## Comparative Contract

| Dimension | PyTorch | JAX | MLX |
|---|---|---|---|
| Default execution | Eager | Eager-like tracing model | Lazy |
| Model style | Stateful `Module` | Pure functions + pytrees | Modules + transforms |
| Randomness | Generator/global state | Explicit keys | Explicit/random APIs |
| Compilation | Optional `torch.compile` | Central `jit` | `compile`, lazy graph |
| Device memory | Explicit device copies | Device arrays/sharding | Unified CPU/GPU memory |
| Ecosystem | Broadest production/research | Strong accelerator research | Apple-silicon focused |

**Recommendation:** choose from workload, hardware, ecosystem, and team
debugging model. **Rationale:** framework philosophy affects architecture and
operations, not just code style.

## Translating This Project

- `GPT.state_dict()` becomes an explicit parameter pytree in idiomatic JAX.
- `_rng_state()` changes radically in JAX because keys travel through calls.
- Python `for` loops may need `lax.scan` for compiled JAX training.
- PyTorch buffer semantics for RoPE become explicit arrays/static config.
- MLX training must choose evaluation boundaries to avoid oversized lazy graphs.
- DDP maps to different sharding/collective APIs; semantics must be re-proven.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| JAX recompiles repeatedly | Changing shapes/static values | Enable compile logs | Stabilize signatures | Shape policy |
| PyTorch compile graph breaks | Python/state side effects | Compiler diagnostics | Refactor hot region | Compile tests |
| MLX memory/latency surprises | Evaluation boundary wrong | Profile `eval` points | Evaluate per step | Explicit lifecycle |
| Port matches shapes, not logits | Semantic drift | Golden tiny inputs | Differential tests | Contract map |

## Lab

Use the existing PyTorch model to expose state distinctions:

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT

m = GPT(ModelConfig(vocab_size=32, context_length=8, d_model=16,
                    num_layers=1, num_heads=2, num_kv_heads=1))
print("parameters", [n for n, _ in m.named_parameters()][:4])
print("buffers", [n for n, _ in m.named_buffers()])
print("state contains RoPE tables",
      any("cos_cached" in k for k in m.state_dict()))
PY
```

Expected: parameters and RoPE buffers are discoverable, while non-persistent
RoPE tables are absent from serialized state.

## Exercises

1. Which current functions are hardest to port idiomatically to JAX?
2. Why is MLX unified memory not equivalent to “data movement is free”?
3. When is PyTorch the conservative recommendation?

## Solutions

1. Stateful training/checkpoint code, hidden RNG capture, dynamic module
   replacement for LoRA, and Python data iteration.
2. CPU/GPU synchronization, bandwidth, graph evaluation, and placement still
   affect latency; shared addressability does not erase hardware costs.
3. When ecosystem breadth, existing CUDA tooling, imperative debugging, model
   compatibility, and team experience dominate transformation elegance.

## Modern LLM Systems Delta

All three ecosystems now support compilation and distributed execution, but
their defaults still shape code. Mature teams isolate framework-specific
kernels/state from model contracts and use differential tests across ports.

## Professional Takeaways

Do not answer “which framework is best?” Answer: for which hardware, scale,
research iteration style, serving target, team, and compatibility constraints?

## Further Exploration

- [PyTorch documentation](https://docs.pytorch.org/docs/stable/)
- [PyTorch compile](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [JAX: Thinking in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)
- [JAX pytrees](https://docs.jax.dev/en/latest/pytrees.html)
- [MLX documentation](https://ml-explore.github.io/mlx/)
- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)

