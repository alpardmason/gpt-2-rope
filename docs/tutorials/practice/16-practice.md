# Practice 16: Framework Philosophy - PyTorch, JAX, and MLX

Companion to [16-framework-philosophy-pytorch-jax-and-mlx.md](../16-framework-philosophy-pytorch-jax-and-mlx.md).
Persist all deliverables to `notes/chapters/16.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`. This
chapter has no CLI entry point; the chapter lab is the entry point, and all
evidence here is CPU-only PyTorch - the JAX/MLX columns stay conceptual.

## 1. Tracing tasks

### Trace A: the lab as a state inventory of `GPT`

Run the chapter lab snippet mentally first: it constructs `GPT` from
[`model.py`](../../../src/gpt2_rope/model.py) and prints
`named_parameters`, `named_buffers`, and a `state_dict` membership check.
Trace where each kind of state is born:

`GPT.__init__` -> `TransformerBlock` -> `GroupedQueryAttention` ->
`RotaryEmbedding.__init__` in [`rope.py`](../../../src/gpt2_rope/rope.py)
(`register_buffer(..., persistent=False)`).

Record:

- The three distinct kinds of mutable state a `GPT` instance owns
  (parameters, buffers, the `training` mode flag) and which of them
  `state_dict()` serializes.
- The names of the RoPE trig buffers (`cos_cached`, `sin_cached`), the flag
  that keeps them out of `state_dict()`, and why a JAX port would instead
  carry them as explicit arrays or recompute them inside a pure function.
- Which line of `GPT.__init__` creates aliasing between two parameters
  (`self.lm_head.weight = self.token_embedding.weight`) - and why a pytree
  of independent arrays cannot express that identity directly.

### Trace B: hidden global state the checkpoint must chase

Trace `seed_everything` in
[`training.py`](../../../src/gpt2_rope/training.py) and `_rng_state` /
`_restore_rng_state` in
[`checkpoint.py`](../../../src/gpt2_rope/checkpoint.py). Record every
global generator they touch (Python `random`, NumPy, torch CPU, CUDA when
available), then contrast with `generate` in
[`generation.py`](../../../src/gpt2_rope/generation.py), which builds a
local `torch.Generator` seeded from `GenerationConfig.seed`. State in one
sentence why the local-generator style is the one that survives a port to
JAX's explicit-key model unchanged.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertion you expect
   `test_seeded_generation_is_deterministic` in
   [`test_lora_generation.py`](../../../tests/test_lora_generation.py) to
   make, and name the mechanism in `generation.py` that makes it pass
   despite sampling with `temperature=0.8`. Then read it and diff against
   your guess.
2. **Lab output prediction.** Predict the three printed lines of the
   chapter lab: the first four parameter names in registration order, the
   buffer names (how many layers, how many buffers each), and the boolean
   of the `state_dict` membership check.
3. **Mutation prediction.** In `rope.py`, change both `register_buffer`
   calls to `persistent=True`. Predict: which lab line flips, and does
   `uv run pytest tests/test_model.py tests/test_checkpoint.py -q` stay
   green (think about whether saver and loader agree on keys)? Verify, then
   revert (`git checkout -- src/gpt2_rope/rope.py`).
4. **Boundary prediction.** Predict the value of `GPT(config).training`
   immediately after construction, after `.eval()`, and after `.train()` -
   and whether the flag propagates to a nested `nn.Dropout` submodule.
   Verify in a REPL with the tiny config.

## 3. Tool walkthrough: eager versus `torch.compile` on the tiny model

- **Why this tool.** "When does an operation execute?" is the chapter's
  second framework question, and `torch.compile` is where PyTorch's eager
  default meets the traced world JAX lives in. Measuring compile latency,
  recompiles, and numerical agreement on a tiny CPU model builds the
  instincts without a GPU. Keep everything tiny: CPU compilation is slow.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import time

import torch

from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT

torch.manual_seed(0)
config = ModelConfig(vocab_size=32, context_length=8, d_model=16,
                     num_layers=1, num_heads=2, num_kv_heads=1, dropout=0.0)
model = GPT(config).eval()
compiled = torch.compile(model)
tokens = torch.randint(0, 32, (2, 8))

with torch.no_grad():
    start = time.perf_counter()
    eager = model(tokens, use_cache=False).logits
    print(f"eager first call   {time.perf_counter() - start:8.3f}s")
    start = time.perf_counter()
    first = compiled(tokens, use_cache=False).logits
    print(f"compiled first call {time.perf_counter() - start:8.3f}s")
    start = time.perf_counter()
    second = compiled(tokens, use_cache=False).logits
    print(f"compiled warm call  {time.perf_counter() - start:8.3f}s")
print("allclose:", torch.allclose(eager, second, atol=1e-5))
PY
```

- **Play.**
  1. Record the ratio of compiled-first-call to warm-call time and explain
     where it goes (graph capture and code generation) - the cost JAX users
     pay at `jit` boundaries.
  2. Trigger recompilation: feed sequence lengths 8, 7, 6 in a loop and
     re-run with the environment variable `TORCH_LOGS=recompiles` set;
     record the recompile reasons printed and connect them to the chapter's
     "changing shapes/static values" failure row.
  3. Confirm the state-mutation hazard: call `model.train()` between two
     compiled calls and observe whether a recompile or guard failure is
     logged - mode flags are exactly the Python side effect tracing models
     must guard.

## 4. Deliverables

Append to `notes/chapters/16.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the eager/compile timing numbers you observed and the
  recompile reason you captured.
- 3-5 why-cards. Seed examples: "Why are the RoPE tables non-persistent
  buffers, and what would they become in JAX?", "What breaks if weight
  tying is exported as two independent arrays and never retied?", "Why
  does hidden global RNG make `_rng_state` necessary in PyTorch but
  meaningless in JAX?"
- Feynman summary: explain to a colleague the five framework questions
  (state, execution, transformation, devices, side effects) using only
  examples from this repository's code.
