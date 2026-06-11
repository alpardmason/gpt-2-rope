# Practice 18: GPT-2 vs Modern SOTA Architectures

Companion to [18-gpt2-vs-modern-sota-architectures.md](../18-gpt2-vs-modern-sota-architectures.md).
Persist all deliverables to `notes/chapters/18.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`. This
chapter changes no code and has no CLI entry point; the chapter lab and the
comparison artifact below are the entry points.

## 1. Tracing tasks

### Trace A: the lab's cache arithmetic against real config defaults

Start from the chapter lab snippet and trace its inputs back to
[`config.py`](../../../src/gpt2_rope/config.py):

lab `kv_bytes` -> `ModelConfig()` defaults -> the `head_dim` and
`query_groups` properties -> the cache shape produced by
`GroupedQueryAttention.forward` in
[`model.py`](../../../src/gpt2_rope/model.py).

Record at each hop:

- The default `num_heads`, `num_kv_heads`, and derived `head_dim` of a
  bare `ModelConfig()`, and the `query_groups` ratio they imply.
- The cache tensor shape `[B, H_kv, T, D]` kept by
  `GroupedQueryAttention.forward` (`present = (key, value)`), and which
  dimension the lab's `2 * hkv * d * 2` formula walks (two tensors, BF16
  bytes).
- Where the GQA-vs-MHA byte ratio `Hkv/Hq` shows up in the lab's printed
  numbers, and why the MLA row bypasses heads entirely (one shared latent
  per token per layer).

### Trace B: `model_preset` to a parameter budget

Trace `model_preset` and `_PRESETS` in
[`config.py`](../../../src/gpt2_rope/config.py), then `GPT.__init__` and
`parameter_count` in [`model.py`](../../../src/gpt2_rope/model.py) for the
`tiny` preset. Record: which four fields each preset pins and which come
from `ModelConfig` defaults (vocabulary, context length); the parameter
groups a block owns (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc`,
`proj`, two LayerNorms); why the tied `lm_head` adds zero parameters to
the total; and which preset rows would change if biases were dropped
Llama-style (`bias=False`).

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_presets_include_tiny_and_gpt2_family` in
   [`test_config.py`](../../../tests/test_config.py) to make - which
   presets it samples and which field of each. Then read it and diff
   against your guess.
2. **Lab output prediction.** Compute, before running, the three lines the
   chapter lab prints: `kv_bytes(12, 64)`, `kv_bytes` for the repo's
   default `num_kv_heads` and `head_dim`, and the 512-dim latent row -
   including both ratios. Then run the lab and check your arithmetic.
3. **Prediction by hand.** Predict the exact `parameter_count()` of
   `GPT(model_preset("tiny"))` by summing: token embedding
   (`vocab_size * d_model`), per-block attention and MLP matrices with
   biases, two LayerNorms per block, and the final LayerNorm (tied head
   adds nothing). Write the total in your notes, then verify in a REPL.
4. **Boundary prediction.** Predict the exact exception type and message
   of `model_preset("gpt2-tiny")` - including what the message lists.
   Verify in a REPL.

## 3. Tool walkthrough: a parameter-count comparison table from presets

- **Why this tool.** "What would you change to make this Llama-class?" is
  answered with budgets, not adjectives. Building the params-and-cache
  table yourself - analytically, validated against the real model - is the
  comparison artifact this chapter exists for, and the same closed-form
  skill prices any architecture diff before you build it.
- **How.** Validate a closed-form count against `GPT`, then tabulate every
  preset without instantiating the billion-parameter ones:

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from gpt2_rope.config import ModelConfig, model_preset
from gpt2_rope.model import GPT


def analytic_count(c: ModelConfig) -> int:
    bias = 1 if c.bias else 0
    kv_width = c.num_kv_heads * c.head_dim
    attention = 2 * (c.d_model * c.d_model + bias * c.d_model)   # q, out
    attention += 2 * (c.d_model * kv_width + bias * kv_width)    # k, v
    mlp = (c.d_model * c.mlp_hidden_size + bias * c.mlp_hidden_size
           + c.mlp_hidden_size * c.d_model + bias * c.d_model)
    block = attention + mlp + 2 * (2 * c.d_model)                # ln_1, ln_2
    return c.vocab_size * c.d_model + c.num_layers * block + 2 * c.d_model


tiny = model_preset("tiny")
print("analytic", analytic_count(tiny),
      "actual", GPT(tiny).parameter_count())
for name in ["tiny", "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
    c = model_preset(name)
    kv = 2 * c.num_kv_heads * c.head_dim * 2
    print(f"{name:12s} params={analytic_count(c):>13,d} "
          f"kv_B_per_token_per_layer={kv:5d} groups={c.query_groups}")
PY
```

- **Play.**
  1. Extend the table with an MHA column by overriding
     `model_preset(name, num_kv_heads=...)` to equal `num_heads`, and
     record the cache ratio each preset's GQA default buys.
  2. Set `bias=False` in the analytic function's input
     (`model_preset("gpt2-small", bias=False)`) and record how many
     parameters vanish - then explain why the chapter says this matters
     more at 70B than at 124M.
  3. Break the geometry: `model_preset("gpt2-xl", num_kv_heads=4)` - 25
     heads do not divide by 4. Record which validator fires and its
     message; this is the same guard the ablation chapter leans on.

## 4. Deliverables

Append to `notes/chapters/18.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the preset table you generated (paste it), plus the analytic
  formula's one mismatch risk you checked (weight tying).
- 3-5 why-cards. Seed examples: "Why does the tied LM head add zero
  parameters but the learned-PE table adds `context_length * d_model`?",
  "What breaks if a SwiGLU port keeps the 4x hidden ratio?", "Why is
  MHA -> GQA -> MLA one cache-bytes story rather than three ideas?"
- Feynman summary: explain to a colleague why every model in the chapter's
  comparison table is still GPT-2's skeleton, and name the two deltas this
  repository already implements versus the three it deliberately leaves
  out.
