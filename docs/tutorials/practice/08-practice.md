# Practice 08: Grouped-Query Attention and SDPA

Companion to [08-grouped-query-attention-and-sdpa.md](../08-grouped-query-attention-and-sdpa.md).
Persist all deliverables to `notes/chapters/08.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `generate` CLI to the kernel boundary

Follow one forward pass into the attention kernel. Start at `generate_text`
in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`generate_text` -> `generate` in
[`generation.py`](../../../src/gpt2_rope/generation.py) -> `GPT.forward` ->
`TransformerBlock.forward` -> `GroupedQueryAttention.forward` in
[`model.py`](../../../src/gpt2_rope/model.py) ->
`functional.scaled_dot_product_attention`.

Record at each hop:

- After `_shape`, the exact shapes of `query`, `key`, and `value` for the
  tiny config (`d_model=32`, `num_heads=4`, `num_kv_heads=2`) on a
  `[2, 6, 32]` input: query is `[2, 4, 6, 8]` but K/V are `[2, 2, 6, 8]`.
  Which projection widths (`q_proj` vs `k_proj`/`v_proj`) created the
  difference?
- The `enable_gqa` decision: it requires `query_groups > 1` AND a CUDA
  device. On CPU, where exactly are K/V expanded with `repeat_interleave`,
  and why does the returned `present` cache stay compact `[B, Hkv, T, D]`?
- At the SDPA call site, which combination of `attn_mask`/`is_causal` is
  used when `past_length == 0` versus when a cached prefix exists.

### Trace B: the offset mask, by hand

Walk the cached branch of `GroupedQueryAttention.forward` with
`past_length=6` and `query_length=4` (so `key_length=10`):
`query_positions = 6 + [0..3]`, `key_positions = [0..9]`, then the
broadcast comparison `key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)`.
Write out the resulting `[4, 10]` Boolean matrix on paper, record the final
`[1, 1, 4, 10]` view, and state in one sentence why `is_causal=True` would
be wrong here (upper-left alignment versus absolute positions).

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_mha_is_gqa_special_case` in
   [`test_model.py`](../../../tests/test_model.py) to make: which config
   override makes GQA degenerate to MHA and what it asserts about the
   output. Then read it and diff against your guess.
2. **Lab output prediction.** Before running the chapter lab, predict all
   printed values: the output shape, the cache K and V shapes, the
   `groups` value, and the cache ratio versus MHA. Then run it.
3. **Mutation prediction.** In `GroupedQueryAttention.forward`, force
   `key_for_attention = key` and `value_for_attention = value`
   unconditionally (delete the `repeat_interleave` fallback). Predict, for
   a CPU run of `tests/test_model.py`: which test fails first in file
   order, the exception type, and roughly what the message says about
   sizes. Predict also which parametrized ablation id (`mha`) still
   passes and why. Verify with
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py`, then revert
   (`git checkout -- src/gpt2_rope/model.py`).
4. **Boundary prediction.** Predict the truth table of the mask
   construction at the boundary `past_length=0`: which branch runs for
   `query_length=1` versus `query_length=6`, and the resulting
   `attention_mask`/`is_causal` pair for each. Verify in a REPL by calling
   the lab's attention module under `breakpoint()` or by reproducing the
   four lines of mask code with `torch.arange` directly.

## 3. Tool walkthrough: `torch.nn.attention.sdpa_kernel` backend selection

- **Why this tool.** `scaled_dot_product_attention` is a dispatcher, not a
  kernel: correctness tests pass identically while performance and small
  numerics differ per backend. Professionals pin backends with this
  context manager to reproduce numerics, isolate regressions, and prove
  which implementation actually ran.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

q = torch.randn(2, 4, 256, 8)
k = torch.randn(2, 4, 256, 8)
v = torch.randn(2, 4, 256, 8)

def run() -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True
    )

default = run()
with sdpa_kernel([SDPBackend.MATH]):
    math_only = run()
    start = time.perf_counter()
    for _ in range(50):
        run()
    print("math backend:", time.perf_counter() - start)
print("max difference:", (default - math_only).abs().max().item())
PY
```

  Record which backends succeed on your machine; do not infer accelerator
  behavior from a CPU result.
- **Play.**
  1. Time the loop under the default dispatcher versus
     `sdpa_kernel([SDPBackend.MATH])` and record the ratio. If they tie on
     your CPU, record that too - absence of a gap is evidence about the
     dispatcher, not a failed experiment.
  2. Break the contract: pass K/V with 2 heads against a 4-head query
     (the un-expanded GQA layout) without `enable_gqa` and record the
     exact `RuntimeError`. This is the diagnostic your prediction task 3
     mutation produces from inside the model.
  3. Reimplement attention manually
     (`softmax(q @ k.transpose(-2, -1) / sqrt(8), dim=-1) @ v` with a
     causal mask) and compare against SDPA with
     `torch.testing.assert_close`. Record the tolerance you needed.

## 4. Deliverables

Append to `notes/chapters/08.md`:

- Tracing log for Traces A and B with all checkpoint answers, including
  the hand-drawn `[4, 10]` mask.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the `sdpa_kernel` snippet you will reuse, plus the backend
  timing observation.
- 3-5 why-cards. Seed examples: "Why is the cache stored compact and
  expanded only at the kernel boundary?", "What breaks if `is_causal=True`
  is used with a non-empty prefix?", "Why does MQA (`num_kv_heads=1`) cut
  cache memory by `1/Hq` but leave parameter count nearly unchanged?"
- Feynman summary: explain to a colleague GQA in three layers - projection
  geometry, persistent cache economics, and backend dispatch - and why
  only the middle layer survives across decode steps.

Tier 2: this chapter has a kata. After the deliverables above, run
`UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start gqa` and
follow [katas/gqa/README.md](../../../katas/gqa/README.md).
