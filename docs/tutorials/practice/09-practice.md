# Practice 09: KV Cache and Autoregressive Generation

Companion to [09-kv-cache-and-autoregressive-generation.md](../09-kv-cache-and-autoregressive-generation.md).
Persist all deliverables to `notes/chapters/09.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `generate` CLI through prefill and decode

Follow one full generation from the command line. Start at `generate_text`
in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`generate_text` -> `_load_model` -> tokenizer encode and the equal-length
batch check -> `GenerationConfig` -> `generate` in
[`generation.py`](../../../src/gpt2_rope/generation.py) -> the prefill call
`model(output, use_cache=True)` -> the decode loop
(`sample_next_token`, then `model(next_token, past_key_values=cache)`).

Record at each hop:

- The input shape to the model in prefill (`[B, T_prompt]`) versus each
  decode step (`[B, 1]`), and where the context-length guard
  (`input_ids.size(1) + config.max_new_tokens > model.config.context_length`)
  fires relative to any compute.
- Per layer, the cache shape after prefill and after `n` decoded tokens
  for the lab config (`num_kv_heads=1`, `head_dim=8`): who appends to it
  (`torch.cat` in `GroupedQueryAttention.forward` in
  [`model.py`](../../../src/gpt2_rope/model.py)), and who carries it
  between steps (`generate`'s local `cache` variable)?
- The `finished` tensor: its dtype and shape, which two places consult it,
  and what finished rows emit while the batch continues.

### Trace B: the sampling pipeline

Trace `sample_next_token` top to bottom: `_apply_repetition_penalty`
(piecewise on sign: negative logits multiplied, positive divided), the
`temperature == 0` greedy early return, top-k thresholding, top-p sorted
cumulative filtering, and the final `torch.multinomial` with the local
`torch.Generator`. Record the order of transforms, why the generator is
created and seeded inside `generate` rather than using global RNG, and
which transforms are skipped for the lab's settings (`top_k=5`, no top-p,
penalty 1.0).

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_seeded_generation_is_deterministic` in
   [`test_lora_generation.py`](../../../tests/test_lora_generation.py) to
   make: what it builds, what it calls twice, and how it compares results.
   Then read it and diff against your guess.
2. **Lab output prediction.** Before running the chapter lab, predict the
   length of each printed list (prompt 3 tokens plus `max_new_tokens=4`),
   whether the two lines are identical, and whether the first three IDs
   are `[1, 2, 3]`. Then run it and explain which line of `generate` makes
   the two runs match.
3. **Mutation prediction.** In the cached branch of
   `GroupedQueryAttention.forward` in
   [`model.py`](../../../src/gpt2_rope/model.py), replace the explicit
   offset mask with `attention_mask = None` and `is_causal = True`.
   Predict: does `test_cached_logits_match_full_forward` in
   [`test_model.py`](../../../tests/test_model.py) fail, and why would a
   test that decoded only one token at a time miss this bug? Verify with
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py`, then revert
   (`git checkout -- src/gpt2_rope/model.py`).
4. **Boundary prediction.** Predict the exact exception type and message of
   calling `generate` on the lab's model (`context_length=16`) with a
   3-token prompt and `GenerationConfig(max_new_tokens=20)`. Verify in a
   REPL, then find the matching guard inside `GPT.forward` and state which
   of the two fires first.

## 3. Tool walkthrough: timing cached versus uncached decoding

- **Why this tool.** The KV cache exists for one reason: without it every
  decode step re-runs attention over the whole sequence, and total work
  grows quadratically. `time.perf_counter` measurements of both decode
  strategies turn that asymptotic claim into a number you measured, which
  is the only kind of performance claim worth repeating.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import time
import torch
from gpt2_rope.config import GenerationConfig, ModelConfig
from gpt2_rope.generation import generate
from gpt2_rope.model import GPT

torch.manual_seed(1)
c = ModelConfig(vocab_size=256, context_length=512, d_model=128,
                num_layers=4, num_heads=4, num_kv_heads=2, dropout=0)
m = GPT(c).eval()
prompt = torch.randint(0, 256, (1, 256))
new_tokens = 64

start = time.perf_counter()
with torch.inference_mode():
    generate(m, prompt, GenerationConfig(max_new_tokens=new_tokens, seed=9))
print("cached:  ", time.perf_counter() - start)

start = time.perf_counter()
with torch.inference_mode():
    output = prompt
    for _ in range(new_tokens):
        logits = m(output, use_cache=False).logits[:, -1]
        output = torch.cat((output, logits.argmax(-1, keepdim=True)), dim=1)
print("uncached:", time.perf_counter() - start)
PY
```

  This is a CPU measurement; do not extrapolate accelerator ratios from it.
- **Play.**
  1. Run the script three times and record the spread, then double the
     prompt length to 512 minus `new_tokens` and record how each variant's
     time moves. Explain the difference via positions computed per step.
  2. Double `new_tokens` instead and record which variant degrades faster.
     Relate it to the `O(T)` versus `O(T^2)` token positions processed.
  3. Count work directly: in the uncached loop, sum `output.size(1)` per
     iteration and compare against `new_tokens * 1` for the cached path.
     Verify the ratio approximately predicts your timing gap.

## 4. Deliverables

Append to `notes/chapters/09.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the timing script you will reuse, plus the cached/uncached
  numbers and the work-count ratio from the play exercises.
- 3-5 why-cards. Seed examples: "Why does the cache equivalence test use a
  multi-token suffix instead of single-token decoding?", "What breaks if
  repetition penalty applies one formula to both logit signs?", "Why does
  a finished batch row keep consuming compute until all rows finish?"
- Feynman summary: explain to a colleague generation as a state machine -
  model, cache, sampling, and batch lifecycle as four separate contracts -
  and why correct logits alone do not make a serving path reliable.

Tier 2: this chapter has a kata. After the deliverables above, run
`UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start kv-cache`
and follow [katas/kv-cache/README.md](../../../katas/kv-cache/README.md).
