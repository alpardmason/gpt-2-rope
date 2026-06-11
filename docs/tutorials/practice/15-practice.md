# Practice 15: LoRA and Supervised Fine-Tuning

Companion to [15-lora-and-supervised-fine-tuning.md](../15-lora-and-supervised-fine-tuning.md).
Persist all deliverables to `notes/chapters/15.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `finetune` CLI to the adapter artifact

Follow a LoRA fine-tune from the command line to the saved adapter. Start
at `finetune` in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`finetune` -> `train_finetuning` in
[`training.py`](../../../src/gpt2_rope/training.py) -> the freeze loop ->
`apply_lora` in [`lora.py`](../../../src/gpt2_rope/lora.py) ->
`GPT.configure_optimizer` -> `save_lora` at the checkpoint boundary.

Record at each hop:

- The exact order of: base-checkpoint load, the
  `parameter.requires_grad = False` loop, and `apply_lora`. Why would
  swapping the last two silently train the whole model?
- The default `lora_targets` tuple on `FineTuningConfig` in
  [`config.py`](../../../src/gpt2_rope/config.py) - which six attribute
  names it lists, and which modules in `model.py` each one matches
  (note `proj` matches the MLP projection, not attention's `out_proj`).
- How `configure_optimizer` ends up seeing only adapter parameters (which
  single condition filters frozen ones out).
- What `step_tokens` counts in this loop
  (`labels[:, 1:].ne(-100)`) versus what the pretraining loop counts, and
  where `save_lora` writes `adapter.safetensors` relative to the full
  checkpoint directory.

### Trace B: `LoRALinear` construction and merge lifecycle

Trace `LoRALinear.__init__`, `forward`, `merge`, and `unmerge` in
[`lora.py`](../../../src/gpt2_rope/lora.py). Record: the shapes of `lora_a`
(`[rank, in_features]`) and `lora_b` (`[out_features, rank]`) and which one
starts at zero; the value of `scale` for `rank=4, alpha=8`; why `forward`
skips the adapter path when `self.merged` is set; and why `merge`/`unmerge`
run under `@torch.no_grad()`.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the sequence of
   assertions you expect `test_lora_merge_is_numerically_equivalent` in
   [`test_lora_generation.py`](../../../tests/test_lora_generation.py) to
   make, including why it must re-initialize `lora_b` before the check
   means anything. Then read it and diff against your guess.
2. **Lab output prediction.** Predict the chapter lab's printed values: the
   `replaced` count for two layers with targets `("q_proj", "v_proj")`, and
   the exact trainable parameter count for `rank=4` given
   `d_model=32`, `num_kv_heads=2`, `head_dim=8` (work out `lora_a`/`lora_b`
   sizes per replaced module; `q_proj` and `v_proj` have different output
   widths). Then run it.
3. **Mutation prediction.** Delete the `self.merged = True` line in
   `LoRALinear.merge`. Predict which assertion of
   `test_lora_merge_is_numerically_equivalent` fails first and what the
   wrong output is (hint: the forward now adds the update on top of merged
   weights). Verify with
   `uv run pytest tests/test_lora_generation.py -q`, then revert
   (`git checkout -- src/gpt2_rope/lora.py`).
4. **Boundary prediction.** Predict the exact exception type and message of
   `apply_lora(model, rank=2, alpha=4.0, target_modules=("does_not_exist",))`
   on a tiny `GPT`. Verify in a REPL.

## 3. Tool walkthrough: state-dict diffing around `apply_lora`

- **Why this tool.** Adapter bugs are layout bugs: wrong targets, renamed
  keys, accidentally trainable bases. Diffing `state_dict` key sets and
  trainable counts before and after replacement is the five-line audit that
  catches all three, and it is exactly what you would run against a vendor
  PEFT library you do not yet trust.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from gpt2_rope.config import ModelConfig
from gpt2_rope.lora import apply_lora
from gpt2_rope.model import GPT

model = GPT(ModelConfig(vocab_size=64, context_length=16, d_model=32,
                        num_layers=2, num_heads=4, num_kv_heads=2))
before = set(model.state_dict())
for parameter in model.parameters():
    parameter.requires_grad = False
replaced = apply_lora(model, rank=4, alpha=8,
                      target_modules=("q_proj", "v_proj"))
after = set(model.state_dict())
print("replaced", replaced)
print("added:", sorted(after - before))
print("removed:", sorted(before - after))
print("trainable", model.parameter_count(trainable_only=True),
      "of", model.parameter_count())
PY
```

- **Play.**
  1. Study the `removed` list: the base weights did not disappear - find
     their new names in the `added` set (the `.base.` infix) and explain
     why `load_lora` therefore requires the same replacement layout to
     exist before loading an adapter.
  2. Re-run with `target_modules=("fc", "proj")` and predict the new
     `replaced` count before looking; reconcile it against the module tree
     of `MLP` in [`model.py`](../../../src/gpt2_rope/model.py).
  3. Reproduce the mismatch guard: save an adapter from a
     `("q_proj", "v_proj")` model with `save_lora`, build a fresh model
     with `apply_lora(..., target_modules=("fc",))`, call `load_lora`, and
     record the `ValueError` - compare with
     `test_load_lora_rejects_mismatched_adapter`.

## 4. Deliverables

Append to `notes/chapters/15.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the state-dict diff snippet you will reuse, plus the observed
  trainable/total parameter ratio.
- 3-5 why-cards. Seed examples: "Why is `lora_b` zero-initialized but
  `lora_a` not?", "What breaks if freezing happens after `apply_lora`?",
  "Why must merge and unmerge run under `no_grad`?"
- Feynman summary: explain to a colleague why a LoRA adapter is only
  interpretable together with its rank, alpha, target list, and base-model
  identity - and what the merge flag protects during deployment.
