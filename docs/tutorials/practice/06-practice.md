# Practice 06: Model Contracts, Initialization, and Weight Tying

Companion to [06-model-contracts-initialization-and-weight-tying.md](../06-model-contracts-initialization-and-weight-tying.md).
Persist all deliverables to `notes/chapters/06.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: construction-time ownership

Trace `GPT.__init__` in [`model.py`](../../../src/gpt2_rope/model.py) line
by line for the chapter lab config (`vocab_size=64`, `d_model=32`,
`num_layers=2`, `num_heads=4`, `num_kv_heads=2`):

`GPT.__init__` -> `self.apply(self._initialize_weights)` ->
`_scale_residual_projections` -> the tying assignment
`self.lm_head.weight = self.token_embedding.weight`.

Record:

- Which modules `_initialize_weights` touches and with what std
  (`initializer_range`), versus the two residual projections per block
  (`attention.out_proj.weight`, `mlp.proj.weight`) that
  `_scale_residual_projections` overwrites with
  `initializer_range / sqrt(2 * num_layers)`.
- The ordering hazard: tying happens AFTER initialization, so `lm_head`'s
  freshly initialized weight is discarded. What would change if the tying
  line ran before `self.apply`?
- Ownership of the rope tables: they live on each
  `GroupedQueryAttention.rope` (chapter 07), not on `GPT`, and are
  non-persistent buffers. Predict, then check, whether any `rope` key
  appears in `GPT(config).state_dict()`.

### Trace B: forward and loss alignment

Trace `GPT.forward` for `input_ids` of shape `[2, 7]` with labels:
shape validation -> `token_embedding` -> block loop -> `final_layer_norm`
-> `lm_head` -> the shifted cross entropy
(`logits[:, :-1]` flattened against `labels[:, 1:]`, `ignore_index=-100`).

Record the tensor shape after each stage, which two `ValueError` guards can
fire before any compute (ndim check, context-length check), and how many
positions actually contribute to the loss for `T=7`.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_model_shapes_tying_and_loss` in
   [`test_model.py`](../../../tests/test_model.py) to make, including the
   logits shape, the tying check, and the per-layer cache shapes for its
   tiny config. Then read it and diff against your guess.
2. **Lab output prediction.** Before running the chapter lab, predict the
   printed logits shape, the loss shape, and the `tied` boolean. Then
   compute `parameter_count()` on paper for the lab config (remember the
   tied head is counted once and every `LayerNorm` has weight and bias)
   and check yourself against the printed value.
3. **Mutation prediction.** Delete the line
   `self.lm_head.weight = self.token_embedding.weight` in
   [`model.py`](../../../src/gpt2_rope/model.py). Predict which assertion
   of `test_model_shapes_tying_and_loss` fails, and whether
   `test_cached_logits_match_full_forward` still passes and why. Verify
   with `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py`, then
   revert (`git checkout -- src/gpt2_rope/model.py`).
4. **Boundary prediction.** Predict the exact exception type and message of
   (a) `model(torch.randint(0, 64, (2, 7, 1)))` and (b)
   `model(torch.randint(0, 64, (2, 7)), labels=torch.randint(0, 64, (2, 6)))`.
   Verify both in a REPL.

## 3. Tool walkthrough: `state_dict` and parameter inspection in a REPL

- **Why this tool.** Checkpoint-shape mismatches, double-counted tied
  weights, and silently untrained modules are all diagnosed the same way:
  by interrogating `state_dict()` and `data_ptr()` directly. This is also
  how you verify that what you save is what you think you save.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run python -i -c "
import torch
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT
m = GPT(ModelConfig(vocab_size=64, context_length=16, d_model=32,
                    num_layers=2, num_heads=4, num_kv_heads=2))
sd = m.state_dict()
for k, v in sd.items():
    print(f'{k:48} {tuple(v.shape)}')
"
```

  Useful probes: `m.parameter_count()`,
  `m.lm_head.weight.data_ptr() == m.token_embedding.weight.data_ptr()`,
  `len(sd)` versus `len(list(m.named_parameters()))`.
- **Play.**
  1. Confirm `lm_head.weight` and `token_embedding.weight` are two
     `state_dict` keys but one storage: equal `data_ptr()`, and
     `named_parameters()` yields one fewer entry than `state_dict()` has
     keys (deduplicated shared parameter).
  2. Search the keys for `rope`, `cos`, or `sin` and find nothing; relate
     this to `register_buffer(..., persistent=False)` and state what would
     happen at `load_state_dict` time if the tables WERE persisted with a
     different `context_length`.
  3. Compute `m.blocks[0].attention.out_proj.weight.std()` and
     `m.blocks[0].attention.q_proj.weight.std()` and verify their ratio is
     approximately `1 / sqrt(2 * num_layers)` = `0.5` for two layers.

## 4. Deliverables

Append to `notes/chapters/06.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the `state_dict` dump command you will reuse, plus the
  parameter-count arithmetic from prediction task 2.
- 3-5 why-cards. Seed examples: "Why is weight tying tested with
  `data_ptr()` instead of `torch.equal`?", "What breaks if residual output
  projections keep the default `initializer_range` std at 48 layers?",
  "Why does the model, not the dataset, own the next-token shift?"
- Feynman summary: explain to a colleague why one shared matrix can serve
  as both token embedding and output head, and what that coupling does to
  gradients flowing into rare-token rows.
