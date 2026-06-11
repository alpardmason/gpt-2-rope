# Practice 23: Post-Training Quantization

Companion to [23-post-training-quantization.md](../23-post-training-quantization.md).
Persist all deliverables to `notes/chapters/23.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `checkpoint quantize` CLI to one INT8 weight

Follow one checkpoint from the command line into the rounding. Start at
`checkpoint_quantize` in [`cli.py`](../../../src/gpt2_rope/cli.py) and
trace:

`checkpoint_quantize` -> `_load_model` -> `quantize_model` in
[`quantization.py`](../../../src/gpt2_rope/quantization.py) ->
`QuantizedLinear.from_linear` -> `save_quantized` ->
`quantization_report`.

Record at each hop:

- In `from_linear`: the shape and dtype of `scales`
  (`weight.abs().amax(dim=1)` - which dimension survives?), the dtype of
  the stored weight, and what `clamp_min(1e-8)` protects against.
- In `QuantizedLinear.__init__`: whether `weight_int8`, `scales`, and
  `bias` are parameters or buffers, and which two `ValueError` guards run.
- In `quantize_model`: how the `lm_head` skip works (`named_children`
  name match) and which module replacement counter the CLI prints via
  `quantization_report`.
- In `save_quantized`: why every tensor is cloned before `save_file`
  (tied embedding storage) and what the `.safetensors.json` sidecar holds.

### Trace B: the load path and the dequantizing forward

Trace `load_quantized` -> `quantize_model` -> `load_state_dict`, then
`QuantizedLinear.forward`. Record:

- Why the model must be restructured with `quantize_model` BEFORE the
  saved tensors are loaded, and why `load_state_dict`'s copy-in-place
  semantics keep tied tensors tied.
- In `forward`: the dtype journey of the weight
  (`int8 -> inputs.dtype`, scaled per channel) and the one-sentence reason
  this saves memory bandwidth but not FLOPs.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_quantize_model_skips_tied_head_and_reports_compression` in
   [`test_quantization.py`](../../../tests/test_quantization.py) to make
   for a 2-layer tiny model: the replaced-module count (count the linears
   per block yourself) and how it proves `lm_head` is still tied to the
   embedding. Then read it and diff against your guess.
2. **Lab output prediction.** Predict the chapter lab's three printed
   lines: quantized module count, approximate compression ratio, and the
   rough size of the max logit delta and argmax agreement at INT8.
3. **Mutation prediction.** If `DEFAULT_SKIP_MODULES` were `()` so the head
   is quantized too, predict every failing test in
   `tests/test_quantization.py` and the first failing assertion in each.
   Verify by temporarily editing `quantization.py`, running
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_quantization.py`, and
   reverting (`git checkout -- src/gpt2_rope/quantization.py`).
4. **Boundary prediction.** Predict the exact exception type and message of
   `QuantizedLinear(torch.zeros(4, 4), torch.ones(4), None)` (a float
   weight) and of `quantize_model(nn.Sequential(nn.ReLU()))`. Verify both
   in a REPL.

## 3. Tool walkthrough: `quantization_report` in a REPL plus `ls -la`

- **Why this tool.** A compression claim must reconcile with bytes on disk.
  Professionals sanity-check every exported artifact with the cheapest
  possible instruments - a byte-accounting report and `ls -la` - before
  trusting dashboards, because the gap between the two is where mistakes
  (and skipped modules) hide.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import torch
from pathlib import Path
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT
from gpt2_rope.quantization import quantize_model, quantization_report, save_quantized

torch.manual_seed(0)
model = GPT(ModelConfig(vocab_size=300, context_length=32, d_model=64,
                        num_layers=2, num_heads=4, num_kv_heads=2)).eval()
torch.save(model.state_dict(), "model-fp32.pt")
quantize_model(model)
save_quantized(model, Path("model-int8.safetensors"))
print(quantization_report(model))
PY
ls -la model-fp32.pt model-int8.safetensors
cat model-int8.safetensors.json
```

- **Play.**
  1. Divide the two `ls -la` sizes and compare against the report's
     `compression_ratio`. Explain the gap in one sentence: the report
     counts only quantized modules, while the files also carry the FP32
     embedding/head and non-linear parameters.
  2. Re-run with `quantize_model(model, skip_modules=())` and record how
     `quantized_modules` and the file size change; state what this does to
     the embedding/head tying contract.
  3. In a REPL, requantize one `nn.Linear` with a single per-tensor scale
     (`weight.abs().max() / 127`) and compare the max reconstruction error
     against `QuantizedLinear.from_linear`'s per-channel version on the
     same weight. Clean up with
     `rm -f model-fp32.pt model-int8.safetensors model-int8.safetensors.json`.

## 4. Deliverables

Append to `notes/chapters/23.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the byte-reconciliation numbers (report ratio vs `ls` ratio)
  and your one-sentence explanation of the gap.
- 3-5 why-cards. Seed examples: "Why is `lm_head` skipped by default?",
  "Why per-output-channel scales instead of one scale per matrix?", "What
  breaks if `save_quantized` did not clone tensors before `save_file`?"
- Feynman summary: explain to a colleague why dequantize-on-forward INT8
  saves memory bandwidth but not FLOPs, and which serving regime that
  trade-off actually accelerates.
