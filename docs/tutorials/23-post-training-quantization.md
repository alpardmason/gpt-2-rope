# 23: Post-Training Quantization

## Objectives and Prerequisites

Quantize trained weights to INT8 for inference, quantify the quality cost
with the evaluation harness, and place weight-only quantization in the wider
GPTQ/AWQ/FP8 landscape. Prerequisite: 06, 13, 20.

**Practice companion:** [23-practice.md](practice/23-practice.md).

**Source map:** [`quantization.py`](../../src/gpt2_rope/quantization.py)
`QuantizedLinear`, `quantize_model`, `quantization_report`,
`save_quantized`, `load_quantized`; [`cli.py`](../../src/gpt2_rope/cli.py)
`checkpoint quantize`;
[`test_quantization.py`](../../tests/test_quantization.py).

## Quantization Contract

Per-output-channel symmetric INT8, weights only:

```text
scale_o = max_i |W[o, i]| / 127          (one FP32 scale per output channel)
Q[o, i] = clamp(round(W[o, i] / scale_o), -127, 127)   (INT8)
forward: y = x @ (Q * scale).T + b       (dequantize per call, FP activations)
```

Invariants:

- `lm_head` is skipped by default: its weight is storage-tied to the token
  embedding, which must stay floating point for the lookup; quantizing it
  would either break tying or quantize the embedding.
- Quantization error per weight is at most `scale_o / 2`; per-channel scales
  isolate outlier rows so one large weight cannot coarsen the whole matrix.
- Storage is INT8 weights plus FP32 scales and biases: about 3.9x smaller
  than FP32 for large matrices (`compression_ratio` in the report).
- This implementation dequantizes in `forward`, so it saves memory and
  bandwidth, not FLOPs -- real INT8 speedups need INT8 GEMM kernels.

## The Quantization Landscape

| Scheme | What is quantized | Calibration | Typical use |
|---|---|---|---|
| Weight-only INT8 (here) | weights, per-channel | none | memory-bound serving |
| GPTQ | weights, 3-4 bit | second-order, per-layer | aggressive compression |
| AWQ | weights, 4 bit | activation-aware scaling | quality at 4 bit |
| INT8 W8A8 (SmoothQuant) | weights + activations | activation statistics | compute-bound speedups |
| FP8 (H100+) | weights + activations | scaling factors | training and serving |
| Quantized KV cache | cache tensors | per-head scales | long-context serving |

**Recommendation:** start with weight-only INT8 -- it is calibration-free,
nearly lossless, and addresses the actual bottleneck of single-stream
inference (weight bandwidth). **Rationale:** decode is memory-bound; weights
dominate bytes moved per token. **Alternatives:** GPTQ/AWQ when 4-bit is
required; W8A8/FP8 when batch sizes make inference compute-bound.

## Verification Is Part of the Artifact

A quantized model without an eval delta is an unverified artifact. The
pipeline here is: quantize, then run the same `eval suite` (chapter 20)
against both checkpoints and record the perplexity delta next to the
`.safetensors.json` report.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Tied weights diverge after load | head quantized despite tying | data_ptr comparison | keep `lm_head` in skip list | tying assertion test |
| One layer dominates error | outlier channels, per-tensor scale | per-channel error histogram | per-channel scales (default) | parity tolerance test |
| No speedup observed | dequant-on-forward still FP GEMM | profile kernel time | INT8 kernels (torch.ao, TensorRT) | document scope honestly |
| Perplexity cliff at 4 bit | naive rounding below 8 bit | bit-width sweep with evals | GPTQ/AWQ calibration | eval delta gate |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_quantization.py -q
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import torch
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT
from gpt2_rope.quantization import quantize_model, quantization_report

torch.manual_seed(0)
model = GPT(ModelConfig(vocab_size=300, context_length=32, d_model=64,
                        num_layers=2, num_heads=4, num_kv_heads=2)).eval()
tokens = torch.randint(0, 300, (2, 16))
with torch.no_grad():
    fp = model(tokens, use_cache=False).logits
quantize_model(model)
with torch.no_grad():
    q8 = model(tokens, use_cache=False).logits
report = quantization_report(model)
print(f"modules={report['quantized_modules']} "
      f"compression={report['compression_ratio']:.2f}x")
print(f"max |logit delta| = {float((q8 - fp).abs().max()):.4f}")
print(f"argmax agreement  = {float((q8.argmax(-1) == fp.argmax(-1)).float().mean()):.3f}")
PY
```

Expected: ~3.9x compression on quantized modules, small logit deltas, and
near-total argmax agreement at INT8.

## Exercises

1. Why per-output-channel scales instead of one scale per weight matrix?
2. Decode moves every weight once per token. Estimate the bandwidth saving
   for GPT-2-small (124M parameters) at FP32 vs INT8 per generated token.
3. Why does quantizing the KV cache matter more than quantizing weights for
   very long contexts at large batch sizes?

## Solutions

1. Rows of a linear layer have very different magnitude ranges; a per-tensor
   scale sized for the largest row wastes most of the INT8 range on small
   rows. Per-channel scales cost `out_features` floats and cut error roughly
   by the spread of row norms.
2. FP32: ~496 MB moved per token; INT8: ~124 MB plus small scale reads --
   about 4x fewer bytes, directly proportional to decode throughput in the
   memory-bound regime.
3. Weight bytes are constant, but cache bytes grow as
   `batch * layers * Hkv * T * D`; past a context length the cache dominates
   total traffic and capacity, so its precision sets the serving ceiling.

## Modern LLM Systems Delta

Production stacks ship GPTQ/AWQ 4-bit weights with fused dequant GEMM
kernels, FP8 on Hopper-class hardware with per-tensor scaling factors
managed by the framework, quantized KV caches, and QLoRA for fine-tuning on
quantized bases. Quantization is also moving into training (FP8 pretraining)
rather than remaining a post-processing step.

## Professional Takeaways

State three things about any quantization scheme before adopting it: what is
quantized (weights/activations/cache), where the scales live (per tensor/
channel/group), and what regime it accelerates (memory- vs compute-bound).
Then demand the eval delta.

## Further Exploration

- [GPTQ: Accurate Post-Training Quantization for GPT](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438)
