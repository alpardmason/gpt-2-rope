# 15: LoRA and Supervised Fine-Tuning

## Objectives and Prerequisites

Understand low-rank module replacement, freezing, merge state, adapter
serialization, and how SFT changes training semantics. Prerequisite: 05, 10-14.

**Source map:** [`lora.py`](../../src/gpt2_rope/lora.py) `LoRALinear`,
`apply_lora`, adapter state/save/load/merge functions;
`train_finetuning` in [`training.py`](../../src/gpt2_rope/training.py);
[`test_lora_generation.py`](../../tests/test_lora_generation.py); and
[`finetune_lora.yaml`](../../configs/finetune_lora.yaml).

## LoRA Contract

For frozen base weight `W`:

```text
y = x W^T + scale * x A^T B^T
A: [rank,in_features]
B: [out_features,rank]
scale = alpha/rank
```

`A` uses Kaiming initialization and `B` starts at zero, so the adapter initially
leaves the base function unchanged. Dropout applies only on the adapter path.

`apply_lora` walks every module's immediate children and replaces matching
`nn.Linear` attributes. Targeting is by local attribute name, making config
concise but potentially broad: every `proj` matches, not one fully qualified
path.

Before replacement, fine-tuning freezes all base parameters. New adapter
parameters default to trainable, and optimizer grouping sees only them.

## Merge State and Serialization

Merge adds `(B @ A) * scale` to base weight in place and sets a flag so forward
does not add it twice. Unmerge subtracts the same update. Numerical equivalence
is tested.

Adapter safetensors contain only parameters ending in `lora_a`/`lora_b`.
Loading requires the same base architecture and the same LoRA replacement
layout to exist first.

SFT differs from pretraining:

- Dataset labels mask prompt/padding.
- Throughput counts supervised tokens.
- Base checkpoint may initialize the model.
- Adapter artifact accompanies the full training checkpoint.
- This implementation does not run validation in its SFT loop.

**Recommendation:** version adapter config with adapter weights.
**Rationale:** rank, alpha, targets, and base-model identity are required to
interpret tensors.

| Adaptation | Trainable state | Inference behavior | Risk |
|---|---:|---|---|
| Full fine-tune | All params | One model | Cost/forgetting |
| LoRA unmerged | Small | Extra matmuls | Layout dependency |
| LoRA merged | Full exported weight | No adapter latency | Harder composition |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| No modules replaced | Wrong target names | Print module tree/count | Correct targets | Fail on zero |
| Output doubles update | Merged flag ignored | Compare merge forward | Skip adapter path | Equivalence test |
| Adapter load mismatch | Different layout/base | Inspect state keys/config | Recreate exact layout | Metadata |
| Too many trainable params | Freeze order wrong | Count trainable params | Freeze before apply | Parameter-budget test |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from gpt2_rope.config import ModelConfig
from gpt2_rope.lora import apply_lora
from gpt2_rope.model import GPT

m = GPT(ModelConfig(vocab_size=64, context_length=16, d_model=32,
    num_layers=2, num_heads=4, num_kv_heads=2))
for p in m.parameters():
    p.requires_grad = False
n = apply_lora(m, rank=4, alpha=8, target_modules=("q_proj", "v_proj"))
print("replaced", n)
print("trainable", m.parameter_count(trainable_only=True),
      "total", m.parameter_count())
PY
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_lora_generation.py -q
```

Expected: four replacements for two layers and a small trainable fraction.

## Exercises

1. Why initialize `B` to zero but not both matrices?
2. Why must merge/unmerge run under `no_grad`?
3. What evaluation is missing before calling an adapter production-ready?

## Solutions

1. Zero `B` gives zero initial update while nonzero `A` permits gradients into
   `B`; both zero can block useful initial gradient flow.
2. They are deployment/state transitions, not differentiable training
   operations, and in-place leaf updates would violate autograd rules.
3. Held-out task quality, regression/safety suites, base comparison, latency,
   memory, merge equivalence, and data/template validation.

## Modern LLM Systems Delta

Modern PEFT includes QLoRA, DoRA, rank adaptation, multiple adapters, quantized
bases, distributed adapter training, serving-time routing, and registries that
bind adapter, base model, tokenizer, and prompt template.

## Professional Takeaways

LoRA is not merely two matrices. Production correctness depends on replacement
scope, frozen-state proof, base identity, adapter metadata, merge lifecycle,
and task evaluation.

## Further Exploration

- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [DoRA](https://arxiv.org/abs/2402.09353)

