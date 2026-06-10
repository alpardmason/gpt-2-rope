# 06: Model Contracts, Initialization, and Weight Tying

## Objectives and Prerequisites

Read a decoder implementation through tensor contracts, parameter ownership,
initialization, and objective semantics. Prerequisite: GPT-2 theory and 00-05.

**Source map:** [`model.py`](../../src/gpt2_rope/model.py) `CausalLMOutput`,
`MLP`, `TransformerBlock`, `GPT`; [`config.py`](../../src/gpt2_rope/config.py)
`ModelConfig`; and [`test_model.py`](../../tests/test_model.py).

## Contracts and Shapes

`GPT.forward` accepts `input_ids: [B,T]` and returns:

```text
logits:          [B,T,V]
loss:            scalar or None
past_key_values: L tuples of compact K/V tensors
```

The block is GPT-2 pre-norm:

```text
x = x + attention(LN1(x))
x = x + MLP(LN2(x))
```

The MLP maps `[B,T,C] -> [B,T,4C] -> [B,T,C]` by default and uses GPT-2's
tanh-approximated GELU. Final layer norm precedes the language-model head.

Pre-norm is a deliberate stability decision, not a given: the original
Transformer and GPT-1 normalized *after* each residual sum (post-norm), which
removes the identity gradient path and makes deep stacks warmup-sensitive.
`ModelConfig.norm_placement` exposes post-norm as a switchable ablation;
chapter 17 runs the comparison and presents the full trade-off table.

Labels are aligned with inputs, then shifted internally:

```python
cross_entropy(logits[:, :-1], labels[:, 1:], ignore_index=-100)
```

This makes the model the single owner of next-token alignment.

## Initialization and Shared Storage

Linear and embedding weights use `N(0, initializer_range)`. Residual output
projections are reinitialized with:

```text
std = initializer_range / sqrt(2 * num_layers)
```

The factor controls residual-stream variance growth across two residual
branches per layer. Input embeddings and `lm_head.weight` are the same
parameter object, reducing parameters and coupling input/output token geometry.

**Recommendation:** test tying with storage identity, not equal values.
**Rationale:** copied weights begin equal but diverge during optimization.

| Choice | Parameters | Semantics |
|---|---:|---|
| Shared parameter (current) | Lower | Updates are intrinsically tied |
| Periodic copy | Same allocation | Can drift between copies |
| Independent matrices | Higher | More flexibility |

## Optimizer Grouping

Trainable parameters with `ndim >= 2` receive weight decay; vectors such as
biases and normalization scales do not. This pragmatic rule avoids brittle
name lists. CUDA availability enables fused AdamW, although actual backend
support still depends on runtime tensors.

Gradient checkpointing disables cache production during training because cache
state and activation recomputation serve conflicting purposes.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Off-by-one objective | Shift in two layers | Tiny token example | One shift owner | Loss test |
| Tied weights diverge | Values copied, storage separate | Compare `data_ptr` | Assign parameter | Identity test |
| Deep model unstable | Residual init unscaled | Activation/grad stats | Scale output projections | Init regression |
| Context overflow | Prompt + cache too long | Inspect lengths | Reject/truncate | Forward validation |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import torch
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT

c = ModelConfig(vocab_size=64, context_length=16, d_model=32,
                num_layers=2, num_heads=4, num_kv_heads=2)
m = GPT(c)
o = m(torch.randint(0, 64, (2, 7)), labels=torch.randint(0, 64, (2, 7)))
print(o.logits.shape, o.loss.shape)
print("tied", m.token_embedding.weight.data_ptr() == m.lm_head.weight.data_ptr())
print("parameters", m.parameter_count())
PY
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py -q
```

Expected: logits `[2,7,64]`, scalar loss, and `tied True`.

## Exercises

1. Why call `.contiguous().view(...)` before cross entropy?
2. Why does parameter counting not double-count tied embeddings?
3. What is lost by returning a plain tuple instead of `CausalLMOutput`?

## Solutions

1. Slicing can create non-contiguous strides; `view` requires compatible
   storage and flattening must preserve logical order.
2. PyTorch module parameter iteration deduplicates shared parameter objects.
3. Named fields, optionality clarity, and an extensible typed interface.

## Modern LLM Systems Delta

Many current decoders use RMSNorm, SwiGLU, no linear biases, larger RoPE
contexts, GQA/MQA, parallel residual variants, and sharded parameters. The
contract-first reading method is unchanged.

## Professional Takeaways

Derive shapes and objective alignment on paper. Explain initialization as a
stability policy, weight tying as parameter ownership, and optimizer grouping
as regularization policy.

## Further Exploration

- [GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [PyTorch `nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)

