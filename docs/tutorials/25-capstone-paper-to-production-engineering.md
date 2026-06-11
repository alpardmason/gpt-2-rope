# 25: Capstone - Paper to Production Engineering

## Objectives and Prerequisites

Produce an implementation-ready design for extending this repository and
defend it like a research engineer. Prerequisite: all previous tutorials.
The expanded toolchain is now in scope: justify your design with ablation
protocol (17), an evaluation plan built on the harness (20), sweep budgets
(21), and -- where relevant -- quantized/serving deployment impact (23, 24).

**Practice companion:** [25-practice.md](practice/25-practice.md).

**Source map:** the full [`src/gpt2_rope`](../../src/gpt2_rope/) package,
[`tests`](../../tests/), [`configs`](../../configs/), [architecture notes](../architecture.md),
and [project decisions](../../AGENTS.md).

## Assignment: Add RoPE Scaling Safely

Design a backward-compatible, optional long-context RoPE scaling feature. You
will not earn credit for pasting a paper equation into `rope.py`. The deliverable
is an engineering proposal, test design, evaluation plan, and small CPU
prototype branch.

Choose one published method, such as position interpolation or YaRN. State the
exact paper/version and distinguish faithfully implemented behavior from local
adaptations.

## Phase 1: Extract the Paper Contract

Write one page containing:

- Claimed capability and assumptions.
- Exact transformation and parameters.
- Training/fine-tuning requirements.
- Reported evaluation tasks and baselines.
- Unsupported extrapolations you will not claim.

Then map mathematical objects to current symbols:

```text
paper position/frequency rule -> RotaryEmbedding tables/forward
maximum context              -> ModelConfig.context_length
cache absolute position      -> offset
experiment parameters        -> frozen config + resolved config
```

## Phase 2: Decision-Complete Design

Recommended public contract:

```text
ModelConfig:
  rope_scaling_type: Literal["none", "<chosen-method>"] = "none"
  rope_scaling_factor: float = 1.0
```

Only add fields required by the chosen paper. Reject invalid combinations in
configuration. Default behavior must reproduce current RoPE tables and logits.
Keep scaling computation owned by `RotaryEmbedding`; attention supplies only
absolute offset.

**Recommendation:** preserve an explicit `"none"` path and differential test.
**Rationale:** a long-context feature must not silently change existing models.

| Approach | Benefit | Risk |
|---|---|---|
| Extend `RotaryEmbedding` | Local ownership | More branches in core math |
| Strategy object | Isolated methods | Premature abstraction for one method |
| New subclass | Clear separation | Construction/config complexity |

For one method, extend the current class. Introduce a strategy only when a
second method creates genuine duplicated complexity.

## Phase 3: Test Pyramid

Unit tests:

- Invalid scaling type/factor is rejected.
- `"none"` produces bitwise/equivalent current tables.
- Chosen method matches hand-computed tiny frequencies.
- Norm preservation remains true.
- Context boundary behavior is explicit.

Component tests:

- Full and cached multi-token logits agree with scaling enabled.
- Save/load configuration reproduces behavior.
- FP32 and autocast outputs are finite on available hardware.

System smoke:

- Tiny model trains for several CPU steps.
- Generate at a position beyond the original training context used by the
  evaluation setup, within configured allocation.
- Existing full suite remains green.

Performance:

- Compare table construction time, forward latency, memory, and compile graph
  behavior against `"none"`.

## Phase 4: Evaluation and Observability

Do not use “the model generated text” as evaluation. Define, using the
harness from chapter 20 (`evaluate_perplexity_files`, `evaluate_passkey`):

- Short-context regression set: quality must not materially regress.
- Long-context task: retrieval/passkey or perplexity by position.
- Baselines: current RoPE at supported context and naive context extension.
- Seeds and confidence intervals.
- Memory/throughput by context length.
- Metrics/config/checkpoint identity sufficient to reproduce each result.

Set acceptance thresholds before running the final experiment. Record failures,
not just the best seed.

## Phase 5: Rollout and Compatibility

- Default remains current behavior.
- Existing YAML files load unchanged.
- Old checkpoints reconstruct with defaults.
- New checkpoints persist scaling configuration.
- README/architecture/tutorial notes explain compatibility.
- Exported weights are not labeled compatible with stock GPT-2.
- Roll back by selecting `"none"` and using a compatible checkpoint.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Short-context regression | Default frequencies changed | Differential logits | Isolate none path | Golden test |
| Cache mismatch | Scaling ignores absolute offset | Prefix/suffix comparison | Position-aware rule | Cache test |
| Good synthetic retrieval, poor LM | Narrow evaluation | Per-position perplexity | Broaden suite | Predeclared metrics |
| Recompile/memory growth | Dynamic tables/shapes | Profiler/compiler logs | Stable cache policy | Performance gate |
| Irreproducible claim | Missing data/config/seed | Artifact audit | Persist provenance | Run manifest |

## Lab: Runnable Baseline

Capture the behavior your extension must preserve:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_model.py -q
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import torch
from gpt2_rope.config import ModelConfig
from gpt2_rope.model import GPT

torch.manual_seed(7)
c = ModelConfig(vocab_size=97, context_length=32, d_model=32,
                num_layers=2, num_heads=4, num_kv_heads=2, dropout=0)
m = GPT(c).eval()
x = torch.randint(0, 97, (2, 10))
with torch.no_grad():
    full = m(x, use_cache=False).logits
    pre = m(x[:, :6], use_cache=True)
    suffix = m(x[:, 6:], past_key_values=pre.past_key_values).logits
print(torch.max(torch.abs(full[:, 6:] - suffix)).item())
PY
```

Expected: a small error within the repository's `2e-5` test tolerance.

## Exercises

1. Write a five-minute design-review opening: problem, evidence, interface,
   risk, and acceptance criteria.
2. Identify the first three tests you would write before implementation.
3. Explain why merely increasing `context_length` is not proof of long-context
   quality.
4. Propose one monitoring signal for research quality and one for systems cost.

## Solutions

1. State the paper claim narrowly, preserve default behavior, put validated
   scaling config in `ModelConfig`, implement inside RoPE, prove table and cache
   semantics, then gate on short/long quality and memory/throughput.
2. Default-table equivalence, hand-computed scaled-frequency case, and cached
   versus full logits with scaling.
3. It allocates larger tables and permits shapes but does not train or validate
   positional extrapolation; attention may degrade badly.
4. Perplexity/retrieval accuracy by position; tokens/s and peak memory by
   context length.

## Interview Defense Checklist

You must be able to explain:

- Why the interface belongs in config and the math belongs in RoPE.
- Which claims come from the paper versus your evaluation.
- How backward compatibility is proven.
- Why cache equivalence is a release blocker.
- What profiling can invalidate the design.
- Which modern alternatives you rejected and why.
- How you would scale tests/checkpoints for FSDP or a serving engine.

## Modern LLM Systems Delta

At industry scale, the same change touches kernel implementations, model
registries, distributed training, serving cache layout, quantization,
evaluation infrastructure, and migration policy. The capstone deliberately
keeps scope small while requiring the same reasoning.

## Professional Takeaways

Paper-to-code work is contract translation under uncertainty. A research
engineer is judged not only by whether an idea runs, but by whether its claims,
compatibility, evidence, performance, and failure modes are legible to others.

## Further Exploration

- [RoFormer](https://arxiv.org/abs/2104.09864)
- [Position Interpolation](https://arxiv.org/abs/2306.15595)
- [YaRN](https://arxiv.org/abs/2309.00071)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [ML reproducibility checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
