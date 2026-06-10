# 20: Evaluation Harnesses and Benchmarks

## Objectives and Prerequisites

Build and reason about the three scoring primitives behind every LLM
leaderboard: windowed perplexity, length-normalized choice logprobs, and
exact-match generation probes. Prerequisite: 09, 11, 14.

**Source map:** [`evaluation.py`](../../src/gpt2_rope/evaluation.py)
`evaluate_perplexity_files`, `continuation_logprob`,
`evaluate_multiple_choice`, `build_passkey_samples`, `evaluate_passkey`;
[`cli.py`](../../src/gpt2_rope/cli.py) `eval suite`;
[`test_evaluation.py`](../../tests/test_evaluation.py).

## Scoring Contracts

**Perplexity** (intrinsic): non-overlapping windows of at most
`context_length` tokens; per-file and token-weighted aggregate;
`exp(min(loss, 20))` guards overflow. Comparable only under the same
tokenizer -- perplexity is per-token, and tokens are tokenizer-defined.

**Multiple choice** (logprob scoring): for each choice, sum
`log p(choice_token_t | question + choice_<t)`; the prefix is conditioned on,
never scored. Length normalization divides by choice token count, otherwise
short choices win systematically:

```text
score(c) = (1/|c|) * sum_t log p(c_t | q, c_<t)
prediction = argmax_c score(c)
```

This is how MMLU, HellaSwag, and ARC are actually scored for base models --
no generation, no parsing, deterministic.

**Passkey retrieval** (extrinsic, generative): plant a key, pad with filler,
greedy-decode, require the key verbatim. Binary, position-controllable, and
the standard probe for whether long-context models actually use their
context.

## Choosing the Right Primitive

| Question | Primitive | Why |
|---|---|---|
| Is pretraining progressing? | perplexity | smooth, cheap, sensitive |
| Does it know X? | multiple choice | deterministic, no parser |
| Can it use its context? | passkey/retrieval | directly behavioral |
| Is it good to talk to? | none of these | needs human/LLM judges |

**Recommendation:** track perplexity continuously, run choice tasks at
checkpoints, treat generation probes as regression gates. **Rationale:** cost
ladder matches information value. **Alternatives:** lm-evaluation-harness for
breadth (hundreds of tasks, standardized prompts) once a model is worth
external comparison; this module is the transparent core of what it does.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Perplexity improves, tasks flat | tiny model/benchmark mismatch | check task difficulty floor | scale-appropriate tasks | accuracy-vs-random baseline |
| Short choices always win | missing length normalization | inspect per-choice scores | normalize by token count | normalization test |
| Scores differ across harnesses | prompt/normalization variants | diff exact scored strings | pin harness version | scoring-contract docs |
| Passkey accuracy 100% trivially | key inside decode budget tail | vary filler length | position sweep | multiple positions |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_evaluation.py -q
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import torch
from gpt2_rope.config import ModelConfig
from gpt2_rope.evaluation import (MultipleChoiceExample, evaluate_multiple_choice,
                                  build_passkey_samples, evaluate_passkey)
from gpt2_rope.model import GPT
from gpt2_rope.tokenizer import ByteBPETokenizer

tok = ByteBPETokenizer.train(["the quick brown fox jumps over the lazy dog"],
                             vocab_size=280, special_tokens=["<|endoftext|>"])
model = GPT(ModelConfig(vocab_size=300, context_length=128, d_model=32,
                        num_layers=1, num_heads=4, num_kv_heads=2)).eval()
device = torch.device("cpu")
mc = [MultipleChoiceExample("the quick", [" brown", " lazy"], 0)]
print(evaluate_multiple_choice(model, tok, mc, device))
samples = build_passkey_samples(4, filler_sentences=2, seed=0)
print(evaluate_passkey(model, tok, samples, device))
PY
```

Expected: a random-weight model scores near chance (0.5) on the binary choice
and 0.0 on passkey -- the harness mechanics, not the model, are under test.

## Exercises

1. Why is perplexity incomparable across tokenizers, and what alternative
   metric fixes this?
2. An un-normalized choice scorer favors which kind of wrong answer, and
   why does normalization introduce its own bias?
3. Why do base-model benchmarks score logprobs instead of generating an
   answer letter and parsing it?

## Solutions

1. Loss is averaged per token; a tokenizer producing fewer, larger tokens
   concentrates more information per token. Bits-per-byte normalizes by
   UTF-8 bytes of the underlying text and is tokenizer-invariant.
2. It favors short choices (fewer negative terms summed). Normalization
   conversely inflates choices whose tokens are individually predictable
   (common words), regardless of semantic fit.
3. Base models follow instructions unreliably; parsing failures would
   measure formatting, not knowledge. Logprob scoring removes the parser
   and the prompt-sensitivity of answer extraction.

## Modern LLM Systems Delta

Production evaluation adds: lm-evaluation-harness or bespoke harnesses with
versioned task definitions, contamination checks against training data
(chapter 19's hashes are the hook), LLM-as-judge for open-ended quality,
agentic/tool-use evals, safety suites, and eval-result registries feeding
release gates. Frontier labs treat eval definitions as code with the same
review discipline as model code.

## Professional Takeaways

A number without its scoring contract (prompt, normalization, harness
version, tokenizer) is not a result. Build evals before training something
you care about; the harness is how you know the run is worth its compute.

## Further Exploration

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- [Landmark Attention / passkey retrieval](https://arxiv.org/abs/2305.16300)
