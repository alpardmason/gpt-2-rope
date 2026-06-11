# 22: Preference Optimization with DPO

## Objectives and Prerequisites

Implement and reason about Direct Preference Optimization: training on
chosen/rejected pairs against a frozen reference model, without a reward
model or reinforcement-learning rollouts. Prerequisite: 05, 15; read the DPO
paper for the derivation -- this chapter covers the system.

**Practice companion:** [22-practice.md](practice/22-practice.md).

**Source map:** [`dpo.py`](../../src/gpt2_rope/dpo.py) `PreferenceDataset`,
`sequence_logprobs`, `dpo_loss`, `train_dpo`;
[`config.py`](../../src/gpt2_rope/config.py) `DPOConfig`;
[`data.py`](../../src/gpt2_rope/data.py) `build_sft_example` (reused);
[`test_dpo.py`](../../tests/test_dpo.py).

## Data and Loss Contracts

Preference data is JSONL:

```json
{"prompt": "...", "chosen": " preferred response", "rejected": " worse response"}
```

Each side is tokenized with the SFT machinery: prompt tokens labeled `-100`,
response tokens supervised, EOS appended. Per-sequence score is the sum of
response-token log-probabilities (`sequence_logprobs`, shape `[B]`).

The objective, with policy `pi`, frozen reference `ref`, and temperature
`beta`:

```text
reward_c = beta * (log pi(chosen)   - log ref(chosen))
reward_r = beta * (log pi(rejected) - log ref(rejected))
loss     = -log sigmoid(reward_c - reward_r)
```

Invariants worth internalizing:

- Only the *margin* is optimized. Both responses' absolute likelihoods can
  fall while the loss improves -- watch `dpo/chosen_reward` for this.
- The reference model anchors the policy: the implicit KL penalty grows as
  the policy drifts, with `beta` setting the exchange rate.
- Two models are resident in memory; the reference runs under `no_grad` and
  never trains.

## DPO vs the Alternatives

| Method | Needs | Compute | Risk profile |
|---|---|---|---|
| SFT only (15) | demonstrations | 1 model | no notion of "worse" |
| RLHF (PPO) | reward model + rollouts | 4 models in flight | reward hacking, instability |
| DPO (here) | preference pairs | 2 models, offline | implicit reward, data-bound |
| IPO/KTO/ORPO | variants of pairs | similar to DPO | different regularization |

**Recommendation:** SFT first, then DPO -- DPO assumes the policy already
places mass on good responses; it reshapes preferences, it does not teach
skills. **Rationale:** the gradient only reweights what the model can
already produce. **Alternatives:** PPO-based RLHF when you need online
exploration or a reusable reward model; KTO when you have thumbs-up/down
rather than pairs.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Margin up, generations worse | both likelihoods collapsing | track `dpo/chosen_reward` | lower LR, raise beta, mix SFT loss | reward decomposition metrics |
| Loss stuck at log(2) | margins near zero | check pair quality/difficulty | better pairs, smaller beta | `dpo/reward_accuracy` trend |
| OOM vs SFT at same batch | reference model forgotten in budget | memory profile | halve batch, LoRA policy | capacity planning |
| Policy gibberish at high LR | KL anchor overpowered | sample generations during training | standard LR is 10-50x below SFT | generation regression gate |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_dpo.py -q
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import torch
from gpt2_rope.dpo import dpo_loss

ref = torch.tensor([-10.0, -10.0])
# Policy already prefers chosen: small loss, positive margin.
good, gm = dpo_loss(torch.tensor([-8.0, -9.0]), torch.tensor([-14.0, -13.0]),
                    ref, ref, beta=0.1)
# Policy prefers rejected: loss above log(2).
bad, bm = dpo_loss(torch.tensor([-14.0, -13.0]), torch.tensor([-8.0, -9.0]),
                   ref, ref, beta=0.1)
print(f"aligned   loss={float(good):.4f} margin={gm['dpo/reward_margin']:+.3f}")
print(f"misaligned loss={float(bad):.4f} margin={bm['dpo/reward_margin']:+.3f}")
print(f"log(2)    {torch.log(torch.tensor(2.0)):.4f}")
PY
```

Expected: aligned loss below `log(2) ~= 0.693`, misaligned above it, margins
symmetric in sign. The unit suite additionally runs a two-step end-to-end
LoRA DPO training on a tiny model.

## Exercises

1. Why does setting `beta` very high make DPO behave like the policy is
   frozen, and very low make it ignore the reference?
2. The chosen response's absolute log-probability falls during training
   while the margin grows. Is the run broken?
3. Why does this implementation reuse `build_sft_example`'s prompt masking
   for both chosen and rejected sequences?

## Solutions

1. `beta` multiplies the log-ratio inside the sigmoid: high `beta` saturates
   the sigmoid for tiny drifts (gradient vanishes, policy pinned to
   reference); low `beta` flattens it so the reference term barely matters.
2. Not necessarily -- only margins are optimized, and likelihood mass can
   shift to other plausible tokens. It becomes a problem when generation
   quality drops; that is why reward decomposition and sampled generations
   are logged, not just the loss.
3. Prompt tokens are identical in both sequences and carry no preference
   signal; scoring them would add identical-in-expectation noise to both
   terms and dilute the margin gradient with prompt-likelihood gradients.

## Modern LLM Systems Delta

Production post-training is a pipeline, not a single loss: SFT -> preference
optimization (DPO/IPO/ORPO or PPO with reward models) -> RL with verifiable
rewards for reasoning (RLVR/GRPO-style), with safety preference data, online
data collection from deployed models, and eval gates (chapter 20) between
every stage. Preference data curation -- not the loss function -- is where
the quality lives.

## Professional Takeaways

Alignment losses are cheap to implement and expensive to monitor. Log the
reward decomposition, sample generations on a schedule, and treat "margin up"
as necessary but never sufficient. Always ask of any preference method: what
anchors the policy, and what is the exchange rate?

## Reimplementation Kata

Tier 2: rebuild `sequence_logprobs` and `dpo_loss` -- the paper-to-code core
-- against the masking, margin, and smoke tests. Start with
`UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start dpo-loss`
and follow [katas/dpo-loss/README.md](../../katas/dpo-loss/README.md).

## Further Exploration

- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
