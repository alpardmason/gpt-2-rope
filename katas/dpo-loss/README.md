# Kata: dpo-loss

Reimplement the DPO objective from a gutted
[`src/gpt2_rope/dpo.py`](../../src/gpt2_rope/dpo.py).
Tutorial: [22](../../docs/tutorials/22-preference-optimization-with-dpo.md).
Estimated effort: a focused half-evening (1-3 hours). The dataset, collation,
and training loop are kept; the paper-to-code translation
(`sequence_logprobs` and `dpo_loss`) is yours.

## Objective

Translate the DPO paper's equation 7 into tested code: per-sequence response
log-probabilities under a masked-label convention, and the sigmoid loss over
implicit reward margins against a frozen reference model.

## Contract

You must satisfy, without editing any other file:

- `sequence_logprobs(model, input_ids, labels)` returns shape `[B]`: the sum
  of log-probabilities of response tokens only. Labels follow the SFT
  convention (`-100` masks prompt and padding); remember the GPT-2 shift -
  the logit at position `t` scores label `t+1`. Compute log-softmax in
  FP32 for numerical stability under autocast.
- `dpo_loss(policy_chosen, policy_rejected, reference_chosen,
  reference_rejected, beta)` computes implicit rewards
  `beta * (policy - reference)` per side, takes
  `-logsigmoid(chosen_reward - rejected_reward).mean()`, and returns the
  loss plus a no-grad diagnostics dict with keys `dpo/reward_margin`,
  `dpo/reward_accuracy`, `dpo/chosen_reward`, `dpo/rejected_reward`.

The skeleton's `# KATA:` comments restate this in place. The dropped
`torch.nn.functional` import must come back.

## Oracle

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_dpo.py -q
UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py check dpo-loss
```

`test_sequence_logprobs_masks_prompt_tokens` pins the masking and shift;
`test_dpo_loss_prefers_wider_margins` pins the objective's direction;
`test_train_dpo_smoke` proves your functions survive the real loop.

## Workflow

Masking test first, then the margin test, then the smoke test, then
mypy/ruff and the full suite. When green,
`git diff main -- src/gpt2_rope/dpo.py` and record the review notes
required by [katas/README.md](../README.md).

## Hint ladder (open one rung at a time)

1. In `sequence_logprobs`, build a boolean `supervised` mask from
   `labels[:, 1:].ne(-100)` and clamp masked targets to a valid index
   before `gather` - gathering at `-100` is the classic crash here. Zero
   the masked positions by multiplication, then sum per row.
2. If the margin test fails by sign: the loss must DECREASE as the chosen
   margin widens; you want `-logsigmoid(margin)`, not `logsigmoid(-margin)`
   confusion - check with a margin of +10 vs +1 by hand.
3. The diagnostics must be detached floats computed under `torch.no_grad()`;
   the smoke test runs backward through the loss only.
