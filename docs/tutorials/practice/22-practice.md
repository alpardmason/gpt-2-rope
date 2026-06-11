# Practice 22: Preference Optimization with DPO

Companion to [22-preference-optimization-with-dpo.md](../22-preference-optimization-with-dpo.md).
Persist all deliverables to `notes/chapters/22.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from the `dpo` CLI to the margin gradient

Follow one preference pair from the command line into the loss. Start at
`dpo` in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`dpo` -> `load_experiment_config` -> `train_dpo` in
[`dpo.py`](../../../src/gpt2_rope/dpo.py) -> `PreferenceDataset` ->
`build_sft_example` in [`data.py`](../../../src/gpt2_rope/data.py) ->
`collate_preferences` -> `sequence_logprobs` -> `dpo_loss`.

Record at each hop:

- How many `GPT` instances `train_dpo` constructs, how the reference gets
  its weights, and the two lines that guarantee it never trains
  (`requires_grad = False` loop plus `eval()`).
- For one batch: the four tensors `collate_preferences` returns, their
  shapes, and which positions of the label tensors carry `-100`.
- The shape `sequence_logprobs` returns (`[B]`) and which calls run inside
  `torch.no_grad()` versus with gradients in the inner loop of `train_dpo`.
- Where `beta` is born (`DPOConfig.beta` in
  [`config.py`](../../../src/gpt2_rope/config.py)) and which optional flag
  swaps full fine-tuning for LoRA on the policy only.

### Trace B: inside `dpo_loss`

Trace `dpo_loss` line by line with symbolic values. Record:

- The formula for `chosen_rewards`, `rejected_rewards`, and `margins`, and
  why the two reference terms do not cancel against each other - each side
  keeps its own anchor.
- Why the loss is `-logsigmoid(margins).mean()` and what its value is when
  every margin is exactly zero.
- Which four diagnostics land in the metrics dict, computed under
  `no_grad`, and which one detects the "both likelihoods collapsing"
  failure from the tutorial.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_dpo_loss_prefers_wider_margins` in
   [`test_dpo.py`](../../../tests/test_dpo.py) to make, including how it
   uses `math.log(2)` as the dividing line and what it expects of
   `dpo/reward_accuracy` in the aligned and misaligned cases. Then read it
   and diff against your guess.
2. **Lab output prediction.** Before running the chapter lab, hand-compute
   the margins for its tensors (`beta=0.1`, policy chosen `[-8, -9]`,
   rejected `[-14, -13]`, reference `-10` everywhere) and predict all three
   printed lines: aligned loss vs `log(2)`, misaligned loss, and the two
   signed margins.
3. **Mutation prediction.** If `sequence_logprobs` dropped the
   `* supervised` mask and summed every token's log-probability, predict
   which assertion of `test_sequence_logprobs_masks_prompt_tokens` fails
   first and what `zero_scores` becomes. Verify by temporarily editing
   `dpo.py`, running `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_dpo.py`,
   and reverting (`git checkout -- src/gpt2_rope/dpo.py`).
4. **Boundary prediction.** `DPOConfig` requires `beta > 0`, but the
   `dpo_loss` function itself does not. Predict the exact loss value and
   `dpo/reward_accuracy` when you call `dpo_loss` with `beta=0.0` on any
   logprob tensors. Verify in a REPL and state in one sentence what `beta=0`
   does to the reference anchor.

## 3. Tool walkthrough: a REPL on `dpo_loss` with hand-built tensors

- **Why this tool.** Alignment losses fail quietly: the scalar goes down
  while the model gets worse. Probing the loss surface with hand-built
  logprob tensors is how you build the intuition to read `dpo/*` metrics in
  a real run - the same skill applies to any preference objective you will
  meet (IPO, KTO, ORPO).
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run python
```

```python
import torch
from gpt2_rope.dpo import dpo_loss

ref = torch.tensor([-10.0, -10.0])
chosen = torch.tensor([-8.0, -9.0])
rejected = torch.tensor([-14.0, -13.0])
for beta in (0.01, 0.1, 0.5, 1.0, 5.0):
    loss, metrics = dpo_loss(chosen, rejected, ref, ref, beta=beta)
    print(f"beta={beta:<4} loss={float(loss):.4f} "
          f"margin={metrics['dpo/reward_margin']:+.3f} "
          f"acc={metrics['dpo/reward_accuracy']:.1f}")
```

- **Play.**
  1. Run the beta sweep above and record where the loss saturates toward 0
     and where it flattens toward `log(2)`. Connect each end to exercise 1
     of the tutorial (high beta pins the policy, low beta ignores the
     reference).
  2. Build the "margin up, both likelihoods down" pathology: keep
     `ref` fixed, set policy chosen to `[-12, -12]` and policy rejected to
     `[-20, -20]`. Confirm the loss is small and `dpo/reward_margin` is
     positive while `dpo/chosen_reward` is negative - the exact signature
     the failure table tells you to alarm on.
  3. Make the two margins disagree in sign (one pair aligned, one
     misaligned) and predict `dpo/reward_accuracy` before printing it.

## 4. Deliverables

Append to `notes/chapters/22.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the beta-sweep snippet you will reuse, plus the recorded
  pathology signature from the play exercise.
- 3-5 why-cards. Seed examples: "Why does DPO keep a frozen reference model
  in memory at all?", "Why is prompt masking reused for both chosen and
  rejected sequences?", "What breaks if you run DPO before any SFT?"
- Feynman summary: explain to a colleague why optimizing only the reward
  margin can lower the chosen response's absolute likelihood, and which
  logged metric tells you whether that has become a problem.

Tier 2: this chapter has a kata. After the deliverables above, run
`UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start dpo-loss`
and follow [katas/dpo-loss/README.md](../../../katas/dpo-loss/README.md).
