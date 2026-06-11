# Practice 11: Building a Correct Pretraining Loop

Companion to [11-building-a-correct-pretraining-loop.md](../11-building-a-correct-pretraining-loop.md).
Persist all deliverables to `notes/chapters/11.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `pretrain` CLI to the commit point of one step

Follow one optimizer step from the command line through the loop's state
machine. Start at `pretrain` in [`cli.py`](../../../src/gpt2_rope/cli.py)
and trace:

`pretrain` -> `load_experiment_config` -> `train_pretraining` in
[`training.py`](../../../src/gpt2_rope/training.py) ->
`_infinite_loader` -> the accumulation micro-loop ->
`save_checkpoint` in [`checkpoint.py`](../../../src/gpt2_rope/checkpoint.py).

Record at each hop:

- Who owns the `progress` dict, which four keys it carries, and the exact
  line where `data_position` increments relative to the `next(stream)` call.
  Why does that ordering matter for resume?
- For `micro_batch_size=2`, `sequence_length=16`,
  `gradient_accumulation_steps=1`, `world_size=1`: what value does
  `step_tokens` accumulate per step, and what tensor shape does
  `next(stream)` yield?
- `accumulated_loss` sums losses already divided by `accumulation`. What
  quantity is logged as `train/loss` - the sum, the mean, or something else?
- After the validation block calls `evaluate(raw_model, ...)`, which single
  call restores training mode, and on which object (`model` or `raw_model`)?
  Why does the distinction matter once DDP or `torch.compile` wraps the
  model?

### Trace B: `evaluate` and `_infinite_loader` contracts

Trace `evaluate` in [`training.py`](../../../src/gpt2_rope/training.py) line
by line and record: the two mode-related effects of
`@torch.inference_mode()` plus `model.eval()`; how per-batch loss is
weighted by `count` before averaging; and where the perplexity exponent is
capped (`math.exp(min(mean_loss, 20))`) and why. Then trace
`_infinite_loader` and state in one sentence what `sampler.set_epoch(epoch)`
changes between passes and who advances `epoch`.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the artifact assertions
   you expect `test_pretraining_smoke_writes_metrics_checkpoints_and_traces`
   in [`test_training.py`](../../../tests/test_training.py) to make: which
   files and directories under the run dir, and which metric keys in
   `metrics.jsonl`. Then read it and diff against your guess.
2. **Lab output prediction.** Predict what the chapter lab prints (one
   number) and whether it is the pre-clip or post-clip gradient norm; check
   the `clip_grad_norm_` return contract before declaring victory.
3. **Mutation prediction.** Delete the `model.train()` call that follows the
   validation block in `train_pretraining`. Predict the failure symptom:
   does `uv run pytest tests/test_training.py` go red, and if not, why does
   this real bug stay invisible at `dropout=0.0`? Verify, then revert
   (`git checkout -- src/gpt2_rope/training.py`) and write down what test
   would be needed to catch it.
4. **Boundary prediction.** Predict the full metrics dict returned by
   `evaluate(model, loader, device, batches=0, precision="fp32")` - reason
   through `max(tokens, 1)` and `math.exp`. Verify in a REPL with a tiny
   `GPT` and a one-batch loader such as
   `[(torch.randint(0, 32, (1, 8)), torch.zeros(1))]`.

## 3. Tool walkthrough: `jq` over training metrics JSONL

- **Why this tool.** `metrics.jsonl` is the loop's flight recorder, and `jq`
  turns it into evidence without a notebook: warmup checks, gradient-norm
  spikes, throughput regressions. Professionals interrogate run logs with
  `jq` in CI and on remote boxes where dashboards do not exist.
- **How.** First produce a tiny run (CPU, a few seconds), then query it:

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from pathlib import Path

from gpt2_rope.config import (DataConfig, ExperimentConfig, ModelConfig,
                              MonitoringConfig, TrainingConfig)
from gpt2_rope.data import prepare_corpus
from gpt2_rope.tokenizer import ByteBPETokenizer
from gpt2_rope.training import train_pretraining

work = Path("runs/practice-11")
work.mkdir(parents=True, exist_ok=True)
tokenizer = ByteBPETokenizer.train(
    ["alpha beta gamma delta epsilon zeta"], vocab_size=280,
    special_tokens=["<|endoftext|>"])
tokenizer.save(work / "tokenizer")
corpus = work / "corpus.txt"
corpus.write_text("\n".join("alpha beta gamma delta epsilon zeta"
                            for _ in range(40)) + "\n", encoding="utf-8")
prepare_corpus([corpus], work / "processed", tokenizer, 0.25)
config = ExperimentConfig(
    model=ModelConfig(vocab_size=300, context_length=16, d_model=16,
                      num_layers=1, num_heads=2, num_kv_heads=1, dropout=0.0),
    data=DataConfig(train_path=work / "processed" / "train.bin",
                    validation_path=work / "processed" / "validation.bin",
                    tokenizer_dir=work / "tokenizer", sequence_length=16),
    training=TrainingConfig(output_dir=work / "run", device="cpu",
                            micro_batch_size=2, gradient_accumulation_steps=1,
                            max_steps=8, warmup_steps=2, learning_rate=1e-3,
                            min_learning_rate=1e-4, precision="fp32",
                            eval_every=4, eval_batches=1, checkpoint_every=4),
    monitoring=MonitoringConfig(log_every=1, tensorboard=False),
)
print(train_pretraining(config))
PY
jq -r 'select(."train/loss") | [.step, ."train/loss", ."train/learning_rate"] | @tsv' \
  runs/practice-11/run/metrics.jsonl
```

- **Play.**
  1. Confirm the warmup from the LR column: with `warmup_steps=2` the first
     two logged learning rates should climb toward `1e-3`. Cross-check the
     values against `cosine_learning_rate` by hand.
  2. Compute the worst gradient norm of the run:
     `jq -s '[.[] | ."train/gradient_norm" | select(.)] | max' runs/practice-11/run/metrics.jsonl`.
  3. Break a key on purpose: query `."train/los"` (typo) and observe that
     `jq` silently yields nothing rather than erroring - record why
     `select(.)` is therefore essential in every pipeline you keep.

## 4. Deliverables

Append to `notes/chapters/11.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the two `jq` queries you will reuse, plus the observed warmup
  LR sequence.
- 3-5 why-cards. Seed examples: "Why is `data_position` incremented before
  the forward pass rather than after the optimizer step?", "What breaks if
  the micro-loss is not divided by the accumulation count?", "Why does the
  `try/finally` protect the logger and process group but not save an
  emergency checkpoint?"
- Feynman summary: explain to a colleague why a training loop is a
  transaction - consume data, build gradients, update, commit progress -
  and what "completed step" must mean for resume to be exact.
