# Practice 20: Evaluation Harnesses and Benchmarks

Companion to [20-evaluation-harnesses-and-benchmarks.md](../20-evaluation-harnesses-and-benchmarks.md).
Persist all deliverables to `notes/chapters/20.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `eval suite` CLI to one choice logprob

Follow one multiple-choice example from the command line into the scoring
sum. Start at `eval_suite` in [`cli.py`](../../../src/gpt2_rope/cli.py) and
trace:

`eval_suite` -> `_load_model` -> `load_multiple_choice_tasks` in
[`evaluation.py`](../../../src/gpt2_rope/evaluation.py) ->
`evaluate_multiple_choice` -> `continuation_logprob`.

Record at each hop:

- Which three malformed-row conditions `load_multiple_choice_tasks` rejects
  and with what message shape (`path:line_number ...`).
- In `evaluate_multiple_choice`, exactly which tokens are conditioned on and
  which are scored for one choice, and where the length normalization
  happens (the `score /= max(len(tokens) - len(prefix), 1)` line).
- In `continuation_logprob`, the shape of `logits` after `[0]` and of
  `log_probs`, and why position `t` is scored from row `t - 1` of the
  log-probability matrix.
- How `eval_suite` namespaces task metrics per file
  (`f"{key}/{path.stem}"`) before printing or writing the JSON report.

### Trace B: the passkey probe from construction to verbatim match

Trace `build_passkey_samples` ->
`evaluate_passkey` -> `generate` in
[`generation.py`](../../../src/gpt2_rope/generation.py). Record:

- How a `PasskeySample` prompt is assembled (key sentence, filler repeats,
  question suffix) and why the seeded `random.Random` makes the suite
  reproducible.
- Where the prompt is truncated to fit the budget
  (`prompt_ids[-budget:]`) and which side of the prompt survives.
- Which `GenerationConfig` settings force greedy decoding, and what exact
  condition counts a sample as correct.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_continuation_logprob_is_a_finite_negative_sum` in
   [`test_evaluation.py`](../../../tests/test_evaluation.py) to make,
   including what it expects when the prefix is made shorter (more scored
   tokens) and which call it expects to raise. Then read it and diff
   against your guess.
2. **Lab output prediction.** Predict the two dictionaries the chapter lab
   prints: every metric key and the values you expect from a random-weight
   model on one binary choice and four passkey probes (chance levels, not
   exact logprobs).
3. **Mutation prediction.** If `continuation_logprob` validated the prefix
   with `0 <= prefix_length` instead of
   `0 < prefix_length < len(token_ids)`, predict which test fails first and
   with what symptom (a missing `ValueError` or a tensor indexing error?).
   Verify by temporarily editing `evaluation.py`, running
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_evaluation.py`, and
   reverting (`git checkout -- src/gpt2_rope/evaluation.py`).
4. **Boundary prediction.** Predict the exact exception type and message
   when `load_multiple_choice_tasks` reads a row whose `answer` is `5` with
   two choices, and when a task file contains only blank lines. Verify in a
   REPL by writing two small files under `/tmp` and calling the loader.

## 3. Tool walkthrough: `jq` over eval reports plus a hand-built task file

- **Why this tool.** A benchmark number without its scoring contract is not
  a result; reading the raw JSON report and the raw task file is how you
  audit both. Professionals routinely hand-craft two-line eval tasks to
  smoke-test a harness before trusting it on real benchmarks.
- **How.** Build a tiny task file by hand, score it, and interrogate the
  report:

```bash
cat > /tmp/tiny-task.jsonl <<'JSONL'
{"question": "the quick", "choices": [" brown", " lazy"], "answer": 0}
{"question": "the lazy", "choices": [" dog", " fox"], "answer": 1}
JSONL
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
import json, torch
from pathlib import Path
from gpt2_rope.config import ModelConfig
from gpt2_rope.evaluation import evaluate_multiple_choice, load_multiple_choice_tasks
from gpt2_rope.model import GPT
from gpt2_rope.tokenizer import ByteBPETokenizer

tok = ByteBPETokenizer.train(["the quick brown fox jumps over the lazy dog"],
                             vocab_size=280, special_tokens=["<|endoftext|>"])
model = GPT(ModelConfig(vocab_size=300, context_length=128, d_model=32,
                        num_layers=1, num_heads=4, num_kv_heads=2)).eval()
examples = load_multiple_choice_tasks(Path("/tmp/tiny-task.jsonl"))
metrics = evaluate_multiple_choice(model, tok, examples, torch.device("cpu"))
Path("/tmp/eval-report.json").write_text(json.dumps(metrics, indent=2))
PY
jq 'to_entries[] | select(.key | startswith("task/"))' /tmp/eval-report.json
```

  With a trained checkpoint from the chapter 11 lab, the CLI produces the
  same report shape:
  `UV_CACHE_DIR=.uv-cache uv run gpt2-rope eval suite configs/tiny.yaml
  <checkpoint> --task-file /tmp/tiny-task.jsonl --output /tmp/report.json`.
- **Play.**
  1. Change one `answer` field to `5` and re-run; record the file-and-line
     diagnostic `load_multiple_choice_tasks` raises and why failing at load
     time beats failing mid-suite.
  2. Re-score the same examples with `length_normalize=False` and compare
     per-choice winners; construct one example where a short choice wins
     only without normalization.
  3. Add a third single-token choice to one row and predict, then check,
     whether accuracy can change without the model changing.

## 4. Deliverables

Append to `notes/chapters/20.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the `jq` query you will reuse, plus the normalization-flip
  example from the play exercise.
- 3-5 why-cards. Seed examples: "Why is perplexity incomparable across
  tokenizers?", "Why are choice scores divided by choice token count?",
  "What breaks if the passkey prompt is truncated from the right instead of
  the left?"
- Feynman summary: explain to a colleague why base-model benchmarks score
  log-probabilities of fixed continuations instead of generating an answer
  and parsing it.
