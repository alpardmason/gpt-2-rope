# Practice 05: Supervised Data and Loss Masking

Companion to [05-supervised-data-and-loss-masking.md](../05-supervised-data-and-loss-masking.md).
Persist all deliverables to `notes/chapters/05.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `finetune` CLI to a masked loss

Follow one SFT example from JSONL to cross entropy. Start at `finetune` in
[`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`finetune` -> `train_finetuning` in
[`training.py`](../../../src/gpt2_rope/training.py) -> `SFTDataset.__init__`
in [`data.py`](../../../src/gpt2_rope/data.py) -> `build_sft_example` ->
`collate_sft` -> `GPT.forward` in
[`model.py`](../../../src/gpt2_rope/model.py).

Record at each hop:

- In `build_sft_example`, the three truncation branches: response too long
  (truncate, force final EOS, drop the prompt), response fits (keep the
  rightmost prompt suffix), and the rejected empty-response case. For each
  branch, which side of the example loses tokens?
- The label construction `[-100] * len(prompt_ids) + response_ids`: which
  exact positions carry training signal, and which guard raises if none do?
- In `GPT.forward`, where the single shift happens
  (`logits[:, :-1]` against `labels[:, 1:]` with `ignore_index=-100`).
  State in one sentence why the dataset must NOT also shift.

### Trace B: dynamic batch assembly

Trace `collate_sft` line by line for a batch of lengths 2 and 4. Record the
allocated shapes of `input_batch` and `label_batch`, the fill values
(`pad_token_id` versus `-100`), and which rows/columns are overwritten by
real data. State why EOS padding is loss-safe here but would not be safe
without masked labels.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_collate_sft_pads_inputs_and_masks_labels` in
   [`test_data.py`](../../../tests/test_data.py) to make, including the
   exact padded rows it checks. Then read it and diff against your guess.
2. **Lab output prediction.** Before running the chapter lab
   (`build_sft_example("question: ", "answer", tok, max_length=16)`),
   predict: will `len(x) == len(y)`, will the final label be EOS, and will
   the supervised count be less than the length? Then run it and record
   the actual three printed lines.
3. **Mutation prediction.** In `collate_sft`, change the `label_batch` fill
   value from `-100` to `pad_token_id`. Predict which row assertion of
   `test_collate_sft_pads_inputs_and_masks_labels` fails first and what
   the mismatched list looks like. Verify with
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_data.py`, then revert
   (`git checkout -- src/gpt2_rope/data.py`).
4. **Boundary prediction.** Predict the exact exception type and message of
   `build_sft_example("prompt", "", tok, max_length=8)`, and predict what
   `build_sft_example` returns when the response alone exceeds
   `max_length` (which IDs survive, and what the last ID is forced to).
   Verify both in a REPL.

## 3. Tool walkthrough: REPL tensor inspection of SFT masks

- **Why this tool.** Loss masking is product behavior: a wrong `-100`
  placement silently trains the model to parrot prompts. The professional
  habit is to visualize one tokenized example - token text, IDs, labels -
  before any SFT launch, exactly as serious teams render chat-template
  previews.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv run python -i -c "
import torch
from gpt2_rope.data import build_sft_example, collate_sft
from gpt2_rope.tokenizer import ByteBPETokenizer
tok = ByteBPETokenizer.train(
    ['question answer'], 280, special_tokens=['<|endoftext|>']
)
x, y = build_sft_example('question: ', 'answer', tok, max_length=16)
for i, l in zip(x, y):
    print(f'{i:>5} {tok.decoder[i]!r:>12} {l:>5}')
"
```

- **Play.**
  1. Read the three-column dump and mark where supervision starts. Confirm
     the boundary matches `len(prompt_ids)` and that the final row is the
     EOS ID with a supervised label.
  2. Shrink `max_length` until the prompt is dropped entirely (response
     plus EOS no longer fits). Record the largest `max_length` that
     triggers the response-truncation branch and what the dump shows.
  3. Build two examples of different lengths, run
     `collate_sft([...], pad_token_id=tok.eos_token_id)`, and print both
     tensors. Verify every padding column has input EOS but label `-100`,
     then count supervised tokens per row with `(labels != -100).sum()`.

## 4. Deliverables

Append to `notes/chapters/05.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the three-column dump command you will reuse, plus the
  truncation boundary you found in play exercise 2.
- 3-5 why-cards. Seed examples: "Why does the truncation policy keep the
  response and sacrifice the prompt?", "What breaks if labels are shifted
  in both the dataset and the model?", "Why is right-padding with EOS safe
  for loss but still visible to attention?"
- Feynman summary: explain to a colleague how `-100` labels turn one
  causal LM objective into instruction tuning, and why supervised-token
  count, not batch size, measures SFT throughput.
