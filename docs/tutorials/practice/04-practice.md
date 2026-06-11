# Practice 04: Pretraining Data Pipelines and Memmap

Companion to [04-pretraining-data-pipelines-and-memmap.md](../04-pretraining-data-pipelines-and-memmap.md).
Persist all deliverables to `notes/chapters/04.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `data prepare` to bytes on disk

Follow one corpus from the command line into binary token streams. Start at
`data_prepare` in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`data_prepare` -> `prepare_corpus` in
[`data.py`](../../../src/gpt2_rope/data.py) -> `read_documents` -> per-split
encode loop -> `np.asarray(..., dtype=np.uint16).tofile` -> `manifest.json`.

Record at each hop:

- How `read_documents` dispatches on suffix: which JSONL keys are accepted
  (`text`, or `prompt` plus `response`) and what exception a record missing
  `response` raises.
- The split arithmetic:
  `split_index = max(1, round(len(documents) * (1 - validation_fraction)))`.
  Compute it by hand for 3 documents at fraction 0.34 and state why the
  split is deterministic but potentially biased.
- Everything the manifest records: `version`, `dtype`, `token_counts`,
  `tokenizer` (the `identity()` hash from chapter 03), `sources`, and
  `source_sha256`. Which of these would expose a silently edited corpus?

### Trace B: the training read path

Trace `train_pretraining` in
[`training.py`](../../../src/gpt2_rope/training.py) ->
`MemmapTokenDataset.__init__` -> `__len__` -> `__getitem__` in
[`data.py`](../../../src/gpt2_rope/data.py).

Record:

- The dtype journey: `uint16` on disk via `np.memmap`, converted to
  `int64` in `__getitem__`. Why does PyTorch embedding lookup require the
  conversion, and why is `.copy()` applied to each window slice?
- The window geometry: a sample reads `sequence_length + 1` tokens and
  returns `(window[:-1], window[1:])`. Which of the two tensors does the
  current pretraining loop actually consume, and who performs the shift?

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_prepare_corpus_and_memmap_batches` in
   [`test_data.py`](../../../tests/test_data.py) to make about the
   manifest, the dataset shapes, and the x/y alignment. Then read it and
   diff against your guess.
2. **Lab output prediction.** Before running the chapter lab (3 documents,
   `--validation-fraction 0.34`), predict the manifest's top-level keys and
   whether `token_counts.validation` is zero or positive. Then run it and
   compute `train.bin`'s expected byte size from the printed token count.
3. **Mutation prediction.** In `MemmapTokenDataset.__getitem__`, make both
   returned tensors `window[:-1]` (drop the shift). Predict which assertion
   of `test_prepare_corpus_and_memmap_batches` fails and what
   `np.testing.assert_array_equal` reports. Verify with
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_data.py`, then revert
   (`git checkout -- src/gpt2_rope/data.py`).
4. **Boundary prediction.** Predict the exact exception type and message of
   constructing `MemmapTokenDataset(path, sequence_length=8)` over a file
   containing only 4 tokens. Verify in a REPL by writing
   `np.arange(4, dtype=np.uint16).tofile(...)` to a temporary file first.

## 3. Tool walkthrough: `numpy.memmap` inspection plus `ls -la`/`du`

- **Why this tool.** Data bugs are cheaper to catch at the byte level than
  after a training run. Being able to derive "this file must be exactly
  `token_count * 2` bytes" and verify it with `ls` is the data-pipeline
  equivalent of a shape check, and `np.memmap` lets you audit a corpus far
  larger than RAM.
- **How.** Run the chapter lab first so `$tmp/data` exists, then:

```bash
ls -la "$tmp/data"
du -h "$tmp/data"
UV_CACHE_DIR=.uv-cache uv run python -i -c "
import json, numpy as np
from pathlib import Path
root = Path('$tmp/data')
manifest = json.loads((root / 'manifest.json').read_text())
tokens = np.memmap(root / 'train.bin', mode='r', dtype=np.uint16)
print(manifest['token_counts'], tokens.size, tokens[:16])
"
```

- **Play.**
  1. Verify `train.bin`'s size in `ls -la` equals
     `token_counts['train'] * 2` exactly, and find every EOS token id in
     the stream with `np.where` - count them against the number of
     training documents.
  2. Reopen `train.bin` with the wrong dtype
     (`np.memmap(..., dtype=np.uint32)`) and record what happens to `size`
     and to the values. State why no error is raised even though every
     token is now garbage.
  3. Attempt `tokens[0] = 1` on the `mode="r"` memmap and record the
     exception. Relate it to why `__getitem__` copies windows before
     handing them to `torch.from_numpy`.

## 4. Deliverables

Append to `notes/chapters/04.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the memmap inspection snippet you will reuse, plus the
  byte-size arithmetic you verified.
- 3-5 why-cards. Seed examples: "Why is EOS appended after every document
  rather than between splits?", "What breaks if a tokenizer with
  `vocab_size > 65_535` feeds a `uint16` stream?", "Why is `__len__` floor
  division instead of rounding up?"
- Feynman summary: explain to a colleague how a multi-gigabyte corpus is
  trained on with near-zero resident memory, and what contract the
  manifest adds on top of the raw bytes.
