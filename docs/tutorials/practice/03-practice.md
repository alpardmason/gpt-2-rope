# Practice 03: Byte-Level BPE From Files to Training

Companion to [03-byte-level-bpe-from-files-to-training.md](../03-byte-level-bpe-from-files-to-training.md).
Persist all deliverables to `notes/chapters/03.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `tokenizer train` to ranked merges on disk

Follow a training run from the command line into the files it writes. Start
at `tokenizer_train` in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`tokenizer_train` -> `ByteBPETokenizer.train` in
[`tokenizer.py`](../../../src/gpt2_rope/tokenizer.py) (word counting via
`GPT2_PATTERN` and `bytes_to_unicode`, then the merge loop) ->
`ByteBPETokenizer.save` -> `identity`.

Record at each hop:

- The vocabulary layout `train` constructs: 256 byte symbols first, then
  learned tokens, then specials. Which line enforces
  `vocab_size >= 256 + len(specials)` and what exact message does it raise?
- The tie-break key in the merge loop:
  `min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))`. State in
  one sentence why the lexicographic second element makes retraining
  deterministic.
- What `identity()` hashes (canonical-JSON encoder, a NUL byte, then the
  merge lines) and which downstream artifact records it (the data-prep
  manifest, chapter 04).

### Trace B: inference-time encoding by rank

Trace `tokenizer_inspect` -> `_tokenizer` -> `ByteBPETokenizer.from_files`
-> `encode` -> `bpe`. Record:

- How `encode` splits out special tokens before applying `GPT2_PATTERN`,
  and what happens to a segment that exactly equals a special token.
- Inside `bpe`, who decides which pair merges next: current pair frequency
  or `merge_ranks`? Note the cache bound (100,000 entries) and that
  eviction clears the entire dictionary rather than the oldest entry.
- In `decode`, why `errors="replace"` exists and which inputs trigger it.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_training_is_deterministic` in
   [`test_tokenizer.py`](../../../tests/test_tokenizer.py) to make: what it
   trains twice and which attributes it compares. Then read it and diff
   against your guess.
2. **Lab output prediction.** Before running the chapter lab's
   `tokenizer inspect --text "café"`, predict: how many bytes UTF-8 needs
   for `é`, whether `token_ids` and `tokens` have the same length, and what
   `round_trip` prints. Then run the lab.
3. **Mutation prediction.** In `ByteBPETokenizer.save`, change the merge
   serialization to write `sorted(self.merges)` instead of `self.merges`.
   Predict which of the two assertions in
   `test_byte_bpe_round_trip_and_persistence` can fail and why the
   round-trip assertion cannot (decoding reverses any segmentation).
   Verify with `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_tokenizer.py`,
   then revert (`git checkout -- src/gpt2_rope/tokenizer.py`).
4. **Boundary prediction.** Predict the exact exception type and message of
   `ByteBPETokenizer.train([], vocab_size=200, special_tokens=["<|endoftext|>"])`
   and of `ByteBPETokenizer.train(["x"], 280).eos_token_id`. Verify both in
   a REPL.

## 3. Tool walkthrough: the `tokenizer` CLI plus `xxd` on asset files

- **Why this tool.** Tokenizer bugs hide in bytes, not in Python objects.
  Hex-dumping `vocab.json` and `merges.txt` shows you exactly what the
  reversible byte mapping persists - the same skill you need when a
  downloaded tokenizer asset fails its checksum or a vocabulary file is
  corrupted in transit.
- **How.**

```bash
tmp="$(mktemp -d)"
printf 'low lower lowest\nnaive café\n' > "$tmp/corpus.txt"
UV_CACHE_DIR=.uv-cache uv run gpt2-rope tokenizer train \
  "$tmp/corpus.txt" "$tmp/tok" --vocab-size 280
UV_CACHE_DIR=.uv-cache uv run gpt2-rope tokenizer inspect \
  "$tmp/tok" --text "café"
head -n 5 "$tmp/tok/merges.txt"
xxd "$tmp/tok/merges.txt" | head -n 10
xxd "$tmp/tok/vocab.json" | head -n 20
```

- **Play.**
  1. In the `xxd` dump of `merges.txt`, locate the `#version: 0.2` header
     and the first learned merge. Cross-check that `from_files` skips the
     header because of its `#` prefix, not its position.
  2. Run `tokenizer inspect --text "café"` and match each entry of the
     `tokens` array to bytes in the hex dump: find the two byte symbols
     that encode `é` (UTF-8 `0xC3 0xA9`) and record the visible characters
     the byte map assigns them.
  3. Break a flag: rerun training with `--vocab-size 200` and record the
     diagnostic. Note which layer rejects it - Typer's `min=257` option
     constraint - and compare its wording with the library's own
     `ValueError` from prediction task 4.

## 4. Deliverables

Append to `notes/chapters/03.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the inspect/xxd pairing you will reuse, plus the byte symbols
  you matched for `é`.
- 3-5 why-cards. Seed examples: "Why must all 256 byte symbols exist even
  if absent from the training corpus?", "Why does encoding follow merge
  rank instead of recounting pair frequencies?", "What breaks if
  `merges.txt` lines are reordered while `vocab.json` is untouched?"
- Feynman summary: explain to a colleague why
  `decode(encode(text)) == text` holds for any valid text even with an
  untrained or badly trained merge list, and which property that invariant
  therefore cannot test.

Tier 2: this chapter has a kata. After the deliverables above, run
`UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start bpe` and
follow [katas/bpe/README.md](../../../katas/bpe/README.md).
