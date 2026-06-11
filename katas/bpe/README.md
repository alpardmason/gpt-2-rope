# Kata: bpe

Reimplement the byte-level BPE core from a gutted
[`src/gpt2_rope/tokenizer.py`](../../src/gpt2_rope/tokenizer.py).
Tutorial: [03](../../docs/tutorials/03-byte-level-bpe-from-files-to-training.md).
Estimated effort: one long evening (3-5 hours). Persistence
(`save`/`from_files`/`identity`) and the constructor are kept; the byte map,
the merge algorithm, encode/decode, and training are yours.

## Objective

Own the algorithm every GPT-2-compatible tokenizer implements: a reversible
byte-to-unicode map covering all 256 bytes, greedy lowest-rank pair merging,
pretokenization-aware encoding with special-token handling, and
deterministic BPE training.

## Contract

You must satisfy, without editing any other file:

- `bytes_to_unicode()` maps every byte 0-255 to a unique visible code point
  exactly as GPT-2 does (printable ASCII and two Latin-1 ranges keep their
  own code points; everything else shifts to `256 + offset`). Missing any
  byte reproduces the documented "tokenizer cannot encode arbitrary text"
  pitfall.
- `ByteBPETokenizer.bpe(token)` repeatedly merges the lowest-ranked
  adjacent pair until none remains in `merge_ranks`, with memoization
  (bounded cache; the constructor provides `self.cache`).
- `encode(text, allowed_special)` splits out special tokens (when allowed),
  applies `GPT2_PATTERN` pretokenization to ordinary segments, maps each
  match through the byte encoder, and emits merged-token ids.
- `decode(token_ids)` inverts the pipeline and replaces invalid UTF-8 with
  the replacement character rather than raising.
- `train(documents, vocab_size, special_tokens)` enforces the minimum
  vocabulary (256 bytes + specials), counts pretokenized words, repeatedly
  merges the most frequent pair (ties broken lexicographically), and lays
  out the vocabulary as byte symbols, then learned merges, then specials.

The skeleton's `# KATA:` comments restate this in place.

## Oracle

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_tokenizer.py -q
# end-to-end through the CLI (train -> inspect -> encode/decode round trip)
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_cli.py -q
UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py check bpe
```

## Workflow

Get the Unicode round-trip test green first (it pins `bytes_to_unicode`,
`encode`, and `decode`), then determinism of `train`, then the CLI workflow
test, then mypy/ruff and the full suite. When green,
`git diff main -- src/gpt2_rope/tokenizer.py` and record the review notes
required by [katas/README.md](../README.md).

## Hint ladder (open one rung at a time)

1. Build `bytes_to_unicode` by listing the three visible ranges
   (`!`-`~`, `¡`-`¬`, `®`-`ÿ`), then assigning `chr(256 + n)` to every byte
   not in them, in byte order. Exactly 256 entries, all distinct.
2. The `bpe` loop: among current adjacent pairs, pick the one with the
   lowest merge rank (`min` with a key that treats unknown pairs as
   infinity); if the best pair is unknown, stop. Rebuild the symbol tuple
   left-to-right, merging non-overlapping occurrences.
3. Training ties: `min` over `(-count, pair)` gives "most frequent, then
   lexicographically smallest" - that exact rule is what makes
   `test_training_is_deterministic` pass.
