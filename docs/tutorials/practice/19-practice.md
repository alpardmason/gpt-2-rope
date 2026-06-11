# Practice 19: Corpus Engineering: Dedup, Filtering, Sharding

Companion to [19-corpus-engineering-dedup-filtering-sharding.md](../19-corpus-engineering-dedup-filtering-sharding.md).
Persist all deliverables to `notes/chapters/19.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `data dedup` CLI to the signature comparison

Follow one corpus file from the command line into the MinHash math. Start at
`data_dedup` in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`data_dedup` -> `read_documents` in
[`data.py`](../../../src/gpt2_rope/data.py) -> `deduplicate_documents` in
[`data_quality.py`](../../../src/gpt2_rope/data_quality.py) ->
`normalized_content_hash` -> `minhash_signature` -> `estimated_jaccard`.

Record at each hop:

- In what order do the exact and near checks run for one document, and why
  does an exact duplicate never reach the MinHash comparison?
- What exactly is one entry of `kept_signatures` (type, length for the CLI
  default `num_hashes=64`), and who owns the list across loop iterations?
- Where do `report.exact_duplicates` and `report.near_duplicates` live
  (`DedupReport` dataclass), and at what point is `report.kept` finally set?
- The near-duplicate scan compares against every kept signature. State the
  asymptotic cost per document and where the tutorial says production
  systems break this with LSH.

### Trace B: from `data shard` to the content-addressed manifest

Trace `data_shard` in [`cli.py`](../../../src/gpt2_rope/cli.py) ->
`write_shards` in
[`data_quality.py`](../../../src/gpt2_rope/data_quality.py) line by line.
Record:

- How `payload` is constructed for one shard (separator, trailing newline)
  and why the SHA-256 is computed over `payload` bytes rather than over the
  document list.
- The exact keys of one entry in `manifest["shards"]` and of the top-level
  manifest written to `shards.json`.
- Which two `ValueError` guards run before any file is written.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_filter_documents_accounts_for_every_rejection` in
   [`test_data_quality.py`](../../../tests/test_data_quality.py) to make:
   the kept list and the exact `rejected` dictionary (which reason names,
   which counts). Then read it and diff against your guess.
2. **Lab output prediction.** Before running the chapter lab, predict the
   filter report (`kept`, every rejection reason and count for the six
   documents), the dedup report (`exact_duplicates`, `near_duplicates`,
   `kept`), and how many shards `shards.json` lists at
   `--documents-per-shard 2`.
3. **Mutation prediction.** If `normalized_content_hash` skipped
   normalization (`canonical = text` instead of the whitespace collapse),
   predict: which assertions in `test_content_hash_is_whitespace_insensitive`
   and `test_exact_and_near_deduplication` fail, and where the
   `base + "  "` document gets caught instead. Verify by temporarily editing
   `data_quality.py`, running
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_data_quality.py`, and
   reverting (`git checkout -- src/gpt2_rope/data_quality.py`).
4. **Boundary prediction.** Predict the exact exception type and message of
   `estimated_jaccard((1, 2), (1,))` and of
   `write_shards([], Path("scratch"))`. Verify both in a REPL
   (`UV_CACHE_DIR=.uv-cache uv run python`).

## 3. Tool walkthrough: `jq` and `shasum` over pipeline reports

- **Why this tool.** Every stage of this pipeline emits JSON precisely so
  that drops are auditable without rereading code. Professionals interrogate
  data-pipeline reports with `jq` the way they interrogate logs with `grep`,
  and a manifest hash you have never verified by hand is a hash you do not
  actually trust.
- **How.** Run the chapter lab's setup, then:

```bash
UV_CACHE_DIR=.uv-cache uv run gpt2-rope data filter corpus_raw.txt \
  corpus_filtered.txt --max-repetition 0.5 --max-non-alpha 0.5 \
  | jq '.rejected'
UV_CACHE_DIR=.uv-cache uv run gpt2-rope data dedup corpus_filtered.txt \
  corpus_clean.txt --near-threshold 0.5 | jq .
UV_CACHE_DIR=.uv-cache uv run gpt2-rope data shard corpus_clean.txt shards/ \
  --documents-per-shard 2
jq '.shards[] | {path, documents, sha256}' shards/shards.json
shasum -a 256 shards/shard-00000.txt
```

- **Play.**
  1. Compare the `shasum -a 256` digest of `shard-00000.txt` against the
     `sha256` field in `shards.json` and confirm they match. Append one
     character to the shard file, re-hash, and record the mismatch - this is
     the "same data" dispute from the failure table, resolved in one line.
  2. Re-run `data dedup` with `--near-threshold 0.95` and then `0.3`, piping
     each report through `jq '.near_duplicates'`. Explain the direction of
     change from the Jaccard-estimate contract.
  3. Break a flag on purpose (`--near-threshold 1.5`) and record the Typer
     range diagnostic; compare it with what a silent clamp would have cost
     you. Clean up with
     `rm -rf corpus_raw.txt corpus_filtered.txt corpus_clean.txt shards/`.

## 4. Deliverables

Append to `notes/chapters/19.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the `jq` query you will reuse, plus the hand-verified shard hash
  from the play exercise.
- 3-5 why-cards. Seed examples: "Why must deduplication run before the
  train/validation split?", "Why is the exact-dedup hash computed over
  whitespace-normalized text?", "What breaks if a filter drops documents
  without incrementing a named counter?"
- Feynman summary: explain to a colleague why `kept + rejected` reconciling
  with the input count is the load-bearing invariant of corpus engineering,
  and how MinHash turns set similarity into signature equality.
