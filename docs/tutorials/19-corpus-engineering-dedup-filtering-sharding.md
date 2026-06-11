# 19: Corpus Engineering: Dedup, Filtering, Sharding

## Objectives and Prerequisites

Treat training data as an engineered artifact: deduplicated, quality-gated,
content-addressed, and sharded, with every removal accounted for.
Prerequisite: 03-04.

**Practice companion:** [19-practice.md](practice/19-practice.md).

**Source map:** [`data_quality.py`](../../src/gpt2_rope/data_quality.py)
`deduplicate_documents`, `filter_documents`, `write_shards`,
`minhash_signature`; [`data.py`](../../src/gpt2_rope/data.py)
`prepare_corpus` manifest `source_sha256`;
[`test_data_quality.py`](../../tests/test_data_quality.py).

## Pipeline Contract

```text
raw files -> filter (reasoned rejections) -> dedup (exact, then near)
          -> shard (content-addressed)    -> prepare_corpus (memmap + manifest)
```

Invariants:

- Every dropped document increments a named counter; `kept + rejected`
  reconciles with input count. Silent data loss is a bug class.
- Exact dedup hashes whitespace-normalized content (SHA-256).
- Near dedup uses MinHash over lowercase word shingles; signature equality
  fraction estimates Jaccard similarity of shingle sets.
- Shards and sources carry SHA-256 digests; a training run can prove which
  bytes it consumed.

## Why Dedup Is Not Optional

Duplicated documents bias the loss toward memorization, corrupt train/
validation splits (leakage), and inflate effective epochs on popular content.
Near-duplicates (boilerplate, mirrored pages, templated text) are the common
case at web scale; exact hashing alone misses them.

**Recommendation:** exact dedup always; near dedup with a threshold around
0.8-0.9 Jaccard, tuned by inspecting borderline pairs. **Rationale:**
false-positive deletions are silent quality loss, so start conservative and
audit. **Alternatives:** suffix-array substring dedup (stronger, costlier);
embedding-based semantic dedup (catches paraphrases, needs a model).

| Method | Catches | Cost | Failure mode |
|---|---|---|---|
| Exact hash | byte-identical | O(n) | trivial edits escape |
| MinHash (here) | high-overlap shingles | O(n^2) pairwise at this scale | threshold tuning |
| MinHash + LSH | same, at scale | O(n) expected | banding parameters |
| Suffix array | repeated substrings | high memory | over-deletion |

The O(n^2) comparison against kept signatures is an explicit teaching
simplification; production buckets signatures with LSH so candidates are
near-constant per document.

## Quality Filters Are Hypotheses

`FilterThresholds` gates length, word-repetition ratio, and non-alphanumeric
ratio. Each is a falsifiable claim about what helps the model; FineWeb-style
pipelines ablate every filter against downstream evals before adopting it.
Filters silently encode language and domain assumptions -- a code corpus
fails `max_non_alpha_ratio` defaults tuned for prose.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Validation loss too good | near-dup leakage across split | cross-split MinHash audit | dedup before splitting | pipeline order test |
| Corpus shrinks 90% | filter threshold wrong for domain | rejection counters by reason | per-domain thresholds | report review gate |
| Dedup nondeterministic | unstable hash/seed | run twice, diff reports | salted deterministic hashes | determinism test |
| "Same data" disputes | no content addressing | nothing to compare | manifest `source_sha256` | hash verification in CI |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from pathlib import Path
base = "the quick brown fox jumps over the lazy dog near the river bank"
docs = [base, base + "  ", base.replace("dog", "cat"),
        "spam spam spam spam spam spam", "$$$ ### %%% @@@",
        "an entirely different document about rotary position embeddings"]
Path("corpus_raw.txt").write_text("\n".join(docs) + "\n")
PY
UV_CACHE_DIR=.uv-cache uv run gpt2-rope data filter corpus_raw.txt corpus_filtered.txt \
  --max-repetition 0.5 --max-non-alpha 0.5
UV_CACHE_DIR=.uv-cache uv run gpt2-rope data dedup corpus_filtered.txt corpus_clean.txt \
  --near-threshold 0.5
UV_CACHE_DIR=.uv-cache uv run gpt2-rope data shard corpus_clean.txt shards/ \
  --documents-per-shard 2
cat shards/shards.json
rm -rf corpus_raw.txt corpus_filtered.txt corpus_clean.txt shards/
```

Expected: the filter report rejects `repetitive` and `non_text` documents;
dedup removes one exact and one near duplicate; `shards.json` lists shards
with document counts and SHA-256 digests.

## Exercises

1. Why must deduplication run before the train/validation split rather than
   after?
2. With 64 hash functions, a true Jaccard of 0.5 yields what distribution of
   the estimated similarity, roughly?
3. The repo's MinHash compares each document against all kept signatures.
   At what corpus size does this become the bottleneck, and what is the
   standard fix?

## Solutions

1. Splitting first can place near-identical documents on both sides; the
   validation metric then rewards memorization and the leak is invisible.
2. Each signature slot matches with probability 0.5, so matches follow
   Binomial(64, 0.5): mean 0.5, standard deviation `sqrt(0.25/64) ~= 0.0625`.
3. Pairwise checks grow quadratically; around 10^5-10^6 documents it
   dominates. LSH banding hashes signature bands into buckets so only
   bucket-collision candidates are compared.

## Modern LLM Systems Delta

Web-scale pipelines (FineWeb, RefinedWeb, Dolma) add: HTML extraction,
language ID, model-based quality scoring, PII scrubbing, licensing and
robots compliance, URL-level dedup, and distributed execution on Spark/Ray.
Dataset versioning moves to content-addressed stores (the `source_sha256`
manifest here is the seed of that idea; DVC and lakehouse tables are the
industrial forms). Filters are validated by ablation training runs, not by
inspection.

## Professional Takeaways

Data work is the highest-leverage, least-glamorous part of LLM engineering.
The professional habits are all accounting: counters for every drop, hashes
for every artifact, reports reviewed like code, and filter changes treated as
experiments with eval evidence.

## Further Exploration

- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [The FineWeb Datasets](https://arxiv.org/abs/2406.17557)
- [Mining of Massive Datasets, ch. 3 (MinHash/LSH)](http://www.mmds.org/)
