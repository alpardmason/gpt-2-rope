# 04: Pretraining Data Pipelines and Memmap

## Objectives and Prerequisites

Build a deterministic path from documents to bounded-memory training windows.
Prerequisite: tutorial 03 and NumPy array basics.

**Practice companion:** [04-practice.md](practice/04-practice.md).

**Source map:** [`data.py`](../../src/gpt2_rope/data.py) `_read_documents`,
`prepare_corpus`, `MemmapTokenDataset`; [`test_data.py`](../../tests/test_data.py);
and `data prepare` in [`cli.py`](../../src/gpt2_rope/cli.py).

## Contracts and Invariants

Input is plain UTF-8 lines or JSONL objects containing `text` or
`prompt`/`response`. Preparation:

```text
documents -> deterministic document split -> encode each document
-> append EOS -> uint16 train.bin/validation.bin + manifest.json
```

The split is order-based, not randomized. Repeating the same ordered inputs and
tokenizer produces the same streams. This is deterministic but can be biased
if source order is correlated with topic or time.

The manifest records format version, dtype, token counts, tokenizer identity,
and source paths. It records provenance, but does not hash source contents.

## Window Geometry

For sequence length `T`, a sample reads `T + 1` tokens:

```text
window: [t0, t1, ..., tT]
x:      [t0, t1, ..., t(T-1)]
y:      [t1, t2, ..., tT]
```

The current training loop ignores `y` and passes `x` as labels because
`GPT.forward` shifts internally. The dataset's shifted target still documents
the conventional public contract.

`np.memmap` maps the file without loading it all into RAM. `__getitem__`
converts one window to `int64` because embedding indices require integer tensor
types supported by PyTorch. `.copy()` avoids exposing a read-only NumPy view.

**Recommendation:** make storage dtype and tokenizer range one validated
contract. **Rationale:** integer overflow silently changes token identity.

| Layout | RAM | I/O | Sampling flexibility |
|---|---:|---:|---:|
| Flat memmap | Low | Sequential-friendly | Fixed contiguous windows |
| In-memory list | High | Low after load | High |
| Sharded indexed format | Low | Scalable | High, more machinery |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Too few samples | Stream shorter than `T+1` | Check manifest/file | Lower T/add data | Constructor validation |
| Topic-skewed validation | Ordered split | Inspect source order | Shuffle documents upstream | Dataset audit |
| Token corruption | ID exceeds uint16 | Compare vocab range | Wider dtype | Config/data contract |
| Resume mismatch | Data changed | Compare manifests/artifacts | Restore exact data | Content hashes |

## Lab

```bash
tmp="$(mktemp -d)"
printf 'alpha beta\ngamma delta\nepsilon zeta\n' > "$tmp/corpus.txt"
UV_CACHE_DIR=.uv-cache uv run gpt2-rope tokenizer train \
  "$tmp/corpus.txt" "$tmp/tok" --vocab-size 280
UV_CACHE_DIR=.uv-cache uv run gpt2-rope data prepare \
  "$tmp/corpus.txt" "$tmp/data" "$tmp/tok" --validation-fraction 0.34
UV_CACHE_DIR=.uv-cache uv run python -c \
  "import json,sys; print(json.load(open(sys.argv[1])))" "$tmp/data/manifest.json"
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_data.py -q
```

Expected: two binary streams and a manifest with stable tokenizer identity.
Debug prompt: calculate file bytes from token count and `uint16`.

## Exercises

1. Why append EOS between documents?
2. Why is `__len__` floor division?
3. Name two provenance fields missing for strict dataset reproducibility.

## Solutions

1. It marks boundaries and prevents ordinary text concatenation from implying
   continuity.
2. Samples are non-overlapping windows and incomplete tails are discarded.
3. Source content hashes and preprocessing code/version; source ordering and
   normalization policy are also valuable.

## Modern LLM Systems Delta

Large pipelines stream, shard, shuffle with deterministic seeds, deduplicate,
filter quality/safety, track licenses, pack documents, and publish immutable
manifests. They often separate raw, normalized, tokenized, and sampled datasets.

## Professional Takeaways

Data layout determines throughput and semantics. Be able to derive exact bytes,
sample count, target alignment, shuffle behavior, and resume implications.

## Further Exploration

- [NumPy memory mapping](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html)
- [The Pile](https://arxiv.org/abs/2101.00027)

