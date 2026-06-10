# 03: Byte-Level BPE From Files to Training

## Objectives and Prerequisites

Understand reversible byte encoding, deterministic BPE learning, persistence,
identity, and supply-chain checks. Prerequisite: tokenizer theory.

**Source map:** [`tokenizer.py`](../../src/gpt2_rope/tokenizer.py)
`bytes_to_unicode`, `ByteBPETokenizer.bpe/encode/decode/train`;
[`assets.py`](../../src/gpt2_rope/assets.py) `download_gpt2_tokenizer`;
[`test_tokenizer.py`](../../tests/test_tokenizer.py); and tokenizer commands in
[`cli.py`](../../src/gpt2_rope/cli.py).

## Contracts and Invariants

- Every byte 0-255 has a reversible Unicode surrogate.
- The base vocabulary therefore needs 256 entries plus special tokens.
- Regex pre-tokenization follows GPT-2's pattern before byte conversion.
- Merge rank, not current pair frequency, drives inference-time BPE.
- Training tie-breaks lexicographically, making equal-frequency merges stable.
- `decode(encode(text)) == text` for valid text, modulo replacement behavior
  for arbitrary invalid UTF-8 token sequences.

```text
Unicode text -> UTF-8 bytes -> reversible visible symbols
-> regex pieces -> ranked BPE merges -> integer IDs
```

`bpe()` caches up to 100,000 token-piece results. Cache eviction clears the
whole dictionary: simple, bounded, and adequate here, but not an LRU.

## Training Versus Encoding

Training repeatedly counts adjacent pairs weighted by word frequency, selects
the best pair, merges every occurrence, and appends the resulting token.
Encoding applies the learned merge list by rank. Confusing these algorithms is
a common implementation bug.

**Recommendation:** retain all 256 byte symbols even if absent from training.
**Rationale:** vocabulary coverage must not depend on the sample corpus.

| Tokenizer | Coverage | Transparency | Throughput |
|---|---:|---:|---:|
| Native Python byte BPE | Complete | High | Low |
| Hugging Face `tokenizers` | Complete | Medium | High |
| Character tokenizer | Complete | High | Poor sequence efficiency |

## Persistence and Identity

`vocab.json` maps string token to ID. `merges.txt` records ordered pairs.
`identity()` hashes canonical JSON plus merges and records vocabulary size.
Downloaded GPT-2 assets have hard-coded SHA-256 values: HTTPS is transport;
the digest verifies content.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Unknown-token lookup | Missing byte symbol | Check base vocab size | Include all bytes | Unicode round-trip test |
| Nondeterministic vocab | Unstable tie-break | Compare merge lists | Deterministic ordering | Duplicate training test |
| Wrong model behavior | Tokenizer mismatch | Compare identity hash | Use matching assets | Persist identity |
| Corrupt download | Asset changed/truncated | SHA-256 mismatch | Reject artifact | Pinned checksums |

## Lab

```bash
tmp="$(mktemp -d)"
printf 'low lower lowest\nnaive café\n' > "$tmp/corpus.txt"
UV_CACHE_DIR=.uv-cache uv run gpt2-rope tokenizer train \
  "$tmp/corpus.txt" "$tmp/tok" --vocab-size 280
UV_CACHE_DIR=.uv-cache uv run gpt2-rope tokenizer inspect \
  "$tmp/tok" --text "café"
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_tokenizer.py -q
```

Expected: the inspection round-trips `café`; repeated training yields identical
vocabulary and merges. Debug prompt: inspect why `é` becomes multiple UTF-8
bytes but still decodes exactly.

## Exercises

1. Why is `encoder` insufficient to reproduce tokenization?
2. What does `allowed_special=False` change?
3. Estimate the naive trainer's scaling bottleneck.

## Solutions

1. Encoding needs ordered merge ranks; the same final token strings do not
   uniquely encode merge history.
2. Special strings are processed as ordinary text rather than injected as one
   special ID.
3. Every merge iteration recounts adjacent pairs and rewrites the corpus
   representation; this Python implementation is educational, not web-scale.

## Modern LLM Systems Delta

Modern models often use SentencePiece BPE/unigram or optimized byte-level
implementations, larger vocabularies, richer special-token policy, parallel
training, and tokenizer version registries. Tokenizer changes still invalidate
model/data compatibility.

## Professional Takeaways

Tokenizer correctness is model correctness. Test arbitrary Unicode, special
tokens, deterministic training, persistence, and artifact identity.

## Further Exploration

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [SentencePiece](https://arxiv.org/abs/1808.06226)

