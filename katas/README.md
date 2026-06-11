# Reimplementation Katas

A kata guts one production module on a dedicated branch and asks you to
reimplement it against the unchanged production test suite, type checker, and
linter. The contract is fixed; only the implementation is yours. This is the
Tier 2 practice loop described in
[docs/tutorials/README.md](../docs/tutorials/README.md).

## Katas

| Kata | Gutted code | Tutorial | Primary oracle |
|---|---|---|---|
| `bpe` | `bytes_to_unicode`, `ByteBPETokenizer.bpe/encode/decode/train` | [03](../docs/tutorials/03-byte-level-bpe-from-files-to-training.md) | `tests/test_tokenizer.py`, `tests/test_cli.py` |
| `rope` | all of [`rope.py`](../src/gpt2_rope/rope.py) | [07](../docs/tutorials/07-rotary-position-embeddings-in-pytorch.md) | `tests/test_model.py` |
| `gqa` | `GroupedQueryAttention.forward` | [08](../docs/tutorials/08-grouped-query-attention-and-sdpa.md) | `tests/test_model.py` |
| `kv-cache` | all of [`generation.py`](../src/gpt2_rope/generation.py) | [09](../docs/tutorials/09-kv-cache-and-autoregressive-generation.md) | `tests/test_lora_generation.py` |
| `checkpoint` | save/load/RNG/export in [`checkpoint.py`](../src/gpt2_rope/checkpoint.py) | [13](../docs/tutorials/13-exact-resume-checkpoint-engineering.md) | `tests/test_checkpoint.py`, `tests/test_training.py` |
| `dpo-loss` | `sequence_logprobs`, `dpo_loss` | [22](../docs/tutorials/22-preference-optimization-with-dpo.md) | `tests/test_dpo.py` |

## Workflow

```bash
# 1. Start (requires a clean worktree; creates branch kata/<name>)
UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py start rope

# 2. Reimplement until the oracle is green
UV_CACHE_DIR=.uv-cache uv run python scripts/make_kata.py check rope

# 3. Full gates before declaring victory
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run mypy
UV_CACHE_DIR=.uv-cache uv run ruff check .

# 4. The review step: compare your solution to the original
git diff main -- src/gpt2_rope/<module>.py

# 5. Discard or archive the branch; main is never modified
git checkout main
```

## Rules

1. No peeking at `main`'s implementation of the gutted code (no `git diff`,
   `git show`, editor history, or external copies) until the oracle is green.
   Reading the tests, the tutorial chapter, and the papers is encouraged --
   that is the contract, not the answer.
2. Do not edit the tests, the config models, or any module other than the
   gutted one. The kata is to satisfy the existing contract, not renegotiate
   it.
3. Type annotations and docstrings in the skeleton are part of the contract;
   keep the signatures.
4. When green, always do the `git diff main` review and record in your
   chapter notes: one place the original is better than yours, and one
   decision you made differently and can defend.
5. If stuck longer than the README's effort estimate, open the hint ladder
   in the kata README one rung at a time. Hints cost nothing; peeking costs
   the kata.

## Mechanics

`scripts/make_kata.py start <name>` copies `katas/<name>/skeleton_*.py` over
the target module on a new `kata/<name>` branch and commits, so the diff
against `main` is exactly the code you must rebuild. Skeletons are verbatim
copies of the current module with kata-target bodies replaced by
`raise NotImplementedError` and `# KATA:` contract comments; if a module
changes on `main`, regenerate the skeleton's untouched parts to match.
Skeletons are ruff-clean but intentionally drop imports only the gutted
bodies used -- re-adding them is part of the exercise. `katas/` is outside
the mypy package scope; mypy applies once the skeleton lands in
`src/gpt2_rope/` on your kata branch.
