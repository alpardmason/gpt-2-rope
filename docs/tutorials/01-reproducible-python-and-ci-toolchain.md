# 01: Reproducible Python and CI Toolchain

## Objectives and Prerequisites

Understand how Python, dependencies, static analysis, tests, and CI form one
reproducibility boundary. Prerequisite: tutorial 00.

**Practice companion:** [01-practice.md](practice/01-practice.md).

**Source map:** [pyproject.toml](../../pyproject.toml),
[uv.lock](../../uv.lock), [.python-version](../../.python-version),
[CI](../../.github/workflows/ci.yml), and [.gitignore](../../.gitignore).

## Contracts and Invariants

- Python is constrained to `>=3.12,<3.13`; local selection is `3.12`.
- `pyproject.toml` is the editable intent; `uv.lock` is the resolved graph.
- `uv sync --frozen` must not silently change the lock.
- Ruff, strict mypy, and pytest are independent definitions of quality.
- Generated environments, caches, runs, secrets, and model weights are ignored.

```toml
[tool.mypy]
python_version = "3.12"
strict = true
packages = ["gpt2_rope"]
```

Strict typing forces library boundaries to state optionality and container
shape. It does not verify tensor dimensions; runtime tests must do that.

**Recommendation:** use the locked environment in local work and CI.
**Rationale:** “works on my machine” often means “resolved a different graph.”

| Choice | Reproducibility | Iteration speed | Risk |
|---|---:|---:|---|
| `uv sync --frozen` | High | High | Lock must be maintained |
| Unlocked resolver | Medium | High | Transitive drift |
| Global Python env | Low | Initially high | Hidden contamination |

## CI Architecture

Three jobs intentionally duplicate environment setup:

```text
checkout -> setup Python/uv -> frozen sync -> one quality gate
```

This costs minutes but gives failure isolation and permits parallel execution.
At larger scale, cache the uv download cache while keeping the lock authoritative.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| uv cache permission error | Sandbox cannot write home cache | Read path in error | Set `UV_CACHE_DIR=.uv-cache` | Standardize command |
| CI passes typing, runtime fails | Types cannot express tensor values | Run tests | Add invariant tests | Layer gates |
| Fresh clone resolves new versions | Lock ignored | Check command/log | Use `--frozen` | CI policy |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run mypy
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run python -c \
  "import sys, torch; print(sys.version.split()[0], torch.__version__)"
```

Expected: all gates pass and Python reports 3.12.x. Accelerator tests may skip.
Debug prompt: why is a skip acceptable for local CPU CI but not evidence that a
CUDA release is safe?

## Exercises

1. Classify Ruff, mypy, and pytest failures by the property they protect.
2. Why should `.env` and `*.safetensors` be ignored?
3. What change requires updating both `pyproject.toml` and `uv.lock`?

## Solutions

1. Ruff: syntactic/style bug patterns; mypy: static interface consistency;
   pytest: executed behavioral claims.
2. `.env` can contain secrets; weights are large generated/supply-chain
   artifacts and require explicit distribution/versioning.
3. Adding, removing, or changing a dependency constraint.

## Modern LLM Systems Delta

Production stacks also pin container images, CUDA/toolkit/driver compatibility,
kernel builds, dataset versions, and cluster topology. A Python lock alone
cannot reproduce an accelerator training run.

## Professional Takeaways

Treat the environment as code. In an interview, distinguish source
reproducibility, dependency reproducibility, numerical reproducibility, and
hardware reproducibility.

## Further Exploration

- [uv projects](https://docs.astral.sh/uv/guides/projects/)
- [Python packaging specification](https://packaging.python.org/en/latest/specifications/)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)

