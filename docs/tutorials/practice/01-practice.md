# Practice 01: Reproducible Python and CI Toolchain

Companion to [01-reproducible-python-and-ci-toolchain.md](../01-reproducible-python-and-ci-toolchain.md).
Persist all deliverables to `notes/chapters/01.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from a CI job to a quality gate

Follow the `test` job in [`ci.yml`](../../../.github/workflows/ci.yml) step
by step: checkout -> `astral-sh/setup-uv` with Python 3.12 ->
`uv sync --frozen` -> `uv run pytest`. Then trace where each step's behavior
is actually defined:

- Which file pins the interpreter range (`requires-python` in
  [`pyproject.toml`](../../../pyproject.toml)) versus the local selection
  ([`.python-version`](../../../.python-version))? Record both values.
- Where does `pytest` get its `-q` flag and its `tests/` search path?
  Find `[tool.pytest.ini_options]` and record `addopts`, `testpaths`, and
  the three custom markers (`cuda`, `mps`, `distributed`).
- Compare the three jobs (`lint`, `type-check`, `test`): which steps are
  byte-identical, and which single line differs? State in one sentence why
  the duplication is intentional.

### Trace B: from the executable name to `main`

Trace how `uv run gpt2-rope` resolves: `[project.scripts]`
`gpt2-rope = "gpt2_rope.cli:main"` in
[`pyproject.toml`](../../../pyproject.toml) -> `main` in
[`cli.py`](../../../src/gpt2_rope/cli.py) -> `logging.basicConfig` ->
`app()`.

Record:

- Which build backend produces the entry point (`hatchling`), and which
  table tells it what to package (`[tool.hatch.build.targets.wheel]`).
- What `main` does besides calling `app()`, and why logging setup belongs
  at the process boundary rather than in library modules.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Lab output prediction.** Before running the chapter lab's final
   command, predict the Python version prefix and the major version of
   `torch` it will print, citing the files that constrain each. Then run it.
2. **Mutation prediction.** Add an unused `import os` to the top of
   [`config_io.py`](../../../src/gpt2_rope/config_io.py). Predict the exact
   Ruff rule code and one-line message that
   `UV_CACHE_DIR=.uv-cache uv run ruff check .` will emit, and predict
   whether `mypy` and `pytest` would also object. Verify, then revert
   (`git checkout -- src/gpt2_rope/config_io.py`).
3. **Mutation prediction.** Remove the `-> None` return annotation from
   `main` in [`cli.py`](../../../src/gpt2_rope/cli.py). Predict the mypy
   strict-mode error code and whether Ruff also flags it. Verify with
   `UV_CACHE_DIR=.uv-cache uv run mypy`, then revert
   (`git checkout -- src/gpt2_rope/cli.py`).
4. **Boundary prediction.** Add a new dependency line such as
   `"requests>=2",` to the `dependencies` list in
   [`pyproject.toml`](../../../pyproject.toml). Predict what
   `UV_CACHE_DIR=.uv-cache uv sync --frozen` does: install it, resolve a new
   graph, or refuse - and predict the wording of the diagnostic. Verify,
   then revert (`git checkout -- pyproject.toml`).

## 3. Tool walkthrough: `uv` lock discipline plus the Ruff/mypy gates

- **Why this tool.** "Works on my machine" usually means "resolved a
  different dependency graph." `uv` makes intent (`pyproject.toml`) and
  resolution (`uv.lock`) separate, inspectable artifacts, and the
  lint/type/test gates each protect a different property. Professionals
  read CI failures by first asking which gate fired.
- **How.**

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen
UV_CACHE_DIR=.uv-cache uv lock --check
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run ruff rule F401
UV_CACHE_DIR=.uv-cache uv run mypy
UV_CACHE_DIR=.uv-cache uv run pytest
```

- **Play.**
  1. Run `uv lock --check` on the clean tree, then repeat it with the
     prediction-task 4 edit in place. Record both outputs and state which
     CI job would have caught the drift.
  2. Time the three gates separately (`time uv run ruff check .`, and so
     on). Record the ordering and explain why CI still runs them in
     parallel jobs rather than fastest-first in one job.
  3. Read `uv run ruff rule F401` and one rule from each other selected
     family in `[tool.ruff.lint]` (`E`, `I`, `UP`, `B`, `SIM`, `RUF`).
     Record one rule whose motivation surprised you.

## 4. Deliverables

Append to `notes/chapters/01.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the `uv lock --check` behavior you observed on clean and
  drifted trees, plus the gate timings.
- 3-5 why-cards. Seed examples: "Why is `uv sync --frozen` required in CI
  but a plain `uv sync` acceptable while editing dependencies?", "What
  breaks if `.env` or `*.safetensors` stop being ignored?", "Why can strict
  mypy pass while a tensor-shape bug ships?"
- Feynman summary: explain to a colleague the difference between source,
  dependency, numerical, and hardware reproducibility, and which of the
  four this chapter's toolchain actually guarantees.
