# Practice 00: Course Map and Engineering Workflow

Companion to [00-course-map-and-engineering-workflow.md](../00-course-map-and-engineering-workflow.md).
Persist all deliverables to `notes/chapters/00.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`.

## 1. Tracing tasks

### Trace A: from `pretrain` CLI to the loss

Follow the chapter's vertical-slice reading order down one workflow. Start at
`pretrain` in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`pretrain` -> `load_experiment_config` in
[`config_io.py`](../../../src/gpt2_rope/config_io.py) ->
`train_pretraining` in [`training.py`](../../../src/gpt2_rope/training.py) ->
`MemmapTokenDataset` in [`data.py`](../../../src/gpt2_rope/data.py) ->
`GPT.forward` in [`model.py`](../../../src/gpt2_rope/model.py).

Record at each hop:

- Where does validation happen relative to expensive work? Name the first
  line of `pretrain` and what has already been proven by the time
  `train_pretraining` starts.
- What does `train_pretraining` return, and which side effects (files,
  directories) does the signature not mention?
- Who computes the loss, and where do `labels` enter? Note that the CLI
  contains zero model mathematics - record which helper functions it does
  own (`_tokenizer`, `_load_model`).

### Trace B: the Typer application tree

Trace how the `gpt2-rope` executable maps to functions. Start at
`[project.scripts]` in [`pyproject.toml`](../../../pyproject.toml), then
follow `main` -> `app` in [`cli.py`](../../../src/gpt2_rope/cli.py) and the
five `app.add_typer` calls (`tokenizer`, `data`, `checkpoint`, `eval`,
`sweep`).

Record:

- Which commands live on the root app (`pretrain`, `finetune`, `dpo`,
  `evaluate`, `generate`, `profile`, `serve`) versus on a sub-app, and what
  the grouping rule appears to be.
- For one sub-app command (`tokenizer train`), which library function does
  all the real work, and what does the command add on top (argument parsing,
  JSON echo)?

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the assertions you expect
   `test_cli_help_exposes_all_workflows` in
   [`test_cli.py`](../../../tests/test_cli.py) to make: how it invokes the
   app and which command names it checks. Then read it and diff against your
   guess.
2. **Lab output prediction.** Before running the chapter lab's
   `uv run gpt2-rope --help`, list every top-level command and command group
   you expect from the tutorial's data-flow diagram alone. Then run it and
   count your hits and misses.
3. **Collection prediction.** Predict which subsystems have a dedicated
   `tests/test_<name>.py` file, then verify with
   `UV_CACHE_DIR=.uv-cache uv run pytest --collect-only -q`. Record any
   subsystem you expected to be untested but is not, and vice versa.
4. **Mutation prediction.** If the decorator `@app.command("generate")` in
   [`cli.py`](../../../src/gpt2_rope/cli.py) were renamed to
   `@app.command("gen")`, predict exactly which assertion of
   `test_cli_help_exposes_all_workflows` fails and what the failure output
   contains. Verify by temporarily editing `cli.py`, running
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_cli.py`, and reverting
   (`git checkout -- src/gpt2_rope/cli.py`).

## 3. Tool walkthrough: `rg` and `git log`/`git show` for codebase archaeology

- **Why this tool.** Engineers spend most of their time reading code they
  did not write. `rg` answers "where is this symbol?" in milliseconds across
  any repository size, and `git log`/`git show` answer "why is it like
  this?" - the question no amount of reading the current state resolves.
- **How.**

```bash
rg -n "^(class|def) " src/gpt2_rope tests
rg -n "add_typer" src/gpt2_rope/cli.py
rg -n "train_pretraining" src tests
git log --oneline -- src/gpt2_rope/model.py
git show --stat HEAD
UV_CACHE_DIR=.uv-cache uv run gpt2-rope --help
```

- **Play.**
  1. List every CLI command without opening the file:
     `rg -n '@(app|\w+_app)\.command' src/gpt2_rope/cli.py`. Compare the
     list against the `--help` output and explain any name that differs
     from its Python function name (for example `generate_text` versus
     `generate`).
  2. Pick one module (`rope.py` or `data.py`), find the commit that
     introduced it with `git log --diff-filter=A --oneline -- <path>`, and
     read that commit with `git show`. Record one design decision visible
     in the original commit.
  3. Break a flag: run `rg "def (" src` and read the regex parse error,
     then fix it with `rg -F "def (" src` or escaping. Record how the
     diagnostic told you the pattern, not the path, was the problem.

## 4. Deliverables

Append to `notes/chapters/00.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the one `rg` invocation you will reuse, plus the commit hash
  and decision you found in the archaeology exercise.
- 3-5 why-cards. Seed examples: "Why must the CLI contain no model
  mathematics?", "Why are lint, typing, and tests separate CI gates instead
  of one job?", "What breaks if a subsystem is reachable only through the
  CLI and has no direct test?"
- Feynman summary: explain to a colleague how to read an unfamiliar
  repository as a set of contracts, using one vertical slice from this
  codebase as the worked example.
