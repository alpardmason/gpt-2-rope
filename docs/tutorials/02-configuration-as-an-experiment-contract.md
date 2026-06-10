# 02: Configuration as an Experiment Contract

## Objectives and Prerequisites

Turn experiment settings into validated, immutable, serializable contracts.
Prerequisite: tutorials 00-01 and basic Pydantic.

**Source map:** [`config.py`](../../src/gpt2_rope/config.py) symbols
`StrictConfig`, `ModelConfig`, `TrainingConfig`, `ExperimentConfig`;
[`config_io.py`](../../src/gpt2_rope/config_io.py)
`load_experiment_config`; [`tiny.yaml`](../../configs/tiny.yaml); and
[`test_config.py`](../../tests/test_config.py).

## Contracts and Invariants

`extra="forbid"` turns misspellings into errors. `frozen=True` prevents a run
from changing its own declared configuration.

```python
class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
```

Model geometry is validated before tensor allocation:

```text
d_model % num_heads == 0
num_heads % num_kv_heads == 0
head_dim % 2 == 0
```

The last rule belongs to RoPE's adjacent-pair rotation. Derived properties
(`head_dim`, `query_groups`, `mlp_hidden_size`) prevent duplicated arithmetic.
Training also enforces `min_learning_rate <= learning_rate`.

Configuration flow:

```text
YAML mapping -> dotted overrides -> JSON-like value parsing
-> ExperimentConfig.model_validate -> resolved_config.json
```

**Recommendation:** validate cross-field constraints next to the owning model.
**Rationale:** failures occur before allocating a model or launching workers.

| Alternative | Strength | Weakness |
|---|---|---|
| Pydantic frozen models | Validation + schema | Runtime dependency |
| Dataclasses | Lightweight | Manual parsing/validation |
| Raw dictionaries | Flexible | Typos and weak contracts |

## Override Semantics

`--set training.max_steps=20` walks nested mappings. Values are parsed with
JSON first, so numbers, booleans, lists, and `null` retain type. A bare string
falls back to text. Shell quoting matters:

```bash
--set training.device='"mps"'
```

The resolved model dump, not just the input YAML, is written into the run
directory. That captures defaults and overrides as provenance.

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Unknown setting rejected | Typo or stale config | Read validation path | Rename/remove | Forbid extras |
| Override becomes string | Invalid JSON/shell quoting | Print resolved config | Quote JSON value | CLI tests |
| RoPE fails later | Odd head dimension | Inspect geometry | Change width/heads | Cross-field validator |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from pathlib import Path
from gpt2_rope.config_io import load_experiment_config

c = load_experiment_config(
    Path("configs/tiny.yaml"),
    ["training.max_steps=7", "model.gradient_checkpointing=true"],
)
print(c.training.max_steps, c.model.gradient_checkpointing, c.model.head_dim)
try:
    c.training.max_steps = 9
except Exception as error:
    print(type(error).__name__)
PY
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_config.py -q
```

Expected: `7 True 32`, then a frozen-instance validation error.

## Exercises

1. Explain why `vocab_size <= 65_535` matches the data representation.
2. Add, on paper, a constraint that warmup cannot exceed max steps.
3. Why persist the resolved configuration in addition to the checkpoint?

## Solutions

1. Prepared token streams use `uint16`, whose maximum representable ID is
   65,535.
2. Add an after-validator to `TrainingConfig` rejecting
   `warmup_steps > max_steps`; test valid, equal, and invalid cases.
3. It is human-readable provenance and can be inspected without deserializing
   executable framework state.

## Modern LLM Systems Delta

Larger stacks use hierarchical config systems, schema migrations, experiment
registries, immutable artifact IDs, and launch-time cluster validation. Avoid
config magic: precedence and resolved values must remain inspectable.

## Professional Takeaways

Configuration is an API. Review its compatibility, validation, provenance, and
override semantics with the same rigor as Python functions.

## Further Exploration

- [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/)
- [Hydra documentation](https://hydra.cc/docs/intro/) for a more powerful,
  higher-complexity alternative
- [Sacred reproducible experiments](https://sacred.readthedocs.io/)

