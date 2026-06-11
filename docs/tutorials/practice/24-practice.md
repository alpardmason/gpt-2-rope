# Practice 24: Inference Serving and Deployment

Companion to [24-inference-serving-and-deployment.md](../24-inference-serving-and-deployment.md).
Persist all deliverables to `notes/chapters/24.md` using
[notes/templates/chapter-notes.md](../../../notes/templates/chapter-notes.md).
Run everything from the repository root with `UV_CACHE_DIR=.uv-cache`. The
serving dependencies are optional; install them first with
`UV_CACHE_DIR=.uv-cache uv sync --extra serving`.

## 1. Tracing tasks

### Trace A: from the `serve` CLI to one batched forward

Follow one HTTP request from the command line into the model. Start at
`serve` in [`cli.py`](../../../src/gpt2_rope/cli.py) and trace:

`serve` (note the guarded import that demands the serving extra) ->
`InferenceService.__init__` in
[`serving.py`](../../../src/gpt2_rope/serving.py) -> `create_app` ->
`submit` -> `validate` -> `_serve_forever` -> `group_compatible` ->
`_run_batch` -> `generate` in
[`generation.py`](../../../src/gpt2_rope/generation.py).

Record at each hop:

- The three 422 conditions `validate` enforces and why all of them run
  before the request ever touches the queue.
- The seven components of `_Pending.batch_key()` and which two requests
  from the tutorial's exercise set can never share a batch.
- Who owns the model: confirm there is exactly one worker task, that the
  forward runs via `asyncio.to_thread`, and how per-request futures get
  their results (or the group's exception).
- The three `serve/*` metric names `_run_batch` logs and the file they
  stream to.

### Trace B: lifecycle through a test client

Trace `test_healthz_and_greedy_generation_is_deterministic` in
[`test_serving.py`](../../../tests/test_serving.py): the `TestClient`
context manager drives `create_app`'s lifespan. Record:

- Where `service.start()` and `service.stop()` are called, and what
  `stop()` does to the worker task and the metrics logger.
- What `/healthz` reports without running the model at all.

## 2. Prediction tasks

Write each prediction in your notes BEFORE running. Then run and record the
outcome.

1. **Test prediction.** Without opening it, write the grouping outcome you
   expect `test_group_compatible_batches_same_shape_and_settings` in
   [`test_serving.py`](../../../tests/test_serving.py) to assert for four
   pending requests: two identical, one with a longer prompt, one with a
   different temperature. Then read it and diff against your guess.
2. **Rejection prediction.** Before opening
   `test_generate_rejects_invalid_requests`, predict which layer rejects
   each of its three payloads: Pydantic field validation on
   `GenerateRequest` or the token-budget logic in `validate`. Then read the
   test and check each against the source.
3. **Mutation prediction.** If `validate` dropped the
   `len(token_ids) > budget` check, predict the failure symptom of the
   over-long prompt case: a 422, a 200 with garbage, or a worker exception
   surfacing through the future (where would the position bound actually
   trip?). Verify by temporarily editing `serving.py`, running
   `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_serving.py`, and
   reverting (`git checkout -- src/gpt2_rope/serving.py`).
4. **Boundary prediction.** `GenerateRequest` sets `extra="forbid"` and
   `max_new_tokens >= 1`. Predict the exact exception type for
   `GenerateRequest(prompt="a", max_new_tokens=0)` and for
   `GenerateRequest(prompt="a", unknown_field=1)`. Verify in a REPL.

## 3. Tool walkthrough: `curl` plus a concurrent load probe

- **Why this tool.** A serving system is judged at its interface, not in
  its source. `curl` against the live endpoint and a crude parallel probe
  are the first instruments every infrastructure engineer reaches for, and
  they are enough to observe micro-batching working (or failing to merge).
- **How.** With a trained checkpoint from the chapters 04 and 11 labs:

```bash
UV_CACHE_DIR=.uv-cache uv run gpt2-rope serve configs/tiny.yaml \
  runs/tiny/checkpoints/step-00010000 --port 8000 \
  --batch-window-ms 50 --metrics-dir runs/serving &
sleep 3
curl -s localhost:8000/healthz | jq .
curl -s -X POST localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt": "The pass key is", "max_new_tokens": 8, "temperature": 0.0}' | jq .
seq 8 | xargs -P 8 -I{} curl -s -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "The pass key is", "max_new_tokens": 8, "temperature": 0.0}' \
  > /dev/null
kill %1
jq '."serve/batch_size"' runs/serving/metrics.jsonl
```

- **Play.**
  1. Run the 8-way probe and record the largest `serve/batch_size` logged.
     Identical prompts and settings share a batch key, so values above 1
     prove the window merged requests.
  2. Vary the prompts (different token lengths) across the 8 requests and
     confirm batch sizes collapse to 1 - the same-shape grouping limit the
     tutorial's exercise 3 quantifies.
  3. Break a request on purpose: send `"max_new_tokens": 0` and an unknown
     JSON field, and record both 422 bodies. Then compare
     `serve/latency_ms` for a singleton batch against a merged batch and
     note what the shared forward pass does to per-request latency.

## 4. Deliverables

Append to `notes/chapters/24.md`:

- Tracing log for Traces A and B with all checkpoint answers.
- Prediction record rows for all four predictions (prediction written
  before, outcome after, one-line post-mortem each).
- Tool log: the `curl` and `xargs -P` commands you will reuse, plus the
  observed `serve/batch_size` distribution from the play exercise.
- 3-5 why-cards. Seed examples: "Why must validation run before the
  queue?", "Why can only same-length prompts share a batch in this
  server?", "What breaks if the forward pass runs on the event-loop
  thread?"
- Feynman summary: explain to a colleague why greedy requests are
  reproducible under batching while sampled requests depend on their batch
  neighbors, and what a per-request fix would require.
