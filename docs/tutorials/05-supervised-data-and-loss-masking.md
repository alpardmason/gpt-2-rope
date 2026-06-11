# 05: Supervised Data and Loss Masking

## Objectives and Prerequisites

Understand SFT example construction as an objective-design problem, not merely
JSON parsing. Prerequisite: tutorials 03-04 and causal cross entropy.

**Practice companion:** [05-practice.md](practice/05-practice.md).

**Source map:** [`data.py`](../../src/gpt2_rope/data.py) `build_sft_example`,
`SFTDataset`, `collate_sft`; [`model.py`](../../src/gpt2_rope/model.py)
`GPT.forward`; [`test_data.py`](../../tests/test_data.py); and
[`finetune_lora.yaml`](../../configs/finetune_lora.yaml).

## Contracts and Invariants

For prompt IDs `P` and response IDs `R`:

```text
input_ids = P + R + EOS
labels    = -100 * len(P) + R + EOS
```

`GPT.forward` shifts logits and labels by one. Therefore a response token at
position `i` is predicted by the logit at `i-1`; prompt and padding labels are
ignored by cross entropy.

Truncation policy preserves supervision:

1. Build response plus EOS.
2. If response alone is too long, truncate it to `max_length`, force final EOS,
   and drop the prompt.
3. Otherwise retain the response and the rightmost prompt suffix that fits.

This policy favors answer learning over full prompt context. It is explicit,
testable, and potentially unsuitable for tasks where early instructions matter.

Dynamic collation pads inputs with EOS and labels with `-100`. EOS as padding is
safe for loss because labels are masked, but this model has no attention
padding mask. Batched shorter examples can attend to right-padding positions
inside their own forward pass; because padding is on the right, supervised
tokens precede it and are unaffected.

**Recommendation:** count supervised tokens, not padded tokens, for SFT
throughput. **Rationale:** only labels other than `-100` produce training signal.

| Truncation choice | Preserves | Main risk |
|---|---|---|
| Response-first (current) | Targets | Loses instruction prefix |
| Prompt-first | Full instruction | Truncates answer signal |
| Drop long examples | Semantics | Distribution shift/waste |

## Failure Analysis

| Symptom | Root cause | Diagnosis | Fix | Prevention |
|---|---|---|---|---|
| Loss trains on prompt | Prompt labels not masked | Inspect labels | Use `-100` | Mask test |
| No learning signal | Entire label row masked | Count labels | Require response | Validation |
| EOS disappears | Blind truncation | Inspect last ID | Force EOS | Boundary test |
| Throughput inflated | Counted padding/prompt | Compare mask count | Count supervised IDs | Metric contract |

## Lab

```bash
UV_CACHE_DIR=.uv-cache uv run python - <<'PY'
from gpt2_rope.data import build_sft_example
from gpt2_rope.tokenizer import ByteBPETokenizer

tok = ByteBPETokenizer.train(
    ["question answer"], 280, special_tokens=["<|endoftext|>"]
)
x, y = build_sft_example("question: ", "answer", tok, max_length=16)
print(x)
print(y)
print("supervised", sum(v != -100 for v in y), "length", len(y))
PY
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_data.py -q
```

Expected: equal lengths, a masked prompt region, at least one supervised token,
and EOS as the final ID.

## Exercises

1. Explain the apparent “double shift” between dataset targets and model loss.
2. Why is empty response rejected?
3. What changes are needed for left padding during batched generation?

## Solutions

1. SFT returns labels aligned to input positions; the model performs the only
   actual shift. `MemmapTokenDataset.y` is not used by current training.
2. It would produce no task-specific supervised target beyond policy-dependent
   EOS and may create an all-masked example.
3. Add an attention mask and position handling that excludes pads; generation
   must select each sequence's last real-token logits.

## Modern LLM Systems Delta

Modern chat SFT uses model-specific templates, role tokens, multi-turn masking,
sequence packing, example weights, contamination checks, and assistant-only
loss policies. Template/version identity belongs in checkpoint provenance.

## Professional Takeaways

Loss masking is product behavior. Always visualize a tokenized example with
token text, IDs, roles, labels, and truncation before launching training.

## Further Exploration

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [PyTorch cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
- [Hugging Face chat templates](https://huggingface.co/docs/transformers/chat_templating)

