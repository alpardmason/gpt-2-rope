from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from gpt2_rope.config import ModelConfig
from gpt2_rope.evaluation import (
    MultipleChoiceExample,
    build_passkey_samples,
    continuation_logprob,
    evaluate_multiple_choice,
    evaluate_passkey,
    evaluate_perplexity_files,
    load_multiple_choice_tasks,
)
from gpt2_rope.model import GPT
from gpt2_rope.tokenizer import ByteBPETokenizer

CPU = torch.device("cpu")


def tiny_tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(
        ["the quick brown fox jumps over the lazy dog"],
        vocab_size=280,
        special_tokens=["<|endoftext|>"],
    )


def tiny_model(context_length: int = 64) -> GPT:
    torch.manual_seed(0)
    return GPT(
        ModelConfig(
            vocab_size=300,
            context_length=context_length,
            d_model=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            dropout=0.0,
        )
    ).eval()


def test_continuation_logprob_is_a_finite_negative_sum() -> None:
    model = tiny_model()
    tokens = list(range(1, 9))
    score = continuation_logprob(model, tokens, prefix_length=4, device=CPU)
    assert math.isfinite(score) and score < 0
    longer = continuation_logprob(model, tokens, prefix_length=2, device=CPU)
    assert longer < score  # more scored tokens accumulate more negative mass

    with pytest.raises(ValueError, match="prefix"):
        continuation_logprob(model, tokens, prefix_length=len(tokens), device=CPU)


def test_load_multiple_choice_tasks_validates(tmp_path: Path) -> None:
    path = tmp_path / "task.jsonl"
    path.write_text(
        json.dumps({"question": "2+2=", "choices": [" 3", " 4"], "answer": 1}) + "\n",
        encoding="utf-8",
    )
    examples = load_multiple_choice_tasks(path)
    assert examples[0].answer == 1

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"question": "q", "choices": [" a", " b"], "answer": 5}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="out of range"):
        load_multiple_choice_tasks(bad)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no examples"):
        load_multiple_choice_tasks(empty)


def test_evaluate_multiple_choice_reports_accuracy() -> None:
    model = tiny_model()
    tokenizer = tiny_tokenizer()
    examples = [
        MultipleChoiceExample("the quick", [" brown", " lazy"], 0),
        MultipleChoiceExample("the lazy", [" dog", " fox"], 1),
    ]
    metrics = evaluate_multiple_choice(model, tokenizer, examples, CPU)
    assert metrics["task/multiple_choice_examples"] == 2.0
    assert 0.0 <= metrics["task/multiple_choice_accuracy"] <= 1.0


def test_evaluate_perplexity_files(tmp_path: Path) -> None:
    model = tiny_model()
    tokenizer = tiny_tokenizer()
    held_out = tmp_path / "validation.txt"
    held_out.write_text(
        "the quick brown fox jumps over the lazy dog " * 8,
        encoding="utf-8",
    )
    metrics = evaluate_perplexity_files(model, tokenizer, [held_out], CPU, sequence_length=16)
    assert metrics["perplexity/aggregate"] > 1.0
    assert metrics["perplexity/validation"] == metrics["perplexity/aggregate"]
    assert metrics["perplexity/tokens"] > 0


def test_passkey_probe_mechanics() -> None:
    samples = build_passkey_samples(3, filler_sentences=1, seed=7)
    assert len({sample.passkey for sample in samples}) >= 1
    assert all(sample.passkey in sample.prompt for sample in samples)
    assert samples == build_passkey_samples(3, filler_sentences=1, seed=7)

    model = tiny_model(context_length=128)
    tokenizer = tiny_tokenizer()
    metrics = evaluate_passkey(model, tokenizer, samples, CPU, max_new_tokens=4)
    assert metrics["task/passkey_samples"] == 3.0
    assert 0.0 <= metrics["task/passkey_accuracy"] <= 1.0
