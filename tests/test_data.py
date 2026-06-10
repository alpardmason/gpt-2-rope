from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gpt2_rope.data import (
    MemmapTokenDataset,
    SFTDataset,
    build_sft_example,
    collate_sft,
    prepare_corpus,
)
from gpt2_rope.tokenizer import ByteBPETokenizer


def tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(
        ["prompt response document text"],
        vocab_size=280,
        special_tokens=["<|endoftext|>"],
    )


def test_prepare_corpus_and_memmap_batches(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    source.write_text("first document\nsecond document\nthird document\n", encoding="utf-8")
    output = tmp_path / "processed"
    manifest = prepare_corpus([source], output, tokenizer(), validation_fraction=0.25)

    assert manifest["dtype"] == "uint16"
    assert (output / "manifest.json").exists()
    train = MemmapTokenDataset(output / "train.bin", sequence_length=4)
    x, y = train[0]
    assert x.shape == y.shape == (4,)
    np.testing.assert_array_equal(y[:-1].numpy(), x[1:].numpy())


def test_sft_mask_and_validation() -> None:
    tok = tokenizer()
    input_ids, labels = build_sft_example(
        "prompt " * 20,
        "response",
        tok,
        max_length=16,
    )
    assert len(input_ids) == len(labels) == 16
    assert labels[-1] == tok.eos_token_id
    assert any(label == -100 for label in labels)
    assert any(label >= 0 for label in labels)

    with pytest.raises(ValueError, match="response"):
        build_sft_example("prompt", "", tok, max_length=8)


def test_jsonl_requires_prompt_and_response(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt": "missing response"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="response"):
        prepare_corpus([bad], tmp_path / "out", tokenizer())


def test_sft_dataset_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sft.jsonl"
    records = [
        {"prompt": "prompt", "response": " response"},
        {"prompt": "text", "response": " document"},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    dataset = SFTDataset(path, tokenizer(), max_length=32)
    assert len(dataset) == 2
    input_ids, labels = dataset[0]
    assert input_ids.shape == labels.shape
    assert bool((labels == -100).any())
    assert bool((labels >= 0).any())

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no examples"):
        SFTDataset(empty, tokenizer(), max_length=32)


def test_collate_sft_pads_inputs_and_masks_labels() -> None:
    short = (torch.tensor([5, 6]), torch.tensor([-100, 6]))
    long = (torch.tensor([1, 2, 3, 4]), torch.tensor([-100, 2, 3, 4]))
    input_batch, label_batch = collate_sft([short, long], pad_token_id=0)
    assert input_batch.shape == label_batch.shape == (2, 4)
    assert input_batch[0].tolist() == [5, 6, 0, 0]
    assert label_batch[0].tolist() == [-100, 6, -100, -100]
    assert label_batch[1].tolist() == [-100, 2, 3, 4]

