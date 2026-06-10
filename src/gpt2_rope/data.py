"""Corpus preparation, memory-mapped pretraining data, and SFT datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from gpt2_rope.tokenizer import ByteBPETokenizer


def read_documents(paths: Sequence[Path]) -> Iterator[str]:
    for path in paths:
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                if "text" in record:
                    yield str(record["text"])
                elif "prompt" in record:
                    if "response" not in record:
                        raise ValueError(f"{path}:{line_number} is missing response")
                    yield f"{record['prompt']}{record['response']}"
                else:
                    raise ValueError(f"{path}:{line_number} requires text or prompt/response")
        else:
            yield from path.read_text(encoding="utf-8").splitlines()


def prepare_corpus(
    paths: Sequence[Path],
    output_dir: Path,
    tokenizer: ByteBPETokenizer,
    validation_fraction: float = 0.01,
) -> dict[str, Any]:
    """Tokenize documents into deterministic uint16 train/validation streams."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    documents = list(read_documents(paths))
    if not documents:
        raise ValueError("corpus contains no documents")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_index = max(1, round(len(documents) * (1 - validation_fraction)))
    split_index = min(split_index, len(documents))
    partitions = {
        "train": documents[:split_index],
        "validation": documents[split_index:],
    }
    counts: dict[str, int] = {}
    for split, split_documents in partitions.items():
        token_ids: list[int] = []
        for document in split_documents:
            token_ids.extend(tokenizer.encode(document))
            token_ids.append(tokenizer.eos_token_id)
        array = np.asarray(token_ids, dtype=np.uint16)
        array.tofile(output_dir / f"{split}.bin")
        counts[split] = int(array.size)
    manifest: dict[str, Any] = {
        "version": 1,
        "dtype": "uint16",
        "token_counts": counts,
        "tokenizer": tokenizer.identity(),
        "sources": [str(path) for path in paths],
        # Content hashes make data provenance verifiable: a resumed or compared
        # run can prove it trained on byte-identical sources.
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


class MemmapTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """Map contiguous token windows without loading the corpus into RAM."""

    def __init__(self, path: Path, sequence_length: int) -> None:
        self.path = path
        self.sequence_length = sequence_length
        self.tokens = np.memmap(path, mode="r", dtype=np.uint16)
        if self.tokens.size < sequence_length + 1:
            raise ValueError(f"{path} has too few tokens for sequence length {sequence_length}")

    def __len__(self) -> int:
        return (self.tokens.size - 1) // self.sequence_length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        window = np.asarray(
            self.tokens[start : start + self.sequence_length + 1],
            dtype=np.int64,
        )
        return torch.from_numpy(window[:-1].copy()), torch.from_numpy(window[1:].copy())


def build_sft_example(
    prompt: str,
    response: str,
    tokenizer: ByteBPETokenizer,
    max_length: int,
) -> tuple[list[int], list[int]]:
    if not response:
        raise ValueError("response must contain at least one character")
    prompt_ids = tokenizer.encode(prompt)
    response_ids = [*tokenizer.encode(response), tokenizer.eos_token_id]
    if len(response_ids) >= max_length:
        response_ids = response_ids[:max_length]
        response_ids[-1] = tokenizer.eos_token_id
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[-(max_length - len(response_ids)) :]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    if not any(label >= 0 for label in labels):
        raise ValueError("example has no supervised response tokens")
    return input_ids, labels


class SFTDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, path: Path, tokenizer: ByteBPETokenizer, max_length: int) -> None:
        self.examples: list[tuple[list[int], list[int]]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "prompt" not in record or "response" not in record:
                raise ValueError(f"{path}:{line_number} requires prompt and response strings")
            self.examples.append(
                build_sft_example(
                    str(record["prompt"]),
                    str(record["response"]),
                    tokenizer,
                    max_length,
                )
            )
        if not self.examples:
            raise ValueError("fine-tuning dataset contains no examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        input_ids, labels = self.examples[index]
        return torch.tensor(input_ids), torch.tensor(labels)


def collate_sft(
    batch: Sequence[tuple[Tensor, Tensor]],
    pad_token_id: int,
) -> tuple[Tensor, Tensor]:
    max_length = max(input_ids.numel() for input_ids, _ in batch)
    input_batch = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    label_batch = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for row, (input_ids, labels) in enumerate(batch):
        input_batch[row, : input_ids.numel()] = input_ids
        label_batch[row, : labels.numel()] = labels
    return input_batch, label_batch
