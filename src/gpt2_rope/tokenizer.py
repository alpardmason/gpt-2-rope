"""Transparent GPT-2-compatible byte-level BPE tokenizer."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

import regex

GPT2_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def bytes_to_unicode() -> dict[int, str]:
    """Map all bytes to reversible visible Unicode code points as GPT-2 does."""
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    byte_values = visible[:]
    code_points = visible[:]
    extra = 0
    for value in range(256):
        if value not in visible:
            byte_values.append(value)
            code_points.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(point) for point in code_points), strict=True))


def _pairs(symbols: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(pairwise(symbols))


class ByteBPETokenizer:
    """Native byte-level BPE supporting GPT-2 vocab/merges file formats."""

    def __init__(
        self,
        encoder: dict[str, int],
        merges: list[tuple[str, str]],
        special_tokens: Iterable[str] = (),
    ) -> None:
        self.encoder = dict(encoder)
        self.decoder = {token_id: token for token, token_id in encoder.items()}
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = tuple(special_tokens)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.cache: dict[str, tuple[str, ...]] = {}
        for token in self.special_tokens:
            if token not in self.encoder:
                raise ValueError(f"special token {token!r} is absent from vocabulary")

    @property
    def eos_token_id(self) -> int:
        if "<|endoftext|>" not in self.encoder:
            raise ValueError("tokenizer has no <|endoftext|> token")
        return self.encoder["<|endoftext|>"]

    def bpe(self, token: str) -> tuple[str, ...]:
        if token in self.cache:
            return self.cache[token]
        symbols = tuple(token)
        if len(symbols) < 2:
            return symbols
        while True:
            candidates = _pairs(symbols)
            pair = min(candidates, key=lambda item: self.merge_ranks.get(item, math_inf()))
            if pair not in self.merge_ranks:
                break
            first, second = pair
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if index + 1 < len(symbols) and symbols[index : index + 2] == pair:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = tuple(merged)
            if len(symbols) == 1:
                break
        if len(self.cache) >= 100_000:
            self.cache.clear()
        self.cache[token] = symbols
        return symbols

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        token_ids: list[int] = []
        if allowed_special and self.special_tokens:
            escaped = "|".join(regex.escape(token) for token in self.special_tokens)
            split_pattern = f"({escaped})"
            segments = regex.split(split_pattern, text)
        else:
            segments = [text]
        for segment in segments:
            if not segment:
                continue
            if segment in self.special_tokens:
                token_ids.append(self.encoder[segment])
                continue
            for match in GPT2_PATTERN.findall(segment):
                byte_token = "".join(self.byte_encoder[value] for value in match.encode("utf-8"))
                token_ids.extend(self.encoder[piece] for piece in self.bpe(byte_token))
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        text = "".join(self.decoder[token_id] for token_id in token_ids)
        data = bytes(self.byte_decoder[character] for character in text)
        return data.decode("utf-8", errors="replace")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "vocab.json").write_text(
            json.dumps(self.encoder, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        merge_lines = ["#version: 0.2", *(f"{left} {right}" for left, right in self.merges)]
        (directory / "merges.txt").write_text("\n".join(merge_lines) + "\n", encoding="utf-8")

    def identity(self) -> dict[str, str | int]:
        payload = json.dumps(self.encoder, sort_keys=True).encode()
        merges = "\n".join(f"{left} {right}" for left, right in self.merges).encode()
        return {
            "vocab_size": len(self.encoder),
            "sha256": hashlib.sha256(payload + b"\0" + merges).hexdigest(),
        }

    @classmethod
    def from_files(
        cls,
        vocab_path: Path,
        merges_path: Path,
        special_tokens: Iterable[str] = ("<|endoftext|>",),
    ) -> ByteBPETokenizer:
        encoder = json.loads(vocab_path.read_text(encoding="utf-8"))
        if not isinstance(encoder, dict):
            raise ValueError("vocab.json must contain an object")
        lines = merges_path.read_text(encoding="utf-8").splitlines()
        merges = [tuple(line.split()) for line in lines if line and not line.startswith("#")]
        if any(len(pair) != 2 for pair in merges):
            raise ValueError("every BPE merge must contain exactly two symbols")
        return cls(
            {str(token): int(token_id) for token, token_id in encoder.items()},
            [(pair[0], pair[1]) for pair in merges],
            special_tokens,
        )

    @classmethod
    def train(
        cls,
        documents: Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = (),
    ) -> ByteBPETokenizer:
        specials = tuple(dict.fromkeys(special_tokens))
        base = bytes_to_unicode()
        minimum_size = 256 + len(specials)
        if vocab_size < minimum_size:
            raise ValueError(f"vocab_size must be at least {minimum_size}")

        word_counts: Counter[tuple[str, ...]] = Counter()
        for document in documents:
            for match in GPT2_PATTERN.findall(document):
                encoded = tuple(base[value] for value in match.encode("utf-8"))
                word_counts[encoded] += 1

        merges: list[tuple[str, str]] = []
        learned_tokens: list[str] = []
        while 256 + len(learned_tokens) + len(specials) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for symbols, count in word_counts.items():
                for pair in pairwise(symbols):
                    pair_counts[pair] += count
            if not pair_counts:
                break
            best_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
            merges.append(best_pair)
            merged_token = "".join(best_pair)
            learned_tokens.append(merged_token)
            updated: Counter[tuple[str, ...]] = Counter()
            for symbols, count in word_counts.items():
                output: list[str] = []
                index = 0
                while index < len(symbols):
                    if index + 1 < len(symbols) and symbols[index : index + 2] == best_pair:
                        output.append(merged_token)
                        index += 2
                    else:
                        output.append(symbols[index])
                        index += 1
                updated[tuple(output)] += count
            word_counts = updated

        tokens = list(base.values()) + learned_tokens + list(specials)
        encoder = {token: token_id for token_id, token in enumerate(tokens)}
        return cls(encoder, merges, specials)


def math_inf() -> float:
    return float("inf")
