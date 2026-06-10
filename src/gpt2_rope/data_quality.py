"""Corpus engineering: exact/near deduplication, quality filters, and shards.

Everything here is native and single-node by design. Production pipelines use
distributed equivalents (MinHashLSH, Bloom filters, Spark/Ray jobs), but the
contracts are identical: deterministic signatures, explicit rejection reasons,
and content-addressed shard manifests.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_WHITESPACE = re.compile(r"\s+")


class FilterThresholds(BaseModel):
    """Heuristic document-quality gates with conservative defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_chars: int = Field(default=8, ge=1)
    max_chars: int = Field(default=100_000, ge=1)
    max_word_repetition_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    max_non_alpha_ratio: float = Field(default=0.6, ge=0.0, le=1.0)


@dataclass(slots=True)
class DedupReport:
    kept: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "kept": self.kept,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
        }


@dataclass(slots=True)
class FilterReport:
    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def as_dict(self) -> dict[str, object]:
        return {"kept": self.kept, "rejected": dict(sorted(self.rejected.items()))}


def normalized_content_hash(text: str) -> str:
    """Whitespace-insensitive SHA-256 used for exact deduplication."""
    canonical = _WHITESPACE.sub(" ", text).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _shingles(text: str, shingle_size: int) -> set[str]:
    words = _WHITESPACE.sub(" ", text).strip().lower().split(" ")
    if len(words) < shingle_size:
        return {" ".join(words)} if words != [""] else set()
    return {" ".join(words[i : i + shingle_size]) for i in range(len(words) - shingle_size + 1)}


def minhash_signature(
    text: str,
    num_hashes: int = 64,
    shingle_size: int = 3,
) -> tuple[int, ...]:
    """Deterministic MinHash signature over lowercase word shingles."""
    shingles = _shingles(text, shingle_size)
    if not shingles:
        return tuple([0] * num_hashes)
    signature: list[int] = []
    for seed in range(num_hashes):
        signature.append(
            min(
                int.from_bytes(
                    hashlib.blake2b(
                        shingle.encode("utf-8"),
                        digest_size=8,
                        salt=seed.to_bytes(2, "little") + b"\0" * 14,
                    ).digest(),
                    "big",
                )
                for shingle in shingles
            )
        )
    return tuple(signature)


def estimated_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("signatures must be non-empty and equal length")
    matches = sum(1 for a, b in zip(left, right, strict=True) if a == b)
    return matches / len(left)


def deduplicate_documents(
    documents: Iterable[str],
    *,
    near_duplicate_threshold: float | None = 0.9,
    num_hashes: int = 64,
    shingle_size: int = 3,
) -> tuple[list[str], DedupReport]:
    """Drop exact duplicates, then near-duplicates above the Jaccard threshold.

    Near-duplicate comparison is O(n^2) against kept documents, which is fine
    at tutorial scale; production systems bucket signatures with LSH instead.
    """
    report = DedupReport()
    kept: list[str] = []
    seen_hashes: set[str] = set()
    kept_signatures: list[tuple[int, ...]] = []
    for document in documents:
        content_hash = normalized_content_hash(document)
        if content_hash in seen_hashes:
            report.exact_duplicates += 1
            continue
        seen_hashes.add(content_hash)
        if near_duplicate_threshold is not None:
            signature = minhash_signature(document, num_hashes, shingle_size)
            if any(
                estimated_jaccard(signature, existing) >= near_duplicate_threshold
                for existing in kept_signatures
            ):
                report.near_duplicates += 1
                continue
            kept_signatures.append(signature)
        kept.append(document)
    report.kept = len(kept)
    return kept, report


def word_repetition_ratio(text: str) -> float:
    """Fraction of words that are repeats of an earlier word (0 = all unique)."""
    words = _WHITESPACE.sub(" ", text).strip().lower().split(" ")
    if not words or words == [""]:
        return 1.0
    counts = Counter(words)
    return 1.0 - len(counts) / len(words)


def non_alpha_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are not alphanumeric."""
    significant = [character for character in text if not character.isspace()]
    if not significant:
        return 1.0
    return sum(1 for character in significant if not character.isalnum()) / len(significant)


def filter_documents(
    documents: Iterable[str],
    thresholds: FilterThresholds | None = None,
) -> tuple[list[str], FilterReport]:
    """Apply heuristic gates and account for every rejection by reason."""
    thresholds = thresholds or FilterThresholds()
    report = FilterReport()
    kept: list[str] = []
    for document in documents:
        length = len(document)
        if length < thresholds.min_chars:
            report.reject("too_short")
        elif length > thresholds.max_chars:
            report.reject("too_long")
        elif word_repetition_ratio(document) > thresholds.max_word_repetition_ratio:
            report.reject("repetitive")
        elif non_alpha_ratio(document) > thresholds.max_non_alpha_ratio:
            report.reject("non_text")
        else:
            kept.append(document)
    report.kept = len(kept)
    return kept, report


def write_shards(
    documents: Sequence[str],
    output_dir: Path,
    *,
    documents_per_shard: int = 10_000,
    prefix: str = "shard",
) -> dict[str, object]:
    """Write fixed-size document shards plus a content-addressed manifest."""
    if not documents:
        raise ValueError("cannot shard an empty document list")
    if documents_per_shard < 1:
        raise ValueError("documents_per_shard must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    for shard_index, start in enumerate(range(0, len(documents), documents_per_shard)):
        chunk = documents[start : start + documents_per_shard]
        payload = "\n".join(chunk) + "\n"
        shard_path = output_dir / f"{prefix}-{shard_index:05d}.txt"
        shard_path.write_text(payload, encoding="utf-8")
        shards.append(
            {
                "path": shard_path.name,
                "documents": len(chunk),
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            }
        )
    manifest: dict[str, object] = {
        "version": 1,
        "total_documents": len(documents),
        "documents_per_shard": documents_per_shard,
        "shards": shards,
    }
    (output_dir / "shards.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
