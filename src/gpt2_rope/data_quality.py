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
    """Heuristic document-quality gates with conservative defaults.

    Each threshold rejects documents that fail a length, repetition, or
    character-composition check. Defaults are tuned for tutorial-scale corpora
    and reject obvious boilerplate or non-text noise without aggressive
    filtering.

    Attributes:
        min_chars: Minimum document length in characters.
        max_chars: Maximum document length in characters.
        max_word_repetition_ratio: Upper bound on the fraction of repeated
            words; see :func:`word_repetition_ratio`.
        max_non_alpha_ratio: Upper bound on the fraction of non-alphanumeric
            characters; see :func:`non_alpha_ratio`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_chars: int = Field(default=8, ge=1)
    max_chars: int = Field(default=100_000, ge=1)
    max_word_repetition_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    max_non_alpha_ratio: float = Field(default=0.6, ge=0.0, le=1.0)


@dataclass(slots=True)
class DedupReport:
    """Aggregate counts from :func:`deduplicate_documents`.

    Attributes:
        kept: Number of documents retained after deduplication.
        exact_duplicates: Documents dropped because their normalized hash
            matched an earlier document.
        near_duplicates: Documents dropped because their MinHash signature
            exceeded the Jaccard threshold against a kept document.
    """

    kept: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0

    def as_dict(self) -> dict[str, int]:
        """Serialize deduplication counters for logging or manifests.

        Returns:
            A mapping of report field names to integer counts.
        """
        return {
            "kept": self.kept,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
        }


@dataclass(slots=True)
class FilterReport:
    """Aggregate counts from :func:`filter_documents`.

    Attributes:
        kept: Number of documents that passed every quality gate.
        rejected: Per-reason rejection counts keyed by gate name
            (for example ``"too_short"`` or ``"repetitive"``).
    """

    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        """Increment the rejection counter for ``reason``.

        Args:
            reason: Stable gate identifier recorded in the report.

        Returns:
            None
        """
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def as_dict(self) -> dict[str, object]:
        """Serialize filter counters for logging or manifests.

        Returns:
            A dictionary with ``kept`` and a sorted ``rejected`` mapping.
        """
        return {"kept": self.kept, "rejected": dict(sorted(self.rejected.items()))}


def normalized_content_hash(text: str) -> str:
    """Compute a whitespace-insensitive SHA-256 digest for exact deduplication.

    Collapses internal whitespace, strips leading and trailing space, then
    hashes the canonical UTF-8 bytes so formatting-only differences map to
    the same key.

    Args:
        text: Raw document text.

    Returns:
        Lowercase hexadecimal SHA-256 digest of the canonical form.
    """
    canonical = _WHITESPACE.sub(" ", text).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _shingles(text: str, shingle_size: int) -> set[str]:
    """Build lowercase word shingles for MinHash fingerprinting.

    Args:
        text: Raw document text.
        shingle_size: Number of consecutive words per shingle.

    Returns:
        The set of distinct shingles; a single shingle when the document is
        shorter than ``shingle_size``, or an empty set for blank input.
    """
    words = _WHITESPACE.sub(" ", text).strip().lower().split(" ")
    if len(words) < shingle_size:
        return {" ".join(words)} if words != [""] else set()
    return {" ".join(words[i : i + shingle_size]) for i in range(len(words) - shingle_size + 1)}


def minhash_signature(
    text: str,
    num_hashes: int = 64,
    shingle_size: int = 3,
) -> tuple[int, ...]:
    """Compute a deterministic MinHash signature over lowercase word shingles.

    Each hash function is a seeded BLAKE2b minimum over shingle digests,
    yielding a fixed-length signature suitable for approximate Jaccard
    comparison.

    Args:
        text: Raw document text.
        num_hashes: Number of independent hash functions (signature length).
        shingle_size: Number of consecutive words per shingle.

    Returns:
        A tuple of ``num_hashes`` integer minimum-hash values; all zeros when
        the document yields no shingles.
    """
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
    """Estimate set Jaccard similarity from two MinHash signatures.

    Uses the fraction of matching hash positions, which is an unbiased
    estimator of Jaccard similarity when signatures are built with the same
    parameters.

    Args:
        left: First MinHash signature.
        right: Second MinHash signature of equal length.

    Returns:
        The fraction of positions where ``left`` and ``right`` agree.

    Raises:
        ValueError: If either signature is empty or the lengths differ.
    """
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

    Exact deduplication uses :func:`normalized_content_hash`. Near-duplicate
    detection compares MinHash signatures against every kept document; this
    is ``O(n^2)`` and fine at tutorial scale—production systems bucket
    signatures with LSH instead.

    Args:
        documents: Corpus in encounter order; only the first occurrence of
            each duplicate is retained.
        near_duplicate_threshold: Minimum estimated Jaccard similarity to
            treat two documents as near-duplicates. Pass ``None`` to skip
            near-duplicate detection and keep exact-dedup only.
        num_hashes: Signature length for :func:`minhash_signature`.
        shingle_size: Shingle width for :func:`minhash_signature`.

    Returns:
        A pair of the kept document list and a :class:`DedupReport` with
        per-category drop counts.
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
    """Measure how repetitive a document is at the word level.

    Args:
        text: Raw document text.

    Returns:
        The fraction of words that repeat an earlier word in the document
        (``0.0`` when every word is unique, ``1.0`` for empty or
        whitespace-only input).
    """
    words = _WHITESPACE.sub(" ", text).strip().lower().split(" ")
    if not words or words == [""]:
        return 1.0
    counts = Counter(words)
    return 1.0 - len(counts) / len(words)


def non_alpha_ratio(text: str) -> float:
    """Measure the fraction of non-alphanumeric characters in a document.

    Args:
        text: Raw document text.

    Returns:
        The fraction of non-whitespace characters that are not alphanumeric
        (``1.0`` when the document contains no significant characters).
    """
    significant = [character for character in text if not character.isspace()]
    if not significant:
        return 1.0
    return sum(1 for character in significant if not character.isalnum()) / len(significant)


def filter_documents(
    documents: Iterable[str],
    thresholds: FilterThresholds | None = None,
) -> tuple[list[str], FilterReport]:
    """Apply heuristic quality gates and account for every rejection by reason.

    Documents are evaluated in order against :class:`FilterThresholds`. The
    first failing gate determines the rejection reason; no document is counted
    against more than one gate.

    Args:
        documents: Corpus to filter in encounter order.
        thresholds: Quality gates to apply; defaults to
            :class:`FilterThresholds` when omitted.

    Returns:
        A pair of the kept document list and a :class:`FilterReport` with
        per-reason rejection counts.
    """
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
    """Write fixed-size document shards plus a content-addressed manifest.

    Each shard is a newline-delimited UTF-8 text file. ``shards.json`` records
    shard paths, document counts, and SHA-256 digests of shard payloads for
    reproducible provenance.

    Args:
        documents: Ordered corpus to partition into shards.
        output_dir: Directory that receives shard files and ``shards.json``.
        documents_per_shard: Maximum documents written to each shard file.
        prefix: Filename stem for shard files (for example ``shard-00000.txt``).

    Returns:
        The manifest dictionary also written to ``output_dir/shards.json``.

    Raises:
        ValueError: If ``documents`` is empty or ``documents_per_shard`` is
            less than one.
    """
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
