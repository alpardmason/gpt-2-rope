from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpt2_rope.data_quality import (
    FilterThresholds,
    deduplicate_documents,
    estimated_jaccard,
    filter_documents,
    minhash_signature,
    non_alpha_ratio,
    normalized_content_hash,
    word_repetition_ratio,
    write_shards,
)


def test_content_hash_is_whitespace_insensitive() -> None:
    assert normalized_content_hash("a  b\tc") == normalized_content_hash(" a b c ")
    assert normalized_content_hash("a b c") != normalized_content_hash("a b d")


def test_minhash_signature_is_deterministic_and_discriminative() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    same = minhash_signature(text)
    assert same == minhash_signature(text)
    near = minhash_signature("the quick brown fox jumps over the lazy cat")
    far = minhash_signature("completely unrelated sentence about language models")
    assert estimated_jaccard(same, near) > estimated_jaccard(same, far)


def test_estimated_jaccard_rejects_invalid_signatures() -> None:
    with pytest.raises(ValueError, match="equal length"):
        estimated_jaccard((1, 2), (1,))


def test_exact_and_near_deduplication() -> None:
    base = "the quick brown fox jumps over the lazy dog near the river bank"
    documents = [
        base,
        base + "  ",  # exact duplicate after whitespace normalization
        base.replace("dog", "cat"),  # near duplicate
        "an entirely different document about rotary position embeddings",
    ]
    kept, report = deduplicate_documents(documents, near_duplicate_threshold=0.5)
    assert report.exact_duplicates == 1
    assert report.near_duplicates == 1
    assert report.kept == len(kept) == 2

    kept_exact_only, report_exact_only = deduplicate_documents(
        documents,
        near_duplicate_threshold=None,
    )
    assert report_exact_only.exact_duplicates == 1
    assert report_exact_only.near_duplicates == 0
    assert len(kept_exact_only) == 3


def test_quality_heuristics() -> None:
    assert word_repetition_ratio("spam spam spam spam") == pytest.approx(0.75)
    assert word_repetition_ratio("all words unique here") == pytest.approx(0.0)
    assert non_alpha_ratio("$$$ ###") == pytest.approx(1.0)
    assert non_alpha_ratio("clean text") == pytest.approx(0.0)


def test_filter_documents_accounts_for_every_rejection() -> None:
    documents = [
        "ok",  # too short
        "x" * 60,  # too long for the test threshold
        "spam spam spam spam spam spam",  # repetitive
        "%$#@! &*()_ +=-~ ^^;;",  # non-text
        "a perfectly reasonable training document",
    ]
    kept, report = filter_documents(
        documents,
        FilterThresholds(
            min_chars=5,
            max_chars=50,
            max_word_repetition_ratio=0.5,
            max_non_alpha_ratio=0.5,
        ),
    )
    assert kept == ["a perfectly reasonable training document"]
    assert report.as_dict() == {
        "kept": 1,
        "rejected": {"non_text": 1, "repetitive": 1, "too_long": 1, "too_short": 1},
    }


def test_write_shards_produces_content_addressed_manifest(tmp_path: Path) -> None:
    documents = [f"document number {index}" for index in range(5)]
    manifest = write_shards(documents, tmp_path, documents_per_shard=2)
    assert manifest["total_documents"] == 5
    shards = manifest["shards"]
    assert isinstance(shards, list) and len(shards) == 3
    on_disk = json.loads((tmp_path / "shards.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    first = tmp_path / "shard-00000.txt"
    assert first.read_text(encoding="utf-8") == "document number 0\ndocument number 1\n"

    with pytest.raises(ValueError, match="empty"):
        write_shards([], tmp_path)
