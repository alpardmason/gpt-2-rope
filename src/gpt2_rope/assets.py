"""Checksum-verified retrieval of the original GPT-2 tokenizer assets."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

GPT2_TOKENIZER_ASSETS = {
    "vocab.json": (
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json",
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    ),
    "merges.txt": (
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe",
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    ),
}


def download_gpt2_tokenizer(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for filename, (url, expected_sha256) in GPT2_TOKENIZER_ASSETS.items():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"checksum mismatch for {filename}: expected {expected_sha256}, "
                f"received {actual_sha256}"
            )
        (output_dir / filename).write_bytes(data)
        checksums[filename] = actual_sha256
    return checksums

