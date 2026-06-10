from __future__ import annotations

from pathlib import Path

from gpt2_rope.tokenizer import ByteBPETokenizer


def test_byte_bpe_round_trip_and_persistence(tmp_path: Path) -> None:
    corpus = ["hello world", "hello rotary attention", "naïve café"]
    tokenizer = ByteBPETokenizer.train(corpus, vocab_size=300, special_tokens=["<|endoftext|>"])

    text = "hello, naïve world!"
    assert tokenizer.decode(tokenizer.encode(text)) == text

    tokenizer.save(tmp_path)
    restored = ByteBPETokenizer.from_files(
        tmp_path / "vocab.json",
        tmp_path / "merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    assert restored.encode(text) == tokenizer.encode(text)


def test_training_is_deterministic() -> None:
    corpus = ["low lower lowest", "newer wider"]
    first = ByteBPETokenizer.train(corpus, vocab_size=280)
    second = ByteBPETokenizer.train(corpus, vocab_size=280)
    assert first.encoder == second.encoder
    assert first.merges == second.merges

