from __future__ import annotations

import json
from pathlib import Path

import torch

from gpt2_rope.config import (
    DataConfig,
    ExperimentConfig,
    FineTuningConfig,
    ModelConfig,
    MonitoringConfig,
    TrainingConfig,
)
from gpt2_rope.data import prepare_corpus
from gpt2_rope.tokenizer import ByteBPETokenizer
from gpt2_rope.training import (
    cosine_learning_rate,
    train_finetuning,
    train_pretraining,
)


def _build_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Create a tokenizer directory and memmap corpus for tiny training runs."""
    tokenizer = ByteBPETokenizer.train(
        ["alpha beta gamma delta epsilon zeta"],
        vocab_size=280,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save(tokenizer_dir)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join("alpha beta gamma delta epsilon zeta" for _ in range(40)) + "\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "processed"
    prepare_corpus([corpus], data_dir, tokenizer, validation_fraction=0.25)
    return tokenizer_dir, data_dir


def _experiment(
    tokenizer_dir: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    max_steps: int,
    resume_from: Path | None = None,
    profile_every: int | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=300,
            context_length=16,
            d_model=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            dropout=0.0,
        ),
        data=DataConfig(
            train_path=data_dir / "train.bin",
            validation_path=data_dir / "validation.bin",
            tokenizer_dir=tokenizer_dir,
            sequence_length=16,
        ),
        training=TrainingConfig(
            output_dir=output_dir,
            device="cpu",
            seed=1337,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=max_steps,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=1,
            precision="fp32",
            eval_every=2,
            eval_batches=1,
            checkpoint_every=2,
            resume_from=resume_from,
        ),
        monitoring=MonitoringConfig(
            log_every=1,
            tensorboard=False,
            profile_every=profile_every,
        ),
    )


def test_pretraining_smoke_writes_metrics_checkpoints_and_traces(tmp_path: Path) -> None:
    tokenizer_dir, data_dir = _build_workspace(tmp_path)
    run_dir = train_pretraining(
        _experiment(
            tokenizer_dir,
            data_dir,
            tmp_path / "run",
            max_steps=4,
            profile_every=3,
        )
    )

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any("train/loss" in record for record in metrics)
    assert any("validation/loss" in record for record in metrics)
    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "checkpoints" / "step-00000002").is_dir()
    assert (run_dir / "checkpoints" / "step-00000004").is_dir()
    assert any((run_dir / "profiler").iterdir())


def test_resume_reproduces_uninterrupted_training(tmp_path: Path) -> None:
    tokenizer_dir, data_dir = _build_workspace(tmp_path)
    full_run = train_pretraining(
        _experiment(tokenizer_dir, data_dir, tmp_path / "full", max_steps=4)
    )
    resumed_run = train_pretraining(
        _experiment(
            tokenizer_dir,
            data_dir,
            tmp_path / "resumed",
            max_steps=4,
            resume_from=full_run / "checkpoints" / "step-00000002",
        )
    )

    full_state = torch.load(
        full_run / "checkpoints" / "step-00000004" / "model.pt",
        weights_only=True,
    )
    resumed_state = torch.load(
        resumed_run / "checkpoints" / "step-00000004" / "model.pt",
        weights_only=True,
    )
    assert full_state.keys() == resumed_state.keys()
    for name, tensor in full_state.items():
        torch.testing.assert_close(resumed_state[name], tensor, atol=0.0, rtol=0.0)


def test_finetuning_smoke_with_validation(tmp_path: Path) -> None:
    tokenizer_dir, data_dir = _build_workspace(tmp_path)
    sft_path = tmp_path / "sft.jsonl"
    records = [
        {"prompt": "alpha beta", "response": " gamma delta"},
        {"prompt": "gamma", "response": " epsilon zeta"},
    ]
    sft_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    base = _experiment(tokenizer_dir, data_dir, tmp_path / "sft-run", max_steps=2)
    config = base.model_copy(
        update={
            "finetuning": FineTuningConfig(
                data_path=sft_path,
                validation_path=sft_path,
                max_length=16,
            )
        }
    )
    run_dir = train_finetuning(config)
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any("train/loss" in record for record in metrics)
    assert any("validation/loss" in record for record in metrics)


def test_cosine_schedule_warmup_and_floor() -> None:
    settings = {
        "warmup_steps": 10,
        "max_steps": 100,
        "max_learning_rate": 1.0,
        "min_learning_rate": 0.1,
    }
    assert cosine_learning_rate(0, **settings) == 0.1 * 1.0
    assert cosine_learning_rate(9, **settings) == 1.0
    assert cosine_learning_rate(100, **settings) == 0.1
    middle = cosine_learning_rate(55, **settings)
    assert 0.1 < middle < 1.0
