from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from gpt2_rope.config import (
    DataConfig,
    DPOConfig,
    ExperimentConfig,
    ModelConfig,
    MonitoringConfig,
    TrainingConfig,
)
from gpt2_rope.dpo import (
    PreferenceDataset,
    collate_preferences,
    dpo_loss,
    sequence_logprobs,
    train_dpo,
)
from gpt2_rope.model import GPT
from gpt2_rope.tokenizer import ByteBPETokenizer


def tiny_tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(
        ["alpha beta gamma delta epsilon zeta"],
        vocab_size=280,
        special_tokens=["<|endoftext|>"],
    )


def write_preferences(path: Path) -> None:
    records = [
        {"prompt": "alpha", "chosen": " beta gamma", "rejected": " zeta"},
        {"prompt": "delta", "chosen": " epsilon", "rejected": " alpha beta"},
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_preference_dataset_and_collation(tmp_path: Path) -> None:
    data_path = tmp_path / "preferences.jsonl"
    write_preferences(data_path)
    dataset = PreferenceDataset(data_path, tiny_tokenizer(), max_length=16)
    assert len(dataset) == 2
    (chosen_ids, chosen_labels), (rejected_ids, rejected_labels) = dataset[0]
    assert chosen_ids.shape == chosen_labels.shape
    assert rejected_ids.shape == rejected_labels.shape
    assert bool((chosen_labels == -100).any())

    batch = [dataset[0], dataset[1]]
    chosen_inputs, chosen_label_batch, rejected_inputs, rejected_label_batch = (
        collate_preferences(batch, pad_token_id=0)
    )
    assert chosen_inputs.shape == chosen_label_batch.shape
    assert rejected_inputs.shape == rejected_label_batch.shape
    assert chosen_inputs.size(0) == rejected_inputs.size(0) == 2


def test_preference_dataset_rejects_malformed_rows(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt": "p", "chosen": "c"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rejected"):
        PreferenceDataset(bad, tiny_tokenizer(), max_length=16)


def test_sequence_logprobs_masks_prompt_tokens() -> None:
    torch.manual_seed(0)
    model = GPT(
        ModelConfig(
            vocab_size=300,
            context_length=16,
            d_model=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            dropout=0.0,
        )
    ).eval()
    input_ids = torch.randint(0, 300, (2, 8))
    labels = input_ids.clone()
    labels[:, :4] = -100
    with torch.no_grad():
        scores = sequence_logprobs(model, input_ids, labels)
        fully_masked = labels.clone()
        fully_masked[:, :] = -100
        zero_scores = sequence_logprobs(model, input_ids, fully_masked)
    assert scores.shape == (2,)
    assert bool((scores < 0).all())
    torch.testing.assert_close(zero_scores, torch.zeros(2))


def test_dpo_loss_prefers_wider_margins() -> None:
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-5.0, -6.0])
    reference = torch.tensor([-3.0, -3.0])
    good_loss, good_metrics = dpo_loss(
        policy_chosen, policy_rejected, reference, reference, beta=0.1
    )
    bad_loss, bad_metrics = dpo_loss(
        policy_rejected, policy_chosen, reference, reference, beta=0.1
    )
    assert float(good_loss) < float(bad_loss)
    assert good_metrics["dpo/reward_accuracy"] == 1.0
    assert bad_metrics["dpo/reward_accuracy"] == 0.0
    assert good_metrics["dpo/reward_margin"] > 0
    assert float(good_loss) < math.log(2) < float(bad_loss)


def test_train_dpo_smoke(tmp_path: Path) -> None:
    tokenizer = tiny_tokenizer()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save(tokenizer_dir)
    data_path = tmp_path / "preferences.jsonl"
    write_preferences(data_path)

    config = ExperimentConfig(
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
            train_path=tmp_path / "unused.bin",
            tokenizer_dir=tokenizer_dir,
            sequence_length=16,
        ),
        training=TrainingConfig(
            output_dir=tmp_path / "dpo-run",
            device="cpu",
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=2,
            learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_steps=1,
            precision="fp32",
            checkpoint_every=2,
        ),
        monitoring=MonitoringConfig(log_every=1, tensorboard=False),
        dpo=DPOConfig(data_path=data_path, max_length=16, beta=0.1, use_lora=True),
    )
    run_dir = train_dpo(config)
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any("dpo/loss" in record for record in metrics)
    assert any("dpo/reward_margin" in record for record in metrics)
    checkpoint = run_dir / "checkpoints" / "step-00000002"
    assert (checkpoint / "model.pt").exists()
    assert (checkpoint / "adapter.safetensors").exists()
