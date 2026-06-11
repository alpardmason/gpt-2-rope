"""Direct Preference Optimization on prompt/chosen/rejected JSONL data.

DPO trains the policy to widen the implicit reward margin between preferred
and dispreferred responses relative to a frozen reference model, without an
explicit reward model or PPO-style rollouts. The loop intentionally reuses
this project's SFT example construction, checkpointing, and monitoring.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset

from gpt2_rope.checkpoint import save_checkpoint
from gpt2_rope.config import ExperimentConfig
from gpt2_rope.data import build_sft_example, collate_sft
from gpt2_rope.lora import apply_lora, save_lora
from gpt2_rope.model import GPT
from gpt2_rope.monitoring import MetricLogger
from gpt2_rope.tokenizer import ByteBPETokenizer
from gpt2_rope.training import (
    _autocast_context,
    _effective_precision,
    cosine_learning_rate,
    resolve_device,
    seed_everything,
)

# KATA(dpo-loss): the torch.nn.functional import used by the removed bodies
# was dropped; re-add it.

PreferencePair = tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]


class PreferenceDataset(Dataset[PreferencePair]):
    """JSONL rows of ``{"prompt", "chosen", "rejected"}`` as masked token pairs."""

    def __init__(self, path: Path, tokenizer: ByteBPETokenizer, max_length: int) -> None:
        self.examples: list[tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not all(
                isinstance(record.get(key), str) for key in ("prompt", "chosen", "rejected")
            ):
                raise ValueError(f"{path}:{line_number} requires prompt, chosen, rejected")
            prompt = str(record["prompt"])
            self.examples.append(
                (
                    build_sft_example(prompt, str(record["chosen"]), tokenizer, max_length),
                    build_sft_example(prompt, str(record["rejected"]), tokenizer, max_length),
                )
            )
        if not self.examples:
            raise ValueError("preference dataset contains no examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreferencePair:
        chosen, rejected = self.examples[index]
        return (
            (torch.tensor(chosen[0]), torch.tensor(chosen[1])),
            (torch.tensor(rejected[0]), torch.tensor(rejected[1])),
        )


def collate_preferences(
    batch: list[PreferencePair],
    pad_token_id: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    chosen_inputs, chosen_labels = collate_sft([pair[0] for pair in batch], pad_token_id)
    rejected_inputs, rejected_labels = collate_sft([pair[1] for pair in batch], pad_token_id)
    return chosen_inputs, chosen_labels, rejected_inputs, rejected_labels


def sequence_logprobs(model: GPT, input_ids: Tensor, labels: Tensor) -> Tensor:
    """Per-sequence sum of response-token log-probabilities, shape ``[B]``."""
    # KATA(dpo-loss): contract:
    # 1. GPT-2 shift: logits[:, :-1] score labels[:, 1:].
    # 2. Only positions where the shifted label != -100 are supervised;
    #    clamp masked targets to a valid index before gather, then zero
    #    their contribution.
    # 3. log_softmax in FP32 (autocast may deliver reduced precision).
    # 4. Sum per sequence -> shape [B].
    raise NotImplementedError("KATA(dpo-loss): implement sequence_logprobs")


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    beta: float,
) -> tuple[Tensor, dict[str, float]]:
    """Sigmoid-DPO objective with implicit-reward diagnostics."""
    # KATA(dpo-loss): contract:
    # 1. Implicit rewards: beta * (policy - reference) per side.
    # 2. Loss: -logsigmoid(chosen_rewards - rejected_rewards).mean().
    # 3. Diagnostics under torch.no_grad(), plain floats, keys
    #    dpo/reward_margin, dpo/reward_accuracy (fraction of positive
    #    margins), dpo/chosen_reward, dpo/rejected_reward.
    raise NotImplementedError("KATA(dpo-loss): implement the DPO objective")


def train_dpo(config: ExperimentConfig) -> Path:
    """Single-process DPO fine-tuning; returns the run directory."""
    if config.dpo is None:
        raise ValueError("dpo configuration is required")
    settings = config.dpo
    device = resolve_device(config.training.device)
    seed_everything(config.training.seed)
    precision = _effective_precision(device, config.training.precision)
    run_dir = config.training.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tokenizer = ByteBPETokenizer.from_files(
        config.data.tokenizer_dir / "vocab.json",
        config.data.tokenizer_dir / "merges.txt",
    )
    dataset = PreferenceDataset(settings.data_path, tokenizer, settings.max_length)
    generator = torch.Generator()
    generator.manual_seed(config.training.seed)
    loader: DataLoader[PreferencePair] = DataLoader(
        dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.data.num_workers,
        collate_fn=lambda batch: collate_preferences(batch, tokenizer.eos_token_id),
    )

    policy = GPT(config.model)
    if settings.base_checkpoint is not None:
        policy.load_state_dict(
            torch.load(
                settings.base_checkpoint / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
    reference = GPT(config.model)
    reference.load_state_dict(policy.state_dict())
    for parameter in reference.parameters():
        parameter.requires_grad = False
    reference.eval()
    if settings.use_lora:
        for parameter in policy.parameters():
            parameter.requires_grad = False
        apply_lora(
            policy,
            rank=settings.lora_rank,
            alpha=settings.lora_alpha,
            dropout=settings.lora_dropout,
            target_modules=settings.lora_targets,
        )
    policy.to(device)
    reference.to(device)

    optimizer = policy.configure_optimizer(
        config.training.learning_rate,
        config.training.weight_decay,
        (config.training.beta1, config.training.beta2),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_learning_rate(
            step,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            max_learning_rate=config.training.learning_rate,
            min_learning_rate=config.training.min_learning_rate,
        )
        / config.training.learning_rate,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")

    accumulation = config.training.gradient_accumulation_steps
    logger = MetricLogger(
        run_dir,
        tensorboard=config.monitoring.tensorboard,
        wandb_project=config.monitoring.wandb_project,
        config=config.model_dump(mode="json"),
    )
    progress = {"step": 0, "tokens": 0, "data_position": 0, "epoch": 0}

    def infinite_batches() -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:
        while True:
            yield from loader

    batches = infinite_batches()
    optimizer.zero_grad(set_to_none=True)
    try:
        while progress["step"] < config.training.max_steps:
            started = time.perf_counter()
            accumulated_loss = 0.0
            step_metrics: dict[str, float] = {}
            for _ in range(accumulation):
                chosen_inputs, chosen_labels, rejected_inputs, rejected_labels = next(batches)
                progress["data_position"] += 1
                chosen_inputs = chosen_inputs.to(device)
                chosen_labels = chosen_labels.to(device)
                rejected_inputs = rejected_inputs.to(device)
                rejected_labels = rejected_labels.to(device)
                with _autocast_context(device, precision):
                    policy_chosen = sequence_logprobs(policy, chosen_inputs, chosen_labels)
                    policy_rejected = sequence_logprobs(policy, rejected_inputs, rejected_labels)
                with torch.no_grad(), _autocast_context(device, precision):
                    reference_chosen = sequence_logprobs(reference, chosen_inputs, chosen_labels)
                    reference_rejected = sequence_logprobs(
                        reference, rejected_inputs, rejected_labels
                    )
                loss, metrics = dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    reference_chosen,
                    reference_rejected,
                    settings.beta,
                )
                loss = loss / accumulation
                scaled: Any = scaler.scale(loss)
                scaled.backward()
                accumulated_loss += float(loss.item())
                step_metrics = metrics
                progress["tokens"] += int(chosen_labels[:, 1:].ne(-100).sum())
            scaler.unscale_(optimizer)
            gradient_norm = float(
                nn.utils.clip_grad_norm_(policy.parameters(), config.training.grad_clip).item()
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress["step"] += 1
            if progress["step"] % config.monitoring.log_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                logger.log(
                    progress["step"],
                    {
                        "dpo/loss": accumulated_loss,
                        "train/learning_rate": float(scheduler.get_last_lr()[0]),
                        "train/gradient_norm": gradient_norm,
                        "train/steps_per_second": 1.0 / elapsed,
                        **step_metrics,
                    },
                )
            if progress["step"] % config.training.checkpoint_every == 0:
                checkpoint_path = run_dir / "checkpoints" / f"step-{progress['step']:08d}"
                save_checkpoint(
                    checkpoint_path,
                    model=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    progress=progress,
                    config=config.model_dump(mode="json"),
                    tokenizer_identity=tokenizer.identity(),
                )
                if settings.use_lora:
                    save_lora(policy, checkpoint_path / "adapter.safetensors")
    finally:
        logger.close()
    return run_dir
