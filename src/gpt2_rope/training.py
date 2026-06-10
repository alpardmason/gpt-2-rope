"""Single-device and single-node DDP training, evaluation, and profiling."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import pickle
import random
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as distributed
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader, DistributedSampler

from gpt2_rope.checkpoint import load_checkpoint, save_checkpoint
from gpt2_rope.config import ExperimentConfig, ProfilingConfig
from gpt2_rope.data import MemmapTokenDataset, SFTDataset, collate_sft
from gpt2_rope.lora import apply_lora, save_lora
from gpt2_rope.model import GPT
from gpt2_rope.monitoring import MetricLogger
from gpt2_rope.tokenizer import ByteBPETokenizer

LOGGER = logging.getLogger("gpt2_rope")


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, rank: int = 0) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cosine_learning_rate(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    max_learning_rate: float,
    min_learning_rate: float,
) -> float:
    if warmup_steps and step < warmup_steps:
        return max_learning_rate * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_learning_rate
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coefficient = 0.5 * (1 + math.cos(math.pi * ratio))
    return min_learning_rate + coefficient * (max_learning_rate - min_learning_rate)


def _autocast_context(
    device: torch.device,
    precision: str,
) -> contextlib.AbstractContextManager[Any]:
    if precision == "fp32" or device.type == "cpu":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _effective_precision(device: torch.device, requested: str) -> str:
    if requested != "auto":
        return requested
    if device.type == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if device.type == "mps":
        return "fp16"
    return "fp32"


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    batches: int,
    precision: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    tokens = 0
    start = time.perf_counter()
    for batch_index, (input_ids, _targets) in enumerate(loader):
        if batch_index >= batches:
            break
        input_ids = input_ids.to(device)
        with _autocast_context(device, precision):
            output = model(input_ids, labels=input_ids, use_cache=False)
        if output.loss is None:
            raise RuntimeError("evaluation model did not return loss")
        labels = input_ids[:, 1:]
        predictions = output.logits[:, :-1].argmax(dim=-1)
        count = labels.numel()
        total_loss += output.loss.item() * count
        correct += int((predictions == labels).sum())
        tokens += count
    elapsed = max(time.perf_counter() - start, 1e-9)
    mean_loss = total_loss / max(tokens, 1)
    return {
        "validation/loss": mean_loss,
        "validation/perplexity": math.exp(min(mean_loss, 20)),
        "validation/token_accuracy": correct / max(tokens, 1),
        "validation/tokens_per_second": tokens / elapsed,
    }


@torch.inference_mode()
def evaluate_sft(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    batches: int,
    precision: str,
) -> dict[str, float]:
    """Masked-label validation: only response tokens contribute to the loss."""
    model.eval()
    total_loss = 0.0
    correct = 0
    tokens = 0
    for batch_index, (input_ids, labels) in enumerate(loader):
        if batch_index >= batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with _autocast_context(device, precision):
            output = model(input_ids, labels=labels, use_cache=False)
        if output.loss is None:
            raise RuntimeError("evaluation model did not return loss")
        shifted = labels[:, 1:]
        supervised = shifted.ne(-100)
        count = int(supervised.sum())
        predictions = output.logits[:, :-1].argmax(dim=-1)
        total_loss += output.loss.item() * count
        correct += int((predictions.eq(shifted) & supervised).sum())
        tokens += count
    mean_loss = total_loss / max(tokens, 1)
    return {
        "validation/loss": mean_loss,
        "validation/perplexity": math.exp(min(mean_loss, 20)),
        "validation/token_accuracy": correct / max(tokens, 1),
    }


def _maybe_profile_step(
    run_dir: Path,
    next_step: int,
    profile_every: int | None,
    device: torch.device,
    enabled: bool,
) -> contextlib.AbstractContextManager[Any]:
    """Profile one full optimizer step every ``profile_every`` steps on rank 0."""
    if not enabled or profile_every is None or next_step % profile_every:
        return contextlib.nullcontext()
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return profile(
        activities=activities,
        on_trace_ready=tensorboard_trace_handler(str(run_dir / "profiler")),
    )


def _infinite_loader(
    loader: DataLoader[tuple[Tensor, Tensor]],
    sampler: DistributedSampler[Any] | None,
    start_epoch: int = 0,
) -> Iterator[tuple[Tensor, Tensor]]:
    epoch = start_epoch
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def train_pretraining(config: ExperimentConfig) -> Path:
    """Run token-based causal LM pretraining and return the run directory."""
    rank, world_size, local_rank = initialize_distributed()
    is_primary = rank == 0
    requested_device = config.training.device
    device = (
        torch.device("cuda", local_rank)
        if world_size > 1 and torch.cuda.is_available()
        else resolve_device(requested_device)
    )
    seed_everything(config.training.seed, rank)
    precision = _effective_precision(device, config.training.precision)
    run_dir = config.training.output_dir
    if is_primary:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    dataset = MemmapTokenDataset(config.data.train_path, config.data.sequence_length)
    sampler: DistributedSampler[Any] = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.training.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.micro_batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    validation_loader = None
    if config.data.validation_path is not None and config.data.validation_path.stat().st_size:
        validation_loader = DataLoader(
            MemmapTokenDataset(config.data.validation_path, config.data.sequence_length),
            batch_size=config.training.micro_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

    raw_model = GPT(config.model).to(device)
    tokenizer = ByteBPETokenizer.from_files(
        config.data.tokenizer_dir / "vocab.json",
        config.data.tokenizer_dir / "merges.txt",
    )
    optimizer = raw_model.configure_optimizer(
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
    scaler = GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision == "fp16",
    )
    progress = {"step": 0, "tokens": 0, "data_position": 0, "epoch": 0}
    if config.training.resume_from is not None:
        restored = load_checkpoint(
            config.training.resume_from,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        progress.update(restored.progress)

    model: Any = raw_model
    if config.training.compile:
        model = torch.compile(model)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    stream = _infinite_loader(loader, sampler, progress["epoch"])
    for _ in range(progress["data_position"]):
        next(stream)

    accumulation = config.training.gradient_accumulation_steps
    logger = MetricLogger(
        run_dir,
        tensorboard=config.monitoring.tensorboard,
        wandb_project=config.monitoring.wandb_project,
        config=config.model_dump(mode="json"),
        enabled=is_primary,
    )
    optimizer.zero_grad(set_to_none=True)
    try:
        while progress["step"] < config.training.max_steps:
            step_start = time.perf_counter()
            accumulated_loss = 0.0
            step_tokens = 0
            profiler_context = _maybe_profile_step(
                run_dir,
                progress["step"] + 1,
                config.monitoring.profile_every,
                device,
                is_primary,
            )
            with profiler_context:
                for micro_step in range(accumulation):
                    input_ids, _targets = next(stream)
                    progress["data_position"] += 1
                    input_ids = input_ids.to(device, non_blocking=True)
                    sync_context = (
                        model.no_sync()
                        if isinstance(model, DistributedDataParallel)
                        and micro_step < accumulation - 1
                        else contextlib.nullcontext()
                    )
                    with sync_context, _autocast_context(device, precision):
                        output = model(input_ids, labels=input_ids, use_cache=False)
                        if output.loss is None:
                            raise RuntimeError("training model did not return loss")
                        loss = output.loss / accumulation
                    scaler.scale(loss).backward()
                    accumulated_loss += float(loss.item())
                    step_tokens += input_ids.numel() * world_size
            scaler.unscale_(optimizer)
            gradient_norm = float(
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.grad_clip,
                ).item()
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress["step"] += 1
            progress["tokens"] += step_tokens

            if is_primary and progress["step"] % config.monitoring.log_every == 0:
                elapsed = max(time.perf_counter() - step_start, 1e-9)
                memory = (
                    torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda"
                    else 0.0
                )
                logger.log(
                    progress["step"],
                    {
                        "train/loss": accumulated_loss,
                        "train/perplexity": math.exp(min(accumulated_loss, 20)),
                        "train/learning_rate": float(scheduler.get_last_lr()[0]),
                        "train/gradient_norm": gradient_norm,
                        "train/tokens_per_second": step_tokens / elapsed,
                        "system/peak_memory_mib": memory,
                        "progress/tokens": progress["tokens"],
                    },
                )
            if (
                is_primary
                and validation_loader is not None
                and progress["step"] % config.training.eval_every == 0
            ):
                logger.log(
                    progress["step"],
                    evaluate(
                        raw_model,
                        validation_loader,
                        device,
                        config.training.eval_batches,
                        precision,
                    ),
                )
                model.train()
            if is_primary and progress["step"] % config.training.checkpoint_every == 0:
                save_checkpoint(
                    run_dir / "checkpoints" / f"step-{progress['step']:08d}",
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    progress=progress,
                    config=config.model_dump(mode="json"),
                    tokenizer_identity=tokenizer.identity(),
                )
                if config.monitoring.cuda_memory_snapshot and device.type == "cuda":
                    try:
                        snapshot: Any = torch.cuda.memory_snapshot()  # type: ignore[no-untyped-call]
                        snapshot_path = run_dir / "cuda-memory"
                        snapshot_path.mkdir(parents=True, exist_ok=True)
                        with (snapshot_path / f"step-{progress['step']:08d}.pickle").open(
                            "wb"
                        ) as snapshot_file:
                            pickle.dump(snapshot, snapshot_file)
                    except Exception:
                        LOGGER.exception("CUDA memory snapshot failed; training continues")
    finally:
        logger.close()
        if distributed.is_initialized():
            distributed.destroy_process_group()
    return run_dir


def train_finetuning(config: ExperimentConfig) -> Path:
    """Run full-parameter or LoRA supervised fine-tuning on prompt/response JSONL."""
    if config.finetuning is None:
        raise ValueError("finetuning configuration is required")
    rank, world_size, local_rank = initialize_distributed()
    is_primary = rank == 0
    device = (
        torch.device("cuda", local_rank)
        if world_size > 1 and torch.cuda.is_available()
        else resolve_device(config.training.device)
    )
    seed_everything(config.training.seed, rank)
    precision = _effective_precision(device, config.training.precision)
    run_dir = config.training.output_dir
    if is_primary:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if distributed.is_initialized():
        distributed.barrier()

    tokenizer = ByteBPETokenizer.from_files(
        config.data.tokenizer_dir / "vocab.json",
        config.data.tokenizer_dir / "merges.txt",
    )
    dataset = SFTDataset(
        config.finetuning.data_path,
        tokenizer,
        config.finetuning.max_length,
    )
    sampler: DistributedSampler[Any] = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.training.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.micro_batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=partial(collate_sft, pad_token_id=tokenizer.eos_token_id),
    )
    validation_loader = None
    if config.finetuning.validation_path is not None:
        validation_loader = DataLoader(
            SFTDataset(
                config.finetuning.validation_path,
                tokenizer,
                config.finetuning.max_length,
            ),
            batch_size=config.training.micro_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=partial(collate_sft, pad_token_id=tokenizer.eos_token_id),
        )
    raw_model = GPT(config.model)
    if config.finetuning.base_checkpoint is not None:
        raw_model.load_state_dict(
            torch.load(
                config.finetuning.base_checkpoint / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
    if config.finetuning.use_lora:
        for parameter in raw_model.parameters():
            parameter.requires_grad = False
        apply_lora(
            raw_model,
            rank=config.finetuning.lora_rank,
            alpha=config.finetuning.lora_alpha,
            dropout=config.finetuning.lora_dropout,
            target_modules=config.finetuning.lora_targets,
        )
    raw_model.to(device)
    optimizer = raw_model.configure_optimizer(
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
    scaler = GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision == "fp16",
    )
    progress = {"step": 0, "tokens": 0, "data_position": 0, "epoch": 0}
    if config.training.resume_from is not None:
        restored = load_checkpoint(
            config.training.resume_from,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        progress.update(restored.progress)
    model: Any = raw_model
    if config.training.compile:
        model = torch.compile(model)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    stream = _infinite_loader(loader, sampler)
    for _ in range(progress["data_position"]):
        next(stream)

    accumulation = config.training.gradient_accumulation_steps
    logger = MetricLogger(
        run_dir,
        tensorboard=config.monitoring.tensorboard,
        wandb_project=config.monitoring.wandb_project,
        config=config.model_dump(mode="json"),
        enabled=is_primary,
    )
    optimizer.zero_grad(set_to_none=True)
    try:
        while progress["step"] < config.training.max_steps:
            started = time.perf_counter()
            accumulated_loss = 0.0
            step_tokens = 0
            for micro_step in range(accumulation):
                input_ids, labels = next(stream)
                progress["data_position"] += 1
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                sync_context = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel)
                    and micro_step < accumulation - 1
                    else contextlib.nullcontext()
                )
                with sync_context, _autocast_context(device, precision):
                    output = model(input_ids, labels=labels, use_cache=False)
                    if output.loss is None:
                        raise RuntimeError("fine-tuning model did not return loss")
                    loss = output.loss / accumulation
                scaler.scale(loss).backward()
                accumulated_loss += float(loss.item())
                step_tokens += int(labels[:, 1:].ne(-100).sum()) * world_size
            scaler.unscale_(optimizer)
            gradient_norm = float(
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.grad_clip,
                ).item()
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress["step"] += 1
            progress["tokens"] += step_tokens
            if is_primary and progress["step"] % config.monitoring.log_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                logger.log(
                    progress["step"],
                    {
                        "train/loss": accumulated_loss,
                        "train/perplexity": math.exp(min(accumulated_loss, 20)),
                        "train/learning_rate": float(scheduler.get_last_lr()[0]),
                        "train/gradient_norm": gradient_norm,
                        "train/supervised_tokens_per_second": step_tokens / elapsed,
                    },
                )
            if (
                is_primary
                and validation_loader is not None
                and progress["step"] % config.training.eval_every == 0
            ):
                logger.log(
                    progress["step"],
                    evaluate_sft(
                        raw_model,
                        validation_loader,
                        device,
                        config.training.eval_batches,
                        precision,
                    ),
                )
                model.train()
            if is_primary and progress["step"] % config.training.checkpoint_every == 0:
                checkpoint_path = run_dir / "checkpoints" / f"step-{progress['step']:08d}"
                save_checkpoint(
                    checkpoint_path,
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    progress=progress,
                    config=config.model_dump(mode="json"),
                    tokenizer_identity=tokenizer.identity(),
                )
                if config.finetuning.use_lora:
                    save_lora(raw_model, checkpoint_path / "adapter.safetensors")
    finally:
        logger.close()
        if distributed.is_initialized():
            distributed.destroy_process_group()
    return run_dir


def run_profiler(
    work: Callable[[], None],
    output_dir: Path,
    config: ProfilingConfig,
    *,
    use_cuda: bool,
) -> None:
    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)
    output_dir.mkdir(parents=True, exist_ok=True)
    with profile(
        activities=activities,
        schedule=schedule(
            wait=config.wait,
            warmup=config.warmup,
            active=config.active,
            repeat=config.repeat,
        ),
        on_trace_ready=tensorboard_trace_handler(str(output_dir)),
        record_shapes=config.record_shapes,
        profile_memory=config.profile_memory,
        with_stack=config.with_stack,
    ) as profiler:
        total = (config.wait + config.warmup + config.active) * config.repeat
        for _ in range(total):
            work()
            profiler.step()
