"""Validated configuration models shared by library and CLI workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    """Reject unknown fields so misspelled experiment settings fail immediately."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictConfig):
    vocab_size: int = Field(default=50_257, ge=1, le=65_535)
    context_length: int = Field(default=1_024, ge=2)
    d_model: int = Field(default=768, ge=8)
    num_layers: int = Field(default=12, ge=1)
    num_heads: int = Field(default=12, ge=1)
    num_kv_heads: int = Field(default=4, ge=1)
    mlp_ratio: float = Field(default=4.0, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    layer_norm_epsilon: float = Field(default=1e-5, gt=0)
    rope_theta: float = Field(default=10_000.0, gt=0)
    initializer_range: float = Field(default=0.02, gt=0)
    bias: bool = True
    gradient_checkpointing: bool = False
    # Ablation switches for comparison labs. Defaults preserve the project's
    # reference architecture: RoPE + pre-norm (+ GQA via num_kv_heads).
    position_encoding: Literal["rope", "learned"] = "rope"
    norm_placement: Literal["pre", "post"] = "pre"

    @model_validator(mode="after")
    def validate_geometry(self) -> ModelConfig:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.position_encoding == "rope" and self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        return self

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def query_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def mlp_hidden_size(self) -> int:
        return int(self.d_model * self.mlp_ratio)


_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"d_model": 128, "num_layers": 4, "num_heads": 4, "num_kv_heads": 2},
    "gpt2-small": {"d_model": 768, "num_layers": 12, "num_heads": 12, "num_kv_heads": 4},
    "gpt2-medium": {"d_model": 1024, "num_layers": 24, "num_heads": 16, "num_kv_heads": 4},
    "gpt2-large": {"d_model": 1280, "num_layers": 36, "num_heads": 20, "num_kv_heads": 4},
    "gpt2-xl": {"d_model": 1600, "num_layers": 48, "num_heads": 25, "num_kv_heads": 5},
}


def model_preset(name: str, **overrides: object) -> ModelConfig:
    """Build a named GPT-2-sized configuration with optional validated overrides."""
    try:
        values: dict[str, object] = dict(_PRESETS[name])
    except KeyError as error:
        choices = sorted(_PRESETS)
        raise ValueError(f"unknown model preset {name!r}; choose from {choices}") from error
    values.update(overrides)
    return ModelConfig.model_validate(values)


class DataConfig(StrictConfig):
    train_path: Path
    validation_path: Path | None = None
    tokenizer_dir: Path
    sequence_length: int = Field(default=1_024, ge=2)
    num_workers: int = Field(default=0, ge=0)


class MonitoringConfig(StrictConfig):
    log_every: int = Field(default=10, ge=1)
    tensorboard: bool = True
    wandb_project: str | None = None
    profile_every: int | None = Field(default=None, ge=1)
    cuda_memory_snapshot: bool = False


class TrainingConfig(StrictConfig):
    output_dir: Path = Path("runs/default")
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    seed: int = 1337
    micro_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    max_steps: int = Field(default=100_000, ge=1)
    learning_rate: float = Field(default=6e-4, gt=0)
    min_learning_rate: float = Field(default=6e-5, ge=0)
    warmup_steps: int = Field(default=2_000, ge=0)
    weight_decay: float = Field(default=0.1, ge=0)
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.95, ge=0, lt=1)
    grad_clip: float = Field(default=1.0, gt=0)
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    compile: bool = False
    eval_every: int = Field(default=500, ge=1)
    eval_batches: int = Field(default=50, ge=1)
    checkpoint_every: int = Field(default=1_000, ge=1)
    resume_from: Path | None = None

    @model_validator(mode="after")
    def validate_schedule(self) -> TrainingConfig:
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        return self


class FineTuningConfig(StrictConfig):
    data_path: Path
    validation_path: Path | None = None
    base_checkpoint: Path | None = None
    max_length: int = Field(default=1_024, ge=2)
    use_lora: bool = False
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: float = Field(default=16.0, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0, lt=1)
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj", "fc", "proj")


class DPOConfig(StrictConfig):
    data_path: Path
    validation_path: Path | None = None
    base_checkpoint: Path | None = None
    max_length: int = Field(default=1_024, ge=2)
    beta: float = Field(default=0.1, gt=0)
    use_lora: bool = False
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: float = Field(default=16.0, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0, lt=1)
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj", "fc", "proj")


class GenerationConfig(StrictConfig):
    max_new_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    repetition_penalty: float = Field(default=1.0, gt=0)
    seed: int = 1337
    eos_token_id: int | None = Field(default=None, ge=0)


class ProfilingConfig(StrictConfig):
    wait: int = Field(default=1, ge=0)
    warmup: int = Field(default=1, ge=0)
    active: int = Field(default=3, ge=1)
    repeat: int = Field(default=1, ge=1)
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False


class ExperimentConfig(StrictConfig):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = TrainingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    finetuning: FineTuningConfig | None = None
    dpo: DPOConfig | None = None
