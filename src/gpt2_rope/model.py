"""GPT-2 decoder blocks with grouped-query attention and RoPE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from gpt2_rope.config import ModelConfig
from gpt2_rope.rope import RotaryEmbedding

KVCache = tuple[Tensor, Tensor]


@dataclass(slots=True)
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None
    past_key_values: tuple[KVCache, ...]


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with fewer key/value heads than query heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        kv_width = config.num_kv_heads * config.head_dim
        self.k_proj = nn.Linear(config.d_model, kv_width, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, kv_width, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.rope: RotaryEmbedding | None = None
        if config.position_encoding == "rope":
            self.rope = RotaryEmbedding(
                config.head_dim,
                config.context_length,
                base=config.rope_theta,
            )

    def _shape(self, tensor: Tensor, heads: int) -> Tensor:
        batch, sequence, _ = tensor.shape
        return tensor.view(batch, sequence, heads, self.config.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, KVCache]:
        batch, query_length, _ = hidden_states.shape
        past_length = 0 if past_key_value is None else past_key_value[0].size(-2)
        query = self._shape(self.q_proj(hidden_states), self.config.num_heads)
        key = self._shape(self.k_proj(hidden_states), self.config.num_kv_heads)
        value = self._shape(self.v_proj(hidden_states), self.config.num_kv_heads)
        if self.rope is not None:
            query, key = self.rope(query, key, offset=past_length)

        if past_key_value is not None:
            # Cache stays compact as [B, H_kv, T, D]; expansion is only a view/copy
            # at the attention kernel boundary on backends without native GQA.
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)
        present = (key, value)

        enable_gqa = self.config.query_groups > 1 and query.device.type == "cuda"
        if self.config.query_groups > 1 and not enable_gqa:
            key_for_attention = key.repeat_interleave(self.config.query_groups, dim=1)
            value_for_attention = value.repeat_interleave(self.config.query_groups, dim=1)
        else:
            key_for_attention = key
            value_for_attention = value

        key_length = key_for_attention.size(-2)
        if past_length == 0:
            attention_mask = None
            is_causal = query_length > 1
        else:
            # SDPA's built-in causal mask is upper-left aligned. Cached decoding
            # needs an offset mask so every new query can see the complete prefix.
            query_positions = past_length + torch.arange(query_length, device=query.device)
            key_positions = torch.arange(key_length, device=query.device)
            attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            attention_mask = attention_mask.view(1, 1, query_length, key_length)
            is_causal = False

        attention = functional.scaled_dot_product_attention(
            query,
            key_for_attention,
            value_for_attention,
            attn_mask=attention_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, query_length, -1)
        output = self.residual_dropout(self.out_proj(attention))
        return output, present if use_cache else (key[:, :, :0], value[:, :, :0])


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.mlp_hidden_size, bias=config.bias)
        self.proj = nn.Linear(config.mlp_hidden_size, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = self.dropout(
            self.proj(functional.gelu(self.fc(hidden_states), approximate="tanh"))
        )
        return cast(Tensor, output)


class TransformerBlock(nn.Module):
    """GPT-2 pre-norm block by default; post-norm is an explicit ablation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_placement = config.norm_placement
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attention = GroupedQueryAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, KVCache]:
        if self.norm_placement == "pre":
            attention_output, present = self.attention(
                self.ln_1(hidden_states),
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            hidden_states = hidden_states + attention_output
            hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        else:
            # Original-Transformer/GPT-1 ordering: normalize after each residual
            # sum. Kept only for ablation labs on training stability at depth.
            attention_output, present = self.attention(
                hidden_states,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            hidden_states = self.ln_1(hidden_states + attention_output)
            hidden_states = self.ln_2(hidden_states + self.mlp(hidden_states))
        return hidden_states, present


class GPT(nn.Module):
    """Decoder-only language model preserving GPT-2's non-positional components."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding: nn.Embedding | None = None
        if config.position_encoding == "learned":
            # Original GPT-2 wpe table; the ablation alternative to RoPE.
            self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        self._scale_residual_projections()
        self.lm_head.weight = self.token_embedding.weight

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        # GPT-2 scales residual output weights to keep variance stable with depth.
        std = self.config.initializer_range / math.sqrt(2 * self.config.num_layers)
        for module in self.blocks:
            block = cast(TransformerBlock, module)
            nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        past_key_values: tuple[KVCache, ...] | None = None,
        use_cache: bool = True,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        if past_length + input_ids.size(1) > self.config.context_length:
            raise ValueError("input sequence and cache exceed configured context length")
        if self.config.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        hidden_states = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = past_length + torch.arange(input_ids.size(1), device=input_ids.device)
            hidden_states = hidden_states + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        presents: list[KVCache] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            if self.config.gradient_checkpointing and self.training:
                hidden_states, present = cast(
                    tuple[Tensor, KVCache],
                    checkpoint(
                        block,
                        hidden_states,
                        past,
                        False,
                        use_reentrant=False,
                    ),
                )
            else:
                hidden_states, present = block(hidden_states, past, use_cache)
            if use_cache:
                presents.append(present)

        logits = self.lm_head(self.final_layer_norm(hidden_states))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            # GPT-2 convention: the logit at position t predicts label t+1.
            loss = functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss, past_key_values=tuple(presents))

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def estimated_parameter_memory_bytes(self, bytes_per_parameter: int = 4) -> int:
        return self.parameter_count() * bytes_per_parameter

    def configure_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.AdamW:
        decay: list[Tensor] = []
        no_decay: list[Tensor] = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
            fused=torch.cuda.is_available(),
        )
