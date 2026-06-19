"""Rotary position embeddings (RoPE) for attention queries and keys."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent feature pairs by 90 degrees in their 2D subspaces."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    #    stack odd and even dims -> flatten (merge) odd and even dims.
    # 2D: [[-a1, a0], [-a3, a2]] -> [-a1, a0, -a3, a2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


# [NOTE] RoPE does not have any learnable parameters. So, it can be shared by all attention layers.
class RotaryEmbedding(nn.Module):
    """Precompute trigonometric tables while keeping them out of checkpoints."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        # 10000^{-2i/d}, i = 0, 2, 4, ..., d/2 - 1
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # 0, ..., m-1
        positions = torch.arange(0, max_position_embeddings, dtype=torch.float32)
        # f \in R^{m * d/2}, [f]_{i, j} = i * 10000^{-2j/d}
        frequencies = torch.outer(positions, inverse_frequency)
        # f: m * d/2 -> \theta: m * d
        angles = torch.repeat_interleave(frequencies, 2, dim=-1)
        self.cos_cached: Tensor
        self.sin_cached: Tensor
        # register as part of the model's state, but not as learnable params
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, query: Tensor, key: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        """Apply positions ``offset:offset+sequence`` to ``[B, H, T, D]`` tensors."""
        sequence_length = query.size(-2)
        # offset will be used for KV cache
        end = offset + sequence_length
        if end > self.max_position_embeddings:
            raise ValueError(
                f"position {end} exceeds configured context {self.max_position_embeddings}"
            )
        cos = self.cos_cached[offset:end].to(dtype=query.dtype, device=query.device)
        sin = self.sin_cached[offset:end].to(dtype=query.dtype, device=query.device)
        cos = cos.view(1, 1, sequence_length, -1)
        sin = sin.view(1, 1, sequence_length, -1)
        # 2D: [q_0, q_1] * [cos, cos] + [-q_1, q_0] * [sin, sin]
        # [q_0 * cost - q_1 * sin, q_1 * cos + q_0 * sin]
        return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin
