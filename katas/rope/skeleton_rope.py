"""Rotary position embeddings (RoPE) for attention queries and keys."""

from __future__ import annotations

from torch import Tensor, nn

# KATA(rope): imports used only by the removed bodies were dropped.
# Re-import what your implementation needs (you will at least need torch).


def rotate_half(x: Tensor) -> Tensor:
    """Rotate adjacent feature pairs by 90 degrees in their 2D subspaces."""
    # KATA(rope): map (x0, x1) -> (-x1, x0) for every adjacent pair along the
    # last dimension, preserving shape. Even indices are the first element of
    # each pair, odd indices the second.
    raise NotImplementedError("KATA(rope): implement rotate_half")


class RotaryEmbedding(nn.Module):
    """Precompute trigonometric tables while keeping them out of checkpoints."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        # KATA(rope): contract for construction time:
        # 1. Reject odd head_dim with ValueError (pairs need an even D).
        # 2. Build inverse frequencies 1 / base**(2i/D) for i in 0..D/2-1
        #    in FP32, take the outer product with positions 0..L-1, and
        #    repeat each angle twice along the feature axis to align with
        #    rotate_half's even/odd pairing -> tables of shape [L, D].
        # 3. Register cos/sin tables so they follow .to(device) but never
        #    appear in state_dict() (the tests assert it is empty).
        # 4. Store max_position_embeddings for the bounds check in forward.
        raise NotImplementedError("KATA(rope): implement table construction")

    def forward(self, query: Tensor, key: Tensor, offset: int = 0) -> tuple[Tensor, Tensor]:
        """Apply positions ``offset:offset+sequence`` to ``[B, H, T, D]`` tensors."""
        # KATA(rope): contract at call time:
        # 1. Raise ValueError when offset + T exceeds the configured maximum.
        # 2. Slice rows offset:offset+T from both tables, cast to the query's
        #    dtype/device, and view as [1, 1, T, D] so one table broadcasts
        #    over batch and over differing Q/K head counts.
        # 3. Return (rot(query), rot(key)) using
        #    rot(x) = x * cos + rotate_half(x) * sin.
        raise NotImplementedError("KATA(rope): implement the rotation")
