"""Stack-level adapter for legacy fixed per-sample [M;V;Q] storage."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .layout import FixedMemoryQueryLayout
from .memory_query import MemoryQueryBlock


def pack_fixed_memory_query(state: Tensor, layout: FixedMemoryQueryLayout) -> Tensor:
    """Convert (B,Sm+1+Sq,E) to flat [all M;all V;all Q], preserving gradients."""
    if not isinstance(layout, FixedMemoryQueryLayout):
        raise ValueError("fixed packing requires FixedMemoryQueryLayout")
    sm, sq, b = layout.memory_length, layout.query_length, layout.batch_size
    if (
        state.ndim != 3
        or state.shape[:2] != (b, sm + 1 + sq)
        or not state.is_floating_point()
        or state.shape[-1] == 0
    ):
        raise ValueError(
            f"per-sample state must have floating shape ({b},{sm + 1 + sq},E)"
        )
    e = state.shape[-1]
    return torch.cat(
        (state[:, :sm].reshape(-1, e), state[:, sm], state[:, sm + 1 :].reshape(-1, e)),
        dim=0,
    )


def unpack_fixed_memory_query(state: Tensor, layout: FixedMemoryQueryLayout) -> Tensor:
    """Convert flat stream-major state to (B,Sm+1+Sq,E), preserving gradients."""
    if not isinstance(layout, FixedMemoryQueryLayout):
        raise ValueError("fixed unpacking requires FixedMemoryQueryLayout")
    if state.ndim != 2:
        raise ValueError("unpacking requires a flat stream-major tensor")
    layout.validate(state, state.shape[-1])
    n, b, e = layout.num_memory_tokens, layout.batch_size, state.shape[-1]
    return torch.cat(
        (
            state[:n].view(b, layout.memory_length, e),
            state[n : n + b].view(b, 1, e),
            state[n + b :].view(b, layout.query_length, e),
        ),
        dim=1,
    )


class FixedMemoryQueryStack(nn.Module):
    """Adapt per-sample fixed input once around an entire native block stack.

    Pass an iterable of MemoryQueryBlock instances. The usual result is two
    (B,S,E) tensors; return_stream_major=True keeps both results flat. Chain all
    layers inside one adapter to avoid conversions between individual blocks.
    """

    def __init__(self, blocks: Iterable[MemoryQueryBlock]):
        super().__init__()
        blocks = list(blocks)
        if not blocks or any(
            not isinstance(block, MemoryQueryBlock) for block in blocks
        ):
            raise ValueError("blocks must contain at least one MemoryQueryBlock")
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor | None = None,
        *,
        layout: FixedMemoryQueryLayout,
        return_stream_major: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if residual is not None and (
            residual.shape != hidden_states.shape
            or residual.device != hidden_states.device
        ):
            raise ValueError("residual must match hidden states in shape and device")
        h = pack_fixed_memory_query(hidden_states, layout)
        r = None if residual is None else pack_fixed_memory_query(residual, layout)
        for block in self.blocks:
            h, r = block(h, r, layout=layout)
        if return_stream_major:
            return h, r
        return unpack_fixed_memory_query(h, layout), unpack_fixed_memory_query(
            r, layout
        )
