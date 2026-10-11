"""Convert per-sample [M;V;Q] storage around a fixed block stack."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .layout import FixedMemoryQueryLayout
from .memory_query import MemoryQueryBlock


def pack_fixed_memory_query(state: Tensor, layout: FixedMemoryQueryLayout) -> Tensor:
    """Convert (B,Sm+1+Sq,E) to flat [all M;all V;all Q], preserving gradients."""
    sm = layout.memory_length
    e = state.shape[-1]
    return torch.cat(
        (state[:, :sm].reshape(-1, e), state[:, sm], state[:, sm + 1 :].reshape(-1, e)),
        dim=0,
    )


def unpack_fixed_memory_query(state: Tensor, layout: FixedMemoryQueryLayout) -> Tensor:
    """Convert flat stream-major state to (B,Sm+1+Sq,E), preserving gradients."""
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

    def validate_inputs(
        self,
        hidden_states: Tensor,
        residual: Tensor | None = None,
        *,
        layout: FixedMemoryQueryLayout,
    ) -> None:
        """Check per-sample state and residual against the first block and layout."""
        if not isinstance(layout, FixedMemoryQueryLayout):
            raise ValueError("fixed stack requires FixedMemoryQueryLayout")
        layout.validate_per_sample(hidden_states, self.blocks[0].memory_mixer.emb_dim)
        if residual is not None and (
            residual.shape != hidden_states.shape
            or residual.device != hidden_states.device
            or not residual.is_floating_point()
        ):
            raise ValueError("residual must match the floating hidden-state shape and device")

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor | None = None,
        *,
        layout: FixedMemoryQueryLayout,
        return_stream_major: bool = False,
    ) -> tuple[Tensor, Tensor]:
        h = pack_fixed_memory_query(hidden_states, layout)
        r = None if residual is None else pack_fixed_memory_query(residual, layout)
        for block in self.blocks:
            h, r = block(h, r, layout=layout)
        if return_stream_major:
            return h, r
        return unpack_fixed_memory_query(h, layout), unpack_fixed_memory_query(
            r, layout
        )
