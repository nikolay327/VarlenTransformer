"""Reusable layouts for memory, one volume token, and density queries."""

from dataclasses import dataclass, field, replace
from itertools import accumulate
from collections.abc import Sequence

import torch
from torch import Tensor

from ._validation import positive_int


@dataclass(frozen=True)
class FixedMemoryQueryLayout:
    """Per-sample state is (B, memory_length + 1 + query_length, E)."""

    memory_length: int
    query_length: int
    x_prefix_length: int

    def __post_init__(self):
        for name in ("memory_length", "query_length", "x_prefix_length"):
            positive_int(name, getattr(self, name))
        if self.x_prefix_length > self.memory_length:
            raise ValueError("x_prefix_length cannot exceed memory_length")

    @classmethod
    def ncse(cls, x_prefix_length: int, query_length: int):
        """M=[x; theta_<D], Q has D tokens, with native rectangular causality."""
        positive_int("x_prefix_length", x_prefix_length)
        positive_int("query_length", query_length)
        return cls(x_prefix_length + query_length - 1, query_length, x_prefix_length)

    @property
    def num_memory_tokens(self):
        return self.memory_length

    def validate(self, hidden_states: Tensor, emb_dim: int):
        expected = self.memory_length + 1 + self.query_length
        if (
            hidden_states.ndim != 3
            or hidden_states.shape[0] == 0
            or hidden_states.shape[1:] != (expected, emb_dim)
            or not hidden_states.is_floating_point()
        ):
            raise ValueError(
                f"fixed state must have floating shape (batch, {expected}, {emb_dim})"
            )


@dataclass(frozen=True)
class PackedMemoryQueryLayout:
    """Stream-major state: [all M samples; one V per sample; all Q samples].

    Build once with from_lengths(), then reuse through the stack. Direct
    construction accepts precomputed metadata without reading its values.
    Its offset/index contents are a caller contract, as in the existing MHA.
    """

    num_memory_tokens: int
    num_query_tokens: int
    cu_seqlens_m: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_x: Tensor
    max_seqlen_m: int
    max_seqlen_q: int
    max_seqlen_x: int
    x_prefix_indices: Tensor
    cu_seqlens_v: Tensor = field(init=False, repr=False)

    def __post_init__(self):
        for name in (
            "num_memory_tokens",
            "num_query_tokens",
            "max_seqlen_m",
            "max_seqlen_q",
            "max_seqlen_x",
        ):
            positive_int(name, getattr(self, name))
        offsets = (self.cu_seqlens_m, self.cu_seqlens_q, self.cu_seqlens_x)
        for name, tensor in zip(
            ("cu_seqlens_m", "cu_seqlens_q", "cu_seqlens_x"), offsets
        ):
            if (
                tensor.ndim != 1
                or tensor.numel() < 2
                or tensor.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    f"{name} must be a 1D int32/int64 tensor with at least two offsets"
                )
            object.__setattr__(self, name, tensor.to(torch.int32).contiguous())
        if len({t.numel() for t in offsets}) != 1:
            raise ValueError("all offset arrays must describe the same batch size")
        if any(
            t.device != self.cu_seqlens_m.device
            for t in (*offsets, self.x_prefix_indices)
        ):
            raise ValueError("all packed layout tensors must have the same device")
        if (
            self.x_prefix_indices.ndim != 1
            or self.x_prefix_indices.dtype != torch.int64
            or not self.batch_size
            <= self.x_prefix_indices.numel()
            <= self.num_memory_tokens
        ):
            raise ValueError(
                "x_prefix_indices must be a nonempty 1D int64 index tensor of valid structural size"
            )
        if min(self.num_memory_tokens, self.num_query_tokens) < self.batch_size:
            raise ValueError("every memory/query sequence must be nonempty")
        object.__setattr__(self, "x_prefix_indices", self.x_prefix_indices.contiguous())
        object.__setattr__(
            self,
            "cu_seqlens_v",
            torch.arange(
                self.batch_size + 1, dtype=torch.int32, device=self.cu_seqlens_m.device
            ),
        )

    @property
    def batch_size(self):
        return self.cu_seqlens_m.numel() - 1

    @property
    def total_tokens(self):
        return self.num_memory_tokens + self.batch_size + self.num_query_tokens

    @classmethod
    def from_lengths(
        cls,
        memory_lengths: Sequence[int],
        query_lengths: Sequence[int],
        x_prefix_lengths: Sequence[int],
        *,
        device=None,
    ):
        """Construct static metadata from host Python lengths, never CUDA reads."""
        if any(
            isinstance(lengths, Tensor)
            for lengths in (memory_lengths, query_lengths, x_prefix_lengths)
        ):
            raise ValueError(
                "from_lengths expects host Python sequences; pass precomputed metadata for tensor lengths"
            )
        ml, ql, xl = map(tuple, (memory_lengths, query_lengths, x_prefix_lengths))
        if not ml or len(ml) != len(ql) or len(ml) != len(xl):
            raise ValueError(
                "memory/query/x lengths must have the same nonzero batch size"
            )
        for lengths in (ml, ql, xl):
            for length in lengths:
                positive_int("sequence length", length)
            if sum(lengths) > 2**31 - 1:
                raise ValueError("packed offsets must fit int32")
        if any(x > m for x, m in zip(xl, ml)):
            raise ValueError("each x-prefix length must not exceed its memory length")
        mo, qo, xo = ([0, *accumulate(lengths)] for lengths in (ml, ql, xl))
        offsets = [
            torch.tensor(o, dtype=torch.int32, device=device) for o in (mo, qo, xo)
        ]
        # output_size avoids repeat_interleave's CUDA-to-host size synchronization.
        shifts = torch.tensor(
            [m - x for m, x in zip(mo[:-1], xo[:-1])], device=device, dtype=torch.int64
        )
        indices = torch.arange(sum(xl), device=device) + torch.repeat_interleave(
            shifts, torch.tensor(xl, device=device), output_size=sum(xl)
        )
        return cls(sum(ml), sum(ql), *offsets, max(ml), max(ql), max(xl), indices)

    @classmethod
    def ncse(
        cls,
        x_prefix_lengths: Sequence[int],
        query_lengths: Sequence[int],
        *,
        device=None,
    ):
        if any(
            isinstance(lengths, Tensor) for lengths in (x_prefix_lengths, query_lengths)
        ):
            raise ValueError(
                "ncse expects host Python lengths; use precomputed metadata for tensor lengths"
            )
        xl, ql = tuple(x_prefix_lengths), tuple(query_lengths)
        if len(xl) != len(ql):
            raise ValueError("x/query lengths must describe the same batch size")
        for length in (*xl, *ql):
            positive_int("sequence length", length)
        return cls.from_lengths(
            [x + q - 1 for x, q in zip(xl, ql)], ql, xl, device=device
        )

    def to(self, device):
        return replace(
            self,
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "cu_seqlens_m",
                    "cu_seqlens_q",
                    "cu_seqlens_x",
                    "x_prefix_indices",
                )
            },
        )

    def validate(self, hidden_states: Tensor, emb_dim: int):
        if (
            hidden_states.ndim != 2
            or hidden_states.shape != (self.total_tokens, emb_dim)
            or not hidden_states.is_floating_point()
        ):
            raise ValueError(
                f"packed state must have floating shape ({self.total_tokens}, {emb_dim})"
            )
        if hidden_states.device != self.cu_seqlens_m.device:
            raise ValueError(
                "packed metadata and hidden states must have the same device"
            )
