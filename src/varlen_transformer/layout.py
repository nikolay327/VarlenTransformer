"""Reusable layouts for memory, one volume token, and density queries."""

from dataclasses import dataclass, field, replace
from itertools import accumulate
from collections.abc import Sequence

import torch
from torch import Tensor

from ._validation import positive_int


@dataclass(frozen=True)
class FixedMemoryQueryLayout:
    """Flat [all M; all V; all Q] state with uniform fixed-kernel geometry."""

    memory_length: int
    query_length: int
    x_prefix_length: int
    batch_size: int = field(kw_only=True)

    def __post_init__(self):
        for name in ("memory_length", "query_length", "x_prefix_length", "batch_size"):
            positive_int(name, getattr(self, name))
        if self.x_prefix_length > self.memory_length:
            raise ValueError("x_prefix_length cannot exceed memory_length")
        if self.memory_length != self.x_prefix_length + self.query_length - 1:
            raise ValueError(
                "NCSE layout requires memory_length = x_prefix_length + query_length - 1"
            )

    @classmethod
    def ncse(cls, x_prefix_length: int, query_length: int, *, batch_size: int):
        """M=[x; theta_<D], Q has D tokens, with native rectangular causality."""
        positive_int("x_prefix_length", x_prefix_length)
        positive_int("query_length", query_length)
        return cls(
            x_prefix_length + query_length - 1,
            query_length,
            x_prefix_length,
            batch_size=batch_size,
        )

    @property
    def num_memory_tokens(self):
        return self.batch_size * self.memory_length

    @property
    def num_query_tokens(self):
        return self.batch_size * self.query_length

    @property
    def total_tokens(self):
        return self.num_memory_tokens + self.batch_size + self.num_query_tokens

    def validate(self, hidden_states: Tensor, emb_dim: int):
        expected = self.total_tokens
        if (
            hidden_states.ndim != 2
            or hidden_states.shape != (expected, emb_dim)
            or not hidden_states.is_floating_point()
        ):
            raise ValueError(
                f"native fixed state must have floating stream-major shape ({expected}, {emb_dim}); "
                "use FixedMemoryQueryStack for per-sample (B,S,E) input"
            )


@dataclass(frozen=True)
class PackedMemoryQueryLayout:
    """Stream-major state: [all M samples; one V per sample; all Q samples].

    Build once with from_lengths(), then reuse through the stack. Direct
    construction verifies CPU offset/index values. Device-resident values require
    explicit trust_metadata=True; use from_lengths() for verified construction.
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
    trust_metadata: bool = field(default=False, kw_only=True)
    metadata_verified: bool = field(default=False, init=False)
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
        if not isinstance(self.trust_metadata, bool):
            raise ValueError("trust_metadata must be an explicit boolean")
        if max(self.num_memory_tokens, self.num_query_tokens) > 2**31 - 1:
            raise ValueError("packed stream totals must fit int32")
        offsets = (self.cu_seqlens_m, self.cu_seqlens_q, self.cu_seqlens_x)
        for name, tensor in zip(
            ("cu_seqlens_m", "cu_seqlens_q", "cu_seqlens_x"), offsets
        ):
            if (
                not isinstance(tensor, Tensor)
                or tensor.ndim != 1
                or tensor.numel() < 2
                or tensor.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    f"{name} must be a 1D int32/int64 tensor with at least two offsets"
                )
        if len({t.numel() for t in offsets}) != 1:
            raise ValueError("all offset arrays must describe the same batch size")
        if not isinstance(self.x_prefix_indices, Tensor):
            raise ValueError("x_prefix_indices must be a tensor")
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
        nx = self.x_prefix_indices.numel()
        if self.num_memory_tokens != nx + self.num_query_tokens - self.batch_size:
            raise ValueError(
                "packed totals violate the NCSE memory = x + query - 1 invariant"
            )
        for maximum, total in (
            (self.max_seqlen_m, self.num_memory_tokens),
            (self.max_seqlen_q, self.num_query_tokens),
            (self.max_seqlen_x, nx),
        ):
            if maximum > total:
                raise ValueError("a sequence maximum cannot exceed its stream total")
        if not self.trust_metadata:
            if self.cu_seqlens_m.device.type != "cpu":
                raise ValueError(
                    "device-resident metadata requires explicit trust_metadata=True; "
                    "use from_lengths() for verified NCSE construction"
                )
            self._verify_cpu_values()
            object.__setattr__(self, "metadata_verified", True)
        for name, tensor in zip(
            ("cu_seqlens_m", "cu_seqlens_q", "cu_seqlens_x"), offsets
        ):
            object.__setattr__(self, name, tensor.to(torch.int32).contiguous())
        object.__setattr__(self, "x_prefix_indices", self.x_prefix_indices.contiguous())
        object.__setattr__(
            self,
            "cu_seqlens_v",
            torch.arange(
                self.batch_size + 1, dtype=torch.int32, device=self.cu_seqlens_m.device
            ),
        )

    def _verify_cpu_values(self):
        offsets = [
            t.tolist()
            for t in (self.cu_seqlens_m, self.cu_seqlens_q, self.cu_seqlens_x)
        ]
        lengths = []
        for values, total, maximum in zip(
            offsets,
            (
                self.num_memory_tokens,
                self.num_query_tokens,
                self.x_prefix_indices.numel(),
            ),
            (self.max_seqlen_m, self.max_seqlen_q, self.max_seqlen_x),
        ):
            if values[0] != 0 or values[-1] != total or total > 2**31 - 1:
                raise ValueError(
                    "offsets must start at zero, end at their total, and fit int32"
                )
            sizes = [b - a for a, b in zip(values, values[1:])]
            if any(size <= 0 or size > maximum for size in sizes):
                raise ValueError(
                    "offsets must describe nonempty samples within the maximum"
                )
            lengths.append(sizes)
        for sample, (m, q, x) in enumerate(zip(*lengths)):
            if m != x + q - 1:
                raise ValueError(
                    f"NCSE sample {sample} requires memory_length = x_prefix_length + query_length - 1"
                )
        expected = [
            index
            for start, length in zip(offsets[0][:-1], lengths[2])
            for index in range(start, start + length)
        ]
        if self.x_prefix_indices.tolist() != expected:
            raise ValueError(
                "x_prefix_indices must select exactly each sample's x-prefix"
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
        for sample, (m, q, x) in enumerate(zip(ml, ql, xl)):
            if m != x + q - 1:
                raise ValueError(
                    f"NCSE sample {sample} requires memory_length = x_prefix_length + query_length - 1"
                )
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
        result = cls(
            sum(ml),
            sum(ql),
            *offsets,
            max(ml),
            max(ql),
            max(xl),
            indices,
            trust_metadata=True,
        )
        object.__setattr__(result, "metadata_verified", True)
        return result

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
        result = replace(
            self,
            trust_metadata=True,
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
        object.__setattr__(result, "metadata_verified", self.metadata_verified)
        return result

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
