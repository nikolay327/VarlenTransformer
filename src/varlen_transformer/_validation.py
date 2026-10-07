"""Structural checks that do not synchronize CUDA tensors to the host."""

import torch
from torch import Tensor


def positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_attention_input(
    x: Tensor, emb_dim: int, cu_seqlens: Tensor | None, max_seqlen: int | None
) -> None:
    if not x.is_floating_point():
        raise ValueError("hidden states must have a floating-point dtype")
    if (cu_seqlens is None) != (max_seqlen is None):
        raise ValueError("cu_seqlens and max_seqlen must be provided together")
    expected_ndim = 3 if cu_seqlens is None else 2
    if x.ndim != expected_ndim or x.shape[-1] != emb_dim:
        shape = (
            "(batch, seqlen, emb_dim)"
            if cu_seqlens is None
            else "(total_tokens, emb_dim)"
        )
        raise ValueError(
            f"hidden states must have shape {shape}, with emb_dim={emb_dim}"
        )
    if any(size == 0 for size in x.shape[:-1]):
        raise ValueError("empty batches and empty token tensors are not supported")
    if cu_seqlens is not None:
        positive_int("max_seqlen", max_seqlen)
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a 1D tensor with at least two offsets")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must have dtype int32 or int64")
        if cu_seqlens.device != x.device:
            raise ValueError("cu_seqlens and hidden states must be on the same device")
