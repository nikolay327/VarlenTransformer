"""Standalone KV-packed FlashAttention with independent query/key lengths."""

import math

import torch
from torch import Tensor, nn

from ._validation import positive_int, validate_cross_attention_input

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError as exc:
    if exc.name != "flash_attn":
        raise
    FlashCrossAttention = None


class CrossMHA(nn.Module):
    """Cross-attention using ordinary PyTorch projection dtype rules.

    Fixed inputs: q=(B,Sq,E), kv=(B,Sk,E). Packed inputs: q=(Tq,E),
    kv=(Tk,E), plus all four sequence arguments. Causal masking uses the
    native bottom-right alignment, including zero attention for fully masked
    rows when Sq>Sk. Offset contents are a caller contract; no host sync occurs.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        positive_int("emb_dim", emb_dim)
        positive_int("num_heads", num_heads)
        if emb_dim % num_heads:
            raise ValueError("emb_dim must be divisible by num_heads")
        if emb_dim // num_heads > 256:
            raise ValueError("FlashAttention head_dim must be at most 256")
        if not math.isfinite(attn_dropout) or not 0.0 <= attn_dropout < 1.0:
            raise ValueError("attn_dropout must be in [0, 1)")
        if FlashCrossAttention is None:
            raise ImportError(
                "FlashAttention 2.6–2.x is required for CrossMHA; use your installed CUDA environment"
            )
        self.emb_dim, self.num_heads = emb_dim, num_heads
        self.head_dim, self.causal = emb_dim // num_heads, causal
        self.attn = FlashCrossAttention(causal=causal, attention_dropout=attn_dropout)
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wkv = nn.Linear(emb_dim, 2 * emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        cu_seqlens_q: Tensor | None = None,
        cu_seqlens_k: Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
    ) -> Tensor:
        validate_cross_attention_input(
            q, kv, self.emb_dim, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        )
        projected_q = self.Wq(q).reshape(*q.shape[:-1], self.num_heads, self.head_dim)
        projected_kv = self.Wkv(kv).reshape(
            *kv.shape[:-1], 2, self.num_heads, self.head_dim
        )
        if cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
            cu_seqlens_k = cu_seqlens_k.to(torch.int32).contiguous()
        out = self.attn(
            projected_q,
            projected_kv,
            causal=self.causal,
            cu_seqlens=cu_seqlens_q,
            max_seqlen=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )
        return self.out_proj(out.reshape_as(q))
