import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ._validation import positive_int, validate_attention_input

try:
    from flash_attn.modules.mha import FlashSelfAttention
except ModuleNotFoundError as exc:
    if exc.name != "flash_attn":
        raise
    FlashSelfAttention = None


class MHA(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        causal: bool = False,
    ):
        super(MHA, self).__init__()

        positive_int("emb_dim", emb_dim)
        positive_int("num_heads", num_heads)
        if emb_dim % num_heads:
            raise ValueError("emb_dim must be divisible by num_heads")
        if emb_dim // num_heads > 256:
            raise ValueError("FlashAttention head_dim must be at most 256")
        if not math.isfinite(attn_dropout) or not 0.0 <= attn_dropout < 1.0:
            raise ValueError("attn_dropout must be in [0, 1)")
        if FlashSelfAttention is None:
            raise ImportError(
                "FlashAttention is required for MHA/Block. Install CUDA-enabled PyTorch, "
                "then run: pip install 'flash-attn>=2.6,<3' --no-build-isolation"
            )

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = self.emb_dim // self.num_heads
        self.causal = causal

        self.attn = FlashSelfAttention(
            causal=self.causal, attention_dropout=attn_dropout
        )
        self.Wqkv = nn.Linear(self.emb_dim, 3 * self.emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        """Attend to fixed ``(B, S, E)`` or packed ``(T, E)`` input."""
        validate_attention_input(x, self.emb_dim, cu_seqlens, max_seqlen)
        qkv = self.Wqkv(x)
        qkv = qkv.reshape(*x.shape[:-1], 3, self.num_heads, self.head_dim).contiguous()
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
        out = self.attn(
            qkv, causal=self.causal, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        ).reshape(*x.shape[:-1], self.emb_dim)
        out = self.out_proj(out)
        return out
