"""Causal memory SA, shared volume/density CA, then one shared FFN."""

import math

import torch
from torch import Tensor, nn

from ._memory_ops import mq_pre_sa, mq_pre_ca, mq_post_ca_mlp
from .cross_mha import CrossMHA
from .layout import FixedMemoryQueryLayout, PackedMemoryQueryLayout
from .mha import MHA
from .mlp import MLP

try:
    from flash_attn import (
        flash_attn_qkvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func,
    )
except ModuleNotFoundError as exc:
    if exc.name != "flash_attn":
        raise
    flash_attn_qkvpacked_func = flash_attn_varlen_qkvpacked_func = None
    flash_attn_kvpacked_func = flash_attn_varlen_kvpacked_func = None


def _attention_options(configuration, training):
    return dict(
        dropout_p=configuration.drop.p if training else 0.0,
        softmax_scale=configuration.softmax_scale,
        window_size=configuration.window_size,
        deterministic=configuration.deterministic,
        softcap=0.0,
        alibi_slopes=None,
        return_attn_probs=False,
    )


class MemoryQueryBlock(nn.Module):
    """Return (new_mlp_branch, updated_residual), both BF16.

    Both inputs and outputs use flat [all M; all V; all Q] storage (N_total,E),
    for fixed and packed layouts. FixedMemoryQueryStack adapts per-sample input.
    The caller supplies M, one V per sample, and Q, and reuses the layout for
    every block. V/Q never feed back into memory or attend to each other. SA is
    always causal. V cross-attention is noncausal and reads only the x-prefix;
    Q uses native rectangular causal attention over the entire memory.
    """

    def __init__(
        self,
        emb_dim: int,
        intermediate_size: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        entry_resid_dropout: float = 0.0,
        sa_resid_dropout: float = 0.0,
        ca_resid_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.memory_mixer = MHA(
            emb_dim, num_heads, attn_dropout=attn_dropout, causal=True
        )
        self.cross_mixer = CrossMHA(
            emb_dim, num_heads, attn_dropout=attn_dropout, causal=True
        )
        self.mlp = MLP(emb_dim, intermediate_size)
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        for p in (entry_resid_dropout, sa_resid_dropout, ca_resid_dropout):
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("residual dropout must be in [0, 1]")
        self.norm_sa = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.norm_ca = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.norm_mlp = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.dropout_input = nn.Dropout(entry_resid_dropout)
        self.dropout_sa = nn.Dropout(sa_resid_dropout)
        self.dropout_ca = nn.Dropout(ca_resid_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor | None = None,
        *,
        layout: FixedMemoryQueryLayout | PackedMemoryQueryLayout,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(layout, (FixedMemoryQueryLayout, PackedMemoryQueryLayout)):
            raise ValueError(
                "layout must be a FixedMemoryQueryLayout or PackedMemoryQueryLayout"
            )
        layout.validate(hidden_states, self.memory_mixer.emb_dim)
        if residual is not None and (
            residual.shape != hidden_states.shape
            or residual.device != hidden_states.device
            or not residual.is_floating_point()
        ):
            raise ValueError(
                "residual must match the floating hidden-state shape and device"
            )
        residual_in = torch.zeros_like(hidden_states) if residual is None else residual
        n, e = layout.num_memory_tokens, self.memory_mixer.emb_dim
        sa, ca = self.memory_mixer, self.cross_mixer
        qkv, r0, *_ = mq_pre_sa(
            hidden_states,
            residual_in,
            self.norm_sa.weight,
            self.norm_sa.bias,
            sa.Wqkv.weight,
            sa.Wqkv.bias,
            self.dropout_input.p,
            self.training,
            self.norm_sa.eps,
            sa.num_heads,
            sa.head_dim,
            n,
        )
        sa_opts = _attention_options(sa.attn, self.training)
        if isinstance(layout, FixedMemoryQueryLayout):
            am = flash_attn_qkvpacked_func(
                qkv.view(
                    layout.batch_size,
                    layout.memory_length,
                    3,
                    sa.num_heads,
                    sa.head_dim,
                ),
                causal=True,
                **sa_opts,
            )
        else:
            am = flash_attn_varlen_qkvpacked_func(
                qkv, layout.cu_seqlens_m, layout.max_seqlen_m, causal=True, **sa_opts
            )
        am = am.reshape(n, e)
        q, kv, rca, *_ = mq_pre_ca(
            am,
            r0,
            sa.out_proj.weight,
            sa.out_proj.bias,
            self.norm_ca.weight,
            self.norm_ca.bias,
            ca.Wq.weight,
            ca.Wq.bias,
            ca.Wkv.weight,
            ca.Wkv.bias,
            self.dropout_sa.p,
            self.training,
            self.norm_ca.eps,
            ca.num_heads,
            ca.head_dim,
            n,
        )
        ca_opts = _attention_options(ca.attn, self.training)
        b = layout.batch_size
        if isinstance(layout, FixedMemoryQueryLayout):
            fixed_kv = kv.view(b, layout.memory_length, 2, ca.num_heads, ca.head_dim)
            av = flash_attn_kvpacked_func(
                q[:b].view(b, 1, ca.num_heads, ca.head_dim),
                fixed_kv[:, : layout.x_prefix_length],
                causal=False,
                **ca_opts,
            )
            aq = flash_attn_kvpacked_func(
                q[b:].view(b, layout.query_length, ca.num_heads, ca.head_dim),
                fixed_kv,
                causal=True,
                **ca_opts,
            )
        else:
            # Project Wkv once; compact only the projected x-prefix for V.
            prefix_kv = kv.index_select(0, layout.x_prefix_indices)
            av = flash_attn_varlen_kvpacked_func(
                q[:b],
                prefix_kv,
                layout.cu_seqlens_v,
                layout.cu_seqlens_x,
                1,
                layout.max_seqlen_x,
                causal=False,
                **ca_opts,
            )
            aq = flash_attn_varlen_kvpacked_func(
                q[b:],
                kv,
                layout.cu_seqlens_q,
                layout.cu_seqlens_m,
                layout.max_seqlen_q,
                layout.max_seqlen_m,
                causal=True,
                **ca_opts,
            )
        query_attn = torch.cat(
            (
                av.reshape(b, ca.num_heads, ca.head_dim),
                aq.reshape(layout.num_query_tokens, ca.num_heads, ca.head_dim),
            ),
            dim=0,
        ).reshape(hidden_states.shape[0] - n, e)
        branch, r1, *_ = mq_post_ca_mlp(
            query_attn,
            rca,
            ca.out_proj.weight,
            ca.out_proj.bias,
            self.norm_mlp.weight,
            self.norm_mlp.bias,
            self.mlp.fc1.weight,
            self.mlp.fc1.bias,
            self.mlp.fc2.weight,
            self.mlp.fc2.bias,
            self.dropout_ca.p,
            self.training,
            self.norm_mlp.eps,
            n,
        )
        return branch, r1
