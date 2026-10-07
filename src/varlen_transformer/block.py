"""
Custom implementation of a bidirectional/causal transformer block with fixed/variable sequence length input
using optimized GPU kernels, flash_attn.flash_attn_qkvpacked_func and flash_attn.flash_attn_varlen_qkvpacked_func.
See https://github.com/dao-ailab/flash-attention

Input, weights, and biases are casted to bf16, while layer_norm computations are kept in fp32 for stability.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import once_differentiable

from ._validation import validate_attention_input

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
except ModuleNotFoundError as exc:
    if exc.name != "flash_attn":
        raise
    flash_attn_qkvpacked_func = flash_attn_varlen_qkvpacked_func = None


def _to_bf16(x: Tensor) -> Tensor:
    return x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)


# Dropout layer customized to add more flexibility later.
# TODO: add an option for epoch-based and fid-/bid-based seed for mask generation.
# Generating the mask outside the fwd and bwd (inside the main class body) is an option.
def _dropout_fwd(x: Tensor, p: float, training: bool) -> tuple[Tensor, Tensor]:
    if (not training) or p == 0.0:
        return x, torch.empty(0, device=x.device, dtype=torch.bool)
    if p == 1.0:
        return torch.zeros_like(x), torch.zeros_like(x, dtype=torch.bool)
    keep = 1.0 - p
    mask = torch.rand(x.shape, device=x.device, dtype=torch.float32) < keep
    y = x * mask.to(dtype=x.dtype) / keep
    return y, mask


def _dropout_bwd(grad_y: Tensor, mask: Tensor, p: float, training: bool) -> Tensor:
    if (not training) or p == 0.0:
        return grad_y
    if p == 1.0:
        return torch.zeros_like(grad_y)
    keep = 1.0 - p
    return grad_y * mask.to(dtype=grad_y.dtype) / keep


# linear
def _linear_fwd(x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
    return torch.ops.aten.linear.default(x, weight, bias)


def _linear_bwd(
    grad_y: Tensor, x: Tensor, weight: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    in_features = x.shape[-1]
    out_features = grad_y.shape[-1]

    x2 = x.reshape(-1, in_features).contiguous()
    gy2 = grad_y.reshape(-1, out_features).contiguous()

    grad_x = grad_y.matmul(weight)

    if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
        grad_w = torch.bmm(
            gy2.transpose(0, 1).unsqueeze(0),
            x2.unsqueeze(0),
            out_dtype=torch.float32,
        ).squeeze(0)
    else:
        # CPU dispatch permits testing the surrounding math without a CUDA kernel.
        grad_w = gy2.float().transpose(0, 1).matmul(x2.float())
    grad_b = gy2.sum(dim=0, dtype=torch.float32)

    return grad_x, grad_w, grad_b


# gelu
def _gelu_fwd_tanh(x: Tensor) -> Tensor:
    return torch.ops.aten.gelu.default(x, approximate="tanh")


def _gelu_bwd_tanh(grad_y: Tensor, x: Tensor) -> Tensor:
    return torch.ops.aten.gelu_backward.default(grad_y, x, approximate="tanh")


# layer_norm
def _layernorm_fwd_fp32(
    x_bf16: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float,
):
    x_fp32 = x_bf16.float()
    w_fp32 = weight.float()
    b_fp32 = bias.float()

    y_fp32, mean_fp32, rstd_fp32 = torch.ops.aten.native_layer_norm.default(
        x_fp32, [x_bf16.shape[-1]], w_fp32, b_fp32, eps
    )
    return y_fp32.to(torch.bfloat16), mean_fp32, rstd_fp32


def _layernorm_bwd_fp32(
    grad_y: Tensor,
    x_bf16: Tensor,
    mean_fp32: Tensor,
    rstd_fp32: Tensor,
    weight: Tensor,
    bias: Tensor,
):
    grad_x_fp32, grad_w_fp32, grad_b_fp32 = (
        torch.ops.aten.native_layer_norm_backward.default(
            grad_y.float(),
            x_bf16.float(),
            [x_bf16.shape[-1]],
            mean_fp32,
            rstd_fp32,
            weight.float(),
            bias.float(),
            [True, True, True],
        )
    )
    return grad_x_fp32, grad_w_fp32, grad_b_fp32


@torch.library.custom_op(
    "legendblock::pre_attn_qkv", mutates_args=(), device_types=("cpu", "cuda")
)
def pre_attn_qkv(
    hidden_states: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    wqkv_weight: Tensor,
    wqkv_bias: Tensor,
    resid_dropout_p: float,
    training: bool,
    norm_eps: float,
    num_heads: int,
    head_dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    hs = _to_bf16(hidden_states)
    res = _to_bf16(residual)

    dropped1, mask1 = _dropout_fwd(hs, resid_dropout_p, training)
    residual1 = res + dropped1

    ln_out, mean, rstd = _layernorm_fwd_fp32(
        residual1, norm_weight, norm_bias, norm_eps
    )

    wqkv_weight_bf16 = _to_bf16(wqkv_weight)
    wqkv_bias_bf16 = _to_bf16(wqkv_bias)
    qkv_lin = _linear_fwd(ln_out, wqkv_weight_bf16, wqkv_bias_bf16)
    qkv = qkv_lin.view(*ln_out.shape[:-1], 3, num_heads, head_dim).contiguous()

    return qkv, residual1, mask1, mean, rstd, ln_out


@pre_attn_qkv.register_fake
def _(
    hidden_states: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    wqkv_weight: Tensor,
    wqkv_bias: Tensor,
    resid_dropout_p: float,
    training: bool,
    norm_eps: float,
    num_heads: int,
    head_dim: int,
):
    qkv = hidden_states.new_empty(
        (*hidden_states.shape[:-1], 3, num_heads, head_dim), dtype=torch.bfloat16
    )
    residual1 = hidden_states.new_empty(hidden_states.shape, dtype=torch.bfloat16)
    mask_shape = hidden_states.shape if training and resid_dropout_p > 0.0 else (0,)
    mask1 = hidden_states.new_empty(mask_shape, dtype=torch.bool)

    stat_shape = (*hidden_states.shape[:-1], 1)
    mean = hidden_states.new_empty(stat_shape, dtype=torch.float32)
    rstd = hidden_states.new_empty(stat_shape, dtype=torch.float32)

    ln_out = hidden_states.new_empty(hidden_states.shape, dtype=torch.bfloat16)
    return qkv, residual1, mask1, mean, rstd, ln_out


def _pre_setup_context(ctx, inputs, output):
    (
        hidden_states,
        residual,
        norm_weight,
        norm_bias,
        wqkv_weight,
        wqkv_bias,
        resid_dropout_p,
        training,
        norm_eps,
        num_heads,
        head_dim,
    ) = inputs

    qkv, residual1, mask1, mean, rstd, ln_out = output
    ctx.mark_non_differentiable(mask1, mean, rstd, ln_out)
    ctx.set_materialize_grads(False)

    ctx.save_for_backward(
        hidden_states,
        residual,
        norm_weight,
        norm_bias,
        wqkv_weight,
        wqkv_bias,
        residual1,
        mask1,
        mean,
        rstd,
        ln_out,
    )
    ctx.resid_dropout_p = resid_dropout_p
    ctx.training = training
    ctx.norm_eps = norm_eps
    ctx.num_heads = num_heads
    ctx.head_dim = head_dim


@once_differentiable
def _pre_backward(ctx, grad_qkv, grad_residual1, *unused_aux_grads):
    (
        hidden_states,
        residual,
        norm_weight,
        norm_bias,
        wqkv_weight,
        wqkv_bias,
        residual1,
        mask1,
        mean,
        rstd,
        ln_out,
    ) = ctx.saved_tensors

    grad_qkv = (
        ln_out.new_zeros((*ln_out.shape[:-1], 3, ctx.num_heads, ctx.head_dim))
        if grad_qkv is None
        else grad_qkv
    )
    grad_residual1 = (
        torch.zeros_like(residual1) if grad_residual1 is None else grad_residual1
    )

    grad_qkv_lin = grad_qkv.reshape(*ln_out.shape[:-1], -1).contiguous()

    wqkv_weight_bf16 = _to_bf16(wqkv_weight)
    grad_ln_out, grad_wqkv_w, grad_wqkv_b = _linear_bwd(
        grad_qkv_lin, ln_out, wqkv_weight_bf16
    )

    grad_residual1_from_ln, grad_norm_w, grad_norm_b = _layernorm_bwd_fp32(
        grad_ln_out, residual1, mean, rstd, norm_weight, norm_bias
    )

    grad_residual1_total = grad_residual1.float() + grad_residual1_from_ln

    grad_hidden = _dropout_bwd(
        grad_residual1_total,
        mask1,
        ctx.resid_dropout_p,
        ctx.training,
    )
    grad_residual = grad_residual1_total

    return (
        grad_hidden.to(hidden_states.dtype),
        grad_residual.to(residual.dtype),
        grad_norm_w.to(norm_weight.dtype),
        grad_norm_b.to(norm_bias.dtype),
        grad_wqkv_w.to(wqkv_weight.dtype),
        grad_wqkv_b.to(wqkv_bias.dtype),
        None,
        None,
        None,
        None,
        None,
    )


pre_attn_qkv.register_autograd(_pre_backward, setup_context=_pre_setup_context)


@torch.library.custom_op(
    "legendblock::post_attn_mlp", mutates_args=(), device_types=("cpu", "cuda")
)
def post_attn_mlp(
    attn_out: Tensor,
    residual1: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,
    resid_dropout_p: float,
    training: bool,
    norm_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    attn_out_bf16 = _to_bf16(attn_out)
    residual1_bf16 = _to_bf16(residual1)

    out_proj_weight_bf16 = _to_bf16(out_proj_weight)
    out_proj_bias_bf16 = _to_bf16(out_proj_bias)
    attn_proj = _linear_fwd(attn_out_bf16, out_proj_weight_bf16, out_proj_bias_bf16)

    dropped2, mask2 = _dropout_fwd(attn_proj, resid_dropout_p, training)
    residual2 = residual1_bf16 + dropped2

    ln_out, mean, rstd = _layernorm_fwd_fp32(
        residual2, norm_weight, norm_bias, norm_eps
    )

    fc1_weight_bf16 = _to_bf16(fc1_weight)
    fc1_bias_bf16 = _to_bf16(fc1_bias)
    gelu_in = _linear_fwd(ln_out, fc1_weight_bf16, fc1_bias_bf16)
    gelu_out = _gelu_fwd_tanh(gelu_in)

    fc2_weight_bf16 = _to_bf16(fc2_weight)
    fc2_bias_bf16 = _to_bf16(fc2_bias)
    out = _linear_fwd(gelu_out, fc2_weight_bf16, fc2_bias_bf16)

    return out, residual2, mask2, mean, rstd, ln_out, gelu_in, gelu_out


@post_attn_mlp.register_fake
def _(
    attn_out: Tensor,
    residual1: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,
    resid_dropout_p: float,
    training: bool,
    norm_eps: float,
):
    out = attn_out.new_empty(attn_out.shape, dtype=torch.bfloat16)
    residual2 = residual1.new_empty(residual1.shape, dtype=torch.bfloat16)
    mask_shape = residual1.shape if training and resid_dropout_p > 0.0 else (0,)
    mask2 = residual1.new_empty(mask_shape, dtype=torch.bool)

    stat_shape = (*residual1.shape[:-1], 1)
    mean = residual1.new_empty(stat_shape, dtype=torch.float32)
    rstd = residual1.new_empty(stat_shape, dtype=torch.float32)

    ln_out = residual1.new_empty(residual1.shape, dtype=torch.bfloat16)
    hidden_dim = fc1_weight.shape[0]
    gelu_in = residual1.new_empty(
        (*residual1.shape[:-1], hidden_dim), dtype=torch.bfloat16
    )
    gelu_out = residual1.new_empty(
        (*residual1.shape[:-1], hidden_dim), dtype=torch.bfloat16
    )
    return out, residual2, mask2, mean, rstd, ln_out, gelu_in, gelu_out


def _post_setup_context(ctx, inputs, output):
    (
        attn_out,
        residual1,
        out_proj_weight,
        out_proj_bias,
        norm_weight,
        norm_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        resid_dropout_p,
        training,
        norm_eps,
    ) = inputs

    out, residual2, mask2, mean, rstd, ln_out, gelu_in, gelu_out = output
    ctx.mark_non_differentiable(mask2, mean, rstd, ln_out, gelu_in, gelu_out)
    ctx.set_materialize_grads(False)

    ctx.save_for_backward(
        attn_out,
        residual1,
        out_proj_weight,
        out_proj_bias,
        norm_weight,
        norm_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        residual2,
        mask2,
        mean,
        rstd,
        ln_out,
        gelu_in,
        gelu_out,
    )
    ctx.resid_dropout_p = resid_dropout_p
    ctx.training = training
    ctx.norm_eps = norm_eps


@once_differentiable
def _post_backward(ctx, grad_out, grad_residual2, *unused_aux_grads):
    (
        attn_out,
        residual1,
        out_proj_weight,
        out_proj_bias,
        norm_weight,
        norm_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        residual2,
        mask2,
        mean,
        rstd,
        ln_out,
        gelu_in,
        gelu_out,
    ) = ctx.saved_tensors

    grad_out = (
        torch.zeros_like(attn_out, dtype=torch.bfloat16)
        if grad_out is None
        else grad_out
    )
    grad_residual2_total = (
        torch.zeros_like(residual2, dtype=torch.float32)
        if grad_residual2 is None
        else grad_residual2.float()
    )

    fc2_weight_bf16 = _to_bf16(fc2_weight)
    grad_gelu_out, grad_fc2_w, grad_fc2_b = _linear_bwd(
        grad_out, gelu_out, fc2_weight_bf16
    )

    grad_gelu_in = _gelu_bwd_tanh(grad_gelu_out, gelu_in)

    fc1_weight_bf16 = _to_bf16(fc1_weight)
    grad_ln_out, grad_fc1_w, grad_fc1_b = _linear_bwd(
        grad_gelu_in, ln_out, fc1_weight_bf16
    )

    grad_residual2_from_ln, grad_norm_w, grad_norm_b = _layernorm_bwd_fp32(
        grad_ln_out, residual2, mean, rstd, norm_weight, norm_bias
    )
    grad_residual2_total = grad_residual2_total + grad_residual2_from_ln

    grad_residual1 = grad_residual2_total
    grad_attn_proj = _dropout_bwd(
        grad_residual2_total, mask2, ctx.resid_dropout_p, ctx.training
    )

    out_proj_weight_bf16 = _to_bf16(out_proj_weight)
    grad_attn_out, grad_outproj_w, grad_outproj_b = _linear_bwd(
        grad_attn_proj.to(torch.bfloat16), _to_bf16(attn_out), out_proj_weight_bf16
    )

    return (
        grad_attn_out.to(attn_out.dtype),
        grad_residual1.to(residual1.dtype),
        grad_outproj_w.to(out_proj_weight.dtype),
        grad_outproj_b.to(out_proj_bias.dtype),
        grad_norm_w.to(norm_weight.dtype),
        grad_norm_b.to(norm_bias.dtype),
        grad_fc1_w.to(fc1_weight.dtype),
        grad_fc1_b.to(fc1_bias.dtype),
        grad_fc2_w.to(fc2_weight.dtype),
        grad_fc2_b.to(fc2_bias.dtype),
        None,
        None,
        None,
    )


post_attn_mlp.register_autograd(_post_backward, setup_context=_post_setup_context)


class Block(nn.Module):
    """Pre-norm block returning ``(mlp_branch, residual_stream)`` in BF16.

    Chain blocks by passing both returned tensors to the next block. After the
    last block, add the branch to the residual before the final normalization.
    Attention requires CUDA; CPU dispatch of the two custom ops is for tests.
    """

    def __init__(self, emb_dim, mixer_cls, mlp_cls, resid_dropout1, resid_dropout2):
        super(Block, self).__init__()

        self.mixer = mixer_cls(emb_dim)
        self.mlp = mlp_cls(emb_dim)

        self.norm_attn = nn.LayerNorm(emb_dim)
        self.norm_mlp = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(resid_dropout1)
        self.dropout2 = nn.Dropout(resid_dropout2)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        validate_attention_input(
            hidden_states, self.mixer.emb_dim, cu_seqlens, max_seqlen
        )
        if residual is not None:
            if (
                residual.shape != hidden_states.shape
                or residual.device != hidden_states.device
            ):
                raise ValueError(
                    "residual must match hidden states in shape and device"
                )
            if not residual.is_floating_point():
                raise ValueError("residual must have a floating-point dtype")
        residual_in = (
            residual if residual is not None else torch.zeros_like(hidden_states)
        )

        # dropout -> resid -> qkv
        qkv, residual1, *_ = torch.ops.legendblock.pre_attn_qkv(
            hidden_states,
            residual_in,
            self.norm_attn.weight,
            self.norm_attn.bias,
            self.mixer.Wqkv.weight,
            self.mixer.Wqkv.bias,
            self.dropout1.p,
            self.training,
            self.norm_attn.eps,
            self.mixer.num_heads,
            self.mixer.head_dim,
        )

        # qkv -> attn
        attn_cfg = self.mixer.attn
        if cu_seqlens is None:
            attn_out = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=attn_cfg.drop.p if self.training else 0.0,
                softmax_scale=attn_cfg.softmax_scale,
                causal=self.mixer.causal,
                window_size=attn_cfg.window_size,
                softcap=0.0,
                alibi_slopes=None,
                deterministic=attn_cfg.deterministic,
                return_attn_probs=False,
            )
        else:
            attn_out = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens.to(torch.int32).contiguous(),
                int(max_seqlen),
                dropout_p=attn_cfg.drop.p if self.training else 0.0,
                softmax_scale=attn_cfg.softmax_scale,
                causal=self.mixer.causal,
                window_size=attn_cfg.window_size,
                softcap=0.0,
                alibi_slopes=None,
                deterministic=attn_cfg.deterministic,
                return_attn_probs=False,
            )
        attn_out = attn_out.reshape(*hidden_states.shape[:-1], self.mixer.emb_dim)

        # linear -> drop -> resid -> mlp
        out, residual2, *_ = torch.ops.legendblock.post_attn_mlp(
            attn_out,
            residual1,
            self.mixer.out_proj.weight,
            self.mixer.out_proj.bias,
            self.norm_mlp.weight,
            self.norm_mlp.bias,
            self.mlp.fc1.weight,
            self.mlp.fc1.bias,
            self.mlp.fc2.weight,
            self.mlp.fc2.bias,
            self.dropout2.p,
            self.training,
            self.norm_mlp.eps,
        )

        return out, residual2


# Eager reference without the custom ops or their explicit mixed-precision policy.
class Block_(nn.Module):
    def __init__(self, emb_dim, mixer_cls, mlp_cls, resid_dropout1, resid_dropout2):
        super(Block_, self).__init__()

        self.mixer = mixer_cls(emb_dim)
        self.mlp = mlp_cls(emb_dim)

        self.norm_attn = nn.LayerNorm(emb_dim)
        self.norm_mlp = nn.LayerNorm(emb_dim)

        self.dropout1 = nn.Dropout(resid_dropout1)
        self.dropout2 = nn.Dropout(resid_dropout2)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        # prenorm path
        dropped = self.dropout1(hidden_states)
        residual = (residual + dropped) if residual is not None else dropped

        hidden_states = self.norm_attn(residual)
        hidden_states = self.mixer(hidden_states, cu_seqlens, max_seqlen)
        residual = residual + self.dropout2(hidden_states)

        hidden_states = self.norm_mlp(residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
