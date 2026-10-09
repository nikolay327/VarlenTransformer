"""Three first-order BF16/FP32 regions surrounding SA and the two CA calls.

All public outputs own storage. Saved norm statistics/intermediates are exposed
only for autograd registration and testing, and are nondifferentiable.

The native block supplies one flat stream-major token matrix. memory_tokens is
the total memory span for the entire batch, so every projection/norm operates
on a contiguous region rather than strided per-sample slices. The arithmetic
helpers also accept leading batch dimensions for direct operator tests.
"""

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .block import (
    _to_bf16,
    _dropout_fwd,
    _dropout_bwd,
    _linear_fwd,
    _linear_bwd,
    _layernorm_fwd_fp32,
    _layernorm_bwd_fp32,
    _gelu_fwd_tanh,
    _gelu_bwd_tanh,
)


def _memory(x, n):
    return x.narrow(-2, 0, n)


def _queries(x, n):
    return x.narrow(-2, n, x.shape[-2] - n)


def _linear(x, w, b):
    # A single token-matrix GEMM, also for noncontiguous fixed-batch slices.
    shape = x.shape
    flat = _to_bf16(x).reshape(-1, shape[-1])
    return _linear_fwd(flat, _to_bf16(w), _to_bf16(b)).reshape(*shape[:-1], w.shape[0])


def _grad(g, like, dtype=torch.bfloat16):
    return torch.zeros_like(like, dtype=dtype) if g is None else g.to(dtype)


def _finish(grads, tensors, scalar_count):
    return (*[g.to(t.dtype) for g, t in zip(grads, tensors)], *([None] * scalar_count))


@torch.library.custom_op(
    "legendblock::mq_pre_sa", mutates_args=(), device_types=("cpu", "cuda")
)
def mq_pre_sa(
    hidden: Tensor,
    residual: Tensor,
    nw: Tensor,
    nb: Tensor,
    ww: Tensor,
    wb: Tensor,
    dropout_p: float,
    training: bool,
    eps: float,
    heads: int,
    dim: int,
    memory_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dropped, mask = _dropout_fwd(_to_bf16(hidden), dropout_p, training)
    r0 = _to_bf16(residual) + dropped
    normalized, mean, rstd = _layernorm_fwd_fp32(
        _memory(r0, memory_tokens), nw, nb, eps
    )
    projected = _linear(normalized, ww, wb)
    qkv = projected.reshape(*normalized.shape[:-1], 3, heads, dim)
    return qkv, r0, mask, mean, rstd, normalized


@mq_pre_sa.register_fake
def _(
    hidden,
    residual,
    nw,
    nb,
    ww,
    wb,
    dropout_p,
    training,
    eps,
    heads,
    dim,
    memory_tokens,
):
    shape = (*hidden.shape[:-2], memory_tokens, hidden.shape[-1])
    return (
        hidden.new_empty((*shape[:-1], 3, heads, dim), dtype=torch.bfloat16),
        hidden.new_empty(hidden.shape, dtype=torch.bfloat16),
        hidden.new_empty(
            hidden.shape if training and dropout_p > 0 else (0,), dtype=torch.bool
        ),
        hidden.new_empty((*shape[:-1], 1), dtype=torch.float32),
        hidden.new_empty((*shape[:-1], 1), dtype=torch.float32),
        hidden.new_empty(shape, dtype=torch.bfloat16),
    )


def _sa_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:6], *output[1:])
    ctx.meta = inputs[6:]
    ctx.mark_non_differentiable(*output[2:])
    ctx.set_materialize_grads(False)


@once_differentiable
def _sa_backward(ctx, gqkv, gr0, *_):
    hidden, residual, nw, nb, ww, wb, r0, mask, mean, rstd, normalized = (
        ctx.saved_tensors
    )
    p, training, eps, heads, dim, n = ctx.meta
    gq = (
        normalized.new_zeros((*normalized.shape[:-1], 3, heads, dim))
        if gqkv is None
        else gqkv
    )
    gn, gww, gwb = _linear_bwd(
        gq.reshape(*normalized.shape[:-1], -1), normalized, _to_bf16(ww)
    )
    gm, gnw, gnb = _layernorm_bwd_fp32(gn, _memory(r0, n), mean, rstd, nw, nb)
    total = _grad(gr0, r0, torch.float32).clone()
    total.narrow(-2, 0, n).add_(gm)
    gh = _dropout_bwd(total, mask, p, training)
    return _finish(
        (gh, total, gnw, gnb, gww, gwb), (hidden, residual, nw, nb, ww, wb), 6
    )


mq_pre_sa.register_autograd(_sa_backward, setup_context=_sa_context)


@torch.library.custom_op(
    "legendblock::mq_pre_ca", mutates_args=(), device_types=("cpu", "cuda")
)
def mq_pre_ca(
    attn_memory: Tensor,
    residual0: Tensor,
    ow: Tensor,
    ob: Tensor,
    nw: Tensor,
    nb: Tensor,
    qw: Tensor,
    qb: Tensor,
    kvw: Tensor,
    kvb: Tensor,
    dropout_p: float,
    training: bool,
    eps: float,
    heads: int,
    dim: int,
    memory_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    projected = _linear(attn_memory, ow, ob)
    dropped, mask = _dropout_fwd(projected, dropout_p, training)
    m1 = _memory(_to_bf16(residual0), memory_tokens) + dropped
    rca = torch.cat((m1, _queries(_to_bf16(residual0), memory_tokens)), dim=-2)
    normalized, mean, rstd = _layernorm_fwd_fp32(rca, nw, nb, eps)
    nq, nm = _queries(normalized, memory_tokens), _memory(normalized, memory_tokens)
    q = _linear(nq, qw, qb).reshape(*nq.shape[:-1], heads, dim)
    kv = _linear(nm, kvw, kvb).reshape(*nm.shape[:-1], 2, heads, dim)
    return q, kv, rca, mask, mean, rstd, normalized


@mq_pre_ca.register_fake
def _(
    attn_memory,
    residual0,
    ow,
    ob,
    nw,
    nb,
    qw,
    qb,
    kvw,
    kvb,
    dropout_p,
    training,
    eps,
    heads,
    dim,
    memory_tokens,
):
    ms = (*residual0.shape[:-2], memory_tokens, residual0.shape[-1])
    qs = (
        *residual0.shape[:-2],
        residual0.shape[-2] - memory_tokens,
        residual0.shape[-1],
    )
    return (
        residual0.new_empty((*qs[:-1], heads, dim), dtype=torch.bfloat16),
        residual0.new_empty((*ms[:-1], 2, heads, dim), dtype=torch.bfloat16),
        residual0.new_empty(residual0.shape, dtype=torch.bfloat16),
        residual0.new_empty(
            ms if training and dropout_p > 0 else (0,), dtype=torch.bool
        ),
        residual0.new_empty((*residual0.shape[:-1], 1), dtype=torch.float32),
        residual0.new_empty((*residual0.shape[:-1], 1), dtype=torch.float32),
        residual0.new_empty(residual0.shape, dtype=torch.bfloat16),
    )


def _ca_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:10], *output[2:])
    ctx.meta = inputs[10:]
    ctx.mark_non_differentiable(*output[3:])
    ctx.set_materialize_grads(False)


@once_differentiable
def _ca_backward(ctx, gq, gkv, grca, *_):
    am, r0, ow, ob, nw, nb, qw, qb, kvw, kvb, rca, mask, mean, rstd, normalized = (
        ctx.saved_tensors
    )
    p, training, eps, heads, dim, n = ctx.meta
    nq, nm = _queries(normalized, n), _memory(normalized, n)
    gq = nq.new_zeros(nq.shape) if gq is None else gq.reshape_as(nq)
    gkv = (
        nm.new_zeros((*nm.shape[:-1], 2 * nm.shape[-1]))
        if gkv is None
        else gkv.reshape(*nm.shape[:-1], -1)
    )
    gnq, gqw, gqb = _linear_bwd(gq, nq, _to_bf16(qw))
    gnm, gkvw, gkvb = _linear_bwd(gkv, nm, _to_bf16(kvw))
    gn = torch.cat((gnm, gnq), dim=-2)
    gr, gnw, gnb = _layernorm_bwd_fp32(gn, rca, mean, rstd, nw, nb)
    total = _grad(grca, rca, torch.float32) + gr
    gprojected = _dropout_bwd(_memory(total, n), mask, p, training).bfloat16()
    gam, gow, gob = _linear_bwd(gprojected, _to_bf16(am), _to_bf16(ow))
    return _finish(
        (gam, total, gow, gob, gnw, gnb, gqw, gqb, gkvw, gkvb),
        (am, r0, ow, ob, nw, nb, qw, qb, kvw, kvb),
        6,
    )


mq_pre_ca.register_autograd(_ca_backward, setup_context=_ca_context)


@torch.library.custom_op(
    "legendblock::mq_post_ca_mlp", mutates_args=(), device_types=("cpu", "cuda")
)
def mq_post_ca_mlp(
    attn_queries: Tensor,
    residual_ca: Tensor,
    ow: Tensor,
    ob: Tensor,
    nw: Tensor,
    nb: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    dropout_p: float,
    training: bool,
    eps: float,
    memory_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    projected = _linear(attn_queries, ow, ob)
    dropped, mask = _dropout_fwd(projected, dropout_p, training)
    rca = _to_bf16(residual_ca)
    rq = _queries(rca, memory_tokens) + dropped
    r1 = torch.cat((_memory(rca, memory_tokens), rq), dim=-2)
    normalized, mean, rstd = _layernorm_fwd_fp32(r1, nw, nb, eps)
    gelu_in = _linear(normalized, w1, b1)
    gelu_out = _gelu_fwd_tanh(gelu_in)
    out = _linear(gelu_out, w2, b2)
    return out, r1, mask, mean, rstd, normalized, gelu_in, gelu_out


@mq_post_ca_mlp.register_fake
def _(
    attn_queries,
    residual_ca,
    ow,
    ob,
    nw,
    nb,
    w1,
    b1,
    w2,
    b2,
    dropout_p,
    training,
    eps,
    memory_tokens,
):
    return (
        residual_ca.new_empty(residual_ca.shape, dtype=torch.bfloat16),
        residual_ca.new_empty(residual_ca.shape, dtype=torch.bfloat16),
        attn_queries.new_empty(
            attn_queries.shape if training and dropout_p > 0 else (0,), dtype=torch.bool
        ),
        residual_ca.new_empty((*residual_ca.shape[:-1], 1), dtype=torch.float32),
        residual_ca.new_empty((*residual_ca.shape[:-1], 1), dtype=torch.float32),
        residual_ca.new_empty(residual_ca.shape, dtype=torch.bfloat16),
        residual_ca.new_empty(
            (*residual_ca.shape[:-1], w1.shape[0]), dtype=torch.bfloat16
        ),
        residual_ca.new_empty(
            (*residual_ca.shape[:-1], w1.shape[0]), dtype=torch.bfloat16
        ),
    )


def _mlp_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:10], *output[1:])
    ctx.meta = inputs[10:]
    ctx.mark_non_differentiable(*output[2:])
    ctx.set_materialize_grads(False)


@once_differentiable
def _mlp_backward(ctx, gout, gr1, *_):
    (
        aq,
        rca,
        ow,
        ob,
        nw,
        nb,
        w1,
        b1,
        w2,
        b2,
        r1,
        mask,
        mean,
        rstd,
        normalized,
        gi,
        go,
    ) = ctx.saved_tensors
    p, training, eps, n = ctx.meta
    gout = _grad(gout, r1)
    ggo, gw2, gb2 = _linear_bwd(gout, go, _to_bf16(w2))
    ggi = _gelu_bwd_tanh(ggo, gi)
    gn, gw1, gb1 = _linear_bwd(ggi, normalized, _to_bf16(w1))
    gr, gnw, gnb = _layernorm_bwd_fp32(gn, r1, mean, rstd, nw, nb)
    total = _grad(gr1, r1, torch.float32) + gr
    gprojected = _dropout_bwd(_queries(total, n), mask, p, training).bfloat16()
    gaq, gow, gob = _linear_bwd(gprojected, _to_bf16(aq), _to_bf16(ow))
    return _finish(
        (gaq, total, gow, gob, gnw, gnb, gw1, gb1, gw2, gb2),
        (aq, rca, ow, ob, nw, nb, w1, b1, w2, b2),
        4,
    )


mq_post_ca_mlp.register_autograd(_mlp_backward, setup_context=_mlp_context)
