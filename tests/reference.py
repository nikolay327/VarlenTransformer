"""Independent PyTorch math, with no calls into the package's helper functions.

Attention accumulates in FP32. The block oracle reproduces the documented BF16
cast boundaries, using ordinary autograd rather than the custom backward rules.
"""

import torch
from torch import nn
from torch.nn import functional as F


def attention(
    qkv, *, causal=False, dropout_p=0.0, softmax_scale=None, window_size=(-1, -1), **_
):
    q, k, v = (t.transpose(1, 2).float() for t in qkv.unbind(dim=2))
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    scores = q @ k.transpose(-1, -2) * scale
    positions = torch.arange(q.shape[-2], device=qkv.device)
    delta = positions[:, None] - positions[None, :]
    allowed = torch.ones_like(delta, dtype=torch.bool)
    if causal:
        allowed = allowed & (delta >= 0)
    left, right = window_size
    if left >= 0:
        allowed = allowed & (delta <= left)
    if right >= 0:
        allowed = allowed & (delta >= -right)
    probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
    probabilities = F.dropout(probabilities, dropout_p, training=dropout_p > 0)
    return (probabilities @ v).transpose(1, 2).to(qkv.dtype)


def packed_attention(qkv, cu_seqlens, max_seqlen, **kwargs):
    offsets = cu_seqlens.tolist()
    assert offsets[0] == 0 and offsets[-1] == qkv.shape[0]
    assert all(0 < b - a <= max_seqlen for a, b in zip(offsets, offsets[1:]))
    return torch.cat(
        [
            attention(qkv[a:b].unsqueeze(0), **kwargs).squeeze(0)
            for a, b in zip(offsets, offsets[1:])
        ]
    )


class ReferenceSelfAttention(nn.Module):
    """A test-only substitute for the FlashSelfAttention dependency."""

    def __init__(self, causal=False, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.drop = nn.Dropout(attention_dropout)
        self.softmax_scale = None
        self.window_size = (-1, -1)
        self.deterministic = False

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        kwargs = dict(
            causal=self.causal if causal is None else causal,
            dropout_p=self.drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            window_size=self.window_size,
        )
        if cu_seqlens is None:
            return attention(qkv, **kwargs)
        return packed_attention(qkv, cu_seqlens, max_seqlen, **kwargs)


def masked_dropout(x, p, training, mask):
    if not training or p == 0.0:
        return x
    if p == 1.0:
        return x * 0
    return x * mask.to(x.dtype) / (1.0 - p)


def pre_attention(args, mask):
    x, residual, nw, nb, w, b, p, training, eps, heads, dim = args
    residual1 = residual.bfloat16() + masked_dropout(x.bfloat16(), p, training, mask)
    normalized = F.layer_norm(
        residual1.float(), (x.shape[-1],), nw.float(), nb.float(), eps
    ).bfloat16()
    qkv = F.linear(normalized, w.bfloat16(), b.bfloat16())
    return qkv.reshape(*x.shape[:-1], 3, heads, dim), residual1


def post_attention(args, mask):
    a, residual, ow, ob, nw, nb, w1, b1, w2, b2, p, training, eps = args
    projected = F.linear(a.bfloat16(), ow.bfloat16(), ob.bfloat16())
    residual2 = residual.bfloat16() + masked_dropout(projected, p, training, mask)
    normalized = F.layer_norm(
        residual2.float(), (a.shape[-1],), nw.float(), nb.float(), eps
    ).bfloat16()
    activated = F.gelu(
        F.linear(normalized, w1.bfloat16(), b1.bfloat16()), approximate="tanh"
    )
    return F.linear(activated, w2.bfloat16(), b2.bfloat16()), residual2


def block_forward(block, x, residual=None, cu_seqlens=None, max_seqlen=None):
    """Deterministic, no-dropout oracle for the full mixed-precision block."""
    assert not block.training or (
        block.dropout1.p == block.dropout2.p == block.mixer.attn.drop.p == 0
    )
    x = x.bfloat16()
    r = torch.zeros_like(x) if residual is None else residual.bfloat16()
    r = r + x
    norm = F.layer_norm(
        r.float(),
        (x.shape[-1],),
        block.norm_attn.weight.float(),
        block.norm_attn.bias.float(),
        block.norm_attn.eps,
    ).bfloat16()
    qkv = F.linear(
        norm, block.mixer.Wqkv.weight.bfloat16(), block.mixer.Wqkv.bias.bfloat16()
    )
    qkv = qkv.reshape(*x.shape[:-1], 3, block.mixer.num_heads, block.mixer.head_dim)
    kwargs = dict(
        causal=block.mixer.causal,
        softmax_scale=block.mixer.attn.softmax_scale,
        window_size=block.mixer.attn.window_size,
    )
    a = (
        attention(qkv, **kwargs)
        if cu_seqlens is None
        else packed_attention(qkv, cu_seqlens, max_seqlen, **kwargs)
    )
    a = a.reshape_as(x)
    r = r + F.linear(
        a, block.mixer.out_proj.weight.bfloat16(), block.mixer.out_proj.bias.bfloat16()
    )
    norm = F.layer_norm(
        r.float(),
        (x.shape[-1],),
        block.norm_mlp.weight.float(),
        block.norm_mlp.bias.float(),
        block.norm_mlp.eps,
    ).bfloat16()
    h = F.gelu(
        F.linear(norm, block.mlp.fc1.weight.bfloat16(), block.mlp.fc1.bias.bfloat16()),
        approximate="tanh",
    )
    return F.linear(
        h, block.mlp.fc2.weight.bfloat16(), block.mlp.fc2.bias.bfloat16()
    ), r


def clone_args(args):
    return tuple(
        a.detach().clone().requires_grad_(a.requires_grad)
        if isinstance(a, torch.Tensor)
        else a
        for a in args
    )


def cross_attention(
    q, kv, *, causal=False, dropout_p=0.0, softmax_scale=None, window_size=(-1, -1), **_
):
    """Independent FP32 rectangular attention, with safe fully-masked rows."""
    query = q.transpose(1, 2).float()
    key, value = (t.transpose(1, 2).float() for t in kv.unbind(dim=2))
    sq, sk = query.shape[-2], key.shape[-2]
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    scores = query @ key.transpose(-1, -2) * scale
    rows = torch.arange(sq, device=q.device)[:, None] + sk - sq
    cols = torch.arange(sk, device=q.device)[None, :]
    allowed = torch.ones((sq, sk), dtype=torch.bool, device=q.device)
    if causal:
        allowed &= cols <= rows
    left, right = window_size
    if left >= 0:
        allowed &= cols >= rows - left
    if right >= 0:
        allowed &= cols <= rows + right
    valid = allowed.any(-1, keepdim=True)
    logits = scores.masked_fill(~allowed, -torch.inf)
    logits = torch.where(valid, logits, torch.zeros_like(logits))
    probabilities = logits.softmax(-1) * valid.to(logits.dtype)
    probabilities = F.dropout(probabilities, dropout_p, training=dropout_p > 0)
    return (probabilities @ value).transpose(1, 2).to(q.dtype)


def packed_cross_attention(
    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
):
    qo, ko = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    assert len(qo) == len(ko)
    assert qo[0] == ko[0] == 0 and qo[-1] == len(q) and ko[-1] == len(kv)
    assert all(0 < b - a <= max_seqlen_q for a, b in zip(qo, qo[1:]))
    assert all(0 < b - a <= max_seqlen_k for a, b in zip(ko, ko[1:]))
    return torch.cat(
        [
            cross_attention(
                q[qa:qb].unsqueeze(0), kv[ka:kb].unsqueeze(0), **kwargs
            ).squeeze(0)
            for qa, qb, ka, kb in zip(qo, qo[1:], ko, ko[1:])
        ]
    )


class ReferenceCrossAttention(nn.Module):
    """Matches FlashCrossAttention's *actual* query metadata argument names."""

    def __init__(self, causal=False, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.drop = nn.Dropout(attention_dropout)
        self.softmax_scale = None
        self.window_size = (-1, -1)
        self.deterministic = False
        self.alibi_slopes = None

    def forward(
        self,
        q,
        kv,
        causal=None,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):
        kwargs = dict(
            causal=self.causal if causal is None else causal,
            dropout_p=self.drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            window_size=self.window_size,
        )
        if cu_seqlens is None:
            return cross_attention(q, kv, **kwargs)
        return packed_cross_attention(
            q, kv, cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k, **kwargs
        )


def cross_mha_forward(module, q, kv, cu_q=None, cu_k=None, max_q=None, max_k=None):
    pq = F.linear(q, module.Wq.weight, module.Wq.bias).reshape(
        *q.shape[:-1], module.num_heads, module.head_dim
    )
    pkv = F.linear(kv, module.Wkv.weight, module.Wkv.bias).reshape(
        *kv.shape[:-1], 2, module.num_heads, module.head_dim
    )
    kwargs = dict(
        causal=module.causal,
        softmax_scale=module.attn.softmax_scale,
        window_size=module.attn.window_size,
    )
    a = (
        cross_attention(pq, pkv, **kwargs)
        if cu_q is None
        else packed_cross_attention(pq, pkv, cu_q, cu_k, max_q, max_k, **kwargs)
    )
    return F.linear(a.reshape_as(q), module.out_proj.weight, module.out_proj.bias)


def memory_query_block_forward(block, x, residual, layout):
    """Independent eager oracle. Packed prefixes are sliced per sample;
    the optimized gather indices are deliberately not used here.
    """
    assert not block.training or all(
        p == 0
        for p in (
            block.dropout_input.p,
            block.dropout_sa.p,
            block.dropout_ca.p,
            block.memory_mixer.attn.drop.p,
            block.cross_mixer.attn.drop.p,
        )
    )

    def norm(value, layer):
        return F.layer_norm(
            value.float(),
            (value.shape[-1],),
            layer.weight.float(),
            layer.bias.float(),
            layer.eps,
        ).bfloat16()

    def linear(value, layer):
        return F.linear(
            value.reshape(-1, value.shape[-1]),
            layer.weight.bfloat16(),
            layer.bias.bfloat16(),
        ).reshape(*value.shape[:-1], layer.weight.shape[0])

    from varlen_transformer.layout import PackedMemoryQueryLayout

    packed = isinstance(layout, PackedMemoryQueryLayout)
    n = layout.num_memory_tokens
    h, r = (
        x.bfloat16(),
        torch.zeros_like(x, dtype=torch.bfloat16)
        if residual is None
        else residual.bfloat16(),
    )
    r0 = r + h
    m0, queries0 = r0.narrow(-2, 0, n), r0.narrow(-2, n, r0.shape[-2] - n)
    mixer = block.memory_mixer
    qkv = linear(norm(m0, block.norm_sa), mixer.Wqkv).reshape(
        *m0.shape[:-1], 3, mixer.num_heads, mixer.head_dim
    )
    opts = dict(
        causal=True,
        softmax_scale=mixer.attn.softmax_scale,
        window_size=mixer.attn.window_size,
    )
    am = (
        packed_attention(qkv, layout.cu_seqlens_m, layout.max_seqlen_m, **opts)
        if packed
        else attention(
            qkv.view(
                layout.batch_size,
                layout.memory_length,
                3,
                mixer.num_heads,
                mixer.head_dim,
            ),
            **opts,
        )
    )
    m1 = m0 + linear(am.reshape_as(m0), mixer.out_proj)
    rca = torch.cat((m1, queries0), dim=-2)
    normalized = norm(rca, block.norm_ca)
    cross = block.cross_mixer
    q_all = linear(normalized.narrow(-2, n, rca.shape[-2] - n), cross.Wq).reshape(
        *queries0.shape[:-1], cross.num_heads, cross.head_dim
    )
    kv_m = linear(normalized.narrow(-2, 0, n), cross.Wkv).reshape(
        *m1.shape[:-1], 2, cross.num_heads, cross.head_dim
    )
    opts = dict(
        softmax_scale=cross.attn.softmax_scale, window_size=cross.attn.window_size
    )
    if not packed:
        b = layout.batch_size
        fixed_memory = kv_m.view(
            b, layout.memory_length, 2, cross.num_heads, cross.head_dim
        )
        av = cross_attention(
            q_all[:b].view(b, 1, cross.num_heads, cross.head_dim),
            fixed_memory[:, : layout.x_prefix_length],
            causal=False,
            **opts,
        )
        aq = cross_attention(
            q_all[b:].view(b, layout.query_length, cross.num_heads, cross.head_dim),
            fixed_memory,
            causal=True,
            **opts,
        )
        av = av.reshape(b, cross.num_heads, cross.head_dim)
        aq = aq.reshape(layout.num_query_tokens, cross.num_heads, cross.head_dim)
    else:
        mo, qo, xo = (
            layout.cu_seqlens_m.tolist(),
            layout.cu_seqlens_q.tolist(),
            layout.cu_seqlens_x.tolist(),
        )
        avs, aqs = [], []
        b = len(mo) - 1
        for i in range(b):
            memory = kv_m[mo[i] : mo[i + 1]].unsqueeze(0)
            avs.append(
                cross_attention(
                    q_all[i : i + 1].unsqueeze(0),
                    memory[:, : xo[i + 1] - xo[i]],
                    causal=False,
                    **opts,
                ).squeeze(0)
            )
            aqs.append(
                cross_attention(
                    q_all[b + qo[i] : b + qo[i + 1]].unsqueeze(0),
                    memory,
                    causal=True,
                    **opts,
                ).squeeze(0)
            )
        av, aq = torch.cat(avs), torch.cat(aqs)
    projected = linear(torch.cat((av, aq), dim=-3).reshape_as(queries0), cross.out_proj)
    r1 = torch.cat((m1, queries0 + projected), dim=-2)
    z = norm(r1, block.norm_mlp)
    branch = linear(F.gelu(linear(z, block.mlp.fc1), approximate="tanh"), block.mlp.fc2)
    return branch, r1


def legacy_fixed_memory_query_reference(
    block, x, residual, memory_length, x_prefix_length
):
    """Independent original per-sample equations, for refactor parity checks.

    This oracle never packs the full state and never calls the native block or
    package arithmetic helpers. Parameters are shared with the tested module.
    """
    assert not block.training or all(
        p == 0
        for p in (
            block.dropout_input.p,
            block.dropout_sa.p,
            block.dropout_ca.p,
            block.memory_mixer.attn.drop.p,
            block.cross_mixer.attn.drop.p,
        )
    )

    def norm(value, layer):
        return F.layer_norm(
            value.float(),
            (value.shape[-1],),
            layer.weight.float(),
            layer.bias.float(),
            layer.eps,
        ).bfloat16()

    def linear(value, layer):
        return F.linear(
            value.reshape(-1, value.shape[-1]),
            layer.weight.bfloat16(),
            layer.bias.bfloat16(),
        ).reshape(*value.shape[:-1], layer.weight.shape[0])

    b, _, e = x.shape
    r0 = x.bfloat16() + (
        torch.zeros_like(x, dtype=torch.bfloat16)
        if residual is None
        else residual.bfloat16()
    )
    m0, queries = r0[:, :memory_length], r0[:, memory_length:]
    sa, ca = block.memory_mixer, block.cross_mixer
    qkv = linear(norm(m0, block.norm_sa), sa.Wqkv).reshape(
        b, memory_length, 3, sa.num_heads, sa.head_dim
    )
    am = attention(
        qkv,
        causal=True,
        softmax_scale=sa.attn.softmax_scale,
        window_size=sa.attn.window_size,
    ).reshape(b, memory_length, e)
    m1 = m0 + linear(am, sa.out_proj)
    rca = torch.cat((m1, queries), dim=1)
    normalized = norm(rca, block.norm_ca)
    q = linear(normalized[:, memory_length:], ca.Wq).reshape(
        b, queries.shape[1], ca.num_heads, ca.head_dim
    )
    kv = linear(normalized[:, :memory_length], ca.Wkv).reshape(
        b, memory_length, 2, ca.num_heads, ca.head_dim
    )
    opts = dict(softmax_scale=ca.attn.softmax_scale, window_size=ca.attn.window_size)
    av = cross_attention(q[:, :1], kv[:, :x_prefix_length], causal=False, **opts)
    aq = cross_attention(q[:, 1:], kv, causal=True, **opts)
    updated = queries + linear(
        torch.cat((av, aq), dim=1).reshape_as(queries), ca.out_proj
    )
    r1 = torch.cat((m1, updated), dim=1)
    z = norm(r1, block.norm_mlp)
    return linear(
        F.gelu(linear(z, block.mlp.fc1), approximate="tanh"), block.mlp.fc2
    ), r1


def pre_args(
    shape=(5, 16),
    *,
    device="cpu",
    dtype=torch.float32,
    p=0.0,
    training=True,
    requires_grad=True,
):
    e = shape[-1]

    def tensor(size, scale=0.2, shift=0.0):
        return (
            torch.randn(size, device=device, dtype=dtype) * scale + shift
        ).requires_grad_(requires_grad)

    return (
        tensor(shape, 1),
        tensor(shape, 0.5),
        tensor((e,), 0.1, 1),
        tensor((e,)),
        tensor((3 * e, e)),
        tensor((3 * e,)),
        p,
        training,
        1e-5,
        2,
        e // 2,
    )


def post_args(
    shape=(5, 16),
    *,
    device="cpu",
    dtype=torch.float32,
    p=0.0,
    training=True,
    requires_grad=True,
):
    e, intermediate = shape[-1], 2 * shape[-1]

    def tensor(size, scale=0.2, shift=0.0):
        return (
            torch.randn(size, device=device, dtype=dtype) * scale + shift
        ).requires_grad_(requires_grad)

    return (
        tensor(shape, 1),
        tensor(shape, 0.5),
        tensor((e, e)),
        tensor((e,)),
        tensor((e,), 0.1, 1),
        tensor((e,)),
        tensor((intermediate, e)),
        tensor((intermediate,)),
        tensor((e, intermediate)),
        tensor((e,)),
        p,
        training,
        1e-5,
    )


def assert_gradients(actual, expected, *, rtol=0.04, atol=0.02, relative_l2=0.03):
    """Check both elementwise error and total relative error from BF16 rounding."""
    assert len(actual) == len(expected)
    for index, (a, b) in enumerate(zip(actual, expected)):
        assert a is not None and b is not None, f"missing gradient {index}"
        assert torch.isfinite(a).all(), f"nonfinite gradient {index}"
        torch.testing.assert_close(
            a.float(),
            b.float(),
            rtol=rtol,
            atol=atol,
            msg=lambda msg, index=index: f"gradient {index}: {msg}",
        )
        denominator = b.float().norm().clamp_min(1e-6)
        assert (a.float() - b.float()).norm() / denominator <= relative_l2, (
            f"relative gradient error {index}"
        )
