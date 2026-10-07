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
