"""Independent eager equations for the three new custom-op regions."""

import torch
from torch.nn import functional as F

from tests.reference import masked_dropout


def _norm(x, nw, nb, eps):
    return F.layer_norm(
        x.float(), (x.shape[-1],), nw.float(), nb.float(), eps
    ).bfloat16()


def _linear(x, w, b):
    return F.linear(
        x.bfloat16().reshape(-1, x.shape[-1]), w.bfloat16(), b.bfloat16()
    ).reshape(*x.shape[:-1], w.shape[0])


def pre_sa_reference(args, mask):
    h, r, nw, nb, w, b, p, train, eps, heads, dim, n = args
    r0 = r.bfloat16() + masked_dropout(h.bfloat16(), p, train, mask)
    nm = _norm(r0.narrow(-2, 0, n), nw, nb, eps)
    return _linear(nm, w, b).reshape(*nm.shape[:-1], 3, heads, dim), r0


def pre_ca_reference(args, mask):
    am, r0, ow, ob, nw, nb, qw, qb, kvw, kvb, p, train, eps, heads, dim, n = args
    r0 = r0.bfloat16()
    m = r0.narrow(-2, 0, n) + masked_dropout(_linear(am, ow, ob), p, train, mask)
    queries = r0.narrow(-2, n, r0.shape[-2] - n)
    rca = torch.cat((m, queries), dim=-2)
    normalized = _norm(rca, nw, nb, eps)
    nq, nm = normalized.narrow(-2, n, rca.shape[-2] - n), normalized.narrow(-2, 0, n)
    return (
        _linear(nq, qw, qb).reshape(*nq.shape[:-1], heads, dim),
        _linear(nm, kvw, kvb).reshape(*nm.shape[:-1], 2, heads, dim),
        rca,
    )


def post_ca_reference(args, mask):
    aq, rca, ow, ob, nw, nb, w1, b1, w2, b2, p, train, eps, n = args
    rca = rca.bfloat16()
    m = rca.narrow(-2, 0, n)
    queries = rca.narrow(-2, n, rca.shape[-2] - n) + masked_dropout(
        _linear(aq, ow, ob), p, train, mask
    )
    r1 = torch.cat((m, queries), dim=-2)
    z = _norm(r1, nw, nb, eps)
    return _linear(F.gelu(_linear(z, w1, b1), approximate="tanh"), w2, b2), r1


def make_args(
    kind,
    shape=(9, 16),
    *,
    dtype=torch.float32,
    device="cpu",
    p=0.0,
    training=True,
    requires_grad=True,
):
    e, n = shape[-1], 5
    ms, qs = (*shape[:-2], n, e), (*shape[:-2], shape[-2] - n, e)

    def tensor(size, scale=0.15, shift=0.0):
        return (
            torch.randn(size, dtype=dtype, device=device) * scale + shift
        ).requires_grad_(requires_grad)

    def norm():
        return (tensor((e,), 0.1, 1), tensor((e,)))

    def linear(out, inp):
        return (tensor((out, inp)), tensor((out,)))

    if kind == "sa":
        return (
            tensor(shape, 1),
            tensor(shape, 0.5),
            *norm(),
            *linear(3 * e, e),
            p,
            training,
            1e-5,
            2,
            e // 2,
            n,
        )
    if kind == "ca":
        return (
            tensor(ms, 1),
            tensor(shape, 0.5),
            *linear(e, e),
            *norm(),
            *linear(e, e),
            *linear(2 * e, e),
            p,
            training,
            1e-5,
            2,
            e // 2,
            n,
        )
    return (
        tensor(qs, 1),
        tensor(shape, 0.5),
        *linear(e, e),
        *norm(),
        *linear(2 * e, e),
        *linear(e, 2 * e),
        p,
        training,
        1e-5,
        n,
    )
