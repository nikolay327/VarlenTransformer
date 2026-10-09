import copy

import pytest
import torch

from tests.reference import cross_attention, cross_mha_forward
from varlen_transformer.cross_mha import CrossMHA


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sq,sk", [(3, 3), (3, 6), (6, 3), (1, 1)])
@pytest.mark.parametrize("heads,dim", [(1, 16), (2, 17), (2, 256)])
def test_cross_module_forward_and_every_gradient(
    cross_reference_backend, packed, causal, sq, sk, heads, dim
):
    module = CrossMHA(heads * dim, heads, causal=causal)
    oracle = copy.deepcopy(module)
    q = torch.randn(
        (2 * sq, heads * dim) if packed else (2, sq, heads * dim), requires_grad=True
    )
    kv = torch.randn(
        (2 * sk, heads * dim) if packed else (2, sk, heads * dim), requires_grad=True
    )
    eq, ekv = q.detach().clone().requires_grad_(), kv.detach().clone().requires_grad_()
    metadata = (
        (torch.tensor([0, sq, 2 * sq]), torch.tensor([0, sk, 2 * sk]), sq, sk)
        if packed
        else (None,) * 4
    )
    actual, expected = (
        module(q, kv, *metadata),
        cross_mha_forward(oracle, eq, ekv, *metadata),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual) / actual.numel() ** 0.5
    ag = torch.autograd.grad(actual, (q, kv, *module.parameters()), gradient)
    eg = torch.autograd.grad(expected, (eq, ekv, *oracle.parameters()), gradient)
    for a, e in zip(ag, eg):
        torch.testing.assert_close(a, e, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("sq,sk", [(2, 5), (5, 2), (4, 4), (1, 7)])
def test_rectangular_bottom_right_geometry_and_zero_rows(sq, sk):
    q = torch.zeros(1, sq, 1, 1, requires_grad=True)
    kv = torch.zeros(1, sk, 2, 1, 1, requires_grad=True)
    with torch.no_grad():
        kv[:, :, 1, 0, 0] = torch.arange(1, sk + 1)
    actual = cross_attention(q, kv, causal=True).flatten()
    expected = torch.tensor(
        [0.0 if i + sk - sq < 0 else (i + sk - sq + 2) / 2 for i in range(sq)]
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(kv.grad).all()


def test_ncse_geometry_has_exact_allowed_prefixes():
    t, d = 3, 4
    q = torch.zeros(1, d, 1, 1)
    kv = torch.zeros(1, t + d - 1, 2, 1, 1)
    kv[:, :, 1, 0, 0] = torch.arange(1, t + d)
    actual = cross_attention(q, kv, causal=True).flatten()
    torch.testing.assert_close(
        actual, torch.tensor([(t + j + 1) / 2 for j in range(d)])
    )


def test_cross_dropout_eval_and_packed_isolation(cross_reference_backend):
    module = CrossMHA(16, 2, attn_dropout=0.3).eval()
    q, kv = (
        torch.randn(5, 16, requires_grad=True),
        torch.randn(9, 16, requires_grad=True),
    )
    meta = (torch.tensor([0, 2, 5]), torch.tensor([0, 3, 9]), 3, 6)
    a = module(q, kv, *meta)
    torch.manual_seed(99)
    torch.testing.assert_close(a, module(q, kv, *meta), rtol=0, atol=0)
    other = kv.detach().clone()
    other[3:] *= 50
    torch.testing.assert_close(a[:2], module(q, other, *meta)[:2], rtol=0, atol=0)
    changed = module.train()(q, kv, *meta)
    assert not torch.equal(a, changed)
    changed.square().sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )


@pytest.mark.parametrize(
    "case",
    [
        "metadata",
        "batch",
        "offset_batch",
        "dtype",
        "rank",
        "max",
        "width",
        "empty",
        "offset_dtype",
    ],
)
def test_cross_validation(cross_reference_backend, case):
    module = CrossMHA(16, 2)
    q, kv, metadata = torch.randn(2, 3, 16), torch.randn(2, 4, 16), {}
    if case == "metadata":
        metadata = {"cu_seqlens_q": torch.tensor([0, 3, 6])}
    if case == "batch":
        kv = kv[:1]
    if case == "dtype":
        kv = kv.double()
    if case == "rank":
        q = q.flatten(0, 1)
    if case == "width":
        kv = kv[..., :8]
    if case == "empty":
        q = q[:, :0]
    if case in ("offset_batch", "max", "offset_dtype"):
        q, kv = q.flatten(0, 1), kv.flatten(0, 1)
        metadata = dict(
            cu_seqlens_q=torch.tensor([0, 3, 6]),
            cu_seqlens_k=torch.tensor([0, 4, 8]),
            max_seqlen_q=3,
            max_seqlen_k=4,
        )
        if case == "offset_batch":
            metadata["cu_seqlens_k"] = torch.tensor([0, 8])
        if case == "max":
            metadata["max_seqlen_k"] = 0
        if case == "offset_dtype":
            metadata["cu_seqlens_k"] = metadata["cu_seqlens_k"].float()
    with pytest.raises(ValueError):
        module(q, kv, **metadata)


@pytest.mark.parametrize("args", [(0, 2), (8, 0), (17, 2), (257, 1)])
def test_cross_dimension_validation_before_dependency(args):
    with pytest.raises(ValueError):
        CrossMHA(*args)


@pytest.mark.parametrize("p", [-0.1, 1.0, float("nan"), float("inf")])
def test_cross_invalid_dropout_before_dependency(p):
    with pytest.raises(ValueError):
        CrossMHA(16, 2, attn_dropout=p)


def test_cross_missing_backend_is_explicit(monkeypatch):
    import varlen_transformer.cross_mha as module

    monkeypatch.setattr(module, "FlashCrossAttention", None)
    with pytest.raises(ImportError, match="FlashAttention"):
        CrossMHA(16, 2)


def test_cross_noncontiguous_inputs_and_independent_offsets(cross_reference_backend):
    cross = CrossMHA(16, 2, causal=True)
    q = torch.randn(5, 16).T.contiguous().T.requires_grad_()
    kv = torch.randn(9, 16).T.contiguous().T.requires_grad_()
    cq = torch.tensor([0, -1, 2, -1, 5, -1], dtype=torch.int64)[::2]
    ck = torch.tensor([0, -1, 3, -1, 9, -1], dtype=torch.int64)[::2]
    actual = cross(q, kv, cq, ck, 3, 6)
    expected = cross_mha_forward(cross, q, kv, cq, ck, 3, 6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(kv.grad).all()
