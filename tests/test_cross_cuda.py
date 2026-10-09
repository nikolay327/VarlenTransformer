"""Real FlashAttention cross-attention and memory/query CUDA integration.

No reference-backend fixture is used here. Expected values use independent
FP32 attention, with explicit BF16 cast boundaries for the optimized block.
"""

import copy

import pytest
import torch

from tests.reference import (
    assert_gradients,
    cross_attention,
    cross_mha_forward,
    packed_cross_attention,
    memory_query_block_forward,
    clone_args,
)
from tests.memory_reference import (
    make_args,
    pre_sa_reference,
    pre_ca_reference,
    post_ca_reference,
)
from varlen_transformer.cross_mha import CrossMHA
from varlen_transformer.memory_query import MemoryQueryBlock
from varlen_transformer.layout import FixedMemoryQueryLayout, PackedMemoryQueryLayout
from varlen_transformer._memory_ops import mq_pre_sa, mq_pre_ca, mq_post_ca_mlp

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("sq,sk", [(3, 3), (3, 7), (7, 3)])
@pytest.mark.parametrize("e,heads", [(16, 1), (34, 2), (512, 2)])
def test_real_cross_mha_forward_input_and_every_parameter_gradient(
    dtype, packed, causal, sq, sk, e, heads
):
    module = CrossMHA(e, heads, causal=causal).to(device="cuda", dtype=dtype)
    assert type(module.attn).__module__.startswith("flash_attn")
    ref = copy.deepcopy(module)
    q = torch.randn(
        (2 * sq, e) if packed else (2, sq, e),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    kv = torch.randn(
        (2 * sk, e) if packed else (2, sk, e),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    eq, ek = q.detach().clone().requires_grad_(), kv.detach().clone().requires_grad_()
    meta = (
        (
            torch.tensor([0, sq, 2 * sq], device="cuda", dtype=torch.int32),
            torch.tensor([0, sk, 2 * sk], device="cuda", dtype=torch.int64),
            sq,
            sk,
        )
        if packed
        else (None,) * 4
    )
    actual, expected = module(q, kv, *meta), cross_mha_forward(ref, eq, ek, *meta)
    tol = 0.015 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol / 2)
    gradient = torch.randn_like(actual) / actual.numel() ** 0.5
    ag = torch.autograd.grad(actual, (q, kv, *module.parameters()), gradient)
    eg = torch.autograd.grad(expected, (eq, ek, *ref.parameters()), gradient)
    assert_gradients(ag, eg, atol=0.01)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("t,d", [(1, 1), (3, 4), (7, 2)])
def test_native_ncse_causal_alignment_and_exact_input_dependencies(packed, t, d):
    from flash_attn import flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func

    sk = t + d - 1
    q = torch.zeros(
        (d, 1, 8) if packed else (1, d, 1, 8),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    kv = torch.zeros(
        (sk, 2, 1, 8) if packed else (1, sk, 2, 1, 8),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with torch.no_grad():
        values = torch.arange(1, sk + 1, device="cuda")[:, None, None]
        if packed:
            kv[:, 1] = values
        else:
            kv[:, :, 1] = values
    if packed:
        cuq = torch.tensor([0, d], dtype=torch.int32, device="cuda")
        cuk = torch.tensor([0, sk], dtype=torch.int32, device="cuda")
        actual = flash_attn_varlen_kvpacked_func(
            q, kv, cuq, cuk, d, sk, causal=True, deterministic=True
        )
        expected = packed_cross_attention(q, kv, cuq, cuk, d, sk, causal=True)
    else:
        actual = flash_attn_kvpacked_func(q, kv, causal=True, deterministic=True)
        expected = cross_attention(q, kv, causal=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result = actual.reshape(d, 1, 8)
    means = torch.tensor(
        [(t + j + 1) / 2 for j in range(d)], device="cuda", dtype=torch.bfloat16
    )
    torch.testing.assert_close(result[:, 0, 0], means, rtol=0, atol=0)
    for j in range(d):
        grad = torch.autograd.grad(result[j].sum(), kv, retain_graph=True)[0].reshape(
            sk, 2, 1, 8
        )
        assert torch.count_nonzero(grad[t + j :]) == 0
        assert torch.count_nonzero(grad[: t + j, 1]) > 0


@pytest.mark.parametrize("packed", [False, True])
def test_native_rectangular_fully_masked_rows_are_zero_with_finite_backward(packed):
    from flash_attn import flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func

    q = torch.randn(5, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(
        2, 2, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    if packed:
        cq, ck = (
            torch.tensor([0, 5], device="cuda", dtype=torch.int32),
            torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        )
        actual = flash_attn_varlen_kvpacked_func(q, kv, cq, ck, 5, 2, causal=True)
    else:
        actual = flash_attn_kvpacked_func(
            q.unsqueeze(0), kv.unsqueeze(0), causal=True
        ).squeeze(0)
    torch.testing.assert_close(actual[:3], torch.zeros_like(actual[:3]), rtol=0, atol=0)
    actual.float().square().sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(kv.grad).all()


@pytest.mark.parametrize("d", [1, 127, 128, 129, 255, 257])
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_cross_tiling_boundaries_and_all_gradients(d, causal):
    from flash_attn import flash_attn_varlen_kvpacked_func

    lengths_q, lengths_k = [1, d], [3, d + 6]
    cuq = torch.tensor([0, 1, 1 + d], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, 3, d + 9], device="cuda", dtype=torch.int32)
    q = torch.randn(
        sum(lengths_q), 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    kv = torch.randn(
        sum(lengths_k),
        2,
        2,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    eq, ek = q.detach().clone().requires_grad_(), kv.detach().clone().requires_grad_()
    actual = flash_attn_varlen_kvpacked_func(
        q, kv, cuq, cuk, d, d + 6, causal=causal, deterministic=True
    )
    expected = packed_cross_attention(eq, ek, cuq, cuk, d, d + 6, causal=causal)
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.015)
    gradient = torch.randn_like(actual) / actual.numel() ** 0.5
    assert_gradients(
        torch.autograd.grad(actual, (q, kv), gradient),
        torch.autograd.grad(expected, (eq, ek), gradient),
        atol=0.005,
    )


def _block_inputs(packed, dtype=torch.float32):
    layout = (
        PackedMemoryQueryLayout.ncse([1, 7, 3], [1, 4, 6], device="cuda")
        if packed
        else FixedMemoryQueryLayout.ncse(7, 5)
    )
    shape = (
        (layout.total_tokens, 32)
        if packed
        else (2, layout.memory_length + 1 + layout.query_length, 32)
    )
    return torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True), layout


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "parameter_dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("with_residual", [False, True])
def test_real_memory_query_block_forward_and_every_gradient(
    packed, dtype, parameter_dtype, with_residual
):
    block = MemoryQueryBlock(32, 64, 2).to(device="cuda", dtype=parameter_dtype)
    assert type(block.memory_mixer.attn).__module__.startswith("flash_attn")
    assert type(block.cross_mixer.attn).__module__.startswith("flash_attn")
    ref = copy.deepcopy(block)
    x, layout = _block_inputs(packed, dtype)
    ex = x.detach().clone().requires_grad_()
    r = torch.randn_like(x, requires_grad=True) if with_residual else None
    er = r.detach().clone().requires_grad_() if r is not None else None
    actual = block(x, r, layout=layout)
    expected = memory_query_block_forward(ref, ex, er, layout)
    for a, b in zip(actual, expected):
        assert a.dtype == torch.bfloat16 and a.shape == x.shape
        torch.testing.assert_close(a, b, rtol=0.03, atol=0.015)
    grad = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual)
    tensors = (x, *((r,) if r is not None else ()), *block.parameters())
    et = (ex, *((er,) if er is not None else ()), *ref.parameters())
    ag = torch.autograd.grad(actual, tensors, grad)
    eg = torch.autograd.grad(expected, et, grad)
    for g, t in zip(ag, tensors):
        assert g.dtype == t.dtype and g.shape == t.shape
    assert_gradients(ag, eg, relative_l2=0.04)


@pytest.mark.parametrize("packed", [False, True])
def test_real_memory_query_stack_and_every_gradient(packed):
    blocks = torch.nn.ModuleList([MemoryQueryBlock(32, 64, 2).cuda() for _ in range(3)])
    refs = copy.deepcopy(blocks)
    x, layout = _block_inputs(packed)
    ex = x.detach().clone().requires_grad_()
    h, r, eh, er = x, None, ex, None
    for block, ref in zip(blocks, refs):
        h, r = block(h, r, layout=layout)
        eh, er = memory_query_block_forward(ref, eh, er, layout)
    actual, expected = h + r, eh + er
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.025)
    gradient = torch.randn_like(actual) / actual.numel() ** 0.5
    assert_gradients(
        torch.autograd.grad(actual, (x, *blocks.parameters()), gradient),
        torch.autograd.grad(expected, (ex, *refs.parameters()), gradient),
        relative_l2=0.04,
    )


@pytest.mark.parametrize(
    "kind,op,oracle,count",
    [
        ("sa", mq_pre_sa, pre_sa_reference, 2),
        ("ca", mq_pre_ca, pre_ca_reference, 3),
        ("mlp", mq_post_ca_mlp, post_ca_reference, 2),
    ],
)
@pytest.mark.parametrize(
    "p,training", [(0.0, True), (0.25, True), (0.25, False), (1.0, True)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_new_custom_ops_forward_and_fp32_accumulated_gradients(
    kind, op, oracle, count, p, training, dtype
):
    args = make_args(
        kind, (2, 9, 16), device="cuda", dtype=dtype, p=p, training=training
    )
    ea = clone_args(args)
    actual = op(*args)
    expected = oracle(ea, actual[count])
    for a, b in zip(actual[:count], expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    gradient = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual[:count])
    tensors = tuple(t for t in args if isinstance(t, torch.Tensor))
    et = tuple(t for t in ea if isinstance(t, torch.Tensor))
    ag = torch.autograd.grad(actual[:count], tensors, gradient)
    eg = torch.autograd.grad(expected, et, gradient)
    assert_gradients(ag, eg)
    if dtype == torch.float32:
        assert all(g.dtype == torch.float32 for g in ag)


@pytest.mark.parametrize(
    "kind,op", [("sa", mq_pre_sa), ("ca", mq_pre_ca), ("mlp", mq_post_ca_mlp)]
)
@pytest.mark.parametrize("p,training", [(0.0, True), (0.25, True), (0.25, False)])
def test_real_new_custom_op_registration(kind, op, p, training):
    result = torch.library.opcheck(
        op, make_args(kind, device="cuda", p=p, training=training)
    )
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.parametrize("packed", [False, True])
def test_real_packed_and_fixed_graph_has_no_forbidden_paths(packed):
    block = MemoryQueryBlock(32, 64, 2).cuda().eval()
    x, layout = _block_inputs(packed)
    baseline = sum(block(x, layout=layout))
    n = layout.num_memory_tokens
    changed = x.detach().clone()
    changed.narrow(-2, n, changed.shape[-2] - n).mul_(20)
    other = sum(block(changed, layout=layout))
    torch.testing.assert_close(
        baseline.narrow(-2, 0, n), other.narrow(-2, 0, n), rtol=0, atol=0
    )
    if packed:
        mo, qo, xo = (
            layout.cu_seqlens_m.tolist(),
            layout.cu_seqlens_q.tolist(),
            layout.cu_seqlens_x.tolist(),
        )
        for s in range(layout.batch_size):
            xv = xo[s + 1] - xo[s]
            gv = torch.autograd.grad(
                baseline[n + s].square().sum(), x, retain_graph=True
            )[0]
            allowed = list(range(mo[s], mo[s] + xv)) + [n + s]
            assert (
                torch.count_nonzero(gv[[i for i in range(len(x)) if i not in allowed]])
                == 0
            )
            assert torch.count_nonzero(gv[mo[s] : mo[s] + xv]) > 0
            for j in range(qo[s + 1] - qo[s]):
                qi = n + layout.batch_size + qo[s] + j
                gx = torch.autograd.grad(
                    baseline[qi].square().sum(), x, retain_graph=True
                )[0]
                allowed = list(range(mo[s], mo[s] + xv + j)) + [qi]
                assert (
                    torch.count_nonzero(
                        gx[[i for i in range(len(x)) if i not in allowed]]
                    )
                    == 0
                )
    else:
        for j in range(layout.query_length):
            gx = torch.autograd.grad(
                baseline[:, n + 1 + j].square().sum(), x, retain_graph=True
            )[0]
            assert torch.count_nonzero(gx[:, layout.x_prefix_length + j : n]) == 0
            assert torch.count_nonzero(gx[:, n : n + 1 + j]) == 0
            assert torch.count_nonzero(gx[:, n + 2 + j :]) == 0
        gv = torch.autograd.grad(baseline[:, n].square().sum(), x)[0]
        assert torch.count_nonzero(gv[:, layout.x_prefix_length : n]) == 0
        assert torch.count_nonzero(gv[:, n + 1 :]) == 0
        assert torch.count_nonzero(gv[:, : layout.x_prefix_length]) > 0


def test_real_cross_dropout_eval_and_sequence_isolation():
    module = CrossMHA(32, 2, attn_dropout=0.3).cuda().bfloat16().eval()
    q = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(9, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cq, ck = (
        torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
        torch.tensor([0, 3, 9], device="cuda", dtype=torch.int32),
    )
    a = module(q, kv, cq, ck, 3, 6)
    torch.testing.assert_close(a, module(q, kv, cq, ck, 3, 6), rtol=0, atol=0)
    other = kv.detach().clone()
    other[3:] *= 20
    torch.testing.assert_close(
        a[:2], module(q, other, cq, ck, 3, 6)[:2], rtol=0, atol=0
    )
    trained = module.train()(q, kv, cq, ck, 3, 6)
    assert not torch.equal(a, trained)
    trained.float().square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )


def test_real_block_dropout_and_optimizer_update():
    block = MemoryQueryBlock(
        32,
        64,
        2,
        attn_dropout=0.2,
        entry_resid_dropout=0.2,
        sa_resid_dropout=0.3,
        ca_resid_dropout=0.4,
    ).cuda()
    x, layout = _block_inputs(True)
    a = block.eval()(x, layout=layout)
    for left, right in zip(a, block(x, layout=layout)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(block.parameters(), lr=0.01)
    before = block.cross_mixer.Wkv.weight.detach().clone()
    h, r = block.train()(x, layout=layout)
    (h + r).float().square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()
    )
    optimizer.step()
    assert not torch.equal(before, block.cross_mixer.Wkv.weight)


@pytest.mark.parametrize("memory_length", [1, 127, 128, 129, 255, 257])
@pytest.mark.parametrize("packed", [False, True])
def test_real_memory_query_block_at_memory_tile_boundaries(memory_length, packed):
    d = min(4, memory_length)
    t = memory_length - d + 1
    layout = (
        PackedMemoryQueryLayout.ncse([t, 1], [d, 1], device="cuda")
        if packed
        else FixedMemoryQueryLayout.ncse(t, d)
    )
    shape = (layout.total_tokens, 16) if packed else (1, memory_length + 1 + d, 16)
    block = MemoryQueryBlock(16, 32, 2).cuda()
    ref = copy.deepcopy(block)
    x = torch.randn(shape, device="cuda", requires_grad=True)
    ex = x.detach().clone().requires_grad_()
    actual = block(x, layout=layout)
    expected = memory_query_block_forward(ref, ex, None, layout)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0.03, atol=0.015)
    grad = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual)
    assert_gradients(
        torch.autograd.grad(actual, (x, *block.parameters()), grad),
        torch.autograd.grad(expected, (ex, *ref.parameters()), grad),
        relative_l2=0.04,
    )


def test_real_fixed_and_uniform_packed_equivalence_and_input_gradients():
    fixed = FixedMemoryQueryLayout.ncse(3, 4)
    packed = PackedMemoryQueryLayout.ncse([3, 3], [4, 4], device="cuda")
    block = MemoryQueryBlock(32, 64, 2).cuda().eval()
    x = torch.randn(2, 11, 32, device="cuda", requires_grad=True)
    px = torch.cat((x[:, :6].flatten(0, 1), x[:, 6], x[:, 7:].flatten(0, 1)))
    a, p = block(x, layout=fixed), block(px, layout=packed)
    for actual, other in zip(a, p):
        repacked = torch.cat(
            (actual[:, :6].flatten(0, 1), actual[:, 6], actual[:, 7:].flatten(0, 1))
        )
        torch.testing.assert_close(repacked, other, rtol=0.025, atol=0.015)
    ga = torch.autograd.grad(
        sum(t.float().square().mean() for t in a), x, retain_graph=True
    )[0]
    gp = torch.autograd.grad(sum(t.float().square().mean() for t in p), x)[0]
    assert_gradients((ga,), (gp,), relative_l2=0.04)
