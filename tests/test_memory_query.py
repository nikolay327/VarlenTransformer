import copy

import pytest
import torch

from tests.reference import assert_gradients, memory_query_block_forward
from varlen_transformer.layout import FixedMemoryQueryLayout, PackedMemoryQueryLayout
from varlen_transformer.memory_query import MemoryQueryBlock


def inputs(packed, *, dtype=torch.float32, device="cpu", requires_grad=True):
    layout = (
        PackedMemoryQueryLayout.ncse([3, 2, 3], [3, 2, 4], device=device)
        if packed
        else FixedMemoryQueryLayout.ncse(3, 4)
    )
    shape = (
        (layout.total_tokens, 16)
        if packed
        else (2, layout.memory_length + 1 + layout.query_length, 16)
    )
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    return x, layout


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
def test_block_outputs_and_every_gradient(
    memory_reference_backend, packed, dtype, with_residual, parameter_dtype
):
    block = MemoryQueryBlock(16, 32, 2).to(dtype=parameter_dtype)
    oracle = copy.deepcopy(block)
    x, layout = inputs(packed, dtype=dtype)
    ex = x.detach().clone().requires_grad_()
    r = torch.randn_like(x, requires_grad=True) if with_residual else None
    er = r.detach().clone().requires_grad_() if r is not None else None
    actual = block(x, r, layout=layout)
    expected = memory_query_block_forward(oracle, ex, er, layout)
    for a, b in zip(actual, expected):
        assert a.dtype == torch.bfloat16 and a.shape == x.shape
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    tensors = (x, *((r,) if r is not None else ()), *block.parameters())
    et = (ex, *((er,) if er is not None else ()), *oracle.parameters())
    grad = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual)
    ag, eg = (
        torch.autograd.grad(actual, tensors, grad),
        torch.autograd.grad(expected, et, grad),
    )
    for g, t in zip(ag, tensors):
        assert g.shape == t.shape and g.dtype == t.dtype
    assert_gradients(ag, eg, relative_l2=0.04)


@pytest.mark.parametrize("packed", [False, True])
def test_stack_chaining_and_all_gradients(memory_reference_backend, packed):
    blocks = torch.nn.ModuleList([MemoryQueryBlock(16, 32, 2) for _ in range(3)])
    refs = copy.deepcopy(blocks)
    x, layout = inputs(packed)
    ex = x.detach().clone().requires_grad_()
    h, r, eh, er = x, None, ex, None
    for block, ref in zip(blocks, refs):
        h, r = block(h, r, layout=layout)
        eh, er = memory_query_block_forward(ref, eh, er, layout)
    actual, expected = h + r, eh + er
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual) / actual.numel() ** 0.5
    ag = torch.autograd.grad(actual, (x, *blocks.parameters()), grad)
    eg = torch.autograd.grad(expected, (ex, *refs.parameters()), grad)
    assert_gradients(ag, eg, relative_l2=0.04)


@pytest.mark.parametrize("depth", [1, 3])
def test_fixed_dependency_graph_perturbations_and_input_jacobian(
    memory_reference_backend, depth
):
    layout = FixedMemoryQueryLayout.ncse(3, 4)
    blocks = [MemoryQueryBlock(16, 32, 2).eval() for _ in range(depth)]
    x = torch.randn(1, 11, 16, requires_grad=True)

    def run(value):
        h, r = value, None
        for block in blocks:
            h, r = block(h, r, layout=layout)
        return h + r

    actual = run(x)
    changed = x.detach().clone()
    changed[:, 6:] *= 20
    torch.testing.assert_close(actual[:, :6], run(changed)[:, :6], rtol=0, atol=0)
    # No V -> Q, Q -> V, or Q -> Q path, even through stacked shared FFNs.
    for index in range(6, 11):
        changed = x.detach().clone()
        changed[:, index] = torch.randn_like(changed[:, index]) * 10
        other = run(changed)
        keep = [i for i in range(11) if i != index]
        torch.testing.assert_close(actual[:, keep], other[:, keep], rtol=0, atol=0)
    for j in range(4):
        gx = torch.autograd.grad(actual[:, 7 + j].square().sum(), x, retain_graph=True)[
            0
        ]
        assert torch.count_nonzero(gx[:, :3]) > 0
        if j:
            assert torch.count_nonzero(gx[:, 3 : 3 + j]) > 0
        assert torch.count_nonzero(gx[:, 3 + j : 6]) == 0
        assert torch.count_nonzero(gx[:, 6 : 7 + j]) == 0
        assert torch.count_nonzero(gx[:, 8 + j :]) == 0
    gv = torch.autograd.grad(actual[:, 6].square().sum(), x)[0]
    assert torch.count_nonzero(gv[:, :3]) > 0
    assert torch.count_nonzero(gv[:, 3:6]) == torch.count_nonzero(gv[:, 7:]) == 0
    changed = x.detach().clone()
    changed[:, 3:6] *= 40
    torch.testing.assert_close(actual[:, 6], run(changed)[:, 6], rtol=0, atol=0)


def test_packed_isolation_prefix_gather_and_dependency_graph(memory_reference_backend):
    block = MemoryQueryBlock(16, 32, 2).eval()
    x, layout = inputs(True)
    h, r = block(x, layout=layout)
    y = h + r
    nm, b = layout.num_memory_tokens, layout.batch_size
    mo, qo = layout.cu_seqlens_m.tolist(), layout.cu_seqlens_q.tolist()
    xl = (3, 2, 3)
    for sample in range(b):
        gv = torch.autograd.grad(y[nm + sample].square().sum(), x, retain_graph=True)[0]
        permitted = list(range(mo[sample], mo[sample] + xl[sample])) + [nm + sample]
        forbidden = [i for i in range(len(x)) if i not in permitted]
        assert torch.count_nonzero(gv[forbidden]) == 0
        assert torch.count_nonzero(gv[mo[sample] : mo[sample] + xl[sample]]) > 0
        for j in range(qo[sample + 1] - qo[sample]):
            index = nm + b + qo[sample] + j
            gx = torch.autograd.grad(y[index].square().sum(), x, retain_graph=True)[0]
            allowed = list(range(mo[sample], mo[sample] + xl[sample] + j)) + [index]
            forbidden = [i for i in range(len(x)) if i not in allowed]
            assert torch.count_nonzero(gx[forbidden]) == 0
    # Change an entire other sample, across all three stream regions.
    changed = x.detach().clone()
    changed[mo[1] : mo[2]] *= 10
    changed[nm + 1] *= 10
    changed[nm + b + qo[1] : nm + b + qo[2]] *= 10
    other = sum(block(changed, layout=layout))
    keep = (
        list(range(mo[0], mo[1])) + [nm] + list(range(nm + b + qo[0], nm + b + qo[1]))
    )
    torch.testing.assert_close(y[keep], other[keep], rtol=0, atol=0)


def test_fixed_and_uniform_packed_equivalence_and_gradient_scatter(
    memory_reference_backend,
):
    fixed = FixedMemoryQueryLayout.ncse(3, 4)
    packed = PackedMemoryQueryLayout.ncse([3, 3], [4, 4])
    block = MemoryQueryBlock(16, 32, 2).eval()
    x = torch.randn(2, 11, 16, requires_grad=True)
    px = torch.cat((x[:, :6].flatten(0, 1), x[:, 6], x[:, 7:].flatten(0, 1)))
    a, p = block(x, layout=fixed), block(px, layout=packed)
    for actual, other in zip(a, p):
        repacked = torch.cat(
            (actual[:, :6].flatten(0, 1), actual[:, 6], actual[:, 7:].flatten(0, 1))
        )
        torch.testing.assert_close(repacked, other, rtol=0, atol=0)
    ga = torch.autograd.grad(
        sum(t.float().square().sum() for t in a), x, retain_graph=True
    )[0]
    gp = torch.autograd.grad(sum(t.float().square().sum() for t in p), x)[0]
    torch.testing.assert_close(ga, gp, rtol=0.02, atol=0.01)


def test_exact_three_kernel_calls_and_shared_fused_operations(
    memory_reference_backend, monkeypatch
):
    import varlen_transformer._memory_ops as ops

    linear_calls, norm_calls = [], []
    linear, norm = ops._linear_fwd, ops._layernorm_fwd_fp32

    def observe_linear(x, w, b):
        linear_calls.append((x.shape, w.shape))
        return linear(x, w, b)

    def observe_norm(x, w, b, eps):
        norm_calls.append((x.shape, w))
        return norm(x, w, b, eps)

    monkeypatch.setattr(ops, "_linear_fwd", observe_linear)
    monkeypatch.setattr(ops, "_layernorm_fwd_fp32", observe_norm)
    block = MemoryQueryBlock(16, 32, 2).eval()
    x, layout = inputs(True, requires_grad=False)
    result = block(x, layout=layout)
    assert [(c["kind"], c["causal"]) for c in memory_reference_backend] == [
        ("sa", True),
        ("ca", False),
        ("ca", True),
    ]
    assert len(linear_calls) == 7 and len(norm_calls) == 3
    n, tail = layout.num_memory_tokens, layout.total_tokens - layout.num_memory_tokens
    assert [s for s, w in linear_calls] == [
        (n, 16),
        (n, 16),
        (tail, 16),
        (n, 16),
        (tail, 16),
        (layout.total_tokens, 16),
        (layout.total_tokens, 32),
    ]
    assert [s for s, w in norm_calls] == [(n, 16), x.shape, x.shape]
    assert (
        norm_calls[1][1] is block.norm_ca.weight
        and norm_calls[2][1] is block.norm_mlp.weight
    )
    assert len([k for k in block.state_dict() if k.endswith("Wq.weight")]) == 1
    assert len([k for k in block.state_dict() if k.endswith("Wkv.weight")]) == 1
    for c in memory_reference_backend:
        assert c["dropout_p"] == 0 and c["softcap"] == 0 and c["alibi_slopes"] is None
    with torch.no_grad():
        block.mlp.fc2.bias.add_(3)
    changed = block(x, layout=layout)
    torch.testing.assert_close(result[1], changed[1], rtol=0, atol=0)
    for region in (
        slice(0, n),
        slice(n, n + layout.batch_size),
        slice(n + layout.batch_size, None),
    ):
        assert not torch.equal(result[0][region], changed[0][region])


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "entry,sa,ca", [(0.0, 0.0, 0.0), (0.2, 0.3, 0.4), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0)]
)
def test_dropout_eval_training_and_finite_gradients(
    memory_reference_backend, packed, entry, sa, ca
):
    block = MemoryQueryBlock(
        16,
        32,
        2,
        attn_dropout=0.2,
        entry_resid_dropout=entry,
        sa_resid_dropout=sa,
        ca_resid_dropout=ca,
    )
    x, layout = inputs(packed)
    a = block.eval()(x, layout=layout)
    torch.manual_seed(66)
    other = block(x, layout=layout)
    for left, right in zip(a, other):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    c = block.train()(x, layout=layout)
    sum(t.float().square().mean() for t in c).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()
    )
    assert torch.isfinite(x.grad).all()
    if entry == 1:
        assert torch.count_nonzero(x.grad) == 0


def test_noncontiguous_frozen_serialization_optimizer(
    memory_reference_backend, tmp_path
):
    block = MemoryQueryBlock(16, 32, 2).eval()
    x, layout = inputs(True, requires_grad=False)
    x = x.T.contiguous().T.detach().requires_grad_()
    path = tmp_path / "memory-query.pt"
    torch.save(block.state_dict(), path)
    other = MemoryQueryBlock(16, 32, 2).eval()
    other.load_state_dict(torch.load(path, weights_only=True))
    for a, b in zip(block(x, layout=layout), other(x, layout=layout)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    before = other.cross_mixer.Wkv.weight.detach().clone()
    optimizer = torch.optim.AdamW(other.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        h, r = other(x, layout=layout)
        (h + r).float().square().mean().backward()
        optimizer.step()
    assert not torch.equal(before, other.cross_mixer.Wkv.weight)
    other.requires_grad_(False)
    x.grad = None
    sum(other(x, layout=layout)).float().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_forward_never_reads_offset_or_index_values(
    memory_reference_backend, monkeypatch
):
    import varlen_transformer.memory_query as mq

    x, layout = inputs(True, requires_grad=False)
    block = MemoryQueryBlock(16, 32, 2).eval()

    def sa(qkv, *args, **kw):
        return qkv.select(-3, 0).clone()

    def ca(q, kv, *args, **kw):
        return q.clone()

    def forbidden(*args, **kw):
        raise AssertionError("host scalar/offset read inside forward")

    monkeypatch.setattr(mq, "flash_attn_varlen_qkvpacked_func", sa)
    monkeypatch.setattr(mq, "flash_attn_varlen_kvpacked_func", ca)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    block(x, layout=layout)


@pytest.mark.parametrize(
    "case", ["layout", "rank", "shape", "integer", "residual_shape", "residual_dtype"]
)
def test_block_validation(memory_reference_backend, case):
    block = MemoryQueryBlock(16, 32, 2)
    x, layout = inputs(False, requires_grad=False)
    residual = None
    if case == "layout":
        layout = None
    if case == "rank":
        x = x.flatten(0, 1)
    if case == "shape":
        x = x[:, :-1]
    if case == "integer":
        x = x.long()
    if case == "residual_shape":
        residual = x[:, :-1]
    if case == "residual_dtype":
        residual = x.long()
    with pytest.raises(ValueError):
        block(x, residual, layout=layout)


@pytest.mark.parametrize(
    "lengths",
    [
        ([], [], []),
        ([2], [1, 2], [1]),
        ([2], [0], [1]),
        ([1], [2], [2]),
        ([True], [1], [1]),
    ],
)
def test_layout_invalid_host_lengths(lengths):
    with pytest.raises(ValueError):
        PackedMemoryQueryLayout.from_lengths(*lengths)


def test_layout_indices_metadata_and_validation():
    layout = PackedMemoryQueryLayout.from_lengths([5, 3, 6], [3, 2, 4], [3, 2, 3])
    assert layout.x_prefix_indices.tolist() == [0, 1, 2, 5, 6, 8, 9, 10]
    assert layout.cu_seqlens_v.tolist() == [0, 1, 2, 3]
    assert layout.to("cpu").total_tokens == 26
    with pytest.raises(ValueError):
        FixedMemoryQueryLayout(2, 2, 3)
    with pytest.raises(ValueError):
        PackedMemoryQueryLayout.from_lengths(torch.tensor([3]), [2], [1])


def test_public_factory_exports_and_configuration(memory_reference_backend):
    from varlen_transformer import create_memory_query_block, CrossMHA

    block = create_memory_query_block(
        16,
        32,
        2,
        attn_dropout=0.2,
        entry_resid_dropout=0.3,
        sa_resid_dropout=0.4,
        ca_resid_dropout=0.5,
        norm_eps=1e-4,
    )
    assert isinstance(block, MemoryQueryBlock) and isinstance(
        block.cross_mixer, CrossMHA
    )
    assert block.memory_mixer.causal and block.cross_mixer.causal
    assert (
        block.dropout_input.p == 0.3
        and block.dropout_sa.p == 0.4
        and block.dropout_ca.p == 0.5
    )
    assert block.cross_mixer.attn.drop.p == block.memory_mixer.attn.drop.p == 0.2
    assert block.norm_ca.eps == block.norm_mlp.eps == block.norm_sa.eps == 1e-4
    default = create_memory_query_block(16, 32, 2)
    assert default.dropout_ca.p == default.dropout_sa.p == default.dropout_input.p == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entry_resid_dropout": -1},
        {"sa_resid_dropout": 1.1},
        {"ca_resid_dropout": float("nan")},
        {"norm_eps": 0},
        {"norm_eps": float("inf")},
    ],
)
def test_invalid_block_configuration(memory_reference_backend, kwargs):
    with pytest.raises(ValueError):
        MemoryQueryBlock(16, 32, 2, **kwargs)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("sm,d,t", [(3, 6, 2), (4, 4, 3), (7, 3, 4)])
def test_non_ncse_block_geometry_is_rejected(packed, sm, d, t):
    with pytest.raises(ValueError, match="NCSE"):
        if packed:
            PackedMemoryQueryLayout.from_lengths(
                [sm, sm + 1], [d, max(1, d - 1)], [t, t]
            )
        else:
            FixedMemoryQueryLayout(sm, d, t)


@pytest.mark.parametrize(
    "case",
    [
        "rank",
        "short",
        "dtype",
        "batch",
        "device",
        "index_rank",
        "index_dtype",
        "index_size",
        "query_count",
        "max",
    ],
)
def test_precomputed_layout_rejects_malformed_structure(case):
    from dataclasses import replace

    layout = PackedMemoryQueryLayout.from_lengths([5, 3], [3, 2], [3, 2])
    changes = {}
    if case == "rank":
        changes["cu_seqlens_q"] = layout.cu_seqlens_q.unsqueeze(0)
    if case == "short":
        changes["cu_seqlens_m"] = torch.tensor([0], dtype=torch.int32)
    if case == "dtype":
        changes["cu_seqlens_x"] = layout.cu_seqlens_x.float()
    if case == "batch":
        changes["cu_seqlens_q"] = torch.tensor([0, 1, 3, 5], dtype=torch.int32)
    if case == "device":
        changes["x_prefix_indices"] = torch.empty(5, dtype=torch.int64, device="meta")
    if case == "index_rank":
        changes["x_prefix_indices"] = layout.x_prefix_indices.unsqueeze(0)
    if case == "index_dtype":
        changes["x_prefix_indices"] = layout.x_prefix_indices.int()
    if case == "index_size":
        changes["x_prefix_indices"] = layout.x_prefix_indices[:1]
    if case == "query_count":
        changes["num_query_tokens"] = 1
    if case == "max":
        changes["max_seqlen_x"] = 0
    with pytest.raises(ValueError):
        replace(layout, **changes)


def test_layout_integer_conversion_nonstrided_offsets_and_state_validation():
    from dataclasses import replace

    layout = PackedMemoryQueryLayout.from_lengths([5, 3], [3, 2], [3, 2])
    padded = torch.tensor([0, -1, 5, -1, 8, -1], dtype=torch.int64)[::2]
    converted = replace(layout, cu_seqlens_m=padded)
    assert (
        converted.cu_seqlens_m.dtype == torch.int32
        and converted.cu_seqlens_m.is_contiguous()
    )
    with pytest.raises(ValueError):
        layout.validate(torch.randn(layout.total_tokens - 1, 16), 16)
    with pytest.raises(ValueError):
        layout.validate(torch.empty(layout.total_tokens, 16, device="meta"), 16)
    with pytest.raises(ValueError):
        PackedMemoryQueryLayout.ncse(torch.tensor([3]), [2])
    with pytest.raises(ValueError):
        PackedMemoryQueryLayout.ncse([3], [2, 3])
    with pytest.raises(ValueError):
        PackedMemoryQueryLayout.from_lengths([2**31], [1], [1])
