import copy

import pytest
import torch

from tests.reference import (
    assert_gradients,
    memory_query_block_forward,
    legacy_fixed_memory_query_reference,
)
from varlen_transformer import (
    FixedMemoryQueryLayout,
    PackedMemoryQueryLayout,
    MemoryQueryBlock,
    FixedMemoryQueryStack,
    pack_fixed_memory_query,
    unpack_fixed_memory_query,
)


def layout_for(packed):
    return (
        PackedMemoryQueryLayout.ncse([1, 3, 2], [1, 4, 2])
        if packed
        else FixedMemoryQueryLayout.ncse(3, 4, batch_size=2)
    )


@pytest.fixture(scope="module")
def cpu_fp16_cast_supported():
    try:
        torch.tensor([1.25], dtype=torch.float32).to(torch.float16)
    except RuntimeError as exc:
        if "Failed to initialize cpuinfo" not in str(exc):
            raise
        return False
    return True


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "parameter_dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_mixed_precision_native_stacks_every_gradient(
    memory_reference_backend,
    cpu_fp16_cast_supported,
    packed,
    depth,
    input_dtype,
    parameter_dtype,
):
    if torch.float16 in (input_dtype, parameter_dtype) and not cpu_fp16_cast_supported:
        pytest.skip("this CPU runtime cannot initialize cpuinfo for FP32-to-FP16 casts")
    layout = layout_for(packed)
    blocks = torch.nn.ModuleList(
        [MemoryQueryBlock(16, 32, 2).to(parameter_dtype) for _ in range(depth)]
    )
    refs = copy.deepcopy(blocks)
    x = torch.randn(layout.total_tokens, 16, dtype=input_dtype, requires_grad=True)
    r = torch.randn(x.shape, dtype=torch.float32, requires_grad=True)
    ex, er = x.detach().clone().requires_grad_(), r.detach().clone().requires_grad_()
    h, residual, eh, eresidual = x, r, ex, er
    for block, ref in zip(blocks, refs):
        h, residual = block(h, residual, layout=layout)
        eh, eresidual = memory_query_block_forward(ref, eh, eresidual, layout)
    actual, expected = (h, residual), (eh, eresidual)
    for a, e in zip(actual, expected):
        assert a.dtype == torch.bfloat16 and a.shape == x.shape
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    gradient = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual)
    tensors, et = (x, r, *blocks.parameters()), (ex, er, *refs.parameters())
    ag = torch.autograd.grad(actual, tensors, gradient)
    eg = torch.autograd.grad(expected, et, gradient)
    assert all(g.dtype == t.dtype for g, t in zip(ag, tensors))
    assert_gradients(ag, eg, relative_l2=0.04)


@pytest.mark.parametrize("depth", [1, 3])
def test_adapter_preserves_original_per_sample_equations(
    memory_reference_backend, depth
):
    layout = FixedMemoryQueryLayout.ncse(3, 4, batch_size=2)
    adapter = FixedMemoryQueryStack([MemoryQueryBlock(16, 32, 2) for _ in range(depth)])
    refs = copy.deepcopy(adapter.blocks)
    x = torch.randn(2, 11, 16, requires_grad=True)
    r = torch.randn_like(x, requires_grad=True)
    ex, er = x.detach().clone().requires_grad_(), r.detach().clone().requires_grad_()
    h, residual = ex, er
    for ref in refs:
        h, residual = legacy_fixed_memory_query_reference(ref, h, residual, 6, 3)
    actual, expected = adapter(x, r, layout=layout), (h, residual)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    gradient = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual)
    assert_gradients(
        torch.autograd.grad(actual, (x, r, *adapter.parameters()), gradient),
        torch.autograd.grad(expected, (ex, er, *refs.parameters()), gradient),
        relative_l2=0.04,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("depth", [1, 3])
def test_stacked_causality_and_isolation_for_both_input_streams(
    memory_reference_backend, packed, depth
):
    layout = layout_for(packed)
    blocks = [MemoryQueryBlock(16, 32, 2).eval() for _ in range(depth)]
    x = torch.randn(layout.total_tokens, 16, requires_grad=True)
    r = torch.randn_like(x, requires_grad=True)
    n, b = layout.num_memory_tokens, layout.batch_size
    if packed:
        mo, qo, xo = (
            layout.cu_seqlens_m.tolist(),
            layout.cu_seqlens_q.tolist(),
            layout.cu_seqlens_x.tolist(),
        )
        xl = [end - start for start, end in zip(xo, xo[1:])]
    else:
        mo = [i * layout.memory_length for i in range(b + 1)]
        qo = [i * layout.query_length for i in range(b + 1)]
        xl = [layout.x_prefix_length] * b

    def run(h, residual):
        for block in blocks:
            h, residual = block(h, residual, layout=layout)
        return h + residual

    actual = run(x, r)
    cx, cr = x.detach().clone(), r.detach().clone()
    cx[n:] *= 20
    cr[n:] *= 10
    torch.testing.assert_close(actual[:n], run(cx, cr)[:n], rtol=0, atol=0)
    for g in torch.autograd.grad(
        actual[:n].float().square().sum(), (x, r), retain_graph=True
    ):
        assert torch.count_nonzero(g[n:]) == 0
    for sample in range(b):
        vi = n + sample
        rows = [(vi, list(range(mo[sample], mo[sample] + xl[sample])) + [vi])]
        for j in range(qo[sample + 1] - qo[sample]):
            qi = n + b + qo[sample] + j
            rows.append(
                (qi, list(range(mo[sample], mo[sample] + xl[sample] + j)) + [qi])
            )
        for row, allowed in rows:
            forbidden = [i for i in range(len(x)) if i not in allowed]
            for g in torch.autograd.grad(
                actual[row].float().square().sum(), (x, r), retain_graph=True
            ):
                assert torch.count_nonzero(g[forbidden]) == 0
                assert torch.count_nonzero(g[mo[sample] : mo[sample] + xl[sample]]) > 0
            cx, cr = x.detach().clone(), r.detach().clone()
            cx[forbidden] *= 20
            cr[forbidden] *= 10
            torch.testing.assert_close(actual[row], run(cx, cr)[row], rtol=0, atol=0)
    for row in range(n, len(x)):
        cx, cr = x.detach().clone(), r.detach().clone()
        cx[row] *= 20
        cr[row] *= 10
        keep = [i for i in range(len(x)) if i != row]
        torch.testing.assert_close(actual[keep], run(cx, cr)[keep], rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_flat_projection_views_kernel_counts_and_prefix_reuse(
    memory_reference_backend, monkeypatch, packed
):
    import varlen_transformer._memory_ops as ops
    import varlen_transformer.memory_query as mq
    import varlen_transformer.stack as adapter

    layout = layout_for(packed)
    blocks = [MemoryQueryBlock(16, 32, 2).eval() for _ in range(3)]
    linears, norms, projections, gathers = [], [], [], []
    linear, norm, pre_sa, pre_ca = (
        ops._linear_fwd,
        ops._layernorm_fwd_fp32,
        mq.mq_pre_sa,
        mq.mq_pre_ca,
    )
    index_select = torch.Tensor.index_select

    def observe_linear(x, w, bias):
        assert x.ndim == 2 and x.is_contiguous()
        linears.append((x.shape, x.data_ptr()))
        return linear(x, w, bias)

    def observe_norm(*args):
        output = norm(*args)
        norms.append(output[0])
        return output

    def observe_sa(*args):
        result = pre_sa(*args)
        projections.append((result[0],))
        return result

    def observe_ca(*args):
        result = pre_ca(*args)
        projections[-1] += (result[0], result[1])
        return result

    def observe_gather(tensor, dim, index):
        if index is layout.x_prefix_indices:
            gathers.append((tensor.shape, index))
        return index_select(tensor, dim, index)

    def forbidden(*args, **kwargs):
        raise AssertionError("layout conversion between native blocks")

    monkeypatch.setattr(ops, "_linear_fwd", observe_linear)
    monkeypatch.setattr(ops, "_layernorm_fwd_fp32", observe_norm)
    monkeypatch.setattr(mq, "mq_pre_sa", observe_sa)
    monkeypatch.setattr(mq, "mq_pre_ca", observe_ca)
    monkeypatch.setattr(adapter, "pack_fixed_memory_query", forbidden)
    monkeypatch.setattr(adapter, "unpack_fixed_memory_query", forbidden)
    if packed:
        monkeypatch.setattr(torch.Tensor, "index_select", observe_gather)
    h, r = torch.randn(layout.total_tokens, 16), None
    for block in blocks:
        h, r = block(h, r, layout=layout)
    assert len(linears) == 21 and len(norms) == 9 and len(memory_reference_backend) == 9
    n, b, tail = (
        layout.num_memory_tokens,
        layout.batch_size,
        layout.total_tokens - layout.num_memory_tokens,
    )
    for i in range(3):
        sa_call, v_call, q_call = memory_reference_backend[3 * i : 3 * i + 3]
        assert [
            (c["kind"], c["causal"], c["packed"]) for c in (sa_call, v_call, q_call)
        ] == [("sa", True, packed), ("ca", False, packed), ("ca", True, packed)]
        ls, ns = linears[7 * i : 7 * i + 7], norms[3 * i : 3 * i + 3]
        assert [s for s, ptr in ls] == [
            (n, 16),
            (n, 16),
            (tail, 16),
            (n, 16),
            (tail, 16),
            (layout.total_tokens, 16),
            (layout.total_tokens, 32),
        ]
        assert ls[0][1] == ns[0].data_ptr()
        assert ls[2][1] == ns[1].data_ptr() + n * 16 * ns[1].element_size()
        assert ls[3][1] == ns[1].data_ptr() and ls[5][1] == ns[2].data_ptr()
        qkv, q, kv = projections[i]
        assert sa_call["qkv"].data_ptr() == qkv.data_ptr()
        assert v_call["q"].data_ptr() == q.data_ptr()
        assert q_call["q"].data_ptr() == q.data_ptr() + b * 16 * q.element_size()
        assert q_call["kv"].data_ptr() == kv.data_ptr()
        if not packed:
            assert v_call["kv"].data_ptr() == kv.data_ptr()
    assert len(gathers) == (3 if packed else 0)
    assert all(index is layout.x_prefix_indices for shape, index in gathers)


@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("flat_result", [False, True])
def test_adapter_converts_only_at_stack_boundaries(
    memory_reference_backend, monkeypatch, depth, with_residual, flat_result
):
    import varlen_transformer.stack as adapter

    layout = FixedMemoryQueryLayout.ncse(3, 4, batch_size=2)
    stack = FixedMemoryQueryStack([MemoryQueryBlock(16, 32, 2) for _ in range(depth)])
    pack, unpack = adapter.pack_fixed_memory_query, adapter.unpack_fixed_memory_query
    calls = []

    def pack_once(*args):
        calls.append("pack")
        return pack(*args)

    def unpack_once(*args):
        calls.append("unpack")
        return unpack(*args)

    monkeypatch.setattr(adapter, "pack_fixed_memory_query", pack_once)
    monkeypatch.setattr(adapter, "unpack_fixed_memory_query", unpack_once)
    x = torch.randn(2, 11, 16, requires_grad=True)
    r = torch.randn_like(x, requires_grad=True) if with_residual else None
    result = stack(x, r, layout=layout, return_stream_major=flat_result)
    assert calls == ["pack"] * (2 if with_residual else 1) + (
        ["unpack"] * 2 if not flat_result else []
    )
    assert len(memory_reference_backend) == 3 * depth
    sum(t.float().square().mean() for t in result).backward()
    assert torch.isfinite(x.grad).all()
    if r is not None:
        assert torch.isfinite(r.grad).all()


@pytest.mark.parametrize("packed", [False, True])
def test_noncontiguous_states_and_frozen_parameters(memory_reference_backend, packed):
    layout = layout_for(packed)
    block = MemoryQueryBlock(16, 32, 2).eval().requires_grad_(False)
    ref = copy.deepcopy(block)
    x = torch.randn(16, layout.total_tokens).T.detach().requires_grad_()
    r = torch.randn_like(x).T.contiguous().T.detach().requires_grad_()
    ex, er = x.detach().clone().requires_grad_(), r.detach().clone().requires_grad_()
    actual, expected = (
        block(x, r, layout=layout),
        memory_query_block_forward(ref, ex, er, layout),
    )
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    gradient = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual)
    assert_gradients(
        torch.autograd.grad(actual, (x, r), gradient),
        torch.autograd.grad(expected, (ex, er), gradient),
    )
    assert all(p.grad is None for p in block.parameters())


def test_adapter_dropout_serialization_and_optimizer(
    memory_reference_backend, tmp_path
):
    layout = FixedMemoryQueryLayout.ncse(3, 4, batch_size=2)

    def make():
        return FixedMemoryQueryStack(
            [
                MemoryQueryBlock(
                    16,
                    32,
                    2,
                    attn_dropout=0.2,
                    entry_resid_dropout=0.3,
                    sa_resid_dropout=0.4,
                    ca_resid_dropout=0.5,
                )
                for _ in range(2)
            ]
        )

    stack = make().eval()
    x = torch.randn(2, 11, 16, requires_grad=True)
    x = x.transpose(1, 2).contiguous().transpose(1, 2).detach().requires_grad_()
    state = tmp_path / "stack.pt"
    torch.save(stack.state_dict(), state)
    other = make().eval()
    other.load_state_dict(torch.load(state, weights_only=True))
    for a, e in zip(stack(x, layout=layout), other(x, layout=layout)):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    before = other.blocks[0].cross_mixer.Wkv.weight.detach().clone()
    optimizer = torch.optim.AdamW(other.parameters(), lr=0.01)
    h, r = other.train()(x, layout=layout)
    (h + r).float().square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in other.parameters()
    )
    optimizer.step()
    assert not torch.equal(before, other.blocks[0].cross_mixer.Wkv.weight)


def test_adapter_and_fixed_metadata_validation():
    with pytest.raises(TypeError):
        FixedMemoryQueryLayout.ncse(3, 4)
    with pytest.raises(ValueError):
        FixedMemoryQueryLayout.ncse(3, 4, batch_size=0)
    with pytest.raises(ValueError):
        FixedMemoryQueryStack([])
    layout = FixedMemoryQueryLayout.ncse(3, 4, batch_size=2)
    for value in (
        torch.randn(11, 16),
        torch.randn(1, 11, 16),
        torch.empty(2, 11, 0),
        torch.ones(2, 11, 16, dtype=torch.int64),
    ):
        with pytest.raises(ValueError):
            pack_fixed_memory_query(value, layout)
    for value in (torch.randn(22), torch.empty(22, 0), torch.randn(21, 16)):
        with pytest.raises(ValueError):
            unpack_fixed_memory_query(value, layout)
    packed = PackedMemoryQueryLayout.ncse([3, 3], [4, 4])
    with pytest.raises(ValueError):
        pack_fixed_memory_query(torch.randn(2, 11, 16), packed)
    with pytest.raises(ValueError):
        unpack_fixed_memory_query(torch.randn(22, 16), packed)
