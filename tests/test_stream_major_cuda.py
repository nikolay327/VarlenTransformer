"""Real kernels only: no reference-attention dependency substitution."""

import copy

import pytest
import torch

from tests.reference import (
    assert_gradients,
    memory_query_block_forward,
    legacy_fixed_memory_query_reference,
)
from varlen_transformer import (
    MemoryQueryBlock,
    FixedMemoryQueryStack,
    FixedMemoryQueryLayout,
    PackedMemoryQueryLayout,
)

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "parameter_dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_real_adapter_matches_legacy_equations_and_all_gradients(
    depth, input_dtype, parameter_dtype
):
    layout = FixedMemoryQueryLayout.ncse(3, 4, batch_size=2)
    stack = FixedMemoryQueryStack(
        [
            MemoryQueryBlock(32, 64, 2).to(device="cuda", dtype=parameter_dtype)
            for _ in range(depth)
        ]
    )
    assert type(stack.blocks[0].cross_mixer.attn).__module__.startswith("flash_attn")
    refs = copy.deepcopy(stack.blocks)
    x = torch.randn(2, 11, 32, device="cuda", dtype=input_dtype, requires_grad=True)
    r = torch.randn(x.shape, device="cuda", dtype=torch.float32, requires_grad=True)
    ex, er = x.detach().clone().requires_grad_(), r.detach().clone().requires_grad_()
    h, residual = ex, er
    for ref in refs:
        h, residual = legacy_fixed_memory_query_reference(ref, h, residual, 6, 3)
    actual, expected = stack(x, r, layout=layout), (h, residual)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0.04, atol=0.025)
    gradient = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual)
    assert_gradients(
        torch.autograd.grad(actual, (x, r, *stack.parameters()), gradient),
        torch.autograd.grad(expected, (ex, er, *refs.parameters()), gradient),
        relative_l2=0.04,
    )


@pytest.mark.parametrize("memory_length", [127, 128, 129])
def test_real_fixed_adapter_at_tile_boundaries(memory_length):
    layout = FixedMemoryQueryLayout.ncse(memory_length - 3, 4, batch_size=2)
    stack = FixedMemoryQueryStack(
        [MemoryQueryBlock(32, 64, 2).cuda() for _ in range(2)]
    )
    refs = copy.deepcopy(stack.blocks)
    x = torch.randn(2, memory_length + 5, 32, device="cuda", requires_grad=True)
    ex = x.detach().clone().requires_grad_()
    h, r = ex, None
    for ref in refs:
        h, r = legacy_fixed_memory_query_reference(
            ref, h, r, memory_length, memory_length - 3
        )
    actual, expected = stack(x, layout=layout), (h, r)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0.04, atol=0.025)
    gradient = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual)
    assert_gradients(
        torch.autograd.grad(actual, (x, *stack.parameters()), gradient),
        torch.autograd.grad(expected, (ex, *refs.parameters()), gradient),
        relative_l2=0.04,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("e,heads", [(16, 1), (34, 2), (512, 2)])
def test_real_native_head_padding_and_noncontiguous_inputs(packed, e, heads):
    layout = (
        PackedMemoryQueryLayout.ncse([3, 3], [4, 4], device="cuda")
        if packed
        else FixedMemoryQueryLayout.ncse(3, 4, batch_size=2)
    )
    block = MemoryQueryBlock(e, 2 * e, heads).cuda()
    ref = copy.deepcopy(block)
    x = torch.randn(e, layout.total_tokens, device="cuda").T.detach().requires_grad_()
    r = torch.randn_like(x, requires_grad=True)
    ex, er = x.detach().clone().requires_grad_(), r.detach().clone().requires_grad_()
    actual, expected = (
        block(x, r, layout=layout),
        memory_query_block_forward(ref, ex, er, layout),
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0.03, atol=0.015)
    gradient = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual)
    assert_gradients(
        torch.autograd.grad(actual, (x, r, *block.parameters()), gradient),
        torch.autograd.grad(expected, (ex, er, *ref.parameters()), gradient),
        relative_l2=0.04,
    )


def test_real_heterogeneous_stack_has_no_forbidden_paths_in_either_state():
    xl, ql, ml = [1, 3, 2], [1, 4, 2], [1, 6, 3]
    layout = PackedMemoryQueryLayout.ncse(xl, ql, device="cuda")
    blocks = [MemoryQueryBlock(32, 64, 2).cuda().eval() for _ in range(3)]
    x = torch.randn(layout.total_tokens, 32, device="cuda", requires_grad=True)
    r = torch.randn_like(x, requires_grad=True)
    h, residual = x, r
    for block in blocks:
        h, residual = block(h, residual, layout=layout)
    y, n, b = h + residual, layout.num_memory_tokens, layout.batch_size
    for g in torch.autograd.grad(y[:n].square().sum(), (x, r), retain_graph=True):
        assert torch.count_nonzero(g[n:]) == 0
    mo, qo = 0, 0
    for sample in range(b):
        rows = [(n + sample, list(range(mo, mo + xl[sample])) + [n + sample])]
        for j in range(ql[sample]):
            row = n + b + qo + j
            rows.append((row, list(range(mo, mo + xl[sample] + j)) + [row]))
        for row, allowed in rows:
            forbidden = [i for i in range(len(x)) if i not in allowed]
            for g in torch.autograd.grad(
                y[row].square().sum(), (x, r), retain_graph=True
            ):
                assert torch.count_nonzero(g[forbidden]) == 0
                assert torch.count_nonzero(g[mo : mo + xl[sample]]) > 0
        mo, qo = mo + ml[sample], qo + ql[sample]


def test_real_device_metadata_requires_trust_and_forward_has_no_host_reads(monkeypatch):
    layout = PackedMemoryQueryLayout.ncse([3, 2], [4, 2], device="cuda")
    block = MemoryQueryBlock(32, 64, 2).cuda()
    x = torch.randn(layout.total_tokens, 32, device="cuda", requires_grad=True)
    names = (
        "num_memory_tokens",
        "num_query_tokens",
        "cu_seqlens_m",
        "cu_seqlens_q",
        "cu_seqlens_x",
        "max_seqlen_m",
        "max_seqlen_q",
        "max_seqlen_x",
        "x_prefix_indices",
    )
    kwargs = {name: getattr(layout, name) for name in names}

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA metadata contents read by production code")

    with monkeypatch.context() as guard:
        for name in ("item", "tolist", "cpu"):
            guard.setattr(torch.Tensor, name, forbidden)
        with pytest.raises(ValueError, match="trust_metadata=True"):
            PackedMemoryQueryLayout(**kwargs)
        trusted = PackedMemoryQueryLayout(**kwargs, trust_metadata=True)
        assert not trusted.metadata_verified
        h, r = block(x, layout=trusted)
        assert h.shape == r.shape == x.shape
    (h + r).float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
