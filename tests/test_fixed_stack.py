import copy

import pytest
import torch

from tests.reference import assert_gradients, memory_query_block_forward
from varlen_transformer import (
    FixedMemoryQueryLayout,
    FixedMemoryQueryStack,
    MemoryQueryBlock,
    pack_fixed_memory_query,
    unpack_fixed_memory_query,
)


def test_conversion_roundtrip_and_gradients():
    layout = FixedMemoryQueryLayout.ncse(3, 4, batch_size=2)
    x = torch.randn(2, 11, 16, requires_grad=True)
    flat = pack_fixed_memory_query(x, layout)
    expected = torch.cat((x[:, :6].reshape(12, 16), x[:, 6], x[:, 7:].reshape(8, 16)))
    torch.testing.assert_close(flat, expected, rtol=0, atol=0)
    restored = unpack_fixed_memory_query(flat, layout)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)
    gradient = torch.randn_like(x)
    torch.testing.assert_close(torch.autograd.grad(restored, x, gradient)[0], gradient)


@pytest.mark.parametrize("with_residual", [False, True])
def test_stack_adapter_matches_independent_native_reference(
    memory_reference_backend, with_residual
):
    layout = FixedMemoryQueryLayout.ncse(3, 4, batch_size=2)
    adapter = FixedMemoryQueryStack([MemoryQueryBlock(16, 32, 2) for _ in range(3)])
    refs = copy.deepcopy(adapter.blocks)
    x = torch.randn(2, 11, 16, requires_grad=True)
    r = torch.randn_like(x, requires_grad=True) if with_residual else None
    ex = x.detach().clone().requires_grad_()
    er = None if r is None else r.detach().clone().requires_grad_()

    def independent_pack(value):
        return torch.cat(
            (value[:, :6].reshape(12, 16), value[:, 6], value[:, 7:].reshape(8, 16))
        )

    h, residual = independent_pack(ex), None if er is None else independent_pack(er)
    for ref in refs:
        h, residual = memory_query_block_forward(ref, h, residual, layout)
    expected = tuple(
        torch.cat(
            (
                t[:12].reshape(2, 6, 16),
                t[12:14].reshape(2, 1, 16),
                t[14:].reshape(2, 4, 16),
            ),
            dim=1,
        )
        for t in (h, residual)
    )
    actual = adapter(x, r, layout=layout)
    for a, e in zip(actual, expected):
        assert a.dtype == torch.bfloat16 and a.shape == x.shape
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    grad = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual)
    tensors = (x, *((r,) if r is not None else ()), *adapter.parameters())
    et = (ex, *((er,) if er is not None else ()), *refs.parameters())
    assert_gradients(
        torch.autograd.grad(actual, tensors, grad),
        torch.autograd.grad(expected, et, grad),
        relative_l2=0.04,
    )


def test_native_rejects_legacy_shape_and_wrapper_can_return_flat(
    memory_reference_backend,
):
    layout = FixedMemoryQueryLayout.ncse(3, 4, batch_size=2)
    block = MemoryQueryBlock(16, 32, 2)
    x = torch.randn(2, 11, 16)
    with pytest.raises(ValueError, match="FixedMemoryQueryStack"):
        block(x, layout=layout)
    adapter = FixedMemoryQueryStack([block])
    h, r = adapter(x, layout=layout, return_stream_major=True)
    assert h.shape == r.shape == (layout.total_tokens, 16)
