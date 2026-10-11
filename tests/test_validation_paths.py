"""Explicit input preparation and validation-free model execution."""

import importlib

import pytest
import torch

from varlen_transformer import (
    CrossMHA,
    FixedMemoryQueryLayout,
    FixedMemoryQueryStack,
    MHA,
    PackedMemoryQueryLayout,
    create_block,
    create_memory_query_block,
)
from varlen_transformer import _fusion


def execution_case(kind):
    def tensor(shape):
        return torch.randn(shape, requires_grad=True)

    if kind.startswith(("block", "mha")):
        packed = kind.endswith("packed")
        model = create_block(16, 32, 2) if kind.startswith("block") else MHA(16, 2)
        shape = (5, 16) if packed else (2, 3, 16)
        args = (tensor(shape),)
        if kind.startswith("block"):
            args += (tensor(shape),)
        metadata = (
            dict(cu_seqlens=torch.tensor([0, 2, 5]), max_seqlen=3)
            if packed else {}
        )
    elif kind.startswith("cross"):
        model = CrossMHA(16, 2, causal=True)
        packed = kind.endswith("packed")
        args = (tensor((5, 16)), tensor((7, 16))) if packed else (
            tensor((2, 3, 16)), tensor((2, 4, 16))
        )
        metadata = (
            dict(cu_seqlens_q=torch.tensor([0, 2, 5]),
                 cu_seqlens_k=torch.tensor([0, 4, 7]),
                 max_seqlen_q=3, max_seqlen_k=4)
            if packed else {}
        )
    else:
        layout = (
            PackedMemoryQueryLayout.autoregressive([2, 1], [2, 3])
            if kind.endswith("packed") else
            FixedMemoryQueryLayout.autoregressive(2, 2, batch_size=2)
        )
        if kind == "stack":
            model = FixedMemoryQueryStack([create_memory_query_block(16, 32, 2) for _ in range(2)])
            shape = (2, layout.memory_length + 1 + layout.query_length, 16)
        else:
            model = create_memory_query_block(16, 32, 2)
            shape = (layout.total_tokens, 16)
        args, metadata = (tensor(shape), tensor(shape)), dict(layout=layout)
    return model, args, metadata


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("kind", [
    "block_fixed", "block_packed", "mha_fixed", "mha_packed",
    "cross_fixed", "cross_packed", "memory_fixed", "memory_packed", "stack",
])
def test_forward_and_backward_bypass_input_validation(
    memory_reference_backend, monkeypatch, training, kind
):
    model, args, metadata = execution_case(kind)
    model.train(training)
    model.validate_inputs(*args, **metadata)

    def forbidden(*args, **kwargs):
        raise AssertionError("input validation during model execution")

    monkeypatch.setattr(model, "validate_inputs", forbidden)
    monkeypatch.setattr(_fusion, "validate_mode", forbidden)
    for name in ("block", "mha", "cross_mha"):
        module = importlib.import_module("varlen_transformer." + name)
        for validator in ("validate_attention_input", "validate_cross_attention_input"):
            if hasattr(module, validator):
                monkeypatch.setattr(module, validator, forbidden)
    monkeypatch.setattr(FixedMemoryQueryLayout, "validate", forbidden)
    monkeypatch.setattr(FixedMemoryQueryLayout, "validate_per_sample", forbidden)
    monkeypatch.setattr(PackedMemoryQueryLayout, "validate", forbidden)

    outputs = model(*args, **metadata)
    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
    sum(output.float().square().mean() for output in outputs).backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in args)
