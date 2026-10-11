"""Python configuration, model isolation, and saved autograd dispatch."""

import copy

import pytest
import torch

from varlen_transformer import (
    FixedMemoryQueryLayout,
    create_block,
    create_memory_query_block,
)
from varlen_transformer import _fusion


def model_inputs(architecture, fusion="off"):
    if architecture == "block":
        model = create_block(16, 32, 2, fusion=fusion)
        shape, metadata = (2, 3, 16), {}
    else:
        model = create_memory_query_block(16, 32, 2, fusion=fusion)
        layout = FixedMemoryQueryLayout.autoregressive(2, 2, batch_size=2)
        shape, metadata = (layout.total_tokens, 16), {"layout": layout}
    return model, torch.randn(shape, requires_grad=True), metadata


@pytest.mark.parametrize("architecture", ["block", "memory_query"])
@pytest.mark.parametrize("mode", sorted(_fusion.MODES))
def test_constructor_configuration_and_checkpoint_portability(
    memory_reference_backend, monkeypatch, architecture, mode
):
    def forbidden():
        raise AssertionError("Construction and CPU execution must not load Triton")

    monkeypatch.setattr(_fusion, "_load", forbidden)
    model, x, metadata = model_inputs(architecture, mode)
    assert model.fusion == mode
    reference = copy.deepcopy(model)
    reference.fusion = "off"
    reference.load_state_dict(model.state_dict(), strict=True)
    assert reference.fusion == "off"
    for actual, expected in zip(model(x, **metadata), reference(x, **metadata)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["block", "memory_query"])
@pytest.mark.parametrize("invalid", ["auto", "unknown", None, True, [], 1])
def test_invalid_configuration_fails_at_creation_or_update(
    memory_reference_backend, architecture, invalid
):
    with pytest.raises(ValueError, match="Invalid fusion mode"):
        model_inputs(architecture, invalid)
    model, _, _ = model_inputs(architecture)
    with pytest.raises(ValueError, match="Invalid fusion mode"):
        model.fusion = invalid
    assert model.fusion == "off"


@pytest.mark.parametrize("architecture,regions", [("block", 2), ("memory_query", 3)])
def test_models_and_pending_backward_keep_independent_modes(
    memory_reference_backend, monkeypatch, architecture, regions
):
    calls = []
    original = _fusion.layernorm_backward

    def observed(*args, **kwargs):
        calls.append(kwargs.get("fusion", args[-1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(_fusion, "layernorm_backward", observed)
    model, x, metadata = model_inputs(architecture, "residual_fwd_bwd")
    other = copy.deepcopy(model)
    other.fusion = "ln_fwd"
    pending = model(x, **metadata)
    other_pending = other(x.detach().clone().requires_grad_(), **metadata)
    # Reconfigure both models and create another graph before consuming any.
    model.fusion = "off"
    other.fusion = "ln_bwd"
    subsequent = model(x.detach().clone().requires_grad_(), **metadata)
    for outputs, expected in (
        (pending, "residual_fwd_bwd"),
        (other_pending, "ln_fwd"),
        (subsequent, "off"),
    ):
        calls.clear()
        sum(t.float().square().mean() for t in outputs).backward()
        assert calls == [expected] * regions


@pytest.mark.parametrize("kind", ["pre", "post", "sa", "ca", "mlp"])
def test_explicit_mode_supports_fake_autograd_and_aot(kind):
    from tests.memory_reference import make_args
    from tests.reference import pre_args, post_args
    from varlen_transformer._memory_ops import mq_pre_sa, mq_pre_ca, mq_post_ca_mlp
    from varlen_transformer.block import pre_attn_qkv, post_attn_mlp

    op, args = {
        "pre": (pre_attn_qkv, pre_args),
        "post": (post_attn_mlp, post_args),
        "sa": (mq_pre_sa, lambda: make_args("sa")),
        "ca": (mq_pre_ca, lambda: make_args("ca")),
        "mlp": (mq_post_ca_mlp, lambda: make_args("mlp")),
    }[kind]
    result = torch.library.opcheck(op, (*args(), "residual_fwd_bwd"))
    assert set(result.values()) == {"SUCCESS"}
