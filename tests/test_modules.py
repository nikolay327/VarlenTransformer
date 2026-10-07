import copy
from functools import partial

import pytest
import torch
from torch.nn import functional as F

from tests.reference import assert_gradients, attention, block_forward, packed_attention
from varlen_transformer import MHA, MLP, Block, create_block
from varlen_transformer.block import Block_
from varlen_transformer.cls import _create_mha_cls, _create_mlp_cls


@pytest.mark.parametrize("causal", [False, True])
def test_factory_sets_architecture_and_defaults(reference_backend, causal):
    block = create_block(32, 96, 4, causal=causal)
    assert (
        isinstance(block, Block)
        and isinstance(block.mixer, MHA)
        and isinstance(block.mlp, MLP)
    )
    assert (
        block.mixer.emb_dim == 32
        and block.mixer.num_heads == 4
        and block.mixer.head_dim == 8
    )
    assert block.mixer.causal is causal and block.mixer.attn.causal is causal
    assert block.mlp.fc1.in_features == block.mlp.fc2.out_features == 32
    assert block.mlp.fc1.out_features == block.mlp.fc2.in_features == 96
    assert block.mixer.attn.drop.p == block.dropout1.p == block.dropout2.p == 0.0


def test_factory_honors_explicit_dropouts_and_partial_classes(reference_backend):
    block = create_block(
        16, 48, 2, attn_dropout=0.1, block_resid_dropout1=0.2, block_resid_dropout2=0.3
    )
    assert (block.mixer.attn.drop.p, block.dropout1.p, block.dropout2.p) == (
        0.1,
        0.2,
        0.3,
    )
    assert _create_mha_cls(2, 0.1, True)(16).causal
    assert _create_mlp_cls(48)(16).fc1.out_features == 48


@pytest.mark.parametrize(
    "emb_dim,heads,message",
    [
        (0, 2, "positive integer"),
        (16, 0, "positive integer"),
        (16, -1, "positive integer"),
        (16, True, "positive integer"),
        (15, 2, "divisible"),
        (16, 2.5, "positive integer"),
        (514, 2, "at most 256"),
    ],
)
def test_attention_rejects_invalid_dimensions_before_loading_flash(
    emb_dim, heads, message
):
    with pytest.raises(ValueError, match=message):
        MHA(emb_dim, heads)


@pytest.mark.parametrize("p", [-0.01, 1.0, float("nan"), float("inf")])
def test_attention_rejects_invalid_dropout(p):
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        MHA(16, 2, attn_dropout=p)


@pytest.mark.parametrize("p1,p2", [(-0.1, 0), (0, 1.1)])
def test_factory_rejects_invalid_residual_dropout(reference_backend, p1, p2):
    with pytest.raises(ValueError):
        create_block(16, 32, 2, block_resid_dropout1=p1, block_resid_dropout2=p2)


def test_missing_flash_dependency_has_installation_guidance(monkeypatch):
    import varlen_transformer.mha as module

    monkeypatch.setattr(module, "FlashSelfAttention", None)
    with pytest.raises(ImportError, match="no-build-isolation"):
        create_block(16, 32, 2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_mha_preserves_layout_and_has_gradients(reference_backend, causal, packed):
    mha = MHA(16, 2, causal=causal)
    shape = (7, 16) if packed else (2, 4, 16)
    x = torch.randn(shape, requires_grad=True)
    cu = torch.tensor([0, 1, 3, 7], dtype=torch.int64) if packed else None
    maximum = 4 if packed else None
    actual = mha(x, cu, maximum)
    qkv = F.linear(x, mha.Wqkv.weight, mha.Wqkv.bias).reshape(*shape[:-1], 3, 2, 8)
    a = (
        packed_attention(qkv, cu, maximum, causal=causal)
        if packed
        else attention(qkv, causal=causal)
    )
    expected = F.linear(a.reshape_as(x), mha.out_proj.weight, mha.out_proj.bias)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in mha.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_block_forward_and_all_gradients_against_eager_math(
    reference_backend, packed, causal, with_residual, dtype
):
    block = create_block(16, 32, 2, causal=causal)
    expected_block = copy.deepcopy(block)
    shape = (9, 16) if packed else (2, 4, 16)
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    residual = torch.randn_like(x, requires_grad=True) if with_residual else None
    expected_residual = (
        residual.detach().clone().requires_grad_() if with_residual else None
    )
    cu = torch.tensor([0, 1, 3, 9], dtype=torch.int32) if packed else None
    maximum = 6 if packed else None
    result = block(x, residual, cu, maximum)
    expected = block_forward(expected_block, expected_x, expected_residual, cu, maximum)
    for a, b in zip(result, expected):
        assert a.dtype == torch.bfloat16 and a.shape == shape
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    gradients = tuple(torch.randn_like(y) / y.numel() ** 0.5 for y in result)
    inputs = (x, *((residual,) if with_residual else ()), *block.parameters())
    reference_inputs = (
        expected_x,
        *((expected_residual,) if with_residual else ()),
        *expected_block.parameters(),
    )
    actual_grads = torch.autograd.grad(result, inputs, gradients)
    expected_grads = torch.autograd.grad(expected, reference_inputs, gradients)
    assert_gradients(actual_grads, expected_grads)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_block_selects_kernel_and_forwards_attention_configuration(
    reference_backend, training, packed
):
    block = create_block(16, 32, 2, causal=True, attn_dropout=0.25).train(training)
    block.mixer.attn.softmax_scale = 0.75
    block.mixer.attn.window_size = (2, 0)
    block.mixer.attn.deterministic = True
    x = torch.randn(7, 16) if packed else torch.randn(2, 4, 16)
    cu = torch.tensor([0, 2, 7], dtype=torch.int64) if packed else None
    block(x, cu_seqlens=cu, max_seqlen=5 if packed else None)
    assert len(reference_backend) == 1
    call = reference_backend[0]
    assert call["kind"] == ("packed" if packed else "fixed")
    assert call["causal"] and call["deterministic"]
    assert call["dropout_p"] == (0.25 if training else 0.0)
    assert call["softmax_scale"] == 0.75 and call["window_size"] == (2, 0)
    assert (
        call["softcap"] == 0.0
        and call["alibi_slopes"] is None
        and not call["return_attn_probs"]
    )
    assert call["qkv"].dtype == torch.bfloat16 and call["qkv"].is_contiguous()
    if packed:
        assert (
            call["cu_seqlens"].dtype == torch.int32
            and call["cu_seqlens"].is_contiguous()
        )
        assert call["max_seqlen"] == 5


def test_block_eval_disables_all_dropout_and_training_changes_masks(reference_backend):
    block = create_block(
        16, 32, 2, attn_dropout=0.2, block_resid_dropout1=0.3, block_resid_dropout2=0.4
    )
    x = torch.randn(2, 8, 16)
    block.eval()
    a, ar = block(x)
    torch.manual_seed(43)
    b, br = block(x)
    assert torch.equal(a, b) and torch.equal(ar, br)
    block.train()
    c, cr = block(x)
    torch.manual_seed(44)
    d, dr = block(x)
    assert not torch.equal(c, d) and not torch.equal(cr, dr)
    (c.float().square().mean() + cr.float().square().mean()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()
    )


def test_stack_preserves_residual_contract_and_backpropagates(reference_backend):
    blocks = torch.nn.ModuleList(
        [create_block(16, 32, 2, causal=True) for _ in range(3)]
    )
    expected_blocks = copy.deepcopy(blocks)
    x = torch.randn(6, 16, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    cu = torch.tensor([0, 1, 6], dtype=torch.int32)
    h, residual = x, None
    eh, er = expected_x, None
    for block, ref in zip(blocks, expected_blocks):
        h, residual = block(h, residual, cu, 5)
        eh, er = block_forward(ref, eh, er, cu, 5)
    actual, expected = h + residual, eh + er
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, (x, *blocks.parameters()), gradient)
    expected_grads = torch.autograd.grad(
        expected, (expected_x, *expected_blocks.parameters()), gradient
    )
    assert_gradients(actual_grads, expected_grads, relative_l2=0.04)


def test_packed_sequences_are_isolated_and_causal_tokens_ignore_future(
    reference_backend,
):
    block = create_block(16, 32, 2, causal=True).eval()
    x = torch.randn(8, 16)
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    baseline = block(x, cu_seqlens=cu, max_seqlen=5)
    changed = x.clone()
    changed[5:] = torch.randn_like(changed[5:]) * 10
    result = block(changed, cu_seqlens=cu, max_seqlen=5)
    for a, b in zip(baseline, result):
        torch.testing.assert_close(a[:5], b[:5], rtol=0, atol=0)
    independent = block(x[:3].unsqueeze(0))
    for a, b in zip(baseline, independent):
        torch.testing.assert_close(a[:3], b.squeeze(0), rtol=0, atol=0)


def test_fixed_and_uniform_packed_batches_agree(reference_backend):
    block = create_block(16, 32, 2).eval()
    x = torch.randn(3, 4, 16)
    fixed = block(x)
    packed = block(
        x.reshape(-1, 16),
        cu_seqlens=torch.tensor([0, 4, 8, 12], dtype=torch.int32),
        max_seqlen=4,
    )
    for a, b in zip(fixed, packed):
        torch.testing.assert_close(a.flatten(0, 1), b, rtol=0, atol=0)


def test_state_dict_roundtrip_and_optimizer_training(reference_backend, tmp_path):
    block = create_block(16, 32, 2).eval()
    x = torch.randn(2, 4, 16)
    path = tmp_path / "weights.pt"
    torch.save(block.state_dict(), path)
    restored = create_block(16, 32, 2).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    for a, b in zip(block(x), restored(x)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    before = restored.mixer.Wqkv.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        h, r = restored.train()(x)
        loss = (h + r).float().square().mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
    assert not torch.equal(restored.mixer.Wqkv.weight, before)


def test_eager_legacy_block_remains_usable_for_both_layouts(reference_backend):
    block = Block_(16, partial(MHA, num_heads=2), partial(MLP, hidden_dim=32), 0.0, 0.0)
    fixed = torch.randn(2, 3, 16, requires_grad=True)
    h, residual = block(fixed)
    assert h.shape == residual.shape == fixed.shape
    h.square().mean().backward()
    assert all(p.grad is not None for p in block.parameters())
    packed = fixed.detach().reshape(-1, 16)
    ph, pr = block(
        packed, cu_seqlens=torch.tensor([0, 3, 6], dtype=torch.int32), max_seqlen=3
    )
    torch.testing.assert_close(ph, h.detach().reshape_as(ph), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        pr, residual.detach().reshape_as(pr), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("kind", ["block", "mha"])
@pytest.mark.parametrize(
    "case",
    [
        "fixed_rank",
        "packed_rank",
        "width",
        "integer",
        "empty",
        "missing_max",
        "missing_cu",
        "max_zero",
        "max_bool",
        "max_float",
        "cu_rank",
        "cu_short",
        "cu_float",
        "cu_device",
    ],
)
def test_attention_input_validation(reference_backend, kind, case):
    module = create_block(16, 32, 2) if kind == "block" else MHA(16, 2)
    x, cu, maximum = torch.randn(5, 16), torch.tensor([0, 2, 5], dtype=torch.int32), 3
    if case == "fixed_rank":
        cu = maximum = None
    elif case == "packed_rank":
        x = x.unsqueeze(0)
    elif case == "width":
        x = torch.randn(5, 15)
    elif case == "integer":
        x = torch.ones(5, 16, dtype=torch.int64)
    elif case == "empty":
        x = torch.empty(0, 16)
    elif case == "missing_max":
        maximum = None
    elif case == "missing_cu":
        cu = None
    elif case == "max_zero":
        maximum = 0
    elif case == "max_bool":
        maximum = True
    elif case == "max_float":
        maximum = 3.0
    elif case == "cu_rank":
        cu = cu.unsqueeze(0)
    elif case == "cu_short":
        cu = torch.tensor([0], dtype=torch.int32)
    elif case == "cu_float":
        cu = cu.float()
    elif case == "cu_device":
        cu = cu.to("meta")
    with pytest.raises(ValueError):
        module(x, cu_seqlens=cu, max_seqlen=maximum)
    assert not reference_backend


@pytest.mark.parametrize("case", ["shape", "device", "dtype"])
def test_residual_input_validation(reference_backend, case):
    block = create_block(16, 32, 2)
    x = torch.randn(2, 3, 16)
    residual = (
        torch.randn(2, 3, 8)
        if case == "shape"
        else torch.empty_like(x, device="meta")
        if case == "device"
        else torch.ones_like(x, dtype=torch.int64)
    )
    with pytest.raises(ValueError, match="residual"):
        block(x, residual)
    assert not reference_backend


def test_noncontiguous_inputs_and_offsets_are_supported(reference_backend):
    block = create_block(16, 32, 2).eval()
    x = torch.randn(5, 32)[:, ::2]
    cu = torch.tensor([0, -1, 2, -1, 5, -1], dtype=torch.int64)[::2]
    assert not x.is_contiguous() and not cu.is_contiguous()
    actual = block(x, cu_seqlens=cu, max_seqlen=3)
    expected = block(x.contiguous(), cu_seqlens=cu.contiguous(), max_seqlen=3)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
