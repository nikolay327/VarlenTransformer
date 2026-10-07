"""Integration tests: these use the real FlashAttention CUDA kernels.

No test here uses the reference_backend fixture. Independent attention math is
used only to calculate expected outputs and gradients.
"""

import copy

import pytest
import torch
from torch.nn import functional as F

from tests.reference import (
    assert_gradients,
    attention,
    block_forward,
    clone_args,
    packed_attention,
    post_args,
    post_attention,
    pre_args,
    pre_attention,
)
from varlen_transformer import MHA, create_block
from varlen_transformer.block import _linear_bwd, post_attn_mlp, pre_attn_qkv

pytestmark = pytest.mark.cuda


def layout(packed, emb_dim, dtype=torch.float32):
    shape = (13, emb_dim) if packed else (2, 7, emb_dim)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    cu = (
        torch.tensor([0, 1, 4, 13], device="cuda", dtype=torch.int32)
        if packed
        else None
    )
    return x, cu, 9 if packed else None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "emb_dim,heads", [(16, 1), (16, 2), (34, 2), (128, 4), (512, 2)]
)
def test_real_mha_forward_and_all_gradients(dtype, packed, causal, emb_dim, heads):
    mha = MHA(emb_dim, heads, causal=causal).to(device="cuda", dtype=dtype)
    assert type(mha.attn).__module__.startswith("flash_attn")
    expected_mha = copy.deepcopy(mha)
    x, cu, maximum = layout(packed, emb_dim, dtype)
    expected_x = x.detach().clone().requires_grad_()
    actual = mha(x, cu, maximum)
    qkv = F.linear(expected_x, expected_mha.Wqkv.weight, expected_mha.Wqkv.bias)
    qkv = qkv.reshape(*x.shape[:-1], 3, heads, emb_dim // heads)
    a = (
        packed_attention(qkv, cu, maximum, causal=causal)
        if packed
        else attention(qkv, causal=causal)
    )
    expected = F.linear(
        a.reshape_as(x), expected_mha.out_proj.weight, expected_mha.out_proj.bias
    )
    tolerance = 0.015 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance / 2)
    gradient = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grads = torch.autograd.grad(actual, (x, *mha.parameters()), gradient)
    expected_grads = torch.autograd.grad(
        expected, (expected_x, *expected_mha.parameters()), gradient
    )
    assert_gradients(actual_grads, expected_grads, atol=0.01)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("with_residual", [False, True])
def test_real_block_forward_and_every_gradient(dtype, packed, causal, with_residual):
    block = create_block(64, 128, 4, causal=causal).cuda()
    expected_block = copy.deepcopy(block)
    x, cu, maximum = layout(packed, 64, dtype)
    expected_x = x.detach().clone().requires_grad_()
    r = torch.randn_like(x, requires_grad=True) if with_residual else None
    expected_r = r.detach().clone().requires_grad_() if with_residual else None
    actual = block(x, r, cu, maximum)
    expected = block_forward(expected_block, expected_x, expected_r, cu, maximum)
    for a, b in zip(actual, expected):
        assert a.dtype == torch.bfloat16 and a.shape == x.shape
        torch.testing.assert_close(a, b, rtol=0.03, atol=0.015)
    gradients = tuple(torch.randn_like(y) / y.numel() ** 0.5 for y in actual)
    actual_inputs = (x, *((r,) if with_residual else ()), *block.parameters())
    expected_inputs = (
        expected_x,
        *((expected_r,) if with_residual else ()),
        *expected_block.parameters(),
    )
    actual_grads = torch.autograd.grad(actual, actual_inputs, gradients)
    expected_grads = torch.autograd.grad(expected, expected_inputs, gradients)
    for grad, inp in zip(actual_grads, actual_inputs):
        assert grad.dtype == inp.dtype and grad.shape == inp.shape
    assert_gradients(actual_grads, expected_grads, relative_l2=0.04)


@pytest.mark.parametrize("maximum", [1, 127, 128, 129, 255, 257, 513, 1025])
@pytest.mark.parametrize("causal", [False, True])
def test_real_packed_attention_at_sequence_tile_boundaries(maximum, causal):
    from flash_attn import flash_attn_varlen_qkvpacked_func

    lengths = [1, max(1, maximum // 3), maximum]
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    qkv = torch.randn(
        sum(lengths), 3, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    expected_qkv = qkv.detach().clone().requires_grad_()
    actual = flash_attn_varlen_qkvpacked_func(
        qkv, cu, maximum, causal=causal, deterministic=True
    )
    expected = packed_attention(expected_qkv, cu, maximum, causal=causal)
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.015)
    gradient = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_gradient = torch.autograd.grad(actual, qkv, gradient)
    expected_gradient = torch.autograd.grad(expected, expected_qkv, gradient)
    assert_gradients(actual_gradient, expected_gradient, atol=0.005)


@pytest.mark.parametrize(
    "op,make_args,reference",
    [
        (pre_attn_qkv, pre_args, pre_attention),
        (post_attn_mlp, post_args, post_attention),
    ],
    ids=["pre", "post"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "p,training", [(0.0, True), (0.25, True), (0.25, False), (1.0, True)]
)
def test_real_custom_op_forward_dropout_and_gradients(
    op, make_args, reference, dtype, p, training
):
    args = make_args((2, 3, 16), device="cuda", dtype=dtype, p=p, training=training)
    expected_args = clone_args(args)
    actual = op(*args)
    expected = reference(expected_args, actual[2])
    for a, b in zip(actual[:2], expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    inputs = tuple(a for a in args if isinstance(a, torch.Tensor))
    expected_inputs = tuple(a for a in expected_args if isinstance(a, torch.Tensor))
    gradients = tuple(torch.randn_like(a) / a.numel() ** 0.5 for a in actual[:2])
    actual_grads = torch.autograd.grad(actual[:2], inputs, gradients)
    expected_grads = torch.autograd.grad(expected, expected_inputs, gradients)
    assert_gradients(actual_grads, expected_grads)


@pytest.mark.parametrize(
    "op,make_args",
    [(pre_attn_qkv, pre_args), (post_attn_mlp, post_args)],
    ids=["pre", "post"],
)
@pytest.mark.parametrize("p,training", [(0.0, True), (0.25, True), (0.25, False)])
def test_real_custom_op_registration_and_compilation(op, make_args, p, training):
    args = make_args((2, 3, 16), device="cuda", p=p, training=training)
    results = torch.library.opcheck(op, args)
    assert set(results.values()) == {"SUCCESS"}


@pytest.mark.parametrize(
    "op,make_args",
    [(pre_attn_qkv, pre_args), (post_attn_mlp, post_args)],
    ids=["pre", "post"],
)
def test_real_custom_op_residual_only_backward(op, make_args):
    args = make_args(device="cuda")
    result = op(*args)
    result[1].float().square().mean().backward()
    assert args[0].grad is not None and torch.isfinite(args[0].grad).all()
    assert args[1].grad is not None and torch.isfinite(args[1].grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_linear_weight_gradient_accumulates_in_fp32(dtype):
    x = torch.randn(2, 7, 16, device="cuda", dtype=dtype)
    gy = torch.randn(2, 7, 32, device="cuda", dtype=dtype)
    w = torch.randn(32, 16, device="cuda", dtype=dtype)
    gx, gw, gb = _linear_bwd(gy, x, w)
    assert gx.dtype == dtype and gw.dtype == gb.dtype == torch.float32
    torch.testing.assert_close(
        gw,
        gy.reshape(-1, 32).float().T @ x.reshape(-1, 16).float(),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(gb, gy.float().sum((0, 1)), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("causal", [False, True])
def test_real_packed_sequences_do_not_influence_each_other(causal):
    block = create_block(32, 64, 2, causal=causal).cuda().eval()
    x = torch.randn(13, 32, device="cuda")
    cu = torch.tensor([0, 4, 13], device="cuda", dtype=torch.int32)
    actual = block(x, cu_seqlens=cu, max_seqlen=9)
    changed = x.clone()
    changed[4:] = torch.randn_like(changed[4:]) * 10
    other = block(changed, cu_seqlens=cu, max_seqlen=9)
    for a, b in zip(actual, other):
        torch.testing.assert_close(a[:4], b[:4], rtol=0, atol=0)
    fixed = block(x[:4].unsqueeze(0))
    for a, b in zip(actual, fixed):
        torch.testing.assert_close(a[:4], b.squeeze(0), rtol=0.015, atol=0.01)


def test_real_causal_block_has_no_future_token_dependency():
    block = create_block(32, 64, 2, causal=True).cuda().eval()
    x = torch.randn(1, 11, 32, device="cuda", requires_grad=True)
    h, residual = block(x)
    gradient = torch.autograd.grad(
        (h[:, :5] + residual[:, :5]).float().square().sum(), x
    )[0]
    assert torch.count_nonzero(gradient[:, 5:]).item() == 0
    changed = x.detach().clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 10
    ch, cr = block(changed)
    torch.testing.assert_close(h[:, :5], ch[:, :5], rtol=0, atol=0)
    torch.testing.assert_close(residual[:, :5], cr[:, :5], rtol=0, atol=0)


def test_real_bidirectional_attention_can_use_future_tokens():
    mha = MHA(16, 2, causal=False).cuda().bfloat16().eval()
    # Zero Q/K yields uniform attention; V and output projection are identities.
    with torch.no_grad():
        mha.Wqkv.weight.zero_()
        mha.Wqkv.weight[32:].copy_(torch.eye(16, device="cuda"))
        mha.Wqkv.bias.zero_()
        mha.out_proj.weight.copy_(torch.eye(16, device="cuda"))
        mha.out_proj.bias.zero_()
    x = torch.zeros(1, 4, 16, device="cuda", dtype=torch.bfloat16)
    changed = x.clone()
    changed[:, -1] = 4
    torch.testing.assert_close(mha(x), torch.zeros_like(x), rtol=0, atol=0)
    torch.testing.assert_close(mha(changed), torch.ones_like(x), rtol=0, atol=0)


def test_real_dropout_is_disabled_in_eval_and_training_remains_finite():
    block = create_block(
        32, 64, 2, attn_dropout=0.2, block_resid_dropout1=0.3, block_resid_dropout2=0.4
    ).cuda()
    x, cu, maximum = layout(True, 32)
    block.eval()
    a = block(x, cu_seqlens=cu, max_seqlen=maximum)
    torch.manual_seed(44)
    b = block(x, cu_seqlens=cu, max_seqlen=maximum)
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    block.train()
    h, residual = block(x, cu_seqlens=cu, max_seqlen=maximum)
    assert not torch.equal(h, a[0])
    (h.float().square().mean() + residual.float().square().mean()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()
    )


def test_real_fixed_and_packed_batches_agree_and_optimizer_updates():
    block = create_block(32, 64, 2, causal=True).cuda().eval()
    x = torch.randn(2, 7, 32, device="cuda")
    fixed = block(x)
    packed = block(
        x.flatten(0, 1),
        cu_seqlens=torch.tensor([0, 7, 14], device="cuda", dtype=torch.int32),
        max_seqlen=7,
    )
    for a, b in zip(fixed, packed):
        torch.testing.assert_close(a.flatten(0, 1), b, rtol=0.015, atol=0.01)
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
    before = block.mixer.Wqkv.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        h, residual = block.train()(x)
        loss = (h + residual).float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
    assert not torch.equal(before, block.mixer.Wqkv.weight)
