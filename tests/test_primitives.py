import pytest
import torch
from torch.nn import functional as F

from varlen_transformer import MLP
from varlen_transformer.block import (
    _dropout_bwd,
    _dropout_fwd,
    _gelu_bwd_tanh,
    _gelu_fwd_tanh,
    _layernorm_bwd_fp32,
    _layernorm_fwd_fp32,
    _linear_bwd,
    _linear_fwd,
    _to_bf16,
)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16]
)
def test_bf16_cast_preserves_values_and_existing_tensor(dtype):
    x = torch.randn(2, 3, dtype=dtype)
    result = _to_bf16(x)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, x.bfloat16(), rtol=0, atol=0)
    if dtype == torch.bfloat16:
        assert result is x


@pytest.mark.parametrize(
    "training,p", [(False, 0.4), (True, 0.0), (True, 0.4), (True, 1.0)]
)
def test_dropout_forward_and_backward_share_mask_without_mutation(training, p):
    x = torch.ones(4096, dtype=torch.bfloat16)
    y, mask = _dropout_fwd(x, p, training)
    gradient = torch.full_like(x, 2)
    if not training or p == 0:
        assert y is x and mask.shape == (0,)
        assert _dropout_bwd(gradient, mask, p, training) is gradient
    elif p == 1:
        assert not y.any() and not mask.any()
        assert not _dropout_bwd(gradient, mask, p, training).any()
    else:
        assert mask.dtype == torch.bool and mask.shape == x.shape
        assert abs(mask.float().mean().item() - (1 - p)) < 0.03
        torch.testing.assert_close(y, mask.bfloat16() / (1 - p), rtol=0, atol=0)
        torch.testing.assert_close(
            _dropout_bwd(gradient, mask, p, training),
            gradient * mask / (1 - p),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(x, torch.ones_like(x), rtol=0, atol=0)


def test_dropout_is_reproducible_and_backward_does_not_resample():
    x = torch.randn(100)
    torch.manual_seed(42)
    y1, mask1 = _dropout_fwd(x, 0.5, True)
    torch.manual_seed(42)
    y2, mask2 = _dropout_fwd(x, 0.5, True)
    torch.testing.assert_close(y1, y2, rtol=0, atol=0)
    assert torch.equal(mask1, mask2)
    torch.manual_seed(7)
    torch.testing.assert_close(
        _dropout_bwd(torch.ones_like(x), mask1, 0.5, True),
        mask1.float() * 2,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8,), (5, 8), (2, 3, 8)])
def test_linear_backward_matches_fp32_accumulation(dtype, shape):
    x = torch.randn(*shape, dtype=dtype)
    w = torch.randn(12, 8, dtype=dtype)
    bias = torch.randn(12, dtype=dtype)
    gy = torch.randn(*shape[:-1], 12, dtype=dtype)
    torch.testing.assert_close(
        _linear_fwd(x, w, bias), F.linear(x, w, bias), rtol=0, atol=0
    )
    torch.testing.assert_close(_linear_fwd(x, w, None), F.linear(x, w), rtol=0, atol=0)
    gx, gw, gb = _linear_bwd(gy, x, w)
    torch.testing.assert_close(gx, gy @ w, rtol=0, atol=0)
    assert gw.dtype == gb.dtype == torch.float32
    torch.testing.assert_close(
        gw,
        gy.reshape(-1, 12).float().T @ x.reshape(-1, 8).float(),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        gb, gy.reshape(-1, 12).float().sum(0), rtol=1e-6, atol=1e-6
    )


def test_linear_backward_handles_noncontiguous_inputs():
    x = torch.randn(4, 16)[:, ::2]
    gy = torch.randn(12, 4).T
    w = torch.randn(12, 8)
    actual = _linear_bwd(gy, x, w)
    expected = torch.autograd.grad(
        F.linear(
            x.requires_grad_(), w.requires_grad_(), torch.zeros(12, requires_grad=True)
        ),
        (x, w),
        gy,
    )
    for a, b in zip(actual[:2], expected):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


def test_gelu_tanh_forward_and_derivative_at_negative_zero_and_positive_values():
    x = torch.tensor(
        [-8.0, -3.0, -1.0, -0.01, 0.0, 0.01, 1.0, 3.0, 8.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    gy = torch.linspace(-1, 1, len(x), dtype=torch.float64)
    expected = 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x**3)))
    torch.testing.assert_close(_gelu_fwd_tanh(x), expected, rtol=1e-12, atol=1e-12)
    (gx,) = torch.autograd.grad(expected, x, gy)
    torch.testing.assert_close(_gelu_bwd_tanh(gy, x), gx, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("shape", [(5, 16), (2, 3, 16)])
def test_layernorm_uses_fp32_statistics_and_gradients(shape):
    x = torch.randn(shape).bfloat16()
    w, b = torch.randn(16), torch.randn(16)
    y, mean, rstd = _layernorm_fwd_fp32(x, w, b, 1e-5)
    assert y.dtype == torch.bfloat16 and mean.dtype == rstd.dtype == torch.float32
    assert mean.shape == rstd.shape == (*shape[:-1], 1)
    torch.testing.assert_close(
        mean, x.float().mean(-1, keepdim=True), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        rstd,
        torch.rsqrt(x.float().var(-1, unbiased=False, keepdim=True) + 1e-5),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        y, F.layer_norm(x.float(), (16,), w, b, 1e-5).bfloat16(), rtol=0, atol=0
    )
    gy = torch.randn(shape).bfloat16()
    xf, wf, bf = [t.float().detach().requires_grad_() for t in (x, w, b)]
    expected = torch.autograd.grad(
        F.layer_norm(xf, (16,), wf, bf, 1e-5), (xf, wf, bf), gy.float()
    )
    for actual, reference in zip(
        _layernorm_bwd_fp32(gy, x, mean, rstd, w, b), expected
    ):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_layernorm_is_finite_for_constant_inputs():
    x = torch.full((3, 16), 1000.0, dtype=torch.bfloat16)
    y, mean, rstd = _layernorm_fwd_fp32(x, torch.ones(16), torch.zeros(16), 1e-5)
    assert torch.isfinite(y).all() and torch.isfinite(rstd).all()
    torch.testing.assert_close(y, torch.zeros_like(y), rtol=0, atol=0)
    assert torch.equal(mean, torch.full_like(mean, 1000))


@pytest.mark.parametrize("shape", [(5, 16), (2, 3, 16)])
def test_mlp_forward_and_all_gradients(shape):
    mlp = MLP(16, 32).double()
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    parameters = tuple(mlp.parameters())
    expected = F.linear(
        F.gelu(F.linear(x, mlp.fc1.weight, mlp.fc1.bias), approximate="tanh"),
        mlp.fc2.weight,
        mlp.fc2.bias,
    )
    actual = mlp(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (x, *parameters), gradient, retain_graph=True)
    refs = torch.autograd.grad(expected, (x, *parameters), gradient)
    for a, b in zip(grads, refs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "emb_dim,hidden_dim", [(0, 8), (-1, 8), (8, 0), (8, -2), (True, 8), (8, 4.5)]
)
def test_mlp_rejects_invalid_dimensions(emb_dim, hidden_dim):
    with pytest.raises(ValueError, match="positive integer"):
        MLP(emb_dim, hidden_dim)
