"""Reference fallback, ownership, precision-boundary, and real CUDA checks."""

import pytest
import torch

from varlen_transformer import _fusion
from varlen_transformer.block import _layernorm_fwd_fp32, _layernorm_bwd_fp32

MODES = ["off", "ln_fwd", "residual_fwd", "ln_bwd", "ln_fwd_bwd", "residual_fwd_bwd"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_keeps_reference_and_does_not_import_backend(monkeypatch, mode, dtype):
    def forbidden():
        raise AssertionError("CPU must not load Triton")
    monkeypatch.setattr(_fusion, "_load", forbidden)
    x = torch.randn(7, 128).bfloat16()
    w, b = torch.randn(128, dtype=dtype), torch.randn(128, dtype=dtype)
    y, m, r = _layernorm_fwd_fp32(x, w, b, 1e-5, mode)
    ref = torch.ops.aten.native_layer_norm.default(x.float(), [128], w.float(), b.float(), 1e-5)
    for a, z in zip((y, m, r), (ref[0].bfloat16(), *ref[1:])):
        torch.testing.assert_close(a, z, rtol=0, atol=0)
    gy = torch.randn_like(x)
    actual = _layernorm_bwd_fp32(gy, x, m, r, w, b, mode)
    expected = torch.ops.aten.native_layer_norm_backward.default(
        gy.float(), x.float(), [128], m, r, w.float(), b.float(), [True, True, True])
    for a, z in zip(actual, expected):
        assert a.dtype == torch.float32
        torch.testing.assert_close(a, z, rtol=0, atol=0)
    assert _fusion.residual_forward(x, x, w, b, 1e-5, fusion=mode) is None


def test_missing_backend_is_cached_without_runtime_controls(monkeypatch):
    monkeypatch.setattr(_fusion, "_backend", None)
    monkeypatch.setattr(_fusion, "_missing", False)
    imports = []

    def missing(*args):
        imports.append(args)
        raise ModuleNotFoundError("Optional Triton is absent", name="triton")

    monkeypatch.setattr(_fusion.importlib, "import_module", missing)
    assert _fusion._load() is None
    assert _fusion._load() is None
    assert len(imports) == 1


@pytest.mark.parametrize("error", [RuntimeError("backend failure"), ModuleNotFoundError("broken backend", name="triton.language")])
def test_backend_errors_propagate(monkeypatch, error):
    monkeypatch.setattr(_fusion, "_backend", None)
    monkeypatch.setattr(_fusion, "_missing", False)

    def broken(*args):
        raise error

    monkeypatch.setattr(_fusion.importlib, "import_module", broken)
    with pytest.raises(type(error), match=str(error)):
        _fusion._load()


def test_backward_fake_boundary_never_launches_triton(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    def forbidden():
        raise AssertionError("Fake kernel must not load a CUDA backend")
    monkeypatch.setattr(_fusion, "_load", forbidden)
    with FakeTensorMode():
        x = torch.empty(17, 128, device="cuda", dtype=torch.bfloat16)
        stat = torch.empty(17, 1, device="cuda")
        weight = torch.empty(128, device="cuda")
        result = _fusion._layernorm_backward_op(x, x, stat, stat, weight)
        assert [v.shape for v in result] == [(17, 128), (128,), (128,)]
        assert all(v.dtype == torch.float32 and v.device.type == "cuda" for v in result)


@pytest.mark.cuda
@pytest.mark.fusion
@pytest.mark.parametrize("e", [128, 512, 1000, 4096])
@pytest.mark.parametrize("rows", [1, 17, 1024])
@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
def test_cuda_norm_forward_backward_against_native(fusion_calls, e, rows, parameter_dtype):
    mode = "ln_fwd_bwd"
    torch.manual_seed(13)
    x = torch.randn(rows, e, device="cuda").bfloat16()
    w = torch.randn(e, device="cuda").to(parameter_dtype)
    b = torch.randn_like(w)
    gy = torch.randn_like(x)
    y, m, r = _layernorm_fwd_fp32(x, w, b, 1e-5, mode)
    ry, rm, rr = torch.ops.aten.native_layer_norm.default(x.float(), [e], w.float(), b.float(), 1e-5)
    torch.testing.assert_close(m, rm, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(r, rr, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(y, ry.bfloat16(), rtol=0.008, atol=0.002)
    assert float((y.float()-ry.bfloat16().float()).norm() / ry.bfloat16().float().norm().clamp_min(1e-12)) < 5e-4
    actual = _layernorm_bwd_fp32(gy, x, m, r, w, b, mode)
    # Use native statistics to isolate the backward implementation itself.
    isolated = _layernorm_bwd_fp32(gy, x, rm, rr, w, b, mode)
    expected = torch.ops.aten.native_layer_norm_backward.default(
        gy.float(), x.float(), [e], rm, rr, w.float(), b.float(), [True, True, True])
    for a, z in zip(isolated, expected):
        assert a.dtype == torch.float32
        torch.testing.assert_close(a, z, rtol=2e-4, atol=5e-5)
        assert float((a-z).norm() / z.norm().clamp_min(1e-12)) < 1e-4
    for a, z in zip(actual, expected):
        torch.testing.assert_close(a, z, rtol=3e-4, atol=1e-4)
    assert fusion_calls["ln_forward"] > 0
    assert fusion_calls["ln_backward"] > 0


@pytest.mark.cuda
@pytest.mark.fusion
@pytest.mark.parametrize("start,length,norm_rows", [(0, 23, 23), (0, 17, 23), (17, 6, 23), (0, 23, 17)])
def test_cuda_partial_residual_rounding_and_storage(fusion_calls, start, length, norm_rows):
    torch.manual_seed(20)
    x = torch.randn(23, 128, device="cuda").bfloat16()
    branch = torch.randn(length, 128, device="cuda").bfloat16()
    # Include BF16 ties, cancellation, and constant rows.
    x[0] = 128
    branch[0] = 0.5
    w, b = torch.ones(128, device="cuda"), torch.zeros(128, device="cuda")
    snapshot = x.clone()
    result = _fusion.residual_forward(x, branch, w, b, 1e-5, start, norm_rows, fusion="residual_fwd")
    assert result is not None
    assert fusion_calls["residual_forward"] == 1
    residual, y, m, r = result
    expected = x.clone()
    expected[start:start+length] = x[start:start+length] + branch
    torch.testing.assert_close(residual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, snapshot, rtol=0, atol=0)
    pointers = [z.untyped_storage().data_ptr() for z in (x, branch, residual, y, m, r)]
    assert len(pointers) == len(set(pointers))
    ry, rm, rr = torch.ops.aten.native_layer_norm.default(expected[:norm_rows].float(), [128], w, b, 1e-5)
    torch.testing.assert_close(y, ry.bfloat16(), rtol=0.008, atol=0.002)
    torch.testing.assert_close(m, rm, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(r, rr, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("kind", ["sa", "ca", "mlp", "pre", "post"])
def test_saved_parameter_versions_still_checked(kind):
    from tests.memory_reference import make_args
    from tests.reference import pre_args, post_args
    from varlen_transformer._memory_ops import mq_pre_sa, mq_pre_ca, mq_post_ca_mlp
    from varlen_transformer.block import pre_attn_qkv, post_attn_mlp
    op, args = {
        "sa": (mq_pre_sa, lambda: make_args("sa")),
        "ca": (mq_pre_ca, lambda: make_args("ca")),
        "mlp": (mq_post_ca_mlp, lambda: make_args("mlp")),
        "pre": (pre_attn_qkv, pre_args),
        "post": (post_attn_mlp, post_args),
    }[kind]
    values = args()
    output = op(*values)[0]
    with torch.no_grad():
        values[2].add_(0.001)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.float().sum().backward()
