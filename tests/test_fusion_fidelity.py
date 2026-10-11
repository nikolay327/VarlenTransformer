"""LayerNorm forward fidelity and backward gradient tests."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from varlen_transformer import _fusion
from varlen_transformer.block import _layernorm_fwd_fp32


@pytest.mark.parametrize("width,rows", [(128, 16385), (2048, 4097)])
def test_backward_group_bound_preserves_native_fallback(monkeypatch, width, rows):
    monkeypatch.setattr(_fusion, "_load", lambda: pytest.fail("Oversized reduction must fall back"))
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        x = torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
        stat = torch.empty(rows, 1, device="cuda")
        w = torch.empty(width, device="cuda")
        assert _fusion.layernorm_backward(x, x, stat, stat, w, w, fusion="ln_bwd") is None


@pytest.mark.parametrize("version", ["2.8.0", "2.9.2", "2.10.0.dev20261010+cu130"])
def test_fusion_dispatch_and_prerequisites_accept_other_torch_versions(version):
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
from types import SimpleNamespace
import torch

torch.__version__ = sys.argv[1]
from varlen_transformer import _fusion
from torch._subclasses.fake_tensor import FakeTensorMode
from tests import conftest

calls = []
result = object()
def forward(*args):
    calls.append("forward")
    return result
def residual_forward(*args):
    calls.append("residual_forward")
    return result
_fusion._load = lambda: SimpleNamespace(
    forward=forward, residual_forward=residual_forward
)
with FakeTensorMode():
    x = torch.empty(23, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.empty(128, device="cuda")
    assert _fusion.layernorm_forward(x, w, w, 1e-5, fusion="ln_fwd") is result
    assert _fusion.residual_forward(x, x, w, w, 1e-5, fusion="residual_fwd") is result
assert calls == ["forward", "residual_forward"]
conftest.cuda_unavailable_reason = lambda: None
conftest.importlib.util.find_spec = lambda name: object()
assert conftest.fusion_unavailable_reason() is None
"""
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [sys.executable, "-c", script, version],
        cwd=root, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["ln_fwd", "residual_fwd", "ln_fwd_bwd", "residual_fwd_bwd"])
def test_small_strict_shape_uses_native_arithmetic(monkeypatch, mode):
    monkeypatch.setattr(_fusion, "_load", lambda: pytest.fail("Small shape must retain native forward"))
    # Fake CUDA lets the dispatch contract run on CPU without a launch.
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        x = torch.empty(2, 9, 16, device="cuda", dtype=torch.bfloat16)
        w = torch.empty(16, device="cuda")
        assert _fusion.layernorm_forward(x, w, w, 1e-5, fusion=mode) is None
        assert _fusion.residual_forward(x, x, w, w, 1e-5, fusion=mode) is None


@pytest.mark.cuda
@pytest.mark.fusion
@pytest.mark.parametrize("shape", [(2,9,16),(2,9,128),(9,128)])
@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
@pytest.mark.parametrize("p,training", [(0.0,True),(0.25,True),(0.25,False),(1.0,True)])
def test_strict_mq_pre_ca_boundary_and_dropout(fusion_calls,shape,dtype,p,training):
    from tests.memory_reference import make_args, pre_ca_reference
    from varlen_transformer._memory_ops import mq_pre_ca
    from tests.reference import clone_args

    torch.manual_seed(1729)
    args = make_args("ca",shape,device="cuda",dtype=dtype,p=p,training=training)
    cloned = clone_args(args)
    torch.manual_seed(711)
    baseline = mq_pre_ca(*cloned)
    rng = torch.cuda.get_rng_state()
    torch.manual_seed(711)
    actual = mq_pre_ca(*args,fusion="residual_fwd_bwd")

    assert torch.equal(rng,torch.cuda.get_rng_state())
    if shape == (9,128):
        assert fusion_calls["residual_forward"] > 0
    elif shape == (2,9,128):
        assert fusion_calls["ln_forward"] > 0
        assert fusion_calls["residual_forward"] == 0
    else:
        assert fusion_calls["ln_forward"] == 0
        assert fusion_calls["residual_forward"] == 0

    for a,b in zip(actual,baseline):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(actual[:3],pre_ca_reference(clone_args(args),actual[3])):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.cuda
@pytest.mark.fusion
@pytest.mark.parametrize("width", [32,64,128,512,1000,4096])
@pytest.mark.parametrize("pdtype", [torch.float32,torch.bfloat16])
@pytest.mark.parametrize("kind", ["random","near_constant","cancellation","bf16_tie","affine_midpoint"])
def test_welford_forward_strict_acceptance(width, pdtype, kind):
    torch.manual_seed(2026)
    x=torch.randn(23,width,device="cuda").bfloat16()
    branch=torch.randn_like(x)
    if kind=="near_constant": x.fill_(128); x[:,0]=129; branch.zero_()
    if kind=="cancellation": branch=-x
    if kind=="bf16_tie": x.fill_(128); branch.fill_(0.5)
    w=torch.randn(width,device="cuda").to(pdtype)
    b=torch.randn_like(w)
    if kind=="affine_midpoint":
        # Place affine outputs at BF16 midpoints to detect rounding differences.
        w.fill_(1)
        z,_,_=torch.ops.aten.native_layer_norm.default((x+branch).float(),[width],w.float(),torch.zeros_like(w,dtype=torch.float32),1e-5)
        lo=z[0].bfloat16()
        hi=torch.nextafter(lo,torch.full_like(lo,float('inf')))
        midpoint=(lo.float()+hi.float())*0.5
        b=(midpoint-z[0]).to(pdtype)
    out=_fusion.residual_forward(x,branch,w,b,1e-5,fusion="residual_fwd")
    assert out is not None,"No accelerated forward was exercised"
    residual,y,mean,rstd=out
    expected=x+branch
    ry,rm,rr=torch.ops.aten.native_layer_norm.default(expected.float(),[width],w.float(),b.float(),1e-5)
    torch.testing.assert_close(residual,expected,rtol=0,atol=0)
    for a,z in zip((y,mean,rstd),(ry.bfloat16(),rm,rr)):
        torch.testing.assert_close(a,z,rtol=0,atol=0)


@pytest.mark.cuda
@pytest.mark.fusion
@pytest.mark.parametrize("width",[128,512])
def test_backward_uses_original_gradient_gate(fusion_calls,width):
    from varlen_transformer.block import _layernorm_bwd_fp32
    x=torch.randn(3200,width,device="cuda").bfloat16()
    grad=torch.randn_like(x)
    w=torch.randn(width,device="cuda");b=torch.randn_like(w)
    _,m,r=torch.ops.aten.native_layer_norm.default(x.float(),[width],w,b,1e-5)
    expected=torch.ops.aten.native_layer_norm_backward.default(grad.float(),x.float(),[width],m,r,w,b,[True,True,True])
    actual=_layernorm_bwd_fp32(grad,x,m,r,w,b,fusion="ln_bwd")
    assert fusion_calls['ln_backward']==1
    for a,z in zip(actual,expected):
        torch.testing.assert_close(a,z,rtol=2e-4,atol=5e-5)
        assert float((a-z).norm()/z.norm().clamp_min(1e-12))<1e-4
