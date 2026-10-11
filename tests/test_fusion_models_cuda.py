"""Same-session reference/candidate comparisons using real FlashAttention."""

import copy
import pytest
import torch

from varlen_transformer import create_block, create_memory_query_block, FixedMemoryQueryLayout, PackedMemoryQueryLayout

pytestmark = [pytest.mark.cuda, pytest.mark.fusion]


def workload(architecture, packed, dtype, pdtype, dropout, depth=2):
    if architecture == "block":
        model = torch.nn.ModuleList([create_block(64,256,4,attn_dropout=0.1 if dropout==0.25 else 0,
            block_resid_dropout1=dropout,block_resid_dropout2=dropout,causal=True) for _ in range(depth)])
        metadata = dict(cu_seqlens=torch.tensor([0,5,13],device="cuda",dtype=torch.int32),max_seqlen=8) if packed else {}
        shape = (13,64) if packed else (2,8,64)
    else:
        model = torch.nn.ModuleList([create_memory_query_block(64,256,4,
            attn_dropout=0.1 if dropout==0.25 else 0,entry_resid_dropout=dropout,
            sa_resid_dropout=dropout,ca_resid_dropout=dropout) for _ in range(depth)])
        layout = PackedMemoryQueryLayout.autoregressive([3,2],[4,2],device="cuda") if packed else FixedMemoryQueryLayout.autoregressive(3,4,batch_size=2)
        metadata = dict(layout=layout)
        shape = (layout.total_tokens,64)
    model = model.to(device="cuda",dtype=pdtype)
    for block in model:
        configs = [block.mixer.attn] if architecture=="block" else [block.memory_mixer.attn,block.cross_mixer.attn]
        for cfg in configs:
            cfg.deterministic = True
    x = torch.randn(shape,device="cuda",dtype=dtype)
    residual = torch.randn_like(x)
    return model,x,residual,metadata


def run(model,x,residual,metadata):
    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_()
    residual = residual.detach().clone().requires_grad_()
    h,r = x,residual
    for block in model:
        h,r = block(h,r,**metadata)
    (h.float().square().mean()+r.float().square().mean()).backward()
    tensors = (h,r,x.grad,residual.grad,*[p.grad for p in model.parameters() if p.requires_grad])
    return [v.detach().clone() for v in tensors],torch.cuda.get_rng_state()

@torch.no_grad()
def compare(a,b):
    for actual,ref in zip(a,b):
        assert actual.dtype==ref.dtype and actual.shape==ref.shape
        torch.testing.assert_close(actual,ref,rtol=0.015,atol=0.002)
        error = float((actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-12))
        assert error < 0.006, f"relative L2 error={error}"


@pytest.mark.parametrize("mode",["ln_fwd","residual_fwd","ln_bwd","ln_fwd_bwd","residual_fwd_bwd"])
@pytest.mark.parametrize("architecture",["block","memory_query"])
@pytest.mark.parametrize("packed",[False,True])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize("pdtype",[torch.float32,torch.bfloat16])
@pytest.mark.parametrize("dropout",[0.0,0.25,1.0])
def test_real_full_model_gradients_rng_and_frozen_parameters(fusion_calls,mode,architecture,packed,dtype,pdtype,dropout):
    torch.manual_seed(41)
    baseline,x,r,metadata = workload(architecture,packed,dtype,pdtype,dropout)
    next(baseline.parameters()).requires_grad_(False)
    candidate = copy.deepcopy(baseline)
    torch.manual_seed(99)
    expected,rng = run(baseline,x,r,metadata)
    for block in candidate:
        block.fusion = mode
    fusion_calls.clear()
    torch.manual_seed(99)
    actual,crng = run(candidate,x,r,metadata)
    compare(actual,expected)
    assert torch.equal(crng,rng), "Fusion altered CUDA RNG advancement"
    key = "residual_forward" if mode.startswith("residual") else "ln_backward" if mode=="ln_bwd" else "ln_forward"
    assert fusion_calls[key] > 0, dict(fusion_calls)
    if mode.endswith("bwd"):
        assert fusion_calls["ln_backward"] > 0, dict(fusion_calls)
    candidate.eval(); baseline.eval()
    with torch.inference_mode():
        eh,er = x,r
        for block in baseline:
            eh,er = block(eh,er,**metadata)
        ah,ar = x,r
        for block in candidate:
            ah,ar = block(ah,ar,**metadata)
        compare((ah,ar),(eh,er))


@pytest.mark.parametrize("architecture",["block","memory_query"])
@pytest.mark.parametrize("depth",[1,4])
def test_real_repeated_optimizer_updates(fusion_calls,architecture,depth):
    torch.manual_seed(51)
    baseline,x,r,metadata = workload(architecture,True,torch.float32,torch.float32,0,depth)
    candidate = copy.deepcopy(baseline)
    for block in candidate:
        block.fusion = "residual_fwd_bwd"
    optimizers = [torch.optim.SGD(m.parameters(),lr=0.001) for m in (baseline,candidate)]
    for _ in range(3):
        expected,_ = run(baseline,x,r,metadata)
        actual,_ = run(candidate,x,r,metadata)
        compare(actual,expected)
        for optimizer in optimizers:
            optimizer.step()
        for a,b in zip(candidate.parameters(),baseline.parameters()):
            torch.testing.assert_close(a,b,rtol=0.001,atol=1e-5)


@pytest.mark.parametrize("architecture", ["block", "memory_query"])
def test_real_backward_preserves_mode_after_model_reconfiguration(fusion_calls, architecture):
    torch.manual_seed(73)
    baseline, x, residual, metadata = workload(architecture, True, torch.float32, torch.float32, 0)
    candidate = copy.deepcopy(baseline)
    for block in candidate:
        block.fusion = "residual_fwd_bwd"
    expected, _ = run(baseline, x, residual, metadata)
    cx = x.detach().clone().requires_grad_()
    cr = residual.detach().clone().requires_grad_()
    h, r = cx, cr
    for block in candidate:
        h, r = block(h, r, **metadata)
    for block in candidate:
        block.fusion = "off"
    fusion_calls.clear()
    (h.float().square().mean() + r.float().square().mean()).backward()
    compare((h, r, cx.grad, cr.grad, *[p.grad for p in candidate.parameters()]), expected)
    assert fusion_calls["ln_backward"] > 0
