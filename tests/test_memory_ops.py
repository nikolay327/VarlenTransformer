import pytest
import torch

from tests.reference import assert_gradients, clone_args
from tests.memory_reference import (
    make_args,
    pre_sa_reference,
    pre_ca_reference,
    post_ca_reference,
)
from varlen_transformer._memory_ops import mq_pre_sa, mq_pre_ca, mq_post_ca_mlp

OPS = [
    ("sa", mq_pre_sa, pre_sa_reference, 2),
    ("ca", mq_pre_ca, pre_ca_reference, 3),
    ("mlp", mq_post_ca_mlp, post_ca_reference, 2),
]


@pytest.mark.parametrize("kind,op,oracle,count", OPS)
@pytest.mark.parametrize("shape", [(9, 16), (2, 9, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "p,training", [(0.0, True), (0.25, True), (0.25, False), (1.0, True)]
)
def test_every_forward_and_gradient(kind, op, oracle, count, shape, dtype, p, training):
    args = make_args(kind, shape, dtype=dtype, p=p, training=training)
    ea = clone_args(args)
    actual = op(*args)
    expected = oracle(ea, actual[count])
    for a, b in zip(actual[:count], expected):
        assert a.dtype == torch.bfloat16
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(not t.requires_grad for t in actual[count:])
    grad = tuple(torch.randn_like(t) / t.numel() ** 0.5 for t in actual[:count])
    tensors = tuple(t for t in args if isinstance(t, torch.Tensor))
    et = tuple(t for t in ea if isinstance(t, torch.Tensor))
    ag = torch.autograd.grad(actual[:count], tensors, grad)
    eg = torch.autograd.grad(expected, et, grad)
    for g, t in zip(ag, tensors):
        assert g.dtype == t.dtype and g.shape == t.shape
    assert_gradients(ag, eg)


@pytest.mark.parametrize("kind,op,oracle,count", OPS)
@pytest.mark.parametrize(
    "p,training", [(0.0, True), (0.25, True), (0.25, False), (1.0, True)]
)
def test_schema_fake_alias_autograd_and_aot(kind, op, oracle, count, p, training):
    result = torch.library.opcheck(op, make_args(kind, p=p, training=training))
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.parametrize("kind,op,oracle,count", OPS)
def test_unused_outputs_noncontiguity_and_all_branches(kind, op, oracle, count):
    for index in range(count):
        args = list(make_args(kind))
        for i in (0, 1):
            args[i] = args[i].T.contiguous().T.detach().requires_grad_()
        ea = clone_args(args)
        actual = op(*args)
        expected = oracle(ea, actual[count])
        tensors = tuple(t for t in args if isinstance(t, torch.Tensor))
        et = tuple(t for t in ea if isinstance(t, torch.Tensor))
        grad = torch.randn_like(actual[index]) / actual[index].numel() ** 0.5
        ag = torch.autograd.grad(actual[index], tensors, grad, allow_unused=True)
        eg = torch.autograd.grad(expected[index], et, grad, allow_unused=True)
        ag = [torch.zeros_like(t) if g is None else g for g, t in zip(ag, tensors)]
        eg = [torch.zeros_like(t) if g is None else g for g, t in zip(eg, et)]
        assert_gradients(ag, eg)


@pytest.mark.parametrize("kind,op,oracle,count", OPS)
def test_no_mutation_aliasing_frozen_params_inference_and_double_backward(
    kind, op, oracle, count
):
    args = make_args(kind, requires_grad=False)
    tensors = tuple(t for t in args if isinstance(t, torch.Tensor))
    snapshots = [t.clone() for t in tensors]
    with torch.inference_mode():
        outputs = op(*args)
    for t, s in zip(tensors, snapshots):
        torch.testing.assert_close(t, s, rtol=0, atol=0)
    pointers = [
        t.untyped_storage().data_ptr() for t in (*tensors, *outputs) if t.numel()
    ]
    assert len(pointers) == len(set(pointers))
    args = list(args)
    ti = 1 if kind == "ca" else 0
    args[ti] = args[ti].detach().requires_grad_()
    y = op(*args)[0]
    first = torch.autograd.grad(y.float().square().sum(), args[ti], create_graph=True)[
        0
    ]
    assert torch.isfinite(first).all()
    with pytest.raises(
        RuntimeError,
        match="differentiate twice|does not require grad|not have been used",
    ):
        torch.autograd.grad(first.sum(), args[ti])


@pytest.mark.parametrize("kind,op,oracle,count", OPS)
def test_dynamic_aot_two_shapes_and_backward(kind, op, oracle, count):
    def region(*args):
        return op(*args)[:count]

    compiled = torch.compile(region, backend="aot_eager", fullgraph=True, dynamic=True)
    for shape in ((2, 9, 16), (3, 11, 16)):
        args = make_args(kind, shape)
        actual, expected = compiled(*args), region(*args)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        sum(t.float().square().mean() for t in actual).backward()
        assert all(
            t.grad is not None and torch.isfinite(t.grad).all()
            for t in args
            if isinstance(t, torch.Tensor)
        )
