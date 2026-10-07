import pytest
import torch

from tests.reference import (
    assert_gradients,
    clone_args,
    post_args,
    post_attention,
    pre_args,
    pre_attention,
)
from varlen_transformer.block import post_attn_mlp, pre_attn_qkv

OPERATORS = [
    (pre_attn_qkv, pre_args, pre_attention),
    (post_attn_mlp, post_args, post_attention),
]


@pytest.mark.parametrize("op,make_args,reference", OPERATORS, ids=["pre", "post"])
@pytest.mark.parametrize("shape", [(5, 16), (2, 3, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "p,training", [(0.0, True), (0.25, True), (0.25, False), (1.0, True)]
)
def test_custom_op_forward_and_every_input_gradient(
    op, make_args, reference, shape, dtype, p, training
):
    args = make_args(shape, dtype=dtype, p=p, training=training)
    expected_args = clone_args(args)
    actual = op(*args)
    expected = reference(expected_args, actual[2])
    for a, b in zip(actual[:2], expected):
        assert a.dtype == torch.bfloat16
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(not auxiliary.requires_grad for auxiliary in actual[2:])
    gradients = tuple(torch.randn_like(a) / a.numel() ** 0.5 for a in actual[:2])
    inputs = tuple(a for a in args if isinstance(a, torch.Tensor))
    expected_inputs = tuple(a for a in expected_args if isinstance(a, torch.Tensor))
    actual_grads = torch.autograd.grad(actual[:2], inputs, gradients)
    expected_grads = torch.autograd.grad(expected, expected_inputs, gradients)
    for g, t in zip(actual_grads, inputs):
        assert g.shape == t.shape and g.dtype == t.dtype
    assert_gradients(actual_grads, expected_grads)


@pytest.mark.parametrize("op,make_args,reference", OPERATORS, ids=["pre", "post"])
@pytest.mark.parametrize("output_index", [0, 1])
def test_unused_main_output_has_a_correct_zero_gradient(
    op, make_args, reference, output_index
):
    """Regression: a residual-only loss previously constructed an impossible QKV view."""
    args = make_args()
    expected_args = clone_args(args)
    actual = op(*args)
    expected = reference(expected_args, actual[2])
    gradient = (
        torch.randn_like(actual[output_index]) / actual[output_index].numel() ** 0.5
    )
    inputs = tuple(a for a in args if isinstance(a, torch.Tensor))
    expected_inputs = tuple(a for a in expected_args if isinstance(a, torch.Tensor))
    actual_grads = torch.autograd.grad(
        actual[output_index], inputs, gradient, allow_unused=True
    )
    expected_grads = torch.autograd.grad(
        expected[output_index], expected_inputs, gradient, allow_unused=True
    )
    # Custom backward returns zeros for weights whose branch was unused.
    expected_grads = tuple(
        torch.zeros_like(t) if g is None else g
        for t, g in zip(expected_inputs, expected_grads)
    )
    actual_grads = tuple(
        torch.zeros_like(t) if g is None else g for t, g in zip(inputs, actual_grads)
    )
    assert_gradients(actual_grads, expected_grads)


@pytest.mark.parametrize("op,make_args,reference", OPERATORS, ids=["pre", "post"])
@pytest.mark.parametrize("p,training", [(0.0, True), (0.25, True), (0.25, False)])
def test_custom_op_schema_fake_shapes_autograd_and_aot_dispatch(
    op, make_args, reference, p, training
):
    args = make_args(p=p, training=training)
    results = torch.library.opcheck(op, args)
    assert set(results.values()) == {"SUCCESS"}


@pytest.mark.parametrize("op,make_args,reference", OPERATORS, ids=["pre", "post"])
def test_custom_op_does_not_modify_or_alias_inputs(op, make_args, reference):
    args = make_args(requires_grad=False)
    tensors = tuple(a for a in args if isinstance(a, torch.Tensor))
    snapshots = tuple(t.clone() for t in tensors)
    outputs = op(*args)
    for tensor, snapshot in zip(tensors, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    storage_pointers = [
        t.untyped_storage().data_ptr() for t in (*tensors, *outputs) if t.numel()
    ]
    assert len(storage_pointers) == len(set(storage_pointers))


@pytest.mark.parametrize("op,make_args,reference", OPERATORS, ids=["pre", "post"])
def test_custom_op_handles_noncontiguous_hidden_states(op, make_args, reference):
    args = list(make_args())
    for index in (0, 1):
        args[index] = args[index].T.contiguous().T.detach().requires_grad_()
        assert not args[index].is_contiguous()
    actual = op(*args)
    expected = reference(args, actual[2])
    for a, b in zip(actual[:2], expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("op,make_args,reference", OPERATORS, ids=["pre", "post"])
def test_custom_op_inference_mode_and_frozen_parameters(op, make_args, reference):
    args = make_args(requires_grad=False)
    with torch.inference_mode():
        result = op(*args)
    assert all(not output.requires_grad for output in result)
    args = list(args)
    args[0] = args[0].detach().requires_grad_()
    result = op(*args)
    result[0].float().square().mean().backward()
    assert args[0].grad is not None and torch.isfinite(args[0].grad).all()
    assert all(t.grad is None for t in args[1:] if isinstance(t, torch.Tensor))


def test_pre_operator_compile_accepts_multiple_fixed_shapes():
    def projection(*tensor_args):
        return pre_attn_qkv(*tensor_args, 0.0, True, 1e-5, 2, 8)[:2]

    compiled = torch.compile(
        projection, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for shape in [(2, 3, 16), (3, 5, 16)]:
        args = pre_args(shape)
        actual = compiled(*args[:6])
        expected = projection(*args[:6])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("op,make_args,reference", OPERATORS, ids=["pre", "post"])
def test_custom_op_rejects_double_backward_instead_of_returning_incomplete_derivatives(
    op, make_args, reference
):
    args = make_args()
    output = op(*args)[0]
    first = torch.autograd.grad(
        output.float().square().sum(), args[0], create_graph=True
    )[0]
    with pytest.raises(
        RuntimeError,
        match="differentiate twice|does not require grad|not have been used",
    ):
        torch.autograd.grad(first.sum(), args[0])
