"""CUDA LayerNorm and residual normalization dispatch with native fallback."""

import importlib

import torch
from torch import Tensor

MODES = frozenset({"off", "ln_fwd", "residual_fwd", "ln_bwd", "ln_fwd_bwd", "residual_fwd_bwd"})
_FORWARD_MODES = MODES - {"off", "ln_bwd"}
_RESIDUAL_MODES = frozenset({"residual_fwd", "residual_fwd_bwd"})
_BACKWARD_MODES = frozenset({"ln_bwd", "ln_fwd_bwd", "residual_fwd_bwd"})
_backend = None
_missing = False


def validate_mode(fusion: str) -> str:
    """Validate configuration when a model is created or reconfigured."""
    if not isinstance(fusion, str) or fusion not in MODES:
        raise ValueError(f"Invalid fusion mode: {fusion!r}; expected one of {sorted(MODES)}")
    return fusion


def _backward_rows(width):
    return min(256,16384 // (1 << (width-1).bit_length()))


def _forward_supported(x, weight, bias):
    return (
        _supported(x, weight, bias)
        and 32 <= x.shape[-1] <= 4096 and x.shape[-1] % 4 == 0
        and (weight.dtype != torch.float32 or weight.storage_offset() % 4 == 0)
        and (bias.dtype != torch.float32 or bias.storage_offset() % 4 == 0)
    )


def _load():
    global _backend, _missing
    if _backend is None and not _missing:
        try:
            _backend = importlib.import_module("._triton_norm", __package__)
        except ModuleNotFoundError as exc:
            if exc.name != "triton":
                raise
            _missing = True
    return _backend


def _supported(x, weight, bias):
    return (
        x.is_cuda and torch.version.hip is None and x.ndim > 0
        and x.dtype == torch.bfloat16 and x.is_contiguous() and x.numel() > 0
        and 1 <= x.shape[-1] <= 4096
        and weight.device == x.device and bias.device == x.device
        and weight.shape == bias.shape == (x.shape[-1],)
        and weight.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and bias.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and weight.is_contiguous() and bias.is_contiguous()
    )


def layernorm_forward(x, weight, bias, eps, fusion="off"):
    if fusion not in _FORWARD_MODES:
        return None
    if not _forward_supported(x, weight, bias):
        return None
    backend = _load()
    if backend is None:
        return None
    return backend.forward(x, weight, bias, eps)


def residual_forward(residual, branch, weight, bias, eps, start=0, norm_rows=None, fusion="off"):
    """Add a BF16 branch and normalize all or part of the residual stream.

    Partial updates use flat stream-major matrices.
    """
    if fusion not in _RESIDUAL_MODES:
        return None
    if branch.ndim == 0 or not _forward_supported(residual, weight, bias):
        return None
    rows = residual.numel() // residual.shape[-1]
    norm_rows = rows if norm_rows is None else norm_rows
    flat_partial = residual.ndim == branch.ndim == 2
    whole = start == 0 and branch.shape == residual.shape and norm_rows == rows
    if not (
        branch.is_contiguous()
        and branch.device == residual.device and branch.dtype == torch.bfloat16
        and branch.shape[-1] == residual.shape[-1] and (whole or flat_partial)
        and 0 <= start <= rows and 0 < norm_rows <= rows
        and start + branch.numel() // branch.shape[-1] <= rows
    ):
        return None
    backend = _load()
    if backend is None:
        return None
    return backend.residual_forward(residual, branch, weight, bias, eps, start, norm_rows)


def layernorm_backward(grad, x, mean, rstd, weight, bias, fusion="off"):
    if fusion not in _BACKWARD_MODES:
        return None
    if not _supported(x, weight, bias):
        return None
    rows = x.numel() // x.shape[-1]
    if not (
        grad.shape == x.shape
        and grad.device == x.device and grad.is_contiguous()
        and grad.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and mean.device == rstd.device == x.device
        and mean.dtype == rstd.dtype == torch.float32
        and mean.is_contiguous() and rstd.is_contiguous()
        and mean.numel() == rstd.numel() == rows
        and rows <= 1024 * _backward_rows(x.shape[-1])
    ):
        return None
    backend = _load()
    if backend is None:
        return None
    return _layernorm_backward_op(grad, x, mean, rstd, weight)


@torch.library.custom_op("varlen_transformer::fusion_ln_backward", mutates_args=(), device_types="cuda")
def _layernorm_backward_op(grad: Tensor, x: Tensor, mean: Tensor, rstd: Tensor,
                           weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Separate the Triton launch from FakeTensor/AOT shape propagation.
    return _load().backward(grad, x, mean, rstd, weight)


@_layernorm_backward_op.register_fake
def _(grad, x, mean, rstd, weight):
    return (x.new_empty(x.shape, dtype=torch.float32),
            weight.new_empty(weight.shape, dtype=torch.float32),
            weight.new_empty(weight.shape, dtype=torch.float32))
