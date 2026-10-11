"""CPU-only orchestration checks for paired fusion benchmarks (no GPU claims)."""

from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "paired_fusion_driver", ROOT / "benchmarks/profile_fusion.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def config(*args):
    return driver.configuration(driver.parser().parse_args(["--architecture", "standard_causal", "--fusion", "ln_bwd", "--output", "placeholder.json", *args]))


def test_configuration_supports_both_standard_attention_modes():
    ca = config("--batch-size", "3", "--layout", "packed", "--sequence-lengths", "7,31,127")
    assert ca["sequence_lengths"] == [7, 31, 127]
    assert ca["total_tokens"] == 165
    other = driver.configuration(driver.parser().parse_args([
        "--architecture", "standard_bidirectional", "--fusion", "residual_fwd_bwd",
        "--output", "other.json",
    ]))
    assert other["sequence_lengths"] == [128] * 8


def test_memory_query_packed_configuration():
    cfg = driver.configuration(driver.parser().parse_args([
        "--architecture", "memory_query", "--layout", "packed",
        "--batch-size", "3", "--memory-lengths", "7,31,127", "--query-lengths", "3,4,5",
        "--fusion", "ln_bwd", "--output", "other.json",
    ]))
    assert cfg["total_tokens"] == 165 + 3 + 12
    assert cfg["query_lengths"] == [3, 4, 5]


@pytest.mark.parametrize("opts", [
    ("--batch-size", "0"),
    ("--emb-dim", "129"),
    ("--execution", "inference"),
    ("--layout", "packed"),
    ("--layout", "fixed", "--sequence-lengths", "7,8,9,10,11,12,13,14"),
    ("--layout", "packed", "--sequence-lengths", "1,2"),
    ("--memory-lengths", "4,5"),
])
def test_bad_standard_shapes_rejected(opts):
    with pytest.raises(ValueError):
        config(*opts)


def test_empty_or_unrecognized_mode_rejected():
    with pytest.raises(SystemExit):
        driver.parser().parse_args(["--architecture", "standard_causal", "--fusion", "off", "--output", "x.json"])


def test_dispatch_wrapper_counts_actual_kernels_and_restores_module():
    class Backend:
        def forward(self, value):
            return value + 1
        def backward(self, value):
            return value - 1
        def residual_forward(self, value):
            return None
    backend = Backend()
    original = backend.forward.__func__
    with driver.count_fused_kernels(backend) as calls:
        assert backend.forward(3) == 4
        assert backend.backward(3) == 2
        assert backend.residual_forward(3) is None
        assert calls == {"forward": 1, "backward": 1}
    assert backend.forward.__func__ is original


def test_dispatch_fails_instead_of_reporting_fallback_as_fused():
    class Backend:
        def forward(self, *args):
            return None
        def residual_forward(self, *args):
            return None
        def backward(self, *args):
            return None
    with pytest.raises(RuntimeError, match="never dispatched"):
        driver.verify_dispatch(SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _: None)),
                               Backend(), "ln_fwd", False,
                               lambda mode: None, lambda: None, lambda: None, 0)


def test_cli_help_requires_no_gpu_libraries(capsys):
    with pytest.raises(SystemExit) as exc:
        driver.parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "standard_bidirectional" in out and "memory_query" in out


@pytest.mark.parametrize("arch,layout", [
    ("standard_causal", "fixed"),
    ("standard_bidirectional", "packed"),
    ("memory_query", "fixed"),
    ("memory_query", "packed"),
])
def test_workload_executes_reference_on_cpu_for_both_architectures(
    reference_backend, memory_reference_backend, arch, layout
):
    import torch
    import varlen_transformer as vt

    arguments = ["--architecture", arch, "--layout", layout, "--batch-size", "2",
                 "--emb-dim", "16", "--num-heads", "2", "--depth", "2",
                 "--fusion", "ln_fwd_bwd", "--output", "placeholder.json"]
    if arch.startswith("standard"):
        arguments += (["--sequence-lengths", "4,7"] if layout == "packed"
                      else ["--sequence-length", "4"])
    else:
        arguments += (["--memory-lengths", "4,7", "--query-lengths", "2,3"]
                      if layout == "packed" else ["--memory-length", "4", "--query-length", "3"])
    cfg = driver.configuration(driver.parser().parse_args(arguments))
    change, call, cleanup, blocks, x = driver.build_model(torch, vt, cfg, torch.device("cpu"))
    change("off")
    out = call()
    assert out.numel() == cfg["total_tokens"] * cfg["emb_dim"]
    assert x.grad is not None and torch.isfinite(x.grad).all()
    cleanup()
    change("ln_fwd_bwd")  # CPU falls back to the original normalization.
    again = call()
    torch.testing.assert_close(again, out, rtol=0, atol=0)
    cleanup()
