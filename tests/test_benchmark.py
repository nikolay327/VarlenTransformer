"""Check report compatibility and timing orchestration without inventing GPU data."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

DRIVER_PATH = Path(__file__).resolve().parents[1] / "benchmarks/profile_memory_query.py"
spec = importlib.util.spec_from_file_location("memory_query_benchmark", DRIVER_PATH)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def config(*arguments):
    return driver.configuration(driver.parser().parse_args(arguments))


def test_configuration_normalizes_fixed_and_heterogeneous_workloads():
    fixed = config("--depth", "1", "--with-residual", "--mode", "train")
    assert fixed["memory_lengths"] == [128] * 4
    assert fixed["x_prefix_lengths"] == [125] * 4
    assert fixed["total_tokens"] == 4 * (128 + 1 + 4)
    assert fixed["intermediate_size"] == 512
    packed = config(
        "--layout",
        "packed",
        "--batch-size",
        "3",
        "--memory-lengths",
        "7,31,127",
        "--query-lengths",
        "3,4,5",
    )
    assert packed["x_prefix_lengths"] == [5, 28, 123]
    assert packed["total_tokens"] == 180


@pytest.mark.parametrize(
    "arguments",
    [
        ["--batch-size", "0"],
        ["--depth", "0"],
        ["--warmup", "0"],
        ["--iterations", "0"],
        ["--runs", "0"],
        ["--device", "-1"],
        ["--intermediate-size", "0"],
        ["--emb-dim", "15", "--num-heads", "4"],
        ["--emb-dim", "512", "--num-heads", "1"],
        ["--memory-length", "3", "--query-length", "4"],
        ["--memory-lengths", "3,4,5,6"],
        ["--implementation", "wrapper", "--layout", "packed"],
        ["--layout", "packed", "--memory-lengths", "3,4"],
        ["--memory-length", str(2**31)],
    ],
)
def test_invalid_workloads_fail_before_cuda_setup(arguments):
    with pytest.raises(ValueError):
        config(*arguments)


@pytest.mark.parametrize("value", ["", "0,3", "1,-2", "1,no", "1,"])
def test_invalid_length_lists_are_cli_errors(value):
    with pytest.raises(SystemExit):
        driver.parser().parse_args(["--memory-lengths", value])


def report_pair():
    metadata = dict(
        hardware=dict(
            gpu_name="test-only GPU",
            gpu_uuid="test-only UUID",
            driver_version="test-only",
            compute_capability=[8, 0],
            total_memory_bytes=1024,
        ),
        software=dict(
            pytorch="test-only",
            cuda="test-only",
            flash_attention="test-only",
            python="test-only",
            tf32=False,
            bf16_reduced_precision_reduction=True,
            cpu_threads=1,
        ),
        driver_sha256="test-only driver digest",
        sampled_workload_sha256="test-only input digest",
        source=dict(commit=driver.BASELINE_COMMIT, dirty=False),
    )
    baseline = dict(
        schema_version=1,
        protocol=driver.PROTOCOL,
        metadata=metadata,
        configuration=config("--implementation", "legacy"),
        measurements=[dict(scope="legacy_fixed", median_ms=8.0)],
    )
    current = copy.deepcopy(baseline)
    current["configuration"]["implementation"] = "native"
    current["metadata"]["source"]["commit"] = "test-only revised commit"
    current["measurements"] = [
        dict(scope="native_core", median_ms=2.0),
        dict(scope="native_end_to_end", median_ms=4.0),
        dict(scope="conversion_pack", median_ms=1.0),
    ]
    return current, baseline


def test_comparison_distinguishes_core_wrapper_and_packed_scopes():
    current, baseline = report_pair()
    compared = driver.compare_reports(current, baseline)
    assert [(row["scope"], row["baseline_over_new_ratio"]) for row in compared] == [
        ("native_core", 4.0),
        ("native_end_to_end", 2.0),
    ]
    current["configuration"]["implementation"] = "wrapper"
    current["measurements"] = [dict(scope="wrapper_end_to_end", median_ms=4.0)]
    assert (
        driver.compare_reports(current, baseline)[0]["baseline_scope"] == "legacy_fixed"
    )
    for report in (current, baseline):
        report["configuration"]["layout"] = "packed"
    current["configuration"]["implementation"] = "native"
    current["measurements"] = [dict(scope="native_packed", median_ms=4.0)]
    baseline["measurements"][0]["scope"] = "legacy_packed"
    assert (
        driver.compare_reports(current, baseline)[0]["baseline_scope"]
        == "legacy_packed"
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("protocol",), "different protocol"),
        (("schema_version",), 2),
        (("metadata", "hardware", "gpu_uuid"), "other GPU"),
        (("metadata", "hardware", "driver_version"), "other driver"),
        (("metadata", "software", "cuda"), "other CUDA"),
        (("metadata", "software", "pytorch"), "other Torch"),
        (("metadata", "software", "flash_attention"), "other FlashAttention"),
        (("metadata", "software", "tf32"), True),
        (("metadata", "software", "cpu_threads"), 2),
        (("metadata", "driver_sha256"), "other driver source"),
        (("metadata", "sampled_workload_sha256"), "other workload"),
        (("metadata", "source", "dirty"), True),
        (("configuration", "mode"), "train"),
        (("configuration", "depth"), 8),
        (("configuration", "input_dtype"), "fp16"),
        (("configuration", "iterations"), 17),
    ],
)
def test_comparison_rejects_incompatible_measurements(path, value):
    current, baseline = report_pair()
    target = current
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        driver.compare_reports(current, baseline)


def test_comparison_requires_recorded_baseline_gpu_identity_and_repeated_runs():
    for condition in (
        "single-run",
        "unknown-GPU",
        "wrong-baseline",
        "dirty-baseline",
        "missing-scope",
    ):
        current, baseline = report_pair()
        if condition == "single-run":
            for report in (current, baseline):
                report["configuration"]["runs"] = 1
        elif condition == "unknown-GPU":
            for report in (current, baseline):
                report["metadata"]["hardware"]["gpu_uuid"] = None
        elif condition == "wrong-baseline":
            baseline["metadata"]["source"]["commit"] = "other commit"
        elif condition == "dirty-baseline":
            baseline["metadata"]["source"]["dirty"] = True
        else:
            baseline["measurements"] = []
        with pytest.raises(ValueError):
            driver.compare_reports(current, baseline)


def test_statistics_reports_pooled_dispersion_throughput_and_each_run():
    stats = driver.timing_statistics([[1.0, 3.0], [5.0, 7.0]], samples=4, tokens=16)
    assert stats["median_ms"] == 4.0
    assert stats["p10_ms"] == pytest.approx(1.6)
    assert stats["p90_ms"] == pytest.approx(6.4)
    assert stats["samples_per_second"] == 1000
    assert stats["tokens_per_second"] == 4000
    assert stats["run_medians_ms"] == [2.0, 6.0]
    with pytest.raises(ValueError, match="zero event latency"):
        driver.timing_statistics([[0.0]], 1, 1)


def test_event_timing_excludes_cleanup_and_waits_before_reading_latency():
    calls = []

    class Event:
        def __init__(self, role):
            self.role = role

        def record(self):
            calls.append(self.role)

        def synchronize(self):
            calls.append("wait")

        def elapsed_time(self, end):
            assert self.role == "start" and end.role == "end"
            assert calls[-1] == "wait"
            calls.append("elapsed")
            return 2.0

    class Cuda:
        events = 0

        def Event(self, *, enable_timing):
            assert enable_timing
            self.events += 1
            return Event("start" if self.events % 2 else "end")

        def synchronize(self, device):
            calls.append("synchronize")

        def memory_allocated(self, device):
            calls.append("allocated")
            return 10

        def reset_peak_memory_stats(self, device):
            calls.append("reset")

        def max_memory_allocated(self, device):
            calls.append("peak")
            return 30

    settings = config("--runs", "2", "--warmup", "1", "--iterations", "2")
    stats = driver.measure(
        SimpleNamespace(cuda=Cuda()),
        lambda: calls.append("call"),
        lambda: calls.append("cleanup"),
        settings,
        0,
    )
    expected_run = ["cleanup", "call", "cleanup", "synchronize", "allocated", "reset"]
    expected_run += ["cleanup", "start", "call", "end", "wait", "elapsed"] * 2 + [
        "peak"
    ]
    assert calls == expected_run * 2 + ["cleanup"]
    assert stats["peak_allocated_bytes"] == 30
    assert stats["peak_incremental_bytes"] == 20
    assert stats["persistent_allocated_bytes"] == 10
    assert stats["latency_ms"] == [[2.0, 2.0], [2.0, 2.0]]


def test_reports_round_trip_json_and_csv(tmp_path):
    current, baseline = report_pair()
    stats = driver.timing_statistics([[2.0, 4.0], [2.0, 4.0]], 4, 532)
    current["measurements"] = [
        dict(
            scope="native_core",
            peak_allocated_bytes=1024,
            peak_incremental_bytes=512,
            persistent_allocated_bytes=512,
            **stats,
        )
    ]
    current["comparisons"] = driver.compare_reports(current, baseline)
    driver.write_reports(current, tmp_path)
    assert json.loads((tmp_path / "report.json").read_text()) == current
    with (tmp_path / "report.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["baseline_over_new_ratio"] == str(8 / 3)
    assert rows[0]["depth"] == "4"
    assert rows[0]["scope"] == "native_core"


def test_driver_help_does_not_import_optional_gpu_dependencies():
    result = subprocess.run(
        [sys.executable, str(DRIVER_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--baseline-json" in result.stdout and "--trace" in result.stdout


def test_no_cuda_is_an_error_and_does_not_write_measurement_files(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        driver, "source_metadata", lambda root: dict(commit="test-only", dirty=False)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as error:
        driver.main(["--output", str(tmp_path)])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", ["inference", "train"])
def test_native_and_wrapper_benchmark_workloads_agree(memory_reference_backend, mode):
    import varlen_transformer as vt

    arguments = [
        "--batch-size",
        "2",
        "--memory-length",
        "4",
        "--query-length",
        "3",
        "--emb-dim",
        "16",
        "--num-heads",
        "2",
        "--depth",
        "2",
        "--with-residual",
        "--mode",
        mode,
    ]
    cfg = config(*arguments)
    cases, conversions, cleanup, fingerprint = driver.build_workload(
        torch, vt, cfg, "cpu"
    )
    assert [scope for scope, _ in cases] == ["native_core", "native_end_to_end"]
    core = cases[0][1]()
    cleanup()
    end_to_end = cases[1][1]()
    cleanup()
    layout = vt.FixedMemoryQueryLayout.ncse(2, 3, batch_size=2)
    for flat, per_sample in zip(core, end_to_end):
        torch.testing.assert_close(
            flat, vt.pack_fixed_memory_query(per_sample, layout), rtol=0, atol=0
        )
    probes = conversions()
    assert [scope for scope, _ in probes] == ["conversion_pack", "conversion_unpack"]
    assert len(probes[0][1]()) == 2 and len(probes[1][1]()) == 2
    cfg["implementation"] = "wrapper"
    wrapped_cases, _, wrapped_cleanup, wrapped_fingerprint = driver.build_workload(
        torch, vt, cfg, "cpu"
    )
    assert wrapped_fingerprint == fingerprint
    assert wrapped_cases[0][0] == "wrapper_end_to_end"
    wrapped = wrapped_cases[0][1]()
    wrapped_cleanup()
    for expected, actual in zip(end_to_end, wrapped):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["inference", "train"])
def test_packed_benchmark_runs_the_heterogeneous_workload(
    memory_reference_backend, mode
):
    import varlen_transformer as vt

    cfg = config(
        "--layout",
        "packed",
        "--batch-size",
        "2",
        "--memory-lengths",
        "4,7",
        "--query-lengths",
        "3,4",
        "--emb-dim",
        "16",
        "--num-heads",
        "2",
        "--depth",
        "2",
        "--mode",
        mode,
    )
    cases, conversions, cleanup, _ = driver.build_workload(torch, vt, cfg, "cpu")
    assert len(cases) == 1 and cases[0][0] == "native_packed"
    assert conversions is None
    h, residual = cases[0][1]()
    cleanup()
    assert h.shape == residual.shape == (20, 16)
    assert torch.isfinite(h).all() and torch.isfinite(residual).all()
