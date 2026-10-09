"""The plot-data extraction is testable without CUDA or Matplotlib."""

import json

import pytest

from benchmarks.visualize_memory_query import chart_data, main


def report(with_baseline=False):
    value = {
        "schema_version": 1,
        "protocol": "varlen-memory-query-cuda-events-v1",
        "configuration": {
            "implementation": "native", "layout": "fixed", "mode": "train", "depth": 4
        },
        "measurements": [
            {
                "scope": "native_core", "median_ms": 2.0, "p10_ms": 1.5,
                "p90_ms": 2.5, "tokens_per_second": 50000,
                "peak_allocated_bytes": 2**20 * 64,
            },
            {
                "scope": "native_end_to_end", "median_ms": 3.0,
                "p10_ms": 2.0, "p90_ms": 4.0,
                "tokens_per_second": 32000,
                "peak_allocated_bytes": 2**20 * 72,
            },
        ],
    }
    if with_baseline:
        value["comparisons"] = [
            {
                "scope": "native_core", "baseline_median_ms": 4.0,
                "new_median_ms": 2.0, "baseline_over_new_ratio": 2.0,
            },
            {
                "scope": "native_end_to_end", "baseline_median_ms": 4.0,
                "new_median_ms": 3.0, "baseline_over_new_ratio": 4 / 3,
            },
        ]
    return value


def test_chart_data_includes_latency_uncertainty_throughput_memory():
    charts = chart_data(report())
    assert set(charts) == {"latency", "throughput", "memory"}
    assert charts["latency"]["labels"] == ["native_core", "native_end_to_end"]
    assert charts["latency"]["xerr"] == [[0.5, 1.0], [0.5, 1.0]]
    assert charts["throughput"]["values"] == [50000, 32000]
    assert charts["memory"]["values"] == [64, 72]


def test_comparison_charts_include_absolute_latency_and_speedup():
    charts = chart_data(report(with_baseline=True))
    assert charts["speedup"]["values"] == [2.0, 4 / 3]
    assert charts["latency_comparison"]["baseline"] == [4.0, 4.0]
    assert charts["latency_comparison"]["current"] == [2.0, 3.0]


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"protocol": "other"},
        {"measurements": []},
        {"measurements": [{"scope": "invalid", "median_ms": 0.0}]},
    ],
)
def test_invalid_reports_are_rejected(change):
    value = report()
    value.update(change)
    with pytest.raises((ValueError, KeyError)):
        chart_data(value)


def test_visualizer_emits_expected_png_files_when_matplotlib_is_installed(tmp_path):
    pytest.importorskip("matplotlib")
    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps(report(with_baseline=True)), encoding="utf-8")
    output = tmp_path / "charts"
    assert main([str(report_file), "--output", str(output)]) == 0
    assert {p.name for p in output.glob("*.png")} == {
        "latency.png", "throughput.png", "memory.png", "speedup.png",
        "latency_comparison.png",
    }
