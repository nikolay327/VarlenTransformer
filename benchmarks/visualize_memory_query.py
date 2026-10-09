"""Create PNG charts from a memory/query CUDA benchmark ``report.json``.

This optional utility uses Matplotlib; the transformer and benchmark driver do
not require it. The charts are written next to the report unless --output is set.
"""

import argparse
import json
from pathlib import Path


def chart_data(report):
    """Extract plotting inputs from the version-1 benchmark report schema."""
    if report.get("schema_version") != 1 or report.get("protocol") != "varlen-memory-query-cuda-events-v1":
        raise ValueError("expected a version-1 memory/query CUDA benchmark report")
    rows = report.get("measurements", [])
    if not rows:
        raise ValueError("the benchmark report contains no measurements")

    labels = [row["scope"] for row in rows]
    medians = [float(row["median_ms"]) for row in rows]
    if any(value <= 0 for value in medians):
        raise ValueError("all measured median latencies must be positive")
    lower = [max(0, m - float(row["p10_ms"])) for m, row in zip(medians, rows)]
    upper = [max(0, float(row["p90_ms"]) - m) for m, row in zip(medians, rows)]
    data = {
        "latency": dict(labels=labels, values=medians, xerr=[lower, upper]),
        "throughput": dict(
            labels=labels,
            values=[float(row["tokens_per_second"]) for row in rows],
        ),
        "memory": dict(
            labels=labels,
            values=[float(row["peak_allocated_bytes"]) / 2**20 for row in rows],
        ),
    }
    comparisons = report.get("comparisons", [])
    if comparisons:
        data["speedup"] = dict(
            labels=[row["scope"] for row in comparisons],
            values=[float(row["baseline_over_new_ratio"]) for row in comparisons],
        )
        data["latency_comparison"] = dict(
            labels=[row["scope"] for row in comparisons],
            baseline=[float(row["baseline_median_ms"]) for row in comparisons],
            current=[float(row["new_median_ms"]) for row in comparisons],
        )
    return data


def _bars(plt, labels, values, xlabel, title, output, *, xerr=None, reference=None):
    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(labels) + 1.5)))
    pos = list(range(len(labels)))
    ax.barh(pos, values, xerr=xerr, capsize=3 if xerr else 0)
    ax.set_yticks(pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    if reference is not None:
        ax.axvline(reference, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Created {output}")


def _compare_latency(plt, data, title, output):
    labels = data["labels"]
    pos = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.7 * len(labels) + 1.5)))
    ax.barh([p - 0.19 for p in pos], data["baseline"], height=0.36, label="Legacy baseline")
    ax.barh([p + 0.19 for p in pos], data["current"], height=0.36, label="New implementation")
    ax.set_yticks(pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Median latency (ms; lower is better)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Created {output}")


def plot_report(report, output):
    """Render charts with the optional dependency loaded only when needed."""
    data = chart_data(report)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Chart generation requires Matplotlib: python -m pip install '.[viz]'"
        ) from exc

    output.mkdir(parents=True, exist_ok=True)
    cfg = report.get("configuration", {})
    context = (
        f"{cfg.get('implementation', '?')} / {cfg.get('layout', '?')} / "
        f"{cfg.get('mode', '?')} / depth {cfg.get('depth', '?')}"
    )
    _bars(
        plt, **data["latency"], xlabel="Median latency (ms), with p10–p90",
        title=f"Memory/query latency — {context}", output=output / "latency.png",
    )
    _bars(
        plt, **data["throughput"], xlabel="Tokens per second",
        title=f"Memory/query throughput — {context}", output=output / "throughput.png",
    )
    _bars(
        plt, **data["memory"], xlabel="Peak allocated GPU memory (MiB)",
        title=f"Memory/query allocation — {context}", output=output / "memory.png",
    )
    if "speedup" in data:
        _bars(
            plt, **data["speedup"], xlabel="Legacy / new median latency (>1 is faster)",
            title=f"Memory/query speedup — {context}",
            output=output / "speedup.png", reference=1.0,
        )
        _compare_latency(
            plt, data["latency_comparison"], f"Legacy versus new — {context}",
            output / "latency_comparison.png",
        )
    print(
        "Conversion scopes are forward-only probes and should not be summed "
        "to reproduce end-to-end latency."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="path to benchmark report.json")
    parser.add_argument(
        "--output", type=Path, help="chart directory (default: next to report.json)"
    )
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        plot_report(report, args.output or args.report.parent)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
