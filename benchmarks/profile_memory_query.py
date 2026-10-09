"""CUDA-event measurements for the recorded baseline and stream-major revision.

Run this same driver against independent source checkouts via --source-root.
All initialization, layout construction, and native entry packing precede timing.
No optimizer update is performed; every training iteration uses identical weights.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

BASELINE_COMMIT = "322bcc65126be61aece6a872ea5508ddd58f7d15"
PROTOCOL = "varlen-memory-query-cuda-events-v1"


def lengths(value):
    try:
        result = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "lengths must be comma-separated positive integers"
        ) from exc
    if not result or any(v <= 0 for v in result):
        raise argparse.ArgumentTypeError("lengths must be positive and nonempty")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument(
        "--implementation", choices=("legacy", "native", "wrapper"), default="native"
    )
    p.add_argument("--layout", choices=("fixed", "packed"), default="fixed")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--memory-length", type=int, default=128)
    p.add_argument("--query-length", type=int, default=4)
    p.add_argument("--memory-lengths", type=lengths)
    p.add_argument("--query-lengths", type=lengths)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--intermediate-size", type=int)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--mode", choices=("inference", "train"), default="inference")
    p.add_argument("--input-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    p.add_argument(
        "--parameter-dtype", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    p.add_argument("--with-residual", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--device", type=int, default=0, help="logical CUDA device index")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument(
        "--output",
        type=Path,
        help="report directory; defaults to source-root/benchmark-results",
    )
    p.add_argument(
        "--baseline-json",
        type=Path,
        help="compatible report collected on the same physical GPU",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="record an explicitly modified source tree",
    )
    p.add_argument(
        "--trace",
        type=Path,
        help="optional Chrome trace, collected separately after timing",
    )
    return p


def configuration(args):
    for name in (
        "batch_size",
        "memory_length",
        "query_length",
        "emb_dim",
        "num_heads",
        "depth",
        "warmup",
        "iterations",
        "runs",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    mid = (
        args.intermediate_size
        if args.intermediate_size is not None
        else 4 * args.emb_dim
    )
    if (
        mid <= 0
        or args.emb_dim % args.num_heads
        or args.emb_dim // args.num_heads > 256
    ):
        raise ValueError("invalid intermediate size or FlashAttention head geometry")
    if args.device < 0:
        raise ValueError("device must be nonnegative")
    if args.implementation == "wrapper" and args.layout != "fixed":
        raise ValueError(
            "the per-sample wrapper is fixed-only; use native for packed state"
        )
    if args.layout == "fixed" and (
        args.memory_lengths is not None or args.query_lengths is not None
    ):
        raise ValueError("per-sample length lists require --layout packed")
    ml = args.memory_lengths or [args.memory_length] * args.batch_size
    ql = args.query_lengths or [args.query_length] * args.batch_size
    if len(ml) != args.batch_size or len(ql) != args.batch_size:
        raise ValueError("length lists must match --batch-size")
    if any(v <= 0 for v in ml + ql):
        raise ValueError("memory and query lengths must be positive")
    xl = [m - q + 1 for m, q in zip(ml, ql)]
    if any(x <= 0 for x in xl) or max(sum(ml), sum(ql)) > 2**31 - 1:
        raise ValueError(
            "NCSE memory must be at least query length, with int32-compatible totals"
        )
    return dict(
        implementation=args.implementation,
        layout=args.layout,
        batch_size=args.batch_size,
        memory_lengths=ml,
        query_lengths=ql,
        x_prefix_lengths=xl,
        total_tokens=sum(ml) + args.batch_size + sum(ql),
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        intermediate_size=mid,
        depth=args.depth,
        mode=args.mode,
        input_dtype=args.input_dtype,
        parameter_dtype=args.parameter_dtype,
        with_residual=args.with_residual,
        deterministic=args.deterministic,
        seed=args.seed,
        dropout=0.0,
        internal_dtype="bf16",
        warmup=args.warmup,
        iterations=args.iterations,
        runs=args.runs,
    )


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    index = int(position)
    return ordered[index] + (
        ordered[min(index + 1, len(ordered) - 1)] - ordered[index]
    ) * (position - index)


def timing_statistics(runs, samples, tokens):
    values = [v for run in runs for v in run]
    if not values or any(v < 0 for v in values):
        raise ValueError("timing samples must be nonempty and nonnegative")
    median = statistics.median(values)
    if median <= 0:
        raise ValueError("zero event latency; increase the measured workload")
    return dict(
        median_ms=median,
        p10_ms=percentile(values, 0.1),
        p90_ms=percentile(values, 0.9),
        samples_per_second=samples * 1000 / median,
        tokens_per_second=tokens * 1000 / median,
        run_medians_ms=[statistics.median(run) for run in runs],
        latency_ms=runs,
    )


def measure(torch, call, cleanup, cfg, device):
    """Cleanup precedes each timed event pair; event completion precedes reading time."""
    runs, peaks, increments, persistent = [], [], [], []
    for _ in range(cfg["runs"]):
        for _ in range(cfg["warmup"]):
            cleanup()
            output = call()
            del output
        cleanup()
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        values = []
        for _ in range(cfg["iterations"]):
            cleanup()
            start.record()
            output = call()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
            del output
        peak = torch.cuda.max_memory_allocated(device)
        runs.append(values)
        peaks.append(peak)
        increments.append(max(0, peak - baseline))
        persistent.append(baseline)
    cleanup()
    return timing_statistics(runs, cfg["batch_size"], cfg["total_tokens"]) | dict(
        peak_allocated_bytes=max(peaks),
        peak_incremental_bytes=max(increments),
        persistent_allocated_bytes=max(persistent),
    )


def source_metadata(root):
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10
        )
        if result.returncode:
            raise ValueError(f"source-root must be a Git checkout: {root}")
        return result.stdout.strip()

    return dict(
        commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain"))
    )


def hardware_metadata(torch, flash, device):
    props = torch.cuda.get_device_properties(device)
    value = getattr(props, "uuid", None)
    uuid = str(value) if value is not None else None
    driver = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [
            [value.strip() for value in line.split(",")]
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        matching = [
            row
            for row in rows
            if uuid is not None
            and row[0].removeprefix("GPU-").lower() == uuid.removeprefix("GPU-").lower()
        ]
        if matching:
            driver = matching[0][1]
        elif uuid is None and len(rows) == 1:
            uuid, driver = rows[0]
        elif rows and len({row[1] for row in rows}) == 1:
            driver = rows[0][1]
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return dict(
        hardware=dict(
            gpu_name=props.name,
            gpu_uuid=uuid,
            device_index=device,
            compute_capability=list(torch.cuda.get_device_capability(device)),
            total_memory_bytes=props.total_memory,
            driver_version=driver,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        ),
        software=dict(
            pytorch=torch.__version__,
            cuda=torch.version.cuda,
            flash_attention=flash.__version__,
            python=platform.python_version(),
            platform=platform.platform(),
            cpu_threads=torch.get_num_threads(),
            tf32=torch.backends.cuda.matmul.allow_tf32,
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        ),
    )


def compare_reports(current, baseline):
    if (
        current["protocol"] != baseline["protocol"]
        or current["schema_version"] != baseline["schema_version"]
    ):
        raise ValueError("benchmark report protocols differ")
    for section, fields in (
        (
            "hardware",
            (
                "gpu_name",
                "gpu_uuid",
                "driver_version",
                "compute_capability",
                "total_memory_bytes",
            ),
        ),
        (
            "software",
            (
                "pytorch",
                "cuda",
                "flash_attention",
                "python",
                "tf32",
                "bf16_reduced_precision_reduction",
                "cpu_threads",
            ),
        ),
    ):
        for field in fields:
            if current["metadata"][section].get(field) != baseline["metadata"][
                section
            ].get(field):
                raise ValueError(f"comparison requires matching {section}.{field}")
    if not current["metadata"]["hardware"].get("gpu_uuid"):
        raise ValueError("cannot verify the same physical GPU: GPU UUID is unavailable")
    for field in ("driver_sha256", "sampled_workload_sha256"):
        if current["metadata"][field] != baseline["metadata"][field]:
            raise ValueError(f"comparison requires the same {field}")
    if (
        current["metadata"]["source"]["dirty"]
        or baseline["metadata"]["source"]["dirty"]
    ):
        raise ValueError("baseline comparison requires clean source checkouts")
    a, b = current["configuration"], baseline["configuration"]
    if {k: v for k, v in a.items() if k != "implementation"} != {
        k: v for k, v in b.items() if k != "implementation"
    }:
        raise ValueError("comparison workloads or measurement settings differ")
    if (
        b["implementation"] != "legacy"
        or baseline["metadata"]["source"]["commit"] != BASELINE_COMMIT
    ):
        raise ValueError(
            "baseline report must measure the recorded original commit with --implementation legacy"
        )
    if a["implementation"] == "legacy":
        raise ValueError("current report must measure native or wrapper execution")
    if min(a["runs"], b["runs"]) < 2:
        raise ValueError("speedup comparison requires repeated runs, not a single run")
    base_scope = "legacy_fixed" if a["layout"] == "fixed" else "legacy_packed"
    base = next(
        (row for row in baseline["measurements"] if row["scope"] == base_scope), None
    )
    if base is None:
        raise ValueError("baseline transformer measurement is missing")
    return [
        dict(
            scope=row["scope"],
            baseline_scope=base_scope,
            baseline_median_ms=base["median_ms"],
            new_median_ms=row["median_ms"],
            baseline_over_new_ratio=base["median_ms"] / row["median_ms"],
        )
        for row in current["measurements"]
        if not row["scope"].startswith("conversion_")
    ]


def write_reports(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "commit",
        "gpu_name",
        "gpu_uuid",
        "pytorch",
        "cuda",
        "driver_version",
        "flash_attention",
        "python",
        "implementation",
        "layout",
        "mode",
        "scope",
        "input_dtype",
        "parameter_dtype",
        "batch_size",
        "memory_lengths",
        "query_lengths",
        "x_prefix_lengths",
        "total_tokens",
        "emb_dim",
        "num_heads",
        "intermediate_size",
        "depth",
        "seed",
        "with_residual",
        "deterministic",
        "warmup",
        "iterations",
        "runs",
        "median_ms",
        "p10_ms",
        "p90_ms",
        "samples_per_second",
        "tokens_per_second",
        "peak_allocated_bytes",
        "peak_incremental_bytes",
        "persistent_allocated_bytes",
        "baseline_over_new_ratio",
    ]
    metadata, cfg = report["metadata"], report["configuration"]
    common = (
        metadata["hardware"]
        | metadata["software"]
        | cfg
        | dict(commit=metadata["source"]["commit"])
    )
    ratios = {
        row["scope"]: row["baseline_over_new_ratio"]
        for row in report.get("comparisons", [])
    }
    with (output / "report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["measurements"]:
            writer.writerow(
                common
                | row
                | dict(baseline_over_new_ratio=ratios.get(row["scope"], ""))
            )


def print_summary(report):
    cfg, hw = report["configuration"], report["metadata"]["hardware"]
    print(
        f"{hw['gpu_name']} | {cfg['implementation']} {cfg['layout']} {cfg['mode']} | depth={cfg['depth']}"
    )
    print(
        "scope                    median ms     p10 / p90 ms      samples/s      tokens/s    peak MiB"
    )
    for row in report["measurements"]:
        print(
            f"{row['scope']:24} {row['median_ms']:9.3f}   {row['p10_ms']:7.3f}/{row['p90_ms']:7.3f}"
            f" {row['samples_per_second']:13.1f} {row['tokens_per_second']:13.1f}"
            f" {row['peak_allocated_bytes'] / 2**20:10.1f}"
        )
    for row in report.get("comparisons", []):
        print(
            f"{row['scope']}: baseline/new median ratio = {row['baseline_over_new_ratio']:.3f}"
        )
    print(
        "Conversion probes measure forward conversion only; end-to-end training includes conversion backward."
    )


def build_workload(torch, vt, cfg, device):
    """Create equivalent weights and canonical inputs before any timed region."""
    legacy, fixed = cfg["implementation"] == "legacy", cfg["layout"] == "fixed"
    revised = hasattr(vt, "FixedMemoryQueryStack")
    if legacy == revised:
        raise ValueError(
            "--implementation does not match --source-root's legacy/native API"
        )
    dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    blocks = torch.nn.ModuleList(
        [
            vt.create_memory_query_block(
                cfg["emb_dim"], cfg["intermediate_size"], cfg["num_heads"]
            ).to(device=device, dtype=dtype[cfg["parameter_dtype"]])
            for _ in range(cfg["depth"])
        ]
    )
    training = cfg["mode"] == "train"
    blocks.train(training)
    for block in blocks:
        block.memory_mixer.attn.deterministic = cfg["deterministic"]
        block.cross_mixer.attn.deterministic = cfg["deterministic"]
    b, e = cfg["batch_size"], cfg["emb_dim"]
    sm, sq, xp = (
        cfg["memory_lengths"][0],
        cfg["query_lengths"][0],
        cfg["x_prefix_lengths"][0],
    )
    if fixed:
        layout = vt.FixedMemoryQueryLayout.ncse(
            xp, sq, **({} if legacy else dict(batch_size=b))
        )
        shape = (b, sm + 1 + sq, e)
    else:
        layout = vt.PackedMemoryQueryLayout.from_lengths(
            cfg["memory_lengths"],
            cfg["query_lengths"],
            cfg["x_prefix_lengths"],
            device=device,
        )
        shape = (cfg["total_tokens"], e)
    x = torch.randn(
        shape, device=device, dtype=dtype[cfg["input_dtype"]], requires_grad=training
    )
    r = torch.randn_like(x, requires_grad=training) if cfg["with_residual"] else None
    tracked = [x, r]

    # Sample every parameter and canonical input to detect incompatible setup.
    # These explicit device-to-host transfers occur during setup, never timing.
    def sample(tensor):
        return dict(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            values=tensor.detach().flatten()[:64].float().cpu().tolist(),
        )

    fingerprint = hashlib.sha256(
        json.dumps(
            dict(
                parameters={
                    name: sample(value) for name, value in blocks.named_parameters()
                },
                input=sample(x),
                residual=None if r is None else sample(r),
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    def native_chain(h, residual):
        for block in blocks:
            h, residual = block(h, residual, layout=layout)
        return h, residual

    def execute(forward):
        with torch.enable_grad() if training else torch.inference_mode():
            h, residual = forward()
            if training:
                (h + residual).float().square().mean().backward()
            return h, residual

    def cleanup():
        blocks.zero_grad(set_to_none=True)
        for value in tracked:
            if value is not None:
                value.grad = None

    cases, conversions = [], None
    if fixed and not legacy:
        adapter = vt.FixedMemoryQueryStack(blocks).train(training)
        if cfg["implementation"] == "native":
            flat_x = (
                vt.pack_fixed_memory_query(x, layout).detach().requires_grad_(training)
            )
            flat_r = (
                None
                if r is None
                else vt.pack_fixed_memory_query(r, layout)
                .detach()
                .requires_grad_(training)
            )
            tracked.extend((flat_x, flat_r))
            cases.append(
                ("native_core", lambda: execute(lambda: native_chain(flat_x, flat_r)))
            )
            cases.append(
                (
                    "native_end_to_end",
                    lambda: execute(lambda: adapter(x, r, layout=layout)),
                )
            )
        else:
            cases.append(
                (
                    "wrapper_end_to_end",
                    lambda: execute(lambda: adapter(x, r, layout=layout)),
                )
            )

        def conversion_cases():
            # Representative BF16 block outputs; setup is outside all event pairs.
            outputs = tuple(
                torch.zeros(
                    (cfg["total_tokens"], e), device=device, dtype=torch.bfloat16
                )
                for _ in range(2)
            )

            def pack():
                with torch.inference_mode():
                    return tuple(
                        vt.pack_fixed_memory_query(t, layout)
                        for t in (x, r)
                        if t is not None
                    )

            def unpack():
                with torch.inference_mode():
                    return tuple(
                        vt.unpack_fixed_memory_query(t, layout) for t in outputs
                    )

            return [("conversion_pack", pack), ("conversion_unpack", unpack)]

        conversions = conversion_cases
    else:
        scope = (
            "legacy_fixed"
            if fixed
            else ("legacy_packed" if legacy else "native_packed")
        )
        cases.append((scope, lambda: execute(lambda: native_chain(x, r))))
    return cases, conversions, cleanup, fingerprint


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = configuration(args)
        root = args.source_root.expanduser().resolve()
        source = source_metadata(root)
        if source["dirty"] and not args.allow_dirty:
            raise ValueError(
                "source tree is modified; commit it or explicitly use --allow-dirty"
            )
        if cfg["implementation"] == "legacy" and source["commit"] != BASELINE_COMMIT:
            raise ValueError("legacy measurement requires the recorded baseline commit")
        sys.path.insert(0, str(root / "src"))
        import torch

        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is required; use your preinstalled PyTorch/FlashAttention GPU environment"
            )
        import flash_attn
        import varlen_transformer as vt

        if not Path(vt.__file__).resolve().is_relative_to(root / "src"):
            raise ValueError("loaded package does not belong to --source-root")
        torch.cuda.set_device(args.device)
        if (
            torch.version.hip is not None
            or torch.cuda.get_device_capability(args.device)[0] < 8
            or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError(
                "the block requires an NVIDIA BF16-capable Ampere-or-newer GPU"
            )
        torch.backends.cuda.matmul.allow_tf32 = False
        cases, conversions, cleanup, fingerprint = build_workload(
            torch, vt, cfg, torch.device("cuda", args.device)
        )
        metadata = hardware_metadata(torch, flash_attn, args.device) | dict(
            source=source,
            performance_baseline_commit=BASELINE_COMMIT,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            sampled_workload_sha256=fingerprint,
        )
        report = dict(
            schema_version=1,
            protocol=PROTOCOL,
            metadata=metadata,
            configuration=cfg,
            methodology=dict(
                timer="CUDA events with per-iteration end-event synchronization",
                loss="mean(square(float(branch + residual)))",
                optimizer_step=False,
                dropout=0.0,
                setup_in_timing=False,
                memory="absolute peak and increment over post-warmup allocations; canonical/native inputs retained",
                conversion="forward-only entry packing and two-output unpacking; full backward is in end-to-end training",
            ),
            measurements=[],
        )
        baseline = None
        if args.baseline_json:
            baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
            # Reject incompatible comparisons before spending time on measurement.
            compare_reports(report, baseline)
        for scope, call in cases:
            report["measurements"].append(
                dict(scope=scope, **measure(torch, call, cleanup, cfg, args.device))
            )
            print(f"Measured {scope}", flush=True)
        if conversions is not None:
            for scope, call in conversions():
                report["measurements"].append(
                    dict(scope=scope, **measure(torch, call, cleanup, cfg, args.device))
                )
            report["conversion_forward_median_ms"] = sum(
                row["median_ms"]
                for row in report["measurements"]
                if row["scope"].startswith("conversion_")
            )
        if baseline is not None:
            report["comparisons"] = compare_reports(report, baseline)
        output = (
            args.output.expanduser().resolve()
            if args.output
            else root / "benchmark-results"
        )
        write_reports(report, output)
        print_summary(report)
        print(f"Reports: {output / 'report.json'} and {output / 'report.csv'}")
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            cleanup()
            torch.cuda.synchronize(args.device)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                with torch.profiler.record_function(cases[0][0]):
                    result = cases[0][1]()
                    torch.cuda.synchronize(args.device)
                    del result
            prof.export_chrome_trace(str(args.trace))
            print(f"Trace: {args.trace}")
    except (ValueError, ImportError, OSError, KeyError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
