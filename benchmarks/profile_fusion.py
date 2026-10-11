"""Paired, unprofiled CUDA-event comparison of fusion off vs one enabled mode.

Both measurements reuse the same model, weights, and input in one process.
Includes standard causal/bidirectional Block and native MemoryQueryBlock.
Dropout is disabled; optimizer.step is excluded from training timings.
CUDA correctness is established separately by pytest --require-fusion.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import statistics

from benchmarks import profile_memory_query as common

FUSIONS = ("ln_fwd", "residual_fwd", "ln_bwd", "ln_fwd_bwd", "residual_fwd_bwd")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=("standard_causal", "standard_bidirectional", "memory_query"), required=True)
    p.add_argument("--layout", choices=("fixed", "packed"), default="fixed")
    p.add_argument("--execution", choices=("train", "inference"), default="train")
    p.add_argument("--fusion", choices=FUSIONS, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sequence-length", type=int, default=128, help="standard fixed sequence length")
    p.add_argument("--memory-length", type=int, default=128, help="MemoryQuery fixed memory length")
    p.add_argument("--query-length", type=int, default=4, help="MemoryQuery fixed query length")
    p.add_argument("--sequence-lengths", type=common.lengths, help="standard packed per-sample lengths")
    p.add_argument("--memory-lengths", type=common.lengths, help="MemoryQuery packed per-sample lengths")
    p.add_argument("--query-lengths", type=common.lengths, help="MemoryQuery packed per-sample lengths")
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--output", type=Path, required=True)
    return p


def configuration(args):
    for field in ("batch_size", "emb_dim", "num_heads", "depth", "warmup", "iterations", "runs"):
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if args.device < 0 or args.emb_dim % args.num_heads or args.emb_dim // args.num_heads > 256:
        raise ValueError("invalid CUDA device or attention head geometry")
    if args.execution == "inference" and args.fusion == "ln_bwd":
        raise ValueError("backward-only fusion cannot be benchmarked in inference")
    if args.architecture.startswith("standard"):
        if args.memory_lengths or args.query_lengths:
            raise ValueError("memory/query lengths only apply to MemoryQuery")
        sl = args.sequence_lengths if args.layout == "packed" else [args.sequence_length] * args.batch_size
        if args.layout == "fixed" and args.sequence_lengths:
            raise ValueError("per-sample lengths require packed layout")
        if not sl or len(sl) != args.batch_size or min(sl) <= 0:
            raise ValueError("standard sequence lengths must match batch size")
        lengths = {"sequence_lengths": sl}
        tokens = sum(sl)
    else:
        if args.sequence_lengths:
            raise ValueError("standard sequence lengths do not apply to MemoryQuery")
        ml = args.memory_lengths if args.layout == "packed" else [args.memory_length] * args.batch_size
        ql = args.query_lengths if args.layout == "packed" else [args.query_length] * args.batch_size
        if args.layout == "fixed" and (args.memory_lengths or args.query_lengths):
            raise ValueError("per-sample lengths require packed layout")
        if not ml or not ql or len(ml) != args.batch_size or len(ql) != args.batch_size:
            raise ValueError("memory/query lengths must match batch size")
        if any(q < 1 or m < q for m, q in zip(ml, ql)):
            raise ValueError("each memory length must be at least its query length")
        lengths = {"memory_lengths": ml, "query_lengths": ql}
        tokens = sum(ml) + args.batch_size + sum(ql)
    return dict(architecture=args.architecture, layout=args.layout, execution=args.execution,
                fusion=args.fusion, batch_size=args.batch_size, emb_dim=args.emb_dim,
                num_heads=args.num_heads, depth=args.depth, warmup=args.warmup,
                iterations=args.iterations, runs=args.runs, seed=args.seed,
                total_tokens=tokens, **lengths)


def build_model(torch, vt, cfg, device):
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    B, E = cfg["batch_size"], cfg["emb_dim"]
    standard = cfg["architecture"].startswith("standard")
    if standard:
        blocks = torch.nn.ModuleList([
            vt.create_block(E, 4 * E, cfg["num_heads"],
                            causal=cfg["architecture"] == "standard_causal")
            for _ in range(cfg["depth"])
        ]).to(device)
        seq = cfg["sequence_lengths"]
        if cfg["layout"] == "fixed":
            shape, offsets, maximum = (B, seq[0], E), None, None
        else:
            shape = (sum(seq), E)
            offsets = torch.tensor([0] + [sum(seq[:i]) for i in range(1, B + 1)],
                                   dtype=torch.int32, device=device)
            maximum = max(seq)
        def chain():
            h, r = x, None
            for block in blocks:
                h, r = block(h, r, cu_seqlens=offsets, max_seqlen=maximum)
            return h + r
    else:
        blocks = torch.nn.ModuleList([
            vt.create_memory_query_block(E, 4 * E, cfg["num_heads"])
            for _ in range(cfg["depth"])
        ]).to(device)
        ml, ql = cfg["memory_lengths"], cfg["query_lengths"]
        if cfg["layout"] == "fixed":
            layout = vt.FixedMemoryQueryLayout.autoregressive(
                ml[0] - ql[0] + 1, ql[0], batch_size=B)
        else:
            layout = vt.PackedMemoryQueryLayout.from_lengths(
                ml, ql, [m-q+1 for m,q in zip(ml, ql)], device=device)
        shape = (cfg["total_tokens"], E)
        def chain():
            h, r = x, None
            for block in blocks:
                h, r = block(h, r, layout=layout)
            return h + r
    training = cfg["execution"] == "train"
    blocks.train(training)
    x = torch.randn(shape, dtype=torch.float32, device=device, requires_grad=training)

    def set_mode(mode):
        for block in blocks:
            block.fusion = mode

    def cleanup():
        blocks.zero_grad(set_to_none=True)
        x.grad = None

    def call():
        with torch.enable_grad() if training else torch.inference_mode():
            y = chain()
            if training:
                y.float().square().mean().backward()
            return y

    return set_mode, call, cleanup, blocks, x


@contextmanager
def count_fused_kernels(backend):
    """Verify actual fused dispatch outside timed regions, including backward."""
    originals = {}
    counts = Counter()
    for name in ("forward", "residual_forward", "backward"):
        originals[name] = getattr(backend, name)
        def counted(*args, _name=name, _orig=originals[name], **kwargs):
            value = _orig(*args, **kwargs)
            if value is not None:
                counts[_name] += 1
            return value
        setattr(backend, name, counted)
    try:
        yield counts
    finally:
        for name, original in originals.items():
            setattr(backend, name, original)


def verify_dispatch(torch, backend, mode, training, set_mode, call, cleanup, device):
    set_mode(mode)
    with count_fused_kernels(backend) as counts:
        cleanup()
        output = call()
        torch.cuda.synchronize(device)
        del output
    cleanup()
    expect_forward = mode in ("ln_fwd", "ln_fwd_bwd", "residual_fwd", "residual_fwd_bwd")
    expect_backward = training and mode in ("ln_bwd", "ln_fwd_bwd", "residual_fwd_bwd")
    if expect_forward and counts["forward"] + counts["residual_forward"] == 0:
        raise RuntimeError(f"fusion={mode} never dispatched a forward Triton kernel; unsupported shape or fallback")
    if mode.startswith("residual_") and counts["residual_forward"] == 0:
        raise RuntimeError("residual fusion mode did not dispatch residual kernel")
    if expect_backward and counts["backward"] == 0:
        raise RuntimeError(f"fusion={mode} never dispatched a backward Triton kernel")
    return dict(counts)


def timed_round(torch, set_mode, call, cleanup, mode, cfg, device):
    set_mode(mode)
    for _ in range(cfg["warmup"]):
        cleanup()
        output = call()
        del output
    cleanup()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    values = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(cfg["iterations"]):
        cleanup()
        start.record()
        output = call()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
        del output
    peak = torch.cuda.max_memory_allocated(device)
    cleanup()
    return values, dict(peak_allocated_bytes=peak,
                        peak_incremental_bytes=max(0, peak-allocated))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = configuration(args)
        import torch
        import flash_attn
        import varlen_transformer as vt
        from varlen_transformer import _fusion
        if not torch.cuda.is_available():
            raise ValueError("this benchmark requires CUDA")
        if _fusion._load() is None:
            raise ValueError("Triton fusion backend is unavailable")
        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        set_mode, call, cleanup, blocks, x = build_model(torch, vt, cfg, device)
        counts = verify_dispatch(torch, _fusion._load(), cfg["fusion"], cfg["execution"] == "train",
                                 set_mode, call, cleanup, device)
        rng = random.Random(cfg["seed"])
        data = {"off": [], cfg["fusion"]: []}
        memory = {"off": [], cfg["fusion"]: []}
        order = []
        for i in range(cfg["runs"]):
            modes = ["off", cfg["fusion"]]
            rng.shuffle(modes)
            order.append(modes[:])
            for mode in modes:
                values, stats = timed_round(torch, set_mode, call, cleanup, mode, cfg, device)
                data[mode].append(values)
                memory[mode].append(stats)
                print(f"Run {i+1}/{cfg['runs']}: {mode} {statistics.median(values):.3f} ms", flush=True)
        stats = {m: common.timing_statistics(v, cfg["batch_size"], cfg["total_tokens"])
                 | dict(peak_allocated_bytes=max(k["peak_allocated_bytes"] for k in memory[m]),
                        peak_incremental_bytes=max(k["peak_incremental_bytes"] for k in memory[m]))
                 for m, v in data.items()}
        ratios = [statistics.median(b)/statistics.median(f)
                  for b, f in zip(data["off"], data[cfg["fusion"]])]
        import triton
        root = Path(__file__).resolve().parents[1]
        report = dict(protocol="varlen-fusion-paired-cuda-events-v1", configuration=cfg,
                      metadata=common.hardware_metadata(torch, flash_attn, args.device)
                      | dict(source=common.source_metadata(root),
                             source_sha256=hashlib.sha256((root / "src/varlen_transformer/_triton_norm.py").read_bytes()).hexdigest(),
                             triton=triton.__version__),
                      fused_dispatch_smoke=counts,
                      measurement=dict(timer="CUDA events", optimizer_step=False,
                                       dropout=0.0, candidate_order=order,
                                       comparisons="identical model, parameters and input; randomized paired rounds",
                                       correctness="separate pytest CUDA fusion suite required"),
                      modes=stats, paired_baseline_over_fused_ratios=ratios,
                      median_paired_baseline_over_fused_ratio=statistics.median(ratios))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(f"Off {stats['off']['median_ms']:.3f} ms | {cfg['fusion']} {stats[cfg['fusion']]['median_ms']:.3f} ms")
        print(f"Median paired baseline/fused ratio {statistics.median(ratios):.4f}")
        print(f"Report: {args.output}")
        return 0
    except (ValueError, ImportError, OSError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
