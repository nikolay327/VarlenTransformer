"""Optional eager LayerNorm kernels with explicit BF16/FP32 boundaries.

The Welford reduction follows PyTorch 2.9.1's vectorized CUDA implementation.
See THIRD_PARTY_NOTICES.md and third_party/PyTorch-LICENSE for attribution.
"""

import torch
import triton
import triton.language as tl

NUM_WARPS = 8

@triton.jit
def _combine(mean_b, m2_b, count_b, mean_a, m2_a, count_a):
    # cuWelfordCombine(dataB, dataA), including its operand order.
    count = count_a + count_b
    coef = tl.div_rn(1.0, tl.where(count > 0, count, 1.0))
    na, nb = count_a * coef, count_b * coef
    delta = mean_b - mean_a
    # CUDA 13 contracts the A product in nA*meanA + nB*meanB.
    # The B product must round separately. Unequal-count merges expose
    # the difference; balanced power-of-two reductions can hide it.
    mean = tl.fma(na, mean_a, nb * mean_b)
    m2 = tl.fma((delta * delta) * count_a, nb, m2_a + m2_b)
    return tl.where(count > 0, mean, 0.0), tl.where(count > 0, m2, 0.0), count


@triton.jit
def _aten_welford(X, BRANCH, row, E: tl.constexpr, ADD: tl.constexpr,
                  START: tl.constexpr, BRANCH_ROWS: tl.constexpr):
    lane = tl.arange(0, 128)
    mean = tl.full((128,), 0.0, tl.float32)
    m2 = tl.full((128,), 0.0, tl.float32)
    count = tl.full((128,), 0.0, tl.float32)
    # ATen reads four consecutive columns per logical thread, in this order,
    # then advances by 128 vectors. Inactive logical threads have count zero.
    for base in range(tl.cdiv(E, 512)):
        for component in tl.static_range(4):
            col = base * 512 + lane * 4 + component
            valid = col < E
            value = tl.load(X + row * E + col, valid, other=0).to(tl.float32)
            if ADD:
                inside = (row >= START) & (row < START + BRANCH_ROWS)
                branch = tl.load(BRANCH + (row - START) * E + col,
                                 valid & inside, other=0).to(tl.float32)
                updated = (value + branch).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
                value = tl.where(inside, updated, value)
            delta = value - mean
            next_count = count + 1.0
            next_mean = tl.fma(delta, tl.div_rn(1.0, next_count), mean)
            next_m2 = tl.fma(delta, value - next_mean, m2)
            mean = tl.where(valid, next_mean, mean)
            m2 = tl.where(valid, next_m2, m2)
            count = tl.where(valid, next_count, count)
    for offset in tl.static_range(5):
        distance = 16 >> offset
        partner = tl.where(lane % 32 + distance < 32, lane + distance, lane)
        ma = tl.gather(mean, partner, 0)
        va = tl.gather(m2, partner, 0)
        ca = tl.gather(count, partner, 0)
        mean, m2, count = _combine(mean, m2, count, ma, va, ca)
    # ATen merges warp 0 with 2, warp 1 with 3, then warp 0 with 1.
    for distance in tl.static_range(2, 0, -1):
        offset = 2 if distance == 2 else 1
        partner = tl.minimum(lane + offset * 32, 127)
        ma = tl.gather(mean, partner, 0)
        va = tl.gather(m2, partner, 0)
        ca = tl.gather(count, partner, 0)
        nm, nv, nc = _combine(mean, m2, count, ma, va, ca)
        merge = (lane % 32 == 0) & (lane // 32 < offset)
        mean, m2, count = tl.where(merge, nm, mean), tl.where(merge, nv, m2), tl.where(merge, nc, count)
    first = tl.full((1,), 0, tl.int32)
    return tl.sum(tl.gather(mean, first, 0), 0), tl.div_rn(tl.sum(tl.gather(m2, first, 0), 0), E)


@triton.jit
def _forward(X, BRANCH, W, BIAS, Y, RES, MEAN, RSTD,
             E: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr,
             ADD: tl.constexpr, START: tl.constexpr, BRANCH_ROWS: tl.constexpr,
             NORM_ROWS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < E
    x = tl.load(X + row * E + col, valid, other=0).to(tl.float32)
    if ADD:
        inside = (row >= START) & (row < START + BRANCH_ROWS)
        branch = tl.load(BRANCH + (row - START) * E + col, valid & inside, other=0).to(tl.float32)
        # Round the residual update to BF16 BEFORE computing moments.
        updated = (x + branch).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        x = tl.where(inside, updated, x)
        tl.store(RES + row * E + col, x, valid)
    if row < NORM_ROWS:
        mean, variance = _aten_welford(X, BRANCH, row, E, ADD, START, BRANCH_ROWS)
        centered = tl.where(valid, x - mean, 0.0)
        rstd = tl.rsqrt(variance + EPS)
        weight = tl.load(W + col, valid, other=0).to(tl.float32)
        bias = tl.load(BIAS + col, valid, other=0).to(tl.float32)
        y = tl.fma(weight, rstd * centered, bias)
        tl.store(Y + row * E + col, y, valid)
        tl.store(MEAN + row, mean)
        tl.store(RSTD + row, rstd)


def _outputs(x, norm_rows):
    e = x.shape[-1]
    rows = x.numel() // e
    shape = x.shape if norm_rows == rows else (norm_rows, e)
    y = torch.empty(shape, device=x.device, dtype=torch.bfloat16)
    mean = torch.empty((*shape[:-1], 1), device=x.device, dtype=torch.float32)
    return y, mean, torch.empty_like(mean)


def forward(x, weight, bias, eps):
    rows, e = x.numel() // x.shape[-1], x.shape[-1]
    y, mean, rstd = _outputs(x, rows)
    _forward[(rows,)](x, x, weight, bias, y, y, mean, rstd,
                     e, triton.next_power_of_2(e), eps, False, 0, 0, rows,
                     num_warps=NUM_WARPS, enable_fp_fusion=False)
    return y, mean, rstd


def residual_forward(residual, branch, weight, bias, eps, start, norm_rows):
    e = residual.shape[-1]
    rows = residual.numel() // e
    out = torch.empty_like(residual)
    y, mean, rstd = _outputs(residual, norm_rows)
    _forward[(rows,)](residual, branch, weight, bias, y, out, mean, rstd,
                     e, triton.next_power_of_2(e), eps, True, start,
                     branch.numel() // e, norm_rows,
                     num_warps=NUM_WARPS, enable_fp_fusion=False)
    return out, y, mean, rstd


@triton.jit
def _backward_partials(GY, X, MEAN, RSTD, W, GX, PW, PB,
                       N: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr,
                       ROWS: tl.constexpr):
    group = tl.program_id(0)
    row = group * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK)
    valid = (row[:, None] < N) & (col[None, :] < E)
    x = tl.load(X + row[:, None] * E + col[None, :], valid, other=0).to(tl.float32)
    dy = tl.load(GY + row[:, None] * E + col[None, :], valid, other=0).to(tl.float32)
    mean = tl.load(MEAN + row, row < N, other=0)
    rstd = tl.load(RSTD + row, row < N, other=0)
    gamma = tl.load(W + col, col < E, other=0).to(tl.float32)
    xhat = tl.where(valid, (x - mean[:, None]) * rstd[:, None], 0.0)
    wdy = dy * gamma[None, :]
    s1 = tl.sum(wdy, axis=1) / E
    s2 = tl.sum(wdy * xhat, axis=1) / E
    dx = (wdy - s1[:, None] - xhat * s2[:, None]) * rstd[:, None]
    tl.store(GX + row[:, None] * E + col[None, :], dx, valid)
    tl.store(PW + group * E + col, tl.sum(dy * xhat, axis=0), col < E)
    tl.store(PB + group * E + col, tl.sum(dy, axis=0), col < E)


@triton.jit
def _backward_reduce(PW, PB, GW, GB, E: tl.constexpr, GROUPS: tl.constexpr,
                     BLOCK_GROUPS: tl.constexpr, COLS: tl.constexpr):
    group = tl.arange(0, BLOCK_GROUPS)
    col = tl.program_id(0) * COLS + tl.arange(0, COLS)
    valid = (group[:, None] < GROUPS) & (col[None, :] < E)
    dw = tl.load(PW + group[:, None] * E + col[None, :], valid, other=0)
    db = tl.load(PB + group[:, None] * E + col[None, :], valid, other=0)
    tl.store(GW + col, tl.sum(dw, axis=0), col < E)
    tl.store(GB + col, tl.sum(db, axis=0), col < E)


def backward(grad, x, mean, rstd, weight):
    n, e = x.numel() // x.shape[-1], x.shape[-1]
    row_tile = min(256,16384 // triton.next_power_of_2(e))
    groups = triton.cdiv(n, row_tile)
    # Avoid an unbounded second reduction and fall back for very large token counts.
    if groups > 1024:
        return None
    gx = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    gw = torch.empty((e,), device=x.device, dtype=torch.float32)
    gb = torch.empty_like(gw)
    partial_w = torch.empty((groups, e), device=x.device, dtype=torch.float32)
    partial_b = torch.empty_like(partial_w)
    _backward_partials[(groups,)](grad, x, mean, rstd, weight, gx, partial_w, partial_b,
                                 n, e, triton.next_power_of_2(e), row_tile,
                                 num_warps=NUM_WARPS, enable_fp_fusion=False)
    _backward_reduce[(triton.cdiv(e, 16),)](
        partial_w, partial_b, gw, gb, e, groups, triton.next_power_of_2(groups), 16,
        num_warps=NUM_WARPS, enable_fp_fusion=False)
    return gx, gw, gb
