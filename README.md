# VarlenTransformer

Pre-norm transformer blocks built on [FlashAttention 2](https://github.com/Dao-AILab/flash-attention), supporting both fixed-length batches and packed variable-length sequences.

The package provides:

- **`Block`** — a standard transformer block with causal or bidirectional self-attention and an MLP.
- **`CrossMHA`** — multi-head cross-attention between separate query and key/value streams.
- **`MemoryQueryBlock`** — causal memory self-attention followed by volume/density cross-attention and a shared MLP.

`MHA`, `MLP`, and the corresponding block factories are also available as public components.

## Installation

The attention backends require Python 3.10+, PyTorch 2.8+, FlashAttention 2.6–2.x, and an NVIDIA GPU with BF16 support (Ampere or newer). CPU execution is supported for the non-CUDA tests. This package contains no separately compiled CUDA extensions.

For an environment with CUDA-enabled PyTorch and FlashAttention already installed:

```bash
python -m pip install -e . --no-deps
```

For a new Linux environment, install a compatible CUDA-enabled PyTorch build from the [PyTorch installation guide](https://pytorch.org/get-started/locally/), followed by FlashAttention and the package:

```bash
python -m pip install packaging psutil ninja
python -m pip install 'flash-attn>=2.6,<3' --no-build-isolation
python -m pip install -e '.[test]'
```

## Self-attention transformer

`create_block` constructs a pre-norm transformer block with causal or bidirectional attention. Inputs are `(B, S, E)` for fixed-length batches or `(N, E)` for packed variable-length sequences.

```python
import torch
from varlen_transformer import create_block

block = create_block(
    emb_dim=128,
    intermediate_size=512,
    num_attention_heads=4,
    causal=True,
).cuda()

x = torch.randn(9, 128, device="cuda", requires_grad=True)
cu_seqlens = torch.tensor([0, 2, 5, 9], device="cuda", dtype=torch.int32)

h, residual = block(x, cu_seqlens=cu_seqlens, max_seqlen=4)
y = h + residual
```

For fixed-length batches, `cu_seqlens` and `max_seqlen` are omitted. For packed batches, `cu_seqlens` marks the boundaries between nonempty sequences. Offset tensors are used without reading their values on the CPU; int64 offsets are converted to contiguous int32.

A block returns `(mlp_branch, residual_stream)`, both BF16. Successive blocks pass both tensors forward, and `h + residual` reconstructs the output of the last block. Any final normalization belongs to the surrounding model.

## Cross-attention

`CrossMHA` uses separate projections for queries (`Wq`) and key/value inputs (`Wkv`), followed by an output projection. It supports noncausal and causal attention.

```python
import torch
from varlen_transformer import CrossMHA

cross = CrossMHA(emb_dim=128, num_heads=4, causal=False).cuda().bfloat16()
q = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
kv = torch.randn(2, 6, 128, device="cuda", dtype=torch.bfloat16)
out = cross(q, kv)  # (2, 4, 128)
```

For packed inputs, queries `(N_q, E)` and keys/values `(N_k, E)` use separate `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, and `max_seqlen_k` metadata. Both streams contain the same number of samples, but their lengths may differ.

For rectangular causal attention, FlashAttention uses bottom-right alignment. Query position `i` attends to key position `k` when `k <= i + S_k - S_q`. Fully masked rows have zero attention output before the learned output projection.

Standalone `MHA` and `CrossMHA` use ordinary PyTorch projection dtype rules, unlike the explicitly BF16-compute block implementations.

## Memory/query transformer

`MemoryQueryBlock` maintains three token regions:

- **Memory (`M`)** undergoes causal self-attention.
- **Volume query (`V`)** attends to the contextualized observation (`x`) prefix of memory, using noncausal cross-attention.
- **Density queries (`Q`)** attend to contextualized memory using rectangular causal cross-attention.

Volume and density queries share the same cross-attention projections. They do not attend to each other and cannot update memory. After the attention operations, a single LayerNorm and MLP are shared by all three token regions.

The layout enforces the autoregressive geometry

$$
S_M = S_x + S_Q - 1
$$

where `S_x` is the length of the observation prefix and `S_Q` is the number of density queries. Density query `j` can therefore attend to the observation prefix and only the preceding `j - 1` conditioning tokens.

### Native stream-major layout

Both fixed and packed `MemoryQueryBlock` calls use a flat `(N_total, E)` state with the token ordering

```text
[all memory tokens; all volume queries; all density queries]
```

The branch and residual streams share this layout across the entire block stack. Fixed-length attention uses views of the contiguous regions and the regular fixed-length FlashAttention kernels; variable-length attention uses packed kernels and sequence offsets.

For a fixed batch, the layout specifies the batch size explicitly:

```python
import torch
from varlen_transformer import FixedMemoryQueryLayout, create_memory_query_block

layout = FixedMemoryQueryLayout.ncse(
    x_prefix_length=3, query_length=4, batch_size=2
)
blocks = torch.nn.ModuleList([
    create_memory_query_block(128, 512, 4).cuda() for _ in range(2)
])

h = torch.randn(layout.total_tokens, 128, device="cuda", requires_grad=True)
residual = None
for block in blocks:
    h, residual = block(h, residual, layout=layout)
y = h + residual
```

`FixedMemoryQueryLayout(memory_length, query_length, x_prefix_length, batch_size=B)` provides the equivalent explicit constructor. Incompatible autoregressive lengths are rejected. `CrossMHA` remains available for general rectangular attention without this constraint.

### Fixed-batch stack adapter

`FixedMemoryQueryStack` accepts the per-sample `(B, S_M + 1 + S_Q, E)` layout, ordered `[M; V; Q]` within each sample. It converts both streams to stream-major form at stack entry and back at stack exit, rather than converting between every block.

```python
from varlen_transformer import FixedMemoryQueryStack

stack = FixedMemoryQueryStack(blocks)
x = torch.randn(2, 11, 128, device="cuda", requires_grad=True)
h, residual = stack(x, layout=layout)
y = h + residual  # (2, 11, 128)
```

The adapter preserves gradients through layout conversion. `return_stream_major=True` returns flat states. `pack_fixed_memory_query` and `unpack_fixed_memory_query` expose the conversions separately.

### Packed variable-length layout

`PackedMemoryQueryLayout` describes heterogeneous memory, observation-prefix, and density-query lengths with reusable metadata. The flat state uses the same stream-major ordering as the fixed native block.

```python
from varlen_transformer import PackedMemoryQueryLayout

layout = PackedMemoryQueryLayout.ncse(
    x_prefix_lengths=[3, 2],
    query_lengths=[4, 2],
    device="cuda",
)
h = torch.randn(layout.total_tokens, 128, device="cuda", requires_grad=True)
residual = None
for block in blocks:
    h, residual = block(h, residual, layout=layout)
y = h + residual
```

Here the two memory lengths are 6 and 3, giving 9 memory tokens, 2 volume queries, and 6 density queries.

`PackedMemoryQueryLayout.from_lengths(...)` constructs and verifies the per-sample autoregressive geometry from host-side sequence lengths. Direct construction with precomputed CPU metadata validates its values. Device-resident precomputed metadata uses `trust_metadata=True`; in that case the caller is responsible for correct offsets, prefix indices, and per-sample lengths. Layouts can be transferred with `.to(device)`.

Attention masks constrain which memory tokens each query can read. The caller-provided initial query embeddings must also be free of target or future conditioning values for the autoregressive dependency to hold.

### Precision and gradients

The block accepts floating-point input and parameter storage dtypes and computes its attention, linear operations, residual updates, dropout, and GELU in BF16. LayerNorm arithmetic and parameter-gradient accumulation use FP32. Outputs are BF16, and the custom backward supports first-order derivatives only.

Each block invokes FlashAttention three times: memory self-attention, volume cross-attention, and density cross-attention. The K/V projection is shared between the two cross-attention paths.

## Testing

The tests cover the standard block, cross-attention, mixed-precision custom operations, memory/query layouts, dependency isolation, packing, gradients, and real CUDA integration.

```bash
python -m pytest -m "not cuda"
python -m pytest -m cuda --require-cuda
python -m pytest --require-cuda
```

`--require-cuda` fails when the required GPU or FlashAttention environment is unavailable, rather than silently skipping CUDA tests. GitHub Actions runs CPU tests automatically; the GPU workflow is manually triggered on a compatible self-hosted runner.

## Local CUDA benchmarking and visualization

`benchmarks/profile_memory_query.py` benchmarks inference and forward/backward
training using CUDA events. It reports median and p10/p90 latency, throughput,
GPU memory allocation, and hardware/software metadata. Setup and initialization
are excluded from timings; training includes backward but not optimizer steps.

Examples from the repository root:

```bash
python benchmarks/profile_memory_query.py \
  --implementation native --layout fixed --depth 4 \
  --output benchmark-results/native-fixed

python benchmarks/profile_memory_query.py \
  --implementation wrapper --layout fixed --depth 4 --mode train \
  --output benchmark-results/wrapper-train

python benchmarks/profile_memory_query.py \
  --layout packed --batch-size 3 \
  --memory-lengths 7,31,127 --query-lengths 3,4,5 \
  --output benchmark-results/packed
```

Native fixed-layout measurements separate the flat core from adapter-inclusive
execution. Each output directory contains `report.json` and `report.csv`.

To create PNG charts from a report, install the optional plotting dependency
(`python -m pip install ".[viz]"`) and run:

```bash
python benchmarks/visualize_memory_query.py benchmark-results/native-fixed/report.json
```

The visualizer plots latency with p10/p90 ranges, throughput, and peak allocated
GPU memory. Reports with baseline comparisons also produce speedup and absolute
latency-comparison charts. Images are saved next to the report unless `--output`
specifies another directory.

For before/after measurements, the driver supports `--implementation legacy`
with `--source-root` pointing to a separate checkout at its recorded baseline
commit, and `--baseline-json` when measuring the revised implementation.
Comparisons require compatible hardware, software, and workload configurations.
Use `--help` for all options or `--trace PATH` for an optional Chrome trace.
The generated data and traces are not committed or used as CI test thresholds.

## AI use

AI assistance was used during the development of this package.

## License

[MIT](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party license information.
