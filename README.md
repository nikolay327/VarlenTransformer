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

layout = FixedMemoryQueryLayout.autoregressive(
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

`FixedMemoryQueryLayout(memory_length, query_length, x_prefix_length, batch_size=B)` provides the equivalent explicit constructor. Incompatible autoregressive lengths are rejected. `CrossMHA` supports general rectangular attention independently of this layout.

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

layout = PackedMemoryQueryLayout.autoregressive(
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

## Input validation

`Block`, `MHA`, `CrossMHA`, `MemoryQueryBlock`, and `FixedMemoryQueryStack` expose `validate_inputs(...)` for checking tensor shapes, dtypes, residuals, and sequence metadata. Validation is performed explicitly during input preparation rather than inside `forward()`.

Layout constructors validate their geometry. `layout.validate(state, emb_dim)` checks flat stream-major storage; `FixedMemoryQueryLayout.validate_per_sample(state, emb_dim)` checks `(B, S_M + 1 + S_Q, E)` storage. The packing and unpacking utilities operate on prepared states and layouts.

## Optional CUDA fusion

`Block`, `MemoryQueryBlock`, `create_block`, and `create_memory_query_block` accept a `fusion` setting for optional Triton LayerNorm and residual-normalization kernels. The default is `"off"`.

```python
from varlen_transformer import create_block, create_memory_query_block

standard = create_block(512, 2048, 8, causal=True, fusion="residual_fwd_bwd").cuda()
memory_query = create_memory_query_block(512, 2048, 8, fusion="residual_fwd").cuda()

standard.fusion = "residual_fwd"
```

Fusion is configured independently for each block. The selected mode is captured for each forward/backward graph and is preserved across `train()` and `eval()`. Fusion settings are not included in `state_dict`.

| Mode | Forward | Backward |
| --- | --- | --- |
| `off` | ATen | ATen |
| `ln_fwd` | Fused Welford LayerNorm | ATen |
| `residual_fwd` | Fused residual + Welford LayerNorm | ATen |
| `ln_bwd` | ATen | Fused FP32 LayerNorm gradients |
| `ln_fwd_bwd` | Fused Welford LayerNorm | Fused FP32 LayerNorm gradients |
| `residual_fwd_bwd` | Fused residual + Welford LayerNorm | Fused FP32 LayerNorm gradients |

Forward fusion uses a vectorized Welford reduction and FP32 affine arithmetic. The Triton forward kernels support contiguous BF16 inputs with embedding widths divisible by four, from 32 to 4096. Unsupported shapes, layouts, and parameter alignments use the ATen implementation.

Backward fusion computes FP32 input and affine gradients using a tiled partial-gradient kernel followed by a bounded reduction. The backward kernel uses four warps and selects `R = min(256, 16384 // next_power_of_2(E))` rows per program, where `E` is the embedding width. The tile size is limited to 16,384 elements, and the final reduction supports at most 1,024 partial groups. Therefore, backward fusion requires `ceil(N / R) <= 1024`, where `N` is the number of flattened token rows.

For example, the maximum supported token counts are 262,144 for `E=64`, 131,072 for `E=128`, 65,536 for `E=256`, 32,768 for `E=512`, and 8,192 for `E=2048`. These are implementation-specific backward-fusion limits, not independent batch-size or sequence-length restrictions. Larger workloads fall back to the native ATen backward implementation, while supported forward fusion can still execute.

Residual sums are rounded to BF16 before normalization. MemoryQuery's partial-residual specialization applies to flat stream-major tensors; batched partial updates use the ATen implementation. Dropout, FlashAttention, linear projections, and the existing first-order autograd interfaces are unchanged.

### Compatibility

The fused forward Welford reduction was developed against **PyTorch 2.9.1+cu130**, with **Triton 3.5.1** on an **NVIDIA RTX 5070 Ti Laptop GPU**. The package requirement is `torch>=2.8`; numerical agreement of the optional fused kernels with other PyTorch and Triton releases is evaluated using the CUDA fusion test suite.

Triton is an optional dependency. The standard ATen implementation is used when fusion is disabled or a fusion specialization is unsupported.

## Testing

The test suite covers fixed and packed attention, standard and MemoryQuery blocks, mixed precision, dependency isolation, custom autograd, gradients, and fusion behavior.

```bash
python -m pytest -m "not cuda"
python -m pytest -m cuda --require-cuda
python -m pytest -m "cuda and fusion" --require-cuda --require-fusion
python -m pytest --require-cuda --require-fusion
```

The fusion CUDA tests compare Welford statistics, BF16 rounding, residual updates, backward gradients, dropout/RNG behavior, and repeated optimizer updates against the reference implementation. The `--require-cuda` and `--require-fusion` options enforce the corresponding hardware and backend requirements; `--require-fusion` also checks that fusion tests were collected.

The CPU suite runs automatically in GitHub Actions. GPU tests, including required fusion coverage, run through the manually triggered self-hosted GPU workflow on the default branch.

## Benchmarking

### Paired fusion comparisons

`benchmarks/profile_fusion.py` compares `fusion="off"` with a selected fusion mode using identical parameters and inputs. It supports standard causal and bidirectional self-attention and MemoryQuery, fixed and heterogeneous packed layouts, inference, and training.

Measurements use paired, randomized CUDA-event runs and record raw timings, per-run statistics, peak allocated GPU memory, fused-kernel dispatch counts, and hardware/software metadata. Training includes backward and excludes optimizer updates; dropout is disabled for timing.

```bash
python -m benchmarks.profile_fusion --architecture standard_causal --layout fixed --execution train --fusion residual_fwd_bwd --output benchmark-results/fusion-standard.json
python -m benchmarks.profile_fusion --architecture standard_bidirectional --layout packed --batch-size 3 --sequence-lengths 7,31,127 --fusion ln_bwd --output benchmark-results/fusion-standard-packed.json
python -m benchmarks.profile_fusion --architecture memory_query --layout fixed --execution train --fusion residual_fwd_bwd --output benchmark-results/fusion-memory-query.json
python -m benchmarks.profile_fusion --architecture memory_query --layout packed --batch-size 3 --memory-lengths 7,31,127 --query-lengths 3,4,5 --fusion ln_bwd --output benchmark-results/fusion-memory-query-packed.json
```

Model dimensions are configurable using `--emb-dim`, `--batch-size`, `--depth`, and `--num-heads`. The default fixed configuration uses embedding dimension 128, batch size 8, depth 4, and four attention heads. The benchmark verifies fused-kernel dispatch during an untimed run.

### MemoryQuery profiling

`benchmarks/profile_memory_query.py` measures MemoryQuery native execution, fixed-layout adapters, and layout conversions. The `--fusion` option selects a single mode per run (`off` by default). Outputs include `report.json` and `report.csv` with latency percentiles, throughput, peak GPU memory, and system metadata. The `--baseline-json` option is for the historical legacy-versus-stream-major comparison against commit `322bcc6`.

```bash
python -m benchmarks.profile_memory_query --implementation native --fusion off --layout fixed --mode train --output benchmark-results/mq-off
python -m benchmarks.profile_memory_query --implementation native --fusion residual_fwd_bwd --layout fixed --mode train --output benchmark-results/mq-fused
```

To generate plots, install the optional visualization dependencies and run:

```bash
python -m pip install ".[viz]"
python -m benchmarks.visualize_memory_query benchmark-results/mq-off/report.json
```

The visualizer produces latency, throughput, and GPU memory charts.

## AI use

AI assistance was used during the development of this package.

## License

[MIT](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party license information.
