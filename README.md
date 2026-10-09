# VarlenTransformer

Pre-norm transformer blocks backed by [FlashAttention 2](https://github.com/Dao-AILab/flash-attention), with fixed-length batches and packed variable-length sequences. The package contains three distinct building blocks:

- `Block`: a conventional transformer block with causal or bidirectional self-attention and an MLP.
- `CrossMHA`: standalone cross-attention between separate query and key/value streams.
- `MemoryQueryBlock`: a specialized architecture with causal memory self-attention, separate volume/density query routes, and a shared MLP.

The original `Block`, `MHA`, `MLP`, and `create_block` APIs remain available alongside the new components.

## Requirements and installation

Python 3.10+, PyTorch 2.8+, FlashAttention 2.6–2.x, and an NVIDIA GPU supporting BF16 (Ampere or newer) are required for real attention execution. CPU environments support the non-CUDA tests; the package does not include compiled extensions of its own.

A preconfigured CUDA/PyTorch/FlashAttention environment supports an editable installation without dependency replacement:

```text
python -m pip install "pytest>=8" "pytest-cov>=5" "build>=1.2" "hatchling>=1.27"
python -m pip install -e . --no-deps
```

For new environments, CUDA-enabled PyTorch is distributed through the [official PyTorch installation options](https://pytorch.org/get-started/locally/). The upstream FlashAttention package has its own CUDA and build requirements. Typical Linux installation commands are:

```bash
python -m pip install packaging psutil ninja
python -m pip install 'flash-attn>=2.6,<3' --no-build-isolation
python -m pip install -e '.[test]'
```

Editable installation uses Hatchling's normal build isolation. `--no-deps` leaves existing PyTorch and FlashAttention installations unchanged.

## Standard transformer block

`create_block` builds the original pre-norm transformer with causal or bidirectional multi-head self-attention:

```python
import torch
from varlen_transformer import create_block

block = create_block(
    emb_dim=128, intermediate_size=512, num_attention_heads=4,
    causal=False,
).cuda()

x = torch.randn(9, 128, device="cuda", requires_grad=True)
cu_seqlens = torch.tensor([0, 2, 5, 9], device="cuda", dtype=torch.int32)
h, residual = block(x, cu_seqlens=cu_seqlens, max_seqlen=4)
y = h + residual
y.float().square().mean().backward()
```

Fixed batches have shape `(batch, seqlen, emb_dim)` and no sequence-offset arguments. Packed batches have shape `(total_tokens, emb_dim)`, with `cu_seqlens` delimiting nonempty samples and `max_seqlen` bounding their lengths. Offset values are trusted without synchronizing CUDA tensors to the host; int64 offsets are converted to contiguous int32.

Each block returns `(mlp_branch, residual_stream)`, both BF16. A block stack propagates both outputs via `h, residual = next_block(h, residual, ...)`. A final `h + residual` reconstructs the updated stream; a separate final normalization remains model-dependent. `MHA` is also available independently.

## Standalone cross-attention

`CrossMHA` projects its query stream and key/value stream separately. It is independent of the memory/query block and supports both noncausal and causal attention:

```python
import torch
from varlen_transformer import CrossMHA

cross = CrossMHA(emb_dim=128, num_heads=4, causal=False).cuda().bfloat16()
q = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
kv = torch.randn(2, 6, 128, device="cuda", dtype=torch.bfloat16)
out = cross(q, kv)  # (2, 4, 128)
```

Packed cross-attention accepts `q=(Tq, E)` and `kv=(Tk, E)` with four independent sequence arguments: `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, and `max_seqlen_k`. Both streams contain the same number of nonempty samples but may have different lengths. Causal rectangular masks follow FlashAttention's bottom-right alignment: query row `i` attends to key row `k` when `k <= i + Sk - Sq`. A fully masked row has zero attention contribution before the output projection.

Standalone `MHA` and `CrossMHA` follow ordinary PyTorch projection dtype rules (typically FP16/BF16 parameters and inputs, or CUDA autocast).

## Memory/query transformer

`MemoryQueryBlock` operates on three token regions per sample: memory `M`, one externally supplied volume query `V`, and density queries `Q`.

- Memory receives **causal self-attention**.
- Volume attends **noncausally** to the `x`-prefix of contextualized memory.
- Density attends **causally** to the full contextualized memory.
- Volume and density share the same cross-attention projections. Neither query region attends to the other or feeds back into memory through attention.
- A single shared tokenwise MLP follows these attention paths.

`MemoryQueryBlock` is a specialized block rather than a replacement for general-purpose `CrossMHA`.

### Fixed-length memory/query layout

Each sample contains `[M; V; Q]` with shape `(batch, memory_length + 1 + query_length, emb_dim)`. The `ncse` convenience constructor sets `memory_length = x_prefix_length + query_length - 1`:

```python
import torch
from varlen_transformer import FixedMemoryQueryLayout, create_memory_query_block

layout = FixedMemoryQueryLayout.ncse(x_prefix_length=3, query_length=4)
blocks = torch.nn.ModuleList([
    create_memory_query_block(128, 512, 4).cuda() for _ in range(2)
])
h = torch.randn(2, 11, 128, device="cuda", requires_grad=True)
residual = None
for block in blocks:
    h, residual = block(h, residual, layout=layout)
y = h + residual
```

For other geometries, `FixedMemoryQueryLayout(memory_length, query_length, x_prefix_length)` specifies the independent lengths.

### Packed memory/query layout

Packed state has shape `(total_tokens, E)` and stores the concatenated regions as `[all M; all V; all Q]`, rather than interleaving samples. The layout tracks sample offsets and the selected memory prefixes. The same layout can be reused across a stack of blocks.

```python
from varlen_transformer import PackedMemoryQueryLayout

layout = PackedMemoryQueryLayout.ncse(
    x_prefix_lengths=[3, 2], query_lengths=[4, 2], device="cuda"
)
h = torch.randn(layout.total_tokens, 128, device="cuda", requires_grad=True)
residual = None
for block in blocks:
    h, residual = block(h, residual, layout=layout)
y = h + residual
```

In this example, the memory lengths are `[6, 3]`; the packed state contains 9 memory tokens, 2 volume tokens and 6 density tokens, totaling 17. General layouts are available through `PackedMemoryQueryLayout.from_lengths(memory_lengths, query_lengths, x_prefix_lengths, device=...)`. The direct packed-layout constructor also supports existing device-side metadata. Layout tensors and state tensors occupy the same device; `.to(device)` produces a device-local layout.

### Precision and limitations

Both memory/query block outputs are BF16. Internal linear operations, residual updates, dropout, attention and GELU use BF16; LayerNorm arithmetic and parameter-gradient accumulation use FP32. The custom autograd regions support first-order gradients, but not double backward.

The memory/query block makes three FlashAttention calls per layer (memory self-attention, volume cross-attention, density cross-attention). Empty memory, prefix and query sequences are not supported. A full-block compilation guarantee and performance benchmark are not implied by custom-operator compilation tests.

## Tests and continuous integration

The test suite covers the original transformer, standalone cross-attention, custom operators, memory/query layouts and blocks, package builds, and CUDA attention integration. CPU tests substitute reference attention where appropriate while preserving the actual custom operators. CUDA tests exercise FlashAttention itself.

```text
python -m pytest -m "not cuda"
python -m pytest -m cuda --require-cuda
python -m pytest --require-cuda
```

The `--require-cuda` option turns unavailable or incompatible CUDA/FlashAttention prerequisites into a test error rather than a skipped GPU suite. Numerical correctness tests are not performance benchmarks.

GitHub Actions contains automated CPU checks and an optional, manually triggered GPU workflow. The GPU workflow uses a self-hosted runner with the `gpu` label and an installed PyTorch/FlashAttention environment. Its operating-system label is defined in `.github/workflows/gpu-tests.yml`.

## AI use

AI assistance was used during the development of this package.

## License

[MIT](LICENSE). Third-party license information for PyTorch and FlashAttention is recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
