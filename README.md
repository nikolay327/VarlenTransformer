# VarlenTransformer

Pre-norm transformer blocks with bidirectional or causal multi-head self-attention,
using [Dao-AILab FlashAttention](https://github.com/Dao-AILab/flash-attention).
Supports packed variable-length sequences and fixed-length batches.

## Install

Requires Python 3.10+, PyTorch 2.8+, FlashAttention 2.6–2.x, and a compatible
NVIDIA GPU with BF16 support (Ampere or newer). Install a CUDA-enabled PyTorch
build following [PyTorch's instructions](https://pytorch.org/get-started/locally/),
then run from this directory:

```bash
python -m pip install packaging psutil ninja
python -m pip install 'flash-attn>=2.6,<3' --no-build-isolation
python -m pip install -e '.[test]'
```

See FlashAttention's upstream requirements for a compatible CUDA toolkit and
driver. CPU-only installation supports the tests described below; attention
execution requires CUDA.

## Use

```python
import torch
from varlen_transformer import create_block

block = create_block(
    emb_dim=128, intermediate_size=512, num_attention_heads=4,
    causal=False,  # True for autoregressive attention
).cuda()

x = torch.randn(9, 128, device="cuda", requires_grad=True)
cu_seqlens = torch.tensor([0, 2, 5, 9], device="cuda", dtype=torch.int32)
h, residual = block(x, cu_seqlens=cu_seqlens, max_seqlen=4)
y = h + residual  # Combine after the last block; apply a final norm if needed.
y.float().square().mean().backward()
```

For fixed batches, pass `x` shaped `(batch, seqlen, emb_dim)` and omit both
sequence arguments. Packed input is `(total_tokens, emb_dim)`; offsets must
start at zero, end at `total_tokens`, increase between nonempty sequences,
and have differences no larger than `max_seqlen`. Offsets and inputs must be
on the same device. Int64 offsets are converted to contiguous int32.
Offset **values are trusted**, avoiding a CUDA-to-host synchronization.

`Block` returns `(mlp_branch, residual_stream)`, both BF16. To stack blocks,
pass both outputs to the next block: `h, residual = next_block(h, residual, ...)`.
Parameters may remain FP32; the block casts linear operations to BF16 and
computes layer normalization and parameter-gradient accumulation in FP32.
The custom backward supports first-order training and rejects double backward.
Standalone `MHA` follows
ordinary PyTorch linear-layer dtype rules; use FP16/BF16 weights and inputs,
or CUDA autocast.

## Test

```bash
python -m pytest -m 'not cuda'      # CPU checks; FlashAttention not required
python -m pytest -m cuda --require-cuda  # Real kernels on your NVIDIA GPU
python -m pytest --require-cuda    # Both suites
```

GitHub Actions runs CPU checks automatically. The included manual GPU workflow
uses a self-hosted runner; setup and coverage details are in [TESTING.md](TESTING.md).

## License

[MIT](LICENSE): anyone may use, modify, redistribute, and sell this package,
including in proprietary products, while retaining the copyright and license
notice. FlashAttention and PyTorch retain their [own licenses](THIRD_PARTY_NOTICES.md).
