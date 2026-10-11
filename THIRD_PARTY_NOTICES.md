# Third-party dependencies

VarlenTransformer calls these separately installed projects. Their licenses
remain their own; this distribution does not bundle their source or binaries.

| Dependency | License | Upstream |
| --- | --- | --- |
| FlashAttention, by Tri Dao and contributors | BSD-3-Clause | [Repository and license](https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE) |
| PyTorch | BSD-3-Clause, with additional notices for bundled components | [Repository and license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Triton, optional CUDA fusion backend | MIT | [Repository and license](https://github.com/triton-lang/triton/blob/v3.5.1/LICENSE) |

## PyTorch-derived Welford reduction

The Welford reduction and logical reduction order in
`src/varlen_transformer/_triton_norm.py` follow the vectorized CUDA LayerNorm
implementation in PyTorch 2.9.1
([source](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu)).
The retained upstream copyright and BSD-3-Clause terms are in
`third_party/PyTorch-LICENSE`, copied from that release. This notice applies
to the source-derived reduction; the project's own contributions remain
under its MIT license. Distribution metadata records both licenses.

If distributing dependencies or copying upstream source into this package,
retain the relevant upstream copyright, license, and third-party notices.
The project's MIT license does not replace those notices.
