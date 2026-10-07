# Testing and GitHub Actions

Install a suitable PyTorch build first, then `python -m pip install -e '.[test]'`.

## CPU suite

```bash
python -m pytest -m 'not cuda' --cov=varlen_transformer --cov-report=term-missing
```

CPU tests run the actual pre-attention and post-attention custom operators,
including their manually registered backward rules. Tests that need attention
explicitly replace the external FlashAttention dependency with independent
FP32 PyTorch attention for that test only. They check surrounding block math,
routing, dropout, layouts, and gradient propagation. They do not validate the
CUDA attention kernels or GPU performance.

Coverage includes shape/dtype validation, parameter factories, BF16 casting,
dropout (training, evaluation, zero, and probability one), FP32 layer-norm
statistics, FP32 parameter-gradient accumulation, independent forward/backward
references for both custom ops, unused outputs, frozen parameters, noncontiguous
inputs, alias/mutation checks, rejection of double backward, `torch.library.opcheck`, dynamic AOT compilation,
fixed/packed equivalence, causal masking, sequence isolation, stacked residuals,
serialization, optimizer steps, and building/installing the wheel and sdist.

## NVIDIA GPU suite

Install FlashAttention as described in the README, then:

```bash
python -m pytest -m cuda --require-cuda
# Run all CPU and GPU tests:
python -m pytest --require-cuda
```

These tests use the real FlashAttention backend; no attention substitute is
installed. They compare outputs and every trainable parameter's gradient with
independent PyTorch math, exercise causal and bidirectional attention, FP16/BF16
MHA, FP32/FP16/BF16 block inputs, fixed and ragged layouts, single-head attention,
odd and maximum supported head dimensions, length-one sequences and
lengths around kernel tiling boundaries, dropout behavior, custom-op registration,
FP32 CUDA gradient accumulation, sequence isolation, and optimizer updates.

`--require-cuda` exits with an error if CUDA, a compatible NVIDIA BF16 GPU, or
FlashAttention is unavailable. Plain `pytest` skips GPU tests on machines without
those prerequisites. An installed but broken FlashAttention build is an error.

The deterministic references use no attention dropout for numeric parity tests;
dropout tests check masks, evaluation behavior, and finite gradients instead of
expecting the same random mask from different implementations. BF16 cast and
FP32-accumulation differences prevent bitwise gradient equality: the reference
checks enforce elementwise tolerances and a relative L2 error bound (normally
3%; up to 4% for full GPU blocks and stacked blocks). FP16 CUDA comparisons use
tighter forward tolerances. Tests are correctness checks, not benchmarks.

## GitHub CPU CI

`.github/workflows/tests.yml` runs on pushes, pull requests, and manual dispatch.
It uses standard `ubuntu-latest` runners, Python 3.10/3.12/3.13, and CPU PyTorch
2.8.0. FlashAttention is not installed. Coverage and JUnit reports are uploaded
as workflow artifacts, and distribution builds are checked.

## Optional GitHub GPU CI

`.github/workflows/gpu-tests.yml` runs only on manual dispatch. It expects a
dedicated Linux x64 self-hosted runner with labels `self-hosted`, `linux`, `x64`,
and `gpu`. A GPU in your personal computer is used by GitHub only after you
register that machine as a runner. Keeping GPU tests local is also fine.

1. Register an up-to-date runner (2.327.1+) in your repository's **Settings → Actions → Runners**,
   following [GitHub's runner instructions](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners).
2. Add the `gpu` label and choose Python 3.10+ with CUDA PyTorch 2.8+ and
   FlashAttention 2.6–2.x already installed. Ensure `python` resolves to that
   environment for the runner service.
3. Start the runner; select **Actions → GPU tests → Run workflow**.

The workflow preserves your preinstalled PyTorch/FlashAttention environment and
uses `--require-cuda`, so a missing backend cannot produce a green skipped run.
Use this dedicated runner only for trusted code. The workflow does not run
automatically on pull requests, and checks out the repository's default branch.
This avoids running an arbitrary dispatched branch or fork PR on your machine.

Eligible GitHub Team/Enterprise organizations can also provision a compatible
[GPU larger runner](https://docs.github.com/en/actions/concepts/runners/larger-runners)
and adapt the runner label and environment setup.
