"""Build and install the real distribution, independently of the source imports."""

import email
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest
from packaging.requirements import Requirement

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    output = tmp_path_factory.mktemp("distributions")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output)],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return next(output.glob("*.whl")), next(output.glob("*.tar.gz"))


def test_wheel_contains_importable_modules_and_license_metadata(distributions):
    wheel, _ = distributions
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        for module in (
            "__init__",
            "block",
            "cls",
            "mha",
            "mlp",
            "_validation",
            "cross_mha",
            "layout",
            "memory_query",
            "_memory_ops",
        ):
            assert f"varlen_transformer/{module}.py" in names
        assert not any("/kernels/" in name or "__pycache__" in name for name in names)
        metadata = email.message_from_bytes(
            archive.read(
                next(name for name in names if name.endswith(".dist-info/METADATA"))
            )
        )
        assert (
            metadata["Name"] == "VarlenTransformer"
            and metadata["License-Expression"] == "MIT"
        )
        assert metadata["Requires-Python"] == ">=3.10"
        requirements = [Requirement(r) for r in metadata.get_all("Requires-Dist")]
        torch_requirement = next(r for r in requirements if r.name == "torch")
        assert (
            "2.8.0" in torch_requirement.specifier
            and "2.7.0" not in torch_requirement.specifier
        )
        flash_requirement = next(r for r in requirements if r.name == "flash-attn")
        assert flash_requirement.marker is not None
        assert flash_requirement.marker.evaluate({"extra": "gpu"})
        assert not flash_requirement.marker.evaluate({"extra": ""})
        assert (
            "2.6.0" in flash_requirement.specifier
            and "3.0.0" not in flash_requirement.specifier
        )
        assert any(name.endswith("/licenses/LICENSE") for name in names)
        assert any(name.endswith("/licenses/THIRD_PARTY_NOTICES.md") for name in names)


def test_sdist_includes_tests_docs_and_workflows(distributions):
    _, sdist = distributions
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        for suffix in (
            "/README.md",
            "/LICENSE",
            "/tests/test_cuda.py",
            "/tests/test_custom_ops.py",
            "/.github/workflows/tests.yml",
            "/.github/workflows/gpu-tests.yml",
        ):
            assert any(name.endswith(suffix) for name in names), suffix
        for obsolete in ("/TESTING.md", "/MEMORY_QUERY.md", "/REVIEW.md"):
            assert not any(name.endswith(obsolete) for name in names), obsolete
        assert not any("/kernels/" in name or "__pycache__" in name for name in names)


def test_installed_wheel_imports_without_optional_flash(distributions, tmp_path):
    wheel, _ = distributions
    install = tmp_path / "installed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(install),
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    script = """
import importlib.abc
from pathlib import Path
import sys
class HideFlash(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flash_attn":
            raise ModuleNotFoundError("FlashAttention deliberately hidden", name="flash_attn")
sys.meta_path.insert(0, HideFlash())
sys.path.insert(0, sys.argv[1])
import torch
import varlen_transformer as vt
assert Path(vt.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert set(vt.__all__) == {"Block", "MHA", "MLP", "create_block", "CrossMHA",
    "MemoryQueryBlock", "FixedMemoryQueryLayout", "PackedMemoryQueryLayout", "create_memory_query_block"}
assert vt.MLP(8, 16)(torch.ones(2, 8)).shape == (2, 8)
try:
    vt.create_block(16, 32, 2)
except ImportError as error:
    assert "no-build-isolation" in str(error)
else:
    raise AssertionError("Creating attention without FlashAttention must fail")
print("Installed wheel import and optional dependency behavior verified")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(install)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_require_cuda_exits_with_an_error_instead_of_skipping():
    script = """
import torch
torch.cuda.is_available = lambda: False
import pytest
raise SystemExit(pytest.main(["tests/test_cuda.py", "-m", "cuda", "--require-cuda", "-q"]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "--require-cuda: CUDA is unavailable" in result.stdout + result.stderr
