import importlib.util

import pytest
import torch
from packaging.version import Version

from tests.reference import ReferenceSelfAttention, attention, packed_attention


@pytest.fixture
def cross_reference_backend(monkeypatch):
    from tests.reference import ReferenceCrossAttention
    import varlen_transformer.cross_mha as cross

    monkeypatch.setattr(cross, "FlashCrossAttention", ReferenceCrossAttention)


@pytest.fixture
def memory_reference_backend(reference_backend, cross_reference_backend, monkeypatch):
    import varlen_transformer.memory_query as memory
    from tests.reference import cross_attention, packed_cross_attention

    calls = []

    def sa_fixed(qkv, **kw):
        calls.append(dict(kind="sa", packed=False, qkv=qkv, **kw))
        return attention(qkv, **kw)

    def sa_packed(qkv, cu, maximum, **kw):
        calls.append(
            dict(kind="sa", packed=True, qkv=qkv, cu=cu, maximum=maximum, **kw)
        )
        return packed_attention(qkv, cu, maximum, **kw)

    def ca_fixed(q, kv, **kw):
        calls.append(dict(kind="ca", packed=False, q=q, kv=kv, **kw))
        return cross_attention(q, kv, **kw)

    def ca_packed(q, kv, cuq, cuk, maxq, maxk, **kw):
        calls.append(
            dict(
                kind="ca",
                packed=True,
                q=q,
                kv=kv,
                cuq=cuq,
                cuk=cuk,
                maxq=maxq,
                maxk=maxk,
                **kw,
            )
        )
        return packed_cross_attention(q, kv, cuq, cuk, maxq, maxk, **kw)

    monkeypatch.setattr(memory, "flash_attn_qkvpacked_func", sa_fixed)
    monkeypatch.setattr(memory, "flash_attn_varlen_qkvpacked_func", sa_packed)
    monkeypatch.setattr(memory, "flash_attn_kvpacked_func", ca_fixed)
    monkeypatch.setattr(memory, "flash_attn_varlen_kvpacked_func", ca_packed)
    return calls


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="Fail instead of skipping when real CUDA/FlashAttention tests cannot run",
    )


def cuda_unavailable_reason():
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if Version(torch.__version__.split("+")[0]) < Version("2.8"):
        return "the CUDA suite requires PyTorch 2.8 or newer"
    if torch.version.hip is not None:
        return "this CUDA suite targets NVIDIA GPUs"
    if torch.cuda.get_device_capability()[0] < 8 or not torch.cuda.is_bf16_supported():
        return "the block requires an NVIDIA Ampere-or-newer GPU with BF16 support"
    if importlib.util.find_spec("flash_attn") is None:
        return "FlashAttention is not installed"
    try:
        import flash_attn
        from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention

        if not Version("2.6") <= Version(flash_attn.__version__) < Version("3"):
            return "the CUDA suite requires FlashAttention 2.6–2.x"

        assert callable(flash_attn.flash_attn_qkvpacked_func) and callable(
            FlashSelfAttention
        )
        assert callable(FlashCrossAttention) and callable(
            flash_attn.flash_attn_kvpacked_func
        )
        assert callable(flash_attn.flash_attn_varlen_kvpacked_func)
    except (ImportError, OSError, RuntimeError, AssertionError) as exc:
        raise pytest.UsageError(f"Installed FlashAttention cannot load: {exc}") from exc
    return None


def pytest_sessionstart(session):
    torch.set_num_threads(1)
    if session.config.getoption("--require-cuda"):
        reason = cuda_unavailable_reason()
        if reason:
            raise pytest.UsageError(
                f"--require-cuda: {reason}. Install a compatible CUDA PyTorch and flash-attn build."
            )


def pytest_collection_modifyitems(config, items):
    cuda_items = [item for item in items if item.get_closest_marker("cuda")]
    if cuda_items:
        reason = cuda_unavailable_reason()
        if reason:
            for item in cuda_items:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def reproducible_randomness():
    torch.manual_seed(1729)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False


@pytest.fixture
def reference_backend(monkeypatch):
    """Patch only dependency boundaries, explicitly and for one CPU test.

    Real package custom-op implementations, registrations and backward rules run
    unchanged. CUDA tests never request this fixture.
    """
    import varlen_transformer.block as block
    import varlen_transformer.mha as mha

    calls = []

    def fixed(qkv, **kwargs):
        calls.append(dict(kind="fixed", qkv=qkv, **kwargs))
        return attention(qkv, **kwargs)

    def packed(qkv, cu_seqlens, max_seqlen, **kwargs):
        calls.append(
            dict(
                kind="packed",
                qkv=qkv,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                **kwargs,
            )
        )
        return packed_attention(qkv, cu_seqlens, max_seqlen, **kwargs)

    monkeypatch.setattr(mha, "FlashSelfAttention", ReferenceSelfAttention)
    monkeypatch.setattr(block, "flash_attn_qkvpacked_func", fixed)
    monkeypatch.setattr(block, "flash_attn_varlen_qkvpacked_func", packed)
    return calls
