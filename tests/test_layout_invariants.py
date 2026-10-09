from dataclasses import replace

import pytest
import torch

from varlen_transformer import FixedMemoryQueryLayout, PackedMemoryQueryLayout


@pytest.mark.parametrize("m,q,x", [(3, 6, 2), (4, 4, 3), (7, 3, 4)])
def test_fixed_rejects_non_autoregressive_geometry(m, q, x):
    with pytest.raises(ValueError, match="memory_length = x_prefix_length"):
        FixedMemoryQueryLayout(m, q, x)


@pytest.mark.parametrize("memory", [[6, 4, 5], [5, 3, 5], [6, 2, 6]])
def test_heterogeneous_builder_checks_each_sample(memory):
    with pytest.raises(ValueError, match="NCSE sample"):
        PackedMemoryQueryLayout.from_lengths(memory, [4, 2, 3], [3, 2, 3])


def raw(layout, **changes):
    names = (
        "num_memory_tokens",
        "num_query_tokens",
        "cu_seqlens_m",
        "cu_seqlens_q",
        "cu_seqlens_x",
        "max_seqlen_m",
        "max_seqlen_q",
        "max_seqlen_x",
        "x_prefix_indices",
    )
    return PackedMemoryQueryLayout(
        **({name: getattr(layout, name) for name in names} | changes)
    )


def test_builders_and_cpu_direct_construction_are_verified():
    layout = PackedMemoryQueryLayout.ncse([3, 2, 3], [4, 2, 3])
    direct = raw(layout)
    assert layout.metadata_verified and direct.metadata_verified
    assert direct.to("cpu").metadata_verified
    assert direct.x_prefix_indices.tolist() == [0, 1, 2, 6, 7, 9, 10, 11]


@pytest.mark.parametrize(
    "case", ["start", "end", "empty", "geometry", "prefix", "maximum"]
)
def test_direct_cpu_metadata_values_are_checked(case):
    layout = PackedMemoryQueryLayout.ncse([3, 2], [4, 2])
    changes = {}
    if case == "start":
        changes["cu_seqlens_q"] = torch.tensor([1, 4, 6])
    if case == "end":
        changes["cu_seqlens_m"] = torch.tensor([0, 6, 8])
    if case == "empty":
        changes["cu_seqlens_q"] = torch.tensor([0, 0, 6])
    if case == "geometry":
        changes["cu_seqlens_m"] = torch.tensor([0, 5, 9])
    if case == "prefix":
        changes["x_prefix_indices"] = torch.tensor([0, 1, 2, 5, 6])
    if case == "maximum":
        changes["max_seqlen_m"] = 5
    with pytest.raises(ValueError):
        raw(layout, **changes)


def test_device_metadata_requires_explicit_trust_without_host_reads(monkeypatch):
    verified = PackedMemoryQueryLayout.ncse([3, 2], [4, 2], device="meta")

    def forbidden(*args, **kwargs):
        raise AssertionError("read of device tensor contents")

    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    with pytest.raises(ValueError, match="explicit trust_metadata=True"):
        raw(verified)
    trusted = raw(verified, trust_metadata=True)
    assert verified.metadata_verified and not trusted.metadata_verified
    assert verified.to("meta").metadata_verified
    assert not trusted.to("meta").metadata_verified


def test_trust_opt_in_keeps_structural_and_global_geometry_checks():
    verified = PackedMemoryQueryLayout.ncse([3, 2], [4, 2])
    trusted = raw(verified, trust_metadata=True)
    assert not trusted.metadata_verified
    with pytest.raises(ValueError, match="totals"):
        replace(trusted, num_memory_tokens=10)
    with pytest.raises(ValueError, match="boolean"):
        raw(verified, trust_metadata="yes")
    with pytest.raises(ValueError, match="stream total"):
        replace(trusted, max_seqlen_x=6)
