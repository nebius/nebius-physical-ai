"""Restricted legacy SONIC variants stay quarantined from all resolvers."""

from __future__ import annotations

import pytest

from npa.deploy.images import (
    container_image_for_tool,
    sonic_image_entry,
    sonic_image_manifest,
)


@pytest.mark.parametrize("target", ["h100", "h200", "l40s", "gpu-l40s-a"])
def test_restricted_legacy_gpu_variants_cannot_resolve(target: str) -> None:
    with pytest.raises(ValueError, match="quarantined"):
        sonic_image_entry(gpu_target=target)
    with pytest.raises(ValueError, match="quarantined"):
        container_image_for_tool("sonic", gpu_target=target)


def test_only_runtime_fetch_variant_is_default_and_active() -> None:
    manifest = sonic_image_manifest()
    assert manifest["default_variant"] == "sonic-k8s-host-mounted"
    active = sonic_image_entry()
    assert active["id"] == "sonic-k8s-host-mounted"
    assert active["status"] == "active"
    assert active["redistribution"] == "public-runtime-fetch"


def test_quarantine_records_why_a_runtime_flag_cannot_fix_baked_bytes() -> None:
    entries = {item["id"]: item for item in sonic_image_manifest()["images"]}
    for variant in ("sonic-l40s-baked", "sonic-mujoco-h100-mvp"):
        entry = entries[variant]
        assert entry["status"] == "quarantined"
        assert entry["redistribution"] == "restricted"
        assert (
            "bake" in entry["quarantine_reason"].lower()
            or "inherit" in entry["quarantine_reason"].lower()
        )
