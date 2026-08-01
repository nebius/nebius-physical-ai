"""License-guarded public-registry publishing.

Nebius CR has no anonymous/public mode, so public exposure means mirroring the
OSS-redistributable image subset to a public registry. These tests lock the
license boundary: the Omniverse-Kit images (isaac-lab, sonic, groot,
sonic-mujoco) must never be selected for a public registry, and the selector
must stay in sync with the packaging contract's redistribution classification.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    DEFAULT_PUBLIC_CONTAINER_REGISTRY,
    OMNIVERSE_RESTRICTED_DERIVED_IMAGES,
    OMNIVERSE_RESTRICTED_TOOLS,
    is_publicly_redistributable,
    omniverse_restricted_image_names,
    public_container_registry,
    publicly_publishable_tools,
)
from npa.deploy.publish_public import build_publish_plan

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "npa" / "docker" / "workbench" / "packaging-contract.yaml"


def test_restricted_tools_are_the_omniverse_images() -> None:
    assert OMNIVERSE_RESTRICTED_TOOLS == frozenset({"isaac-lab", "sonic", "groot"})


def test_public_set_excludes_every_restricted_tool() -> None:
    public = set(publicly_publishable_tools())
    assert public.isdisjoint(OMNIVERSE_RESTRICTED_TOOLS)
    for tool in OMNIVERSE_RESTRICTED_TOOLS:
        assert not is_publicly_redistributable(tool)


def test_public_set_includes_the_oss_tools() -> None:
    public = set(publicly_publishable_tools())
    for tool in ("lerobot", "genesis", "cosmos", "fiftyone", "lancedb", "rerun-viewer", "lichtblick"):
        assert tool in public, tool
    # Everything not Omniverse-restricted is public.
    assert public == set(CONTAINER_IMAGE_NAMES) - OMNIVERSE_RESTRICTED_TOOLS


def test_publish_plan_never_targets_a_restricted_image() -> None:
    plan = build_publish_plan(target_registry="ghcr.io/example/workbench")
    names = {item.source_ref.rsplit("/", 1)[-1].split(":", 1)[0] for item in plan}
    for restricted in ("npa-isaac-lab", "npa-sonic", "npa-groot", "npa-sonic-mujoco"):
        assert restricted not in names
    # Targets are all under the requested public registry.
    for item in plan:
        assert item.target_ref.startswith("ghcr.io/example/workbench/")


def test_publish_plan_requires_a_target() -> None:
    with pytest.raises(ValueError):
        build_publish_plan(target_registry="")


def test_public_registry_defaults_to_ghcr(monkeypatch) -> None:
    monkeypatch.delenv("NPA_PUBLIC_REGISTRY", raising=False)
    assert public_container_registry() == DEFAULT_PUBLIC_CONTAINER_REGISTRY
    assert DEFAULT_PUBLIC_CONTAINER_REGISTRY.startswith("ghcr.io/")


def test_public_registry_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "docker.io/nebius/workbench")
    assert public_container_registry() == "docker.io/nebius/workbench"


def test_publish_plan_targets_public_registry_by_default() -> None:
    plan = build_publish_plan(target_registry=DEFAULT_PUBLIC_CONTAINER_REGISTRY)
    assert len(plan) == 16
    for item in plan:
        assert item.target_ref.startswith(DEFAULT_PUBLIC_CONTAINER_REGISTRY + "/npa-")


def test_restricted_image_names_cover_every_contract_restricted_image() -> None:
    """The operator-facing excluded list must name every restricted image, derived
    variants included, without any caller hardcoding them."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_restricted = {
        name
        for name, entry in contract["images"].items()
        if entry.get("redistribution") == "restricted"
    }
    names = omniverse_restricted_image_names()
    assert names == sorted(names), "names must be stable/sorted for operator output"
    assert contract_restricted <= set(names), sorted(contract_restricted - set(names))
    # Derived variants are not canonical tools, so they never reach the public set.
    assert set(OMNIVERSE_RESTRICTED_DERIVED_IMAGES).isdisjoint(CONTAINER_IMAGE_NAMES)
    assert set(OMNIVERSE_RESTRICTED_DERIVED_IMAGES).isdisjoint(publicly_publishable_tools())


def test_selector_matches_packaging_contract_classification() -> None:
    """Every image the packaging contract marks ``restricted`` must resolve to a
    tool that the selector also treats as non-public (kept in sync)."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    # contract image keys that map onto canonical tool keys
    for image_name, entry in contract["images"].items():
        if entry.get("redistribution") != "restricted":
            continue
        # sonic-mujoco is a sonic variant (covered by the "sonic" restriction)
        tool = "sonic" if image_name == "sonic-mujoco" else image_name
        if tool in CONTAINER_IMAGE_NAMES:
            assert not is_publicly_redistributable(tool), image_name
        else:
            # non-canonical restricted image (e.g. sonic-mujoco) must map to a
            # restricted canonical tool
            assert tool in OMNIVERSE_RESTRICTED_TOOLS, image_name
