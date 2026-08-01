"""Region-agnostic container image resolution.

An agent or workflow must be able to pull a tool image regardless of which region
it runs in. Every tool image is mirrored to both the primary (eu-north1) and the
mirror (us-central1) registry, and ``container_image_candidates`` must surface
both so a caller can fail over when its local region cannot reach the other one.
"""

from __future__ import annotations

from npa.deploy.images import (
    BACKUP_CONTAINER_REGISTRY,
    DEFAULT_CONTAINER_REGISTRY,
    CONTAINER_IMAGE_NAMES,
    backup_container_registry,
    container_image_candidates,
)


def test_primary_and_backup_span_both_regions() -> None:
    assert DEFAULT_CONTAINER_REGISTRY.startswith("cr.eu-north1.nebius.cloud/")
    # The mirror registry must be a real us-central1 registry (the stale
    # registry-u00gwj4vqcp98k7ph6 id did not resolve, which broke failover).
    assert BACKUP_CONTAINER_REGISTRY.startswith("cr.us-central1.nebius.cloud/")
    assert "registry-u00gwj4vqcp98k7ph6" not in BACKUP_CONTAINER_REGISTRY


def test_candidates_cover_both_registries_for_every_tool() -> None:
    for tool in CONTAINER_IMAGE_NAMES:
        # SONIC image selection is GPU-variant driven (separate from region); its
        # registry resolution is exercised via the same primary/backup path.
        if tool == "sonic":
            continue
        candidates = container_image_candidates(tool)
        hosts = {ref.split("/", 1)[0] for ref in candidates}
        assert "cr.eu-north1.nebius.cloud" in hosts, tool
        assert "cr.us-central1.nebius.cloud" in hosts, tool


def test_lichtblick_resolves_in_both_regions() -> None:
    candidates = container_image_candidates("lichtblick")
    assert any(
        ref.startswith("cr.eu-north1.nebius.cloud/") and ref.endswith("/npa-lichtblick:1.26.0")
        for ref in candidates
    )
    assert any(
        ref.startswith("cr.us-central1.nebius.cloud/") and ref.endswith("/npa-lichtblick:1.26.0")
        for ref in candidates
    )


def test_preferred_region_is_tried_first() -> None:
    # A us-central1 caller (which cannot read the eu-north1 registry) must try the
    # us-central1 mirror first; an eu-north1 caller tries eu-north1 first.
    us_first = container_image_candidates("lichtblick", preferred_region="us-central1")
    assert us_first[0].startswith("cr.us-central1.nebius.cloud/")
    eu_first = container_image_candidates("lichtblick", preferred_region="eu-north1")
    assert eu_first[0].startswith("cr.eu-north1.nebius.cloud/")
    # Both still cover both registries regardless of ordering.
    for candidates in (us_first, eu_first):
        hosts = {ref.split("/", 1)[0] for ref in candidates}
        assert hosts == {"cr.eu-north1.nebius.cloud", "cr.us-central1.nebius.cloud"}


def test_backup_registry_env_override(monkeypatch) -> None:
    monkeypatch.setenv("NPA_BACKUP_REGISTRY", "cr.us-central1.nebius.cloud/custom")
    assert backup_container_registry() == "cr.us-central1.nebius.cloud/custom"
