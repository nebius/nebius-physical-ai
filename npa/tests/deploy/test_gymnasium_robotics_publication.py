from __future__ import annotations

from pathlib import Path

from npa.deploy import images
from npa.deploy.publish_public import PublishItem, verify_validated_publication

ROOT = Path(__file__).resolve().parents[3]


def test_phase_a_candidate_is_eligible_but_not_registered_or_publishable() -> None:
    tool = "gymnasium-robotics"
    assert images.is_publicly_redistributable(tool)
    assert tool in images.PRE_REGISTRATION_PUBLICATION_QUARANTINE_TOOLS
    assert tool not in images.PUBLICATION_QUARANTINE_TOOLS
    assert tool not in images.CONTAINER_IMAGE_NAMES
    assert tool not in images.SUPPORTED_TOOL_VERSIONS
    assert tool not in images.publicly_publishable_tools()


def test_publication_guard_names_every_withheld_evidence_boundary() -> None:
    item = PublishItem(
        tool="gymnasium-robotics",
        source_ref="ghcr.io/example/npa-gymnasium-robotics:dev-" + "a" * 40,
        target_ref="ghcr.io/example/npa-gymnasium-robotics:phase-a-unbuilt",
    )
    ok, reason = verify_validated_publication(item)
    assert not ok
    for token in (
        "pre-registration",
        "corresponding-source",
        "accepted manifest",
        "supported tag",
        "architecture",
    ):
        assert token in reason


def test_no_accepted_manifest_or_withheld_catalog_record_exists() -> None:
    assert not (
        ROOT / "npa/src/npa/deploy/gymnasium_robotics_image_manifest.json"
    ).exists()
    assert not (
        ROOT / "npa/scripts/image_byte_scan/public_policies/gymnasium-robotics-v1.json"
    ).exists()
    assert (
        "gymnasium-robotics"
        not in (ROOT / "npa/docker/workbench/sm120-images.json")
        .read_text(encoding="utf-8")
        .lower()
    )
