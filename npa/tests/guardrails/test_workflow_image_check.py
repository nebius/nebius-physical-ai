from __future__ import annotations

from pathlib import Path

from npa.guardrails.skypilot import (
    image_refs_for_workflows,
    resolve_workflow_image,
    unresolved_image_placeholders,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_workflow_image_extraction_finds_skypilot_images() -> None:
    """The extractor's contract, pinned against a FROZEN task rather than the catalog.

    This used to assert that `npa-sonic` appeared among the shipped templates' images, which
    quietly made the extractor's test a reason the SONIC templates could not be retired. What is
    actually under test is that a raw SkyPilot task's image references are found at all, so it
    reads a fixture that will not move (tests/fixtures/skypilot/README.md).
    """

    fixtures = REPO_ROOT / "npa" / "tests" / "fixtures" / "skypilot"
    images = image_refs_for_workflows(sorted(fixtures.glob("*.yaml")))

    assert images
    assert any("npa-sonic" in image for image in images)


def test_retired_catalog_does_not_reappear_for_image_check() -> None:
    """Use fixtures for image extraction; do not revive the retired catalog."""

    workflow_dir = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"
    assert not workflow_dir.exists()


def test_registry_placeholder_resolution_is_local_check_ready() -> None:
    image = "<your-registry>/npa:tag"

    resolved = resolve_workflow_image(image, registry="registry.example/operator")

    assert resolved == "registry.example/operator/npa:tag"
    assert not unresolved_image_placeholders(resolved)


def test_image_check_classifies_operator_placeholders_as_seam() -> None:
    assert unresolved_image_placeholders(
        "<your-registry>/npa:<tag>"
    )
    assert unresolved_image_placeholders("${POLICY_IMAGE}")


def test_workflow_image_extraction_resolves_env_default(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        """
name: env-image
execution: serial
---
name: task
resources:
  image_id: docker:${NPA_WORKBENCH_IMAGE}
envs:
  NPA_WORKBENCH_IMAGE: registry.example/npa:tag
""",
        encoding="utf-8",
    )

    assert image_refs_for_workflows([workflow]) == ["registry.example/npa:tag"]
