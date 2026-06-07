from __future__ import annotations

from pathlib import Path

from npa.guardrails.skypilot import (
    image_refs_for_workflows,
    resolve_workflow_image,
    unresolved_image_placeholders,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_workflow_image_extraction_finds_skypilot_images() -> None:
    workflow_dir = REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot"
    images = image_refs_for_workflows(sorted(workflow_dir.glob("*.yaml")))

    assert images
    assert any("npa-sonic" in image for image in images)


def test_registry_placeholder_resolution_is_local_check_ready() -> None:
    image = "cr.eu-north1.nebius.cloud/<your-registry-id>/npa:tag"

    resolved = resolve_workflow_image(image, registry_id="registry-test")

    assert resolved == "cr.eu-north1.nebius.cloud/registry-test/npa:tag"
    assert not unresolved_image_placeholders(resolved)


def test_image_check_classifies_operator_placeholders_as_seam() -> None:
    assert unresolved_image_placeholders("cr.eu-north1.nebius.cloud/<your-registry-id>/npa:<tag>")
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
