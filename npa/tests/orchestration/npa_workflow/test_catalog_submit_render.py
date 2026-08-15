"""Production-path render coverage for every shipped npa.workflow spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_DIR = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SHIPPED_SPECS = sorted(WORKFLOW_DIR.glob("*.yaml"))
TEST_REGISTRY = "cr.ci.invalid/workbench"
TEST_BAKED_IMAGE = f"{TEST_REGISTRY}/npa-runtime@sha256:{'0' * 64}"


@pytest.mark.parametrize("spec_path", SHIPPED_SPECS, ids=lambda path: path.name)
def test_shipped_catalog_prepares_for_submit(
    spec_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch specs that validate structurally but fail the real submit renderer."""

    assert SHIPPED_SPECS, "expected shipped npa.workflow specs"
    monkeypatch.setenv("NPA_REGISTRY", TEST_REGISTRY)
    monkeypatch.setenv(
        "NPA_PUBLIC_REGISTRY", "ghcr.io/nebius/nebius-physical-ai"
    )
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://ci-fixtures/npa-source")
    spec = load_spec(spec_path)
    requires_baked_image = str(spec.config.get("require_baked_npa") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    prepared = prepare_npa_workflow_for_submit(
        spec_path,
        run_id=f"catalog-render-{spec_path.stem}",
        assume_decision="promote_checkpoint",
        render_options=SkypilotRenderOptions(
            registry=TEST_REGISTRY,
            # Specs that fail closed on image provenance require the same
            # immutable operator input in CI that production submit requires.
            image_overrides=(
                {"*": TEST_BAKED_IMAGE}
                if requires_baked_image
                else {}
            ),
            materialize_registry_secrets=False,
        ),
    )
    try:
        rendered = prepared.skypilot_yaml_path.read_text(encoding="utf-8")
        assert "execution: serial" in rendered
        assert prepared.plan.steps
    finally:
        prepared.temp_dir.cleanup()
