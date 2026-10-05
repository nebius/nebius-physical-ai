"""Verify PAIDF's exact default images against the anonymous public registry."""

import os
from pathlib import Path

import pytest

from npa.deploy.publish_public import anonymous_digest
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    plan_images,
)
from npa.orchestration.skypilot.registry_preflight import fetch_image_config_metadata

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1",
        reason="Set NPA_INTEGRATION_E2E=1 for anonymous registry verification.",
    ),
]
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "spec_path",
    [
        "workflows/testing/physical-ai-data-factory.yaml",
        "workflows/testing/nvidia-paidf-vda-cosmos-transfer25.yaml",
        "workflows/main/paidf-cosmos3.yaml",
    ],
)
def test_paidf_repaired_defaults_resolve_anonymously(spec_path: str) -> None:
    spec = load_spec(REPO_ROOT / spec_path)
    plan = build_plan(
        spec, run_id="public-image-proof", assume_decision="promote_checkpoint"
    )
    images = plan_images(
        spec, plan.steps, run_id="public-image-proof", options=SkypilotRenderOptions()
    )
    candidates = [image for image in images if ":dev-" in image]
    assert candidates
    for image in candidates:
        tag, recorded_digest = image.split("@", 1)
        for reference in (tag, image):
            ok, resolved_digest = anonymous_digest(reference)
            assert ok, f"anonymous manifest resolution failed for {reference}"
            assert resolved_digest == recorded_digest
        config_digest, labels = fetch_image_config_metadata(image)
        assert config_digest == recorded_digest
        assert labels["org.opencontainers.image.revision"] == tag.rsplit(":dev-", 1)[1]
        assert (
            labels["org.nebius.npa.skypilot-bootstrap-contract"] == "skypilot-0.12.2-v1"
        )
        assert "npa.base_image" not in {key.lower() for key in labels}
