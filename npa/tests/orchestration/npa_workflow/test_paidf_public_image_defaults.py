"""Cover repaired PAIDF defaults and precedence without relaxing old-image policy."""

from pathlib import Path

import pytest

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    container_image_for_tool,
    public_release_manifest,
)
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    plan_images,
    resolve_task_image,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
PAIDF_SPECS = (
    "workflows/testing/physical-ai-data-factory.yaml",
    "workflows/testing/nvidia-paidf-vda-cosmos-transfer25.yaml",
    "workflows/main/paidf-cosmos3.yaml",
)


@pytest.mark.parametrize("spec_path", PAIDF_SPECS)
@pytest.mark.parametrize("decision", ["promote_checkpoint", "loop_back"])
def test_paidf_plans_repaired_public_images(spec_path: str, decision: str) -> None:
    spec = load_spec(REPO_ROOT / spec_path)
    plan = build_plan(spec, run_id="repaired-images", assume_decision=decision)
    images = plan_images(
        spec, plan.steps, run_id="repaired-images", options=SkypilotRenderOptions()
    )
    repaired_digests = {
        CONTAINER_IMAGE_NAMES[tool]: entry["published_digest"]
        for tool, entry in public_release_manifest()[
            "workflow_validation_candidates"
        ].items()
    }
    selected = {}
    for image in images:
        name = image.rsplit("/", 1)[-1].split(":", 1)[0]
        if name in repaired_digests:
            assert image.startswith("ghcr.io/nebius/nebius-physical-ai/")
            assert ":dev-" in image
            assert image.endswith("@" + repaired_digests[name])
            selected[name] = image
    assert "npa-cosmos-evaluator" in selected
    if spec_path.endswith("paidf-cosmos3.yaml"):
        assert "npa-cosmos3" in selected
    elif decision == "promote_checkpoint":
        assert "npa-cosmos-curate" in selected


@pytest.mark.parametrize("tool", ["cosmos3", "cosmos-evaluator", "cosmos-curate"])
def test_old_paidf_release_defaults_stay_quarantined(tool: str) -> None:
    with pytest.raises(ValueError, match="no consumable public release"):
        container_image_for_tool(tool)
    with pytest.raises(ValueError, match="quarantined"):
        container_image_for_tool(tool, tag="old-release")


@pytest.mark.parametrize(
    "tool_ref, tool",
    [
        ("workbench.cosmos3.generate_variants", "cosmos3"),
        ("workbench.cosmos_evaluator.evaluate", "cosmos-evaluator"),
        ("workbench.cosmos_curate.curate", "cosmos-curate"),
    ],
)
def test_paidf_explicit_operator_registry_still_wins(tool_ref: str, tool: str) -> None:
    registry = "registry.example.invalid/operator/workbench"
    assert resolve_task_image(
        tool_ref, {}, options=SkypilotRenderOptions(registry=registry)
    ) == container_image_for_tool(tool, registry=registry)


@pytest.mark.parametrize("registry", ["quay.io/example/workbench", "docker.io/example"])
@pytest.mark.parametrize(
    "tool_ref",
    [
        "workbench.cosmos3.generate_variants",
        "workbench.cosmos_evaluator.evaluate",
        "workbench.cosmos_curate.curate",
    ],
)
def test_other_public_registries_cannot_select_official_candidates(
    registry: str, tool_ref: str
) -> None:
    with pytest.raises(NpaWorkflowError, match="no consumable public release"):
        resolve_task_image(
            tool_ref, {}, options=SkypilotRenderOptions(registry=registry)
        )


@pytest.mark.parametrize(
    "selector", ["*", "workbench.cosmos3", "workbench.cosmos3.generate_variants"]
)
def test_paidf_explicit_image_override_still_wins(selector: str) -> None:
    image = "registry.example.invalid/operator/custom@sha256:" + "a" * 64
    assert (
        resolve_task_image(
            "workbench.cosmos3.generate_variants",
            {},
            options=SkypilotRenderOptions(image_overrides={selector: image}),
        )
        == image
    )


def test_paidf_resource_image_and_preflight_digest_pin_still_win() -> None:
    image = "registry.example.invalid/operator/custom:checked"
    digest_image = image + "@sha256:" + "b" * 64
    assert (
        resolve_task_image(
            "workbench.cosmos3.generate_variants",
            {"image": image},
            options=SkypilotRenderOptions(image_digest_pins={image: digest_image}),
        )
        == digest_image
    )


def test_paidf_tool_uri_and_official_registry_use_same_repaired_default() -> None:
    options = SkypilotRenderOptions(registry="ghcr.io/nebius/nebius-physical-ai/")
    tool_ref = "workbench.cosmos3.generate_variants"
    expected = resolve_task_image(tool_ref, {}, options=options)
    assert (
        resolve_task_image(tool_ref, {"image": "tool://cosmos3"}, options=options)
        == expected
    )
