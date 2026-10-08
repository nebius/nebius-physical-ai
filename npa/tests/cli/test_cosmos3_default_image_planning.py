"""Exercise stock Cosmos3 image selection through real CLI planning paths."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.deploy.images import public_release_manifest
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    plan_images,
    resolve_task_image,
    tool_image_key,
    workflow_validation_candidate_selections,
)
from npa.orchestration.skypilot.registry_preflight import (
    ImagePullCheck,
    KubernetesPullTarget,
)


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "workflows/testing/cosmos3-generate.yaml"
COSMOS3_CANDIDATE_TOOL_REFS = frozenset(
    public_release_manifest()["workflow_validation_candidates"]["cosmos3"][
        "validation_tool_refs"
    ]
)
QUARANTINED_COSMOS3_TOOL_REFS = tuple(
    tool_ref
    for tool_ref in sorted(TOOL_CATALOG)
    if tool_image_key(tool_ref) == "cosmos3"
    and tool_ref not in COSMOS3_CANDIDATE_TOOL_REFS
)


def _candidate_image() -> str:
    candidate = public_release_manifest()["workflow_validation_candidates"]["cosmos3"]
    return (
        "ghcr.io/nebius/nebius-physical-ai/npa-cosmos3:dev-"
        + candidate["development_sha"]
        + "@"
        + candidate["published_digest"]
    )


def test_stock_generate_plans_the_exact_governed_image_without_overrides() -> None:
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="stock-cosmos3")
    images = plan_images(
        spec, plan.steps, run_id="stock-cosmos3", options=SkypilotRenderOptions()
    )
    assert images == [_candidate_image()]
    assert [
        selection.to_dict()
        for selection in workflow_validation_candidate_selections(
            spec, plan.steps, run_id="stock-cosmos3", options=SkypilotRenderOptions()
        )
    ] == [
        {
            "tool_ref": "workbench.cosmos3.generate",
            "image": _candidate_image(),
            "release_status": "workflow_validation_candidate",
            "selection_scope": "planned_steps",
        }
    ]
    assert not workflow_validation_candidate_selections(
        spec,
        plan.steps,
        run_id="stock-cosmos3",
        options=SkypilotRenderOptions(
            image_overrides={
                "*": "registry.example.invalid/operator/custom@sha256:" + "a" * 64
            }
        ),
    )


def test_candidate_selection_skips_an_unresolvable_step(mocker) -> None:
    """Candidate disclosure must not turn a normal resolution error into a gate."""

    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="stock-cosmos3")
    mocker.patch(
        "npa.orchestration.npa_workflow.skypilot_render.resolve_task_image",
        side_effect=NpaWorkflowError("synthetic image resolution failure"),
    )

    assert not workflow_validation_candidate_selections(
        spec, plan.steps, run_id="stock-cosmos3", options=SkypilotRenderOptions()
    )


def test_stock_generate_submit_plan_uses_image_resolution() -> None:
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SPEC),
            "--plan-only",
            "--no-deploy-if-absent",
            "--infra",
            "k8s/stock-test",
            "--run-id",
            "stock-cosmos3",
            "--var",
            "bucket=stock-test-bucket",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "status: PLANNED" in result.output
    assert "no consumable public release" not in result.output
    assert "workflow validation candidate, not an accepted release" in result.output


def test_stock_generate_submit_plan_json_reports_candidate_status() -> None:
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(SPEC),
            "--plan-only",
            "--no-deploy-if-absent",
            "--infra",
            "k8s/stock-test",
            "--run-id",
            "stock-cosmos3-json",
            "--var",
            "bucket=stock-test-bucket",
            "--output-format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["workflow_validation_candidates"] == [
        {
            "tool_ref": "workbench.cosmos3.generate",
            "image": _candidate_image(),
            "release_status": "workflow_validation_candidate",
            "selection_scope": "planned_steps",
        }
    ]
    assert payload["workflow_validation_candidates_status"] == "available"


def test_stock_generate_preflight_probes_the_governed_image(mocker) -> None:
    checks = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        return_value=[],
    )
    bootstrap = mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts",
        return_value=[],
    )
    mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.resolve_kubernetes_pull_target",
        return_value=KubernetesPullTarget(namespace="stock-test"),
    )
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(SPEC),
            "--infra",
            "k8s/stock-test",
            "--var",
            "bucket=stock-test-bucket",
        ],
    )
    assert result.exit_code == 0, result.output
    assert set(checks.call_args.args[0]) == {_candidate_image()}
    assert bootstrap.called
    assert "workflow validation candidate, not an accepted release" in result.output


def test_stock_generate_preflight_json_reports_candidate_status(mocker) -> None:
    checks = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        return_value=[
            ImagePullCheck(
                image=_candidate_image(),
                status="ok",
                digest="sha256:" + "a" * 64,
            )
        ],
    )
    mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts",
        return_value=[],
    )
    mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.resolve_kubernetes_pull_target",
        return_value=KubernetesPullTarget(namespace="stock-test"),
    )

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(SPEC),
            "--infra",
            "k8s/stock-test",
            "--var",
            "bucket=stock-test-bucket",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert checks.call_args.args[0] == [_candidate_image()]
    assert json.loads(result.stdout)[0]["release_status"] == (
        "workflow_validation_candidate"
    )
    assert json.loads(result.stdout)[0]["selection_scope"] == "reachable_branches"


def test_stock_generate_plan_render_json_reports_candidate_status() -> None:
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(SPEC),
            "--check-render",
            "--run-id",
            "stock-cosmos3-render",
            "--var",
            "bucket=stock-test-bucket",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["workflow_validation_candidates"] == [
        {
            "tool_ref": "workbench.cosmos3.generate",
            "image": _candidate_image(),
            "release_status": "workflow_validation_candidate",
            "selection_scope": "planned_steps",
        }
    ]
    assert payload["workflow_validation_candidates_status"] == "available"


@pytest.mark.parametrize("tool_ref", QUARANTINED_COSMOS3_TOOL_REFS)
def test_other_cosmos3_capabilities_keep_the_quarantine(tool_ref: str) -> None:
    assert tool_ref in TOOL_CATALOG
    assert tool_ref not in COSMOS3_CANDIDATE_TOOL_REFS
    with pytest.raises(NpaWorkflowError, match="no consumable public release"):
        resolve_task_image(tool_ref, {}, options=SkypilotRenderOptions())


def test_every_non_candidate_cosmos3_catalog_ref_is_quarantined() -> None:
    """Keep candidate scope tied to the catalog rather than a hand-curated list."""

    assert QUARANTINED_COSMOS3_TOOL_REFS
