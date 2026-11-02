"""Cover governed candidate disclosure without relaxing Cosmos3 quarantine."""

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


ROOT = Path(__file__).resolve().parents[3]
STOCK_SPEC = ROOT / "workflows/testing/cosmos3-generate.yaml"
PAIDF_SPEC = ROOT / "workflows/main/paidf-cosmos3.yaml"
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


def _paidf_plan():
    spec = load_spec(PAIDF_SPEC)
    return spec, build_plan(
        spec,
        run_id="candidate-disclosure",
        assume_decision="promote_checkpoint",
    )


def test_stock_generate_default_stays_quarantined() -> None:
    spec = load_spec(STOCK_SPEC)
    plan = build_plan(spec, run_id="stock-cosmos3")

    assert not workflow_validation_candidate_selections(
        spec, plan.steps, run_id="stock-cosmos3", options=SkypilotRenderOptions()
    )
    with pytest.raises(NpaWorkflowError, match="no consumable public release"):
        plan_images(
            spec, plan.steps, run_id="stock-cosmos3", options=SkypilotRenderOptions()
        )


def test_paidf_candidate_selection_reports_the_resolved_image() -> None:
    spec, plan = _paidf_plan()
    selections = {
        selection.tool_ref: selection.to_dict()
        for selection in workflow_validation_candidate_selections(
            spec,
            plan.steps,
            run_id="candidate-disclosure",
            options=SkypilotRenderOptions(),
        )
    }

    assert selections["workbench.cosmos3.generate_variants"] == {
        "tool_ref": "workbench.cosmos3.generate_variants",
        "image": _candidate_image(),
        "release_status": "workflow_validation_candidate",
        "selection_scope": "planned_steps",
    }
    assert "workbench.cosmos3.generate" not in selections


def test_candidate_selection_skips_an_unresolvable_step(mocker) -> None:
    """Candidate disclosure must not turn a normal resolution error into a gate."""

    spec, plan = _paidf_plan()
    mocker.patch(
        "npa.orchestration.npa_workflow.skypilot_render.resolve_task_image",
        side_effect=NpaWorkflowError("synthetic image resolution failure"),
    )

    assert not workflow_validation_candidate_selections(
        spec,
        plan.steps,
        run_id="candidate-disclosure",
        options=SkypilotRenderOptions(),
    )


def test_stock_generate_submit_plan_stays_quarantined() -> None:
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(STOCK_SPEC),
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

    assert result.exit_code == 1
    assert "no consumable public release" in result.output
    assert "status: PLANNED" not in result.output


def test_stock_generate_preflight_stays_quarantined_before_external_checks(
    mocker,
) -> None:
    pulls = mocker.patch(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials"
    )
    contracts = mocker.patch(
        "npa.cli.workbench.workflow._preflight_image_bootstrap_contracts"
    )

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "preflight-images",
            str(STOCK_SPEC),
            "--infra",
            "k8s/stock-test",
            "--var",
            "bucket=stock-test-bucket",
        ],
    )

    assert result.exit_code == 1
    assert "no consumable public release" in result.output
    pulls.assert_not_called()
    contracts.assert_not_called()


def test_paidf_plan_render_json_reports_candidate_status(monkeypatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://unit/npa")
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(PAIDF_SPEC),
            "--check-render",
            "--run-id",
            "candidate-disclosure-render",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert any(
        candidate["tool_ref"] == "workbench.cosmos3.generate_variants"
        and candidate["image"] == _candidate_image()
        and candidate["selection_scope"] == "planned_steps"
        for candidate in payload["workflow_validation_candidates"]
    )
    assert payload["workflow_validation_candidates_status"] == "available"


@pytest.mark.parametrize("tool_ref", QUARANTINED_COSMOS3_TOOL_REFS)
def test_non_candidate_cosmos3_capabilities_keep_the_quarantine(tool_ref: str) -> None:
    assert tool_ref in TOOL_CATALOG
    assert tool_ref not in COSMOS3_CANDIDATE_TOOL_REFS
    with pytest.raises(NpaWorkflowError, match="no consumable public release"):
        resolve_task_image(tool_ref, {}, options=SkypilotRenderOptions())


def test_non_candidate_cosmos3_catalog_ref_parametrization_is_not_empty() -> None:
    """Keep candidate scope tied to the catalog rather than a hand-curated list."""

    assert QUARANTINED_COSMOS3_TOOL_REFS
