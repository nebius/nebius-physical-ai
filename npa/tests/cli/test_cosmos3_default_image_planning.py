"""Exercise stock Cosmos3 image selection through real CLI planning paths."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.deploy.images import public_release_manifest
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    plan_images,
)
from npa.orchestration.skypilot.registry_preflight import KubernetesPullTarget


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "workflows/testing/cosmos3-generate.yaml"


def _candidate_image():
    candidate = public_release_manifest()["workflow_validation_candidates"]["cosmos3"]
    return (
        "ghcr.io/nebius/nebius-physical-ai/npa-cosmos3:dev-"
        + candidate["development_sha"]
        + "@"
        + candidate["published_digest"]
    )


def test_stock_generate_plans_the_exact_governed_image_without_overrides():
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="stock-cosmos3")
    images = plan_images(
        spec, plan.steps, run_id="stock-cosmos3", options=SkypilotRenderOptions()
    )
    assert images == [_candidate_image()]


def test_stock_generate_submit_plan_uses_image_resolution():
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


def test_stock_generate_preflight_probes_the_governed_image(mocker):
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


@pytest.mark.parametrize("action", ["checkpoint_eval", "policy_eval", "policy_train"])
def test_other_cosmos3_capabilities_keep_the_quarantine(action):
    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.skypilot_render import resolve_task_image

    with pytest.raises(NpaWorkflowError, match="no consumable public release"):
        resolve_task_image(
            "workbench.cosmos3." + action, {}, options=SkypilotRenderOptions()
        )
