"""Verify the FLUX step has a real command, image route, and declared artifacts."""

from pathlib import Path

from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import tool_image_key
from npa.orchestration.npa_workflow.spec import load_spec
from npa.deploy.images import UNVALIDATED_PUBLICATION_TOOLS, publicly_publishable_tools
from npa.sdk.workbench import flux_action
from npa.workbench.flux_action import runner

ROOT = Path(__file__).resolve().parents[3]


def test_workflow_reaches_native_cli_and_routes_gpu_image():
    spec = load_spec(ROOT / "workflows/testing/flux-action-finetune.yaml")
    plan = build_plan(spec, run_id="flux-contract")
    assert plan
    entry = TOOL_CATALOG["workbench.flux_action.finetune"]
    assert entry.argv_template[:4] == ["npa", "workbench", "flux-action", "finetune"]
    assert entry.multi_node_mode == "forbidden"
    assert tool_image_key("workbench.flux_action.finetune") == "flux-action"
    assert "flux-action" in UNVALIDATED_PUBLICATION_TOOLS
    assert "flux-action" not in publicly_publishable_tools()


def test_sdk_uses_same_request_contract(monkeypatch):
    seen = []

    def run(request, *, dry_run):
        seen.append(request)
        return {"status": "planned"}

    monkeypatch.setattr(runner, "finetune", run)
    assert (
        flux_action.finetune(
            input_path="s3://test-bucket/data",
            recipe_uri="s3://test-bucket/recipe.json",
            output_path="s3://test-bucket/run",
            processes=3,
            dry_run=True,
        )["status"]
        == "planned"
    )
    assert seen[0].processes == 3
