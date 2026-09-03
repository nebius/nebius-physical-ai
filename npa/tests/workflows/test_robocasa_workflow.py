"""Workflow tests for the RoboCasa native workbench."""

from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG, argv_for_tool

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "robocasa-smoke.yaml"


def test_workflow_validates() -> None:
    spec = load_spec(WORKFLOW)
    validate_spec(spec)
    assert spec.name == "robocasa-smoke"
    assert spec.initial == "task-registration"


def test_workflow_expands_all_states() -> None:
    spec = load_spec(WORKFLOW)
    plan = build_plan(spec, run_id="test")
    states = [step.state for step in plan.steps]
    assert states == [
        "task-registration",
        "asset-availability",
        "egl-env-reset",
        "random-rollout",
    ]


def test_workflow_dependency_order_is_topological() -> None:
    spec = load_spec(WORKFLOW)
    assert spec.states["task-registration"].needs == []
    assert spec.states["asset-availability"].needs == ["task-registration"]
    assert spec.states["egl-env-reset"].needs == ["asset-availability"]
    assert spec.states["random-rollout"].needs == ["egl-env-reset"]


def test_robocasa_toolrefs_render() -> None:
    for tool_ref in (
        "workbench.robocasa.task_registration",
        "workbench.robocasa.asset_availability",
        "workbench.robocasa.egl_env_reset",
        "workbench.robocasa.random_rollout",
    ):
        assert tool_ref in TOOL_CATALOG
        argv = argv_for_tool(tool_ref)
        assert argv, tool_ref
        assert argv[:4] == ["npa", "workbench", "robocasa", "run"]
        assert "--capability" in argv
        assert "--output-path" in argv
        assert "--service" in argv
        assert "--endpoint" in argv


def test_random_rollout_toolref_includes_iterations() -> None:
    argv = argv_for_tool("workbench.robocasa.random_rollout")
    assert "--iterations" in argv
    assert "--num-envs" in argv


DATA_POLICY = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "robocasa-data-policy.yaml"


def test_data_policy_workflow_validates() -> None:
    spec = load_spec(DATA_POLICY)
    validate_spec(spec)
    assert spec.name == "robocasa-data-policy"
    assert spec.initial == "trajectory-export"


def test_data_policy_workflow_expands_all_states() -> None:
    spec = load_spec(DATA_POLICY)
    plan = build_plan(spec, run_id="test")
    states = [step.state for step in plan.steps]
    assert states == [
        "trajectory-export",
        "lerobot-convert",
        "policy-train",
        "policy-eval",
        "insights",
    ]


def test_data_policy_uses_real_toolrefs() -> None:
    spec = load_spec(DATA_POLICY)
    tool_refs = {
        state.tool_ref
        for state in spec.states.values()
        if state.tool_ref
    }
    assert "workbench.robocasa.trajectory_export" in tool_refs
    assert "workbench.lerobot.policy_train" in tool_refs
    assert "workbench.robocasa.policy_eval" in tool_refs
    assert "workbench.insights.ingest_run" in tool_refs
    for ref in tool_refs:
        assert ref in TOOL_CATALOG


def test_data_policy_trajectory_export_toolref_renders() -> None:
    argv = argv_for_tool("workbench.robocasa.trajectory_export")
    assert argv[:4] == ["npa", "workbench", "robocasa", "run"]
    assert "--capability" in argv
    assert "kitchen_trajectory_export" in argv
    assert "--num-envs" in argv


def test_data_policy_uses_panda_omron_and_disjoint_robocasa_eval() -> None:
    spec = load_spec(DATA_POLICY)
    assert spec.config["dataset_robot"] == "panda_omron"
    train = set(spec.config["train_env_ids"].split(","))
    heldout = set(spec.config["heldout_env_ids"].split(","))
    assert len(train) >= 2
    assert len(heldout) >= 2
    assert train.isdisjoint(heldout)
    argv = argv_for_tool("workbench.robocasa.policy_eval")
    assert "kitchen_policy_eval" in argv
    assert "--checkpoint-uri" in argv
    assert "--train-env-ids" in argv
    assert "--heldout-env-ids" in argv
    assert "--env-id" not in argv


def test_data_policy_routes_raw_cpu_stages_to_native_images() -> None:
    spec = load_spec(DATA_POLICY)
    assert spec.states["lerobot-convert"].resources == "convert-cpu"
    assert spec.resources["convert-cpu"]["image"] == "tool://lerobot"
    assert spec.states["insights"].resources == "insights-cpu"
    assert spec.resources["insights-cpu"]["image"] == "tool://lancedb"
