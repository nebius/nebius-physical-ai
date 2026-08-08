"""GR00T N1.7 closed-loop PushT workflow contract coverage."""

from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.scheduler import build_scheduler_plan
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit


REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = (
    REPO_ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "groot-1-7-finetune.yaml"
)


def _option_value(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_groot_workflow_is_closed_loop_paired_pusht() -> None:
    spec = load_spec(SPEC_PATH)
    plan = build_plan(spec, run_id="groot-pusht-closed-loop")

    assert spec.name == "groot-1-7-finetune"
    assert [item.state for item in plan.steps] == [
        "resolve_task_contract",
        "evaluate_baseline_closed_loop",
        "evaluate_trained_closed_loop",
        "analyze_paired_outcomes",
        "render_task_rollouts",
        "emit_mcap",
        "emit_rrd",
        "publish",
    ]
    assert [item.tool_ref for item in plan.steps] == [
        "workflow.groot.resolve_task_contract",
        "workbench.groot.evaluate_baseline_closed_loop",
        "workbench.groot.evaluate_trained_closed_loop",
        "workflow.groot.analyze_paired_outcomes",
        "workflow.groot.render_task_rollouts",
        "workflow.groot.emit_task_mcap",
        "workflow.groot.emit_task_rrd",
        "workflow.groot.publish_task_performance",
    ]
    assert int(spec.config["paired_episodes"]) >= 20
    assert spec.config["horizon"] == "300"
    assert "groot17-pusht-final" in str(spec.config["final_seed_namespace"])
    for step in plan.steps[1:3]:
        assert step.resources_profile["accelerators"].endswith(":1")
        assert _option_value(step.argv, "--episodes") == "24"
        assert _option_value(step.argv, "--horizon") == "300"
        assert _option_value(step.argv, "--policy-batch-size") == "4"
        assert _option_value(step.argv, "--run-id") == "groot-pusht-closed-loop"
    assert "checkpoints/baseline/" in _option_value(plan.steps[1].argv, "--checkpoint-uri")
    assert "checkpoints/posttrain/" in _option_value(plan.steps[2].argv, "--checkpoint-uri")


def test_groot_workflow_reaches_plan_scheduler_and_vendor_render(monkeypatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source/npa")
    prepared = prepare_npa_workflow_for_submit(
        SPEC_PATH,
        run_id="groot-task-performance-render",
        render_options=SkypilotRenderOptions(
            registry="cr.example.invalid/workbench",
            materialize_registry_secrets=False,
        ),
    )
    try:
        scheduler = build_scheduler_plan(
            prepared.spec,
            prepared.plan.steps,
            run_id="groot-task-performance-render",
        )
        assert [task["name"] for task in scheduler["tasks"]] == [
            step.state for step in prepared.plan.steps
        ]
        assert scheduler["tasks"][1]["resources"]["accelerators"].endswith(":1")
        assert scheduler["tasks"][2]["resources"]["accelerators"].endswith(":1")

        documents = [
            doc
            for doc in yaml.safe_load_all(prepared.skypilot_yaml_path.read_text())
            if doc
        ]
        assert len(documents) == 9
        stages = documents[1:]
        assert "groot_task_performance resolve-task-contract" in stages[0]["run"]
        assert "groot_task_performance evaluate-closed-loop" in stages[1]["run"]
        assert "--phase baseline" in stages[1]["run"]
        assert "--phase trained" in stages[2]["run"]
        assert "gym-pusht==0.1.6" in stages[0]["setup"]
        assert "gym-pusht==0.1.6" in stages[1]["setup"]
        assert "pymunk==6.11.1" in stages[0]["setup"]
        assert "opencv-python-headless==4.10.0.84" in stages[0]["setup"]
        assert "transformers==4.57.3" in stages[1]["setup"]
        assert "groot_task_performance analyze-paired-outcomes" in stages[3]["run"]
        assert "groot_task_performance render-task-rollouts" in stages[4]["run"]
        assert "groot_task_performance emit-mcap" in stages[5]["run"]
        assert "groot_task_performance emit-rrd" in stages[6]["run"]
        assert "groot_task_performance publish" in stages[7]["run"]
        assert "[viz]" in stages[6]["setup"]
        assert stages[1]["config"]["kubernetes"]["pod_config"]["spec"][
            "securityContext"
        ] == {"runAsUser": 0, "runAsGroup": 0}
    finally:
        prepared.temp_dir.cleanup()
