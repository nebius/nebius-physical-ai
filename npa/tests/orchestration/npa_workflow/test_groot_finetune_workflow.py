"""GR00T N1.7 retrain/select/final closed-loop PushT workflow coverage."""

from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.scheduler import build_scheduler_plan
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit


REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = REPO_ROOT / "npa/workflows/workbench/npa-workflows/groot-1-7-finetune.yaml"


def _option_value(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_groot_workflow_retrains_selects_and_tests_closed_loop() -> None:
    spec = load_spec(SPEC_PATH)
    plan = build_plan(spec, run_id="groot-pusht-closed-loop")

    assert [item.state for item in plan.steps] == [
        "resolve_task_contract",
        "prepare_retraining_split",
        "retrain_task_policy",
        "resolve_trained_checkpoint",
        "evaluate_validation_baseline",
        "evaluate_validation_candidate",
        "analyze_validation_outcomes",
        "select_checkpoint",
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
        "workflow.groot.prepare_split",
        "workbench.groot.finetune",
        "workflow.groot.resolve_trained_checkpoint",
        "workbench.groot.evaluate_baseline_closed_loop.validation",
        "workbench.groot.evaluate_trained_closed_loop.validation",
        "workflow.groot.analyze_validation_outcomes",
        "workflow.groot.select_checkpoint",
        "workbench.groot.evaluate_baseline_closed_loop",
        "workbench.groot.evaluate_trained_closed_loop.selected",
        "workflow.groot.analyze_paired_outcomes",
        "workflow.groot.render_task_rollouts",
        "workflow.groot.emit_task_mcap",
        "workflow.groot.emit_task_rrd",
        "workflow.groot.publish_task_performance",
    ]
    assert int(spec.config["max_steps"]) == 6000
    assert int(spec.config["train_episodes"]) == 180
    assert int(spec.config["heldout_episodes"]) == 26
    assert int(spec.config["validation_episodes"]) >= 20
    assert int(spec.config["paired_episodes"]) >= 20
    assert spec.config["validation_seed_namespace"] != spec.config["final_seed_namespace"]
    assert _option_value(plan.steps[1].argv, "--max-steps") == "6000"
    assert _option_value(plan.steps[2].argv, "--max-steps") == "6000"
    assert plan.steps[2].resources_profile["accelerators"].endswith(":7")
    for step in (plan.steps[4], plan.steps[5], plan.steps[8], plan.steps[9]):
        assert step.resources_profile["accelerators"].endswith(":1")
        assert _option_value(step.argv, "--episodes") == "24"
        assert _option_value(step.argv, "--horizon") == "300"
    assert "checkpoints/baseline/" in _option_value(plan.steps[8].argv, "--checkpoint-uri")
    assert "selected-checkpoint.json" in _option_value(
        plan.steps[9].argv, "--checkpoint-ref-uri"
    )


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
            prepared.spec, prepared.plan.steps, run_id="groot-task-performance-render"
        )
        assert [task["name"] for task in scheduler["tasks"]] == [
            step.state for step in prepared.plan.steps
        ]
        assert scheduler["tasks"][2]["resources"]["accelerators"].endswith(":7")

        documents = [doc for doc in yaml.safe_load_all(prepared.skypilot_yaml_path.read_text()) if doc]
        assert len(documents) == 16
        stages = documents[1:]
        assert "groot_task_performance resolve-task-contract" in stages[0]["run"]
        assert "groot_learning prepare-split" in stages[1]["run"]
        assert "workbench groot finetune" in stages[2]["run"]
        assert "groot_task_performance resolve-trained-checkpoint" in stages[3]["run"]
        assert "--phase baseline" in stages[4]["run"]
        assert "--phase trained" in stages[5]["run"]
        assert "groot_task_performance analyze-paired-outcomes" in stages[6]["run"]
        assert "groot_task_performance select-checkpoint" in stages[7]["run"]
        assert "groot_task_performance analyze-paired-outcomes" in stages[10]["run"]
        assert "groot_task_performance render-task-rollouts" in stages[11]["run"]
        assert "groot_task_performance emit-mcap" in stages[12]["run"]
        assert "groot_task_performance emit-rrd" in stages[13]["run"]
        assert "groot_task_performance publish" in stages[14]["run"]
        assert "gym-pusht==0.1.6" in stages[4]["setup"]
        assert "transformers==4.57.3" in stages[5]["setup"]
        assert "[viz]" in stages[13]["setup"]
        assert stages[4]["config"]["kubernetes"]["pod_config"]["spec"][
            "securityContext"
        ] == {"runAsUser": 0, "runAsGroup": 0}
    finally:
        prepared.temp_dir.cleanup()
