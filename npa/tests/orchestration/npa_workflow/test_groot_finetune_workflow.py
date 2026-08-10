"""GR00T N1.7 retrain/select/final closed-loop PushT workflow coverage."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
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
        "access_capacity_preflight",
        "resolve_task_contract",
        "prepare_retraining_split",
        "evaluate_offline_baseline",
        "retrain_task_policy",
        "resolve_trained_checkpoint",
        "probe_checkpoint_2500",
        "probe_checkpoint_5000",
        "probe_checkpoint_10000",
        "select_offline_checkpoint",
        "compare_offline_learning",
        "emit_offline_mcap",
        "emit_offline_rrd",
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
        "workflow.groot.preflight_rigor",
        "workflow.groot.resolve_task_contract",
        "workflow.groot.prepare_split",
        "workbench.groot.baseline_eval",
        "workbench.groot.finetune",
        "workflow.groot.resolve_trained_checkpoint",
        "workbench.groot.probe_2500_eval",
        "workbench.groot.probe_5000_eval",
        "workbench.groot.probe_10000_eval",
        "workflow.groot.select_offline_checkpoint",
        "workflow.groot.compare_learning",
        "workflow.groot.emit_learning_mcap",
        "workflow.groot.emit_learning_rrd",
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
    assert int(spec.config["max_steps"]) == 10000
    assert int(spec.config["train_episodes"]) == 154
    assert int(spec.config["heldout_episodes"]) == 26
    assert int(spec.config["final_episodes"]) == 26
    assert int(spec.config["validation_episodes"]) >= 24
    assert int(spec.config["paired_episodes"]) >= 24
    assert int(spec.config["global_batch_size"]) >= 128
    assert int(spec.config["global_batch_size"]) == (
        int(spec.config["gpu_count"])
        * int(spec.config["per_device_batch_size"])
        * int(spec.config["gradient_accumulation_steps"])
    )
    assert (
        spec.config["validation_seed_namespace"] != spec.config["final_seed_namespace"]
    )
    assert _option_value(plan.steps[2].argv, "--max-steps") == "10000"
    assert _option_value(plan.steps[4].argv, "--max-steps") == "10000"
    assert plan.steps[4].resources_profile["accelerators"].endswith(":8")
    trainer_argv = plan.steps[4].argv
    assert _option_value(trainer_argv, "--per-device-batch-size") == "1"
    assert _option_value(trainer_argv, "--gradient-accumulation-steps") == "16"
    override_values = [
        trainer_argv[index + 1]
        for index, value in enumerate(trainer_argv[:-1])
        if value == "--override"
    ]
    assert {
        "tune-projector=true",
        "tune-diffusion-model=true",
    } <= set(override_values)
    assert spec.config["action_representation"] == "ABSOLUTE"
    assert all("denoising" not in value for step in plan.steps for value in step.argv)
    for step in (plan.steps[13], plan.steps[14], plan.steps[17], plan.steps[18]):
        assert step.resources_profile["accelerators"].endswith(":1")
        assert _option_value(step.argv, "--episodes") == "24"
        assert _option_value(step.argv, "--horizon") == "300"
    assert "offline/baseline/evaluation.json" in _option_value(
        plan.steps[17].argv, "--checkpoint-ref-uri"
    )
    assert "selected-checkpoint.json" in _option_value(
        plan.steps[18].argv, "--checkpoint-ref-uri"
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
        assert scheduler["tasks"][4]["resources"]["accelerators"].endswith(":8")

        documents = [
            doc
            for doc in yaml.safe_load_all(prepared.skypilot_yaml_path.read_text())
            if doc
        ]
        assert len(documents) == 25
        stages = documents[1:]
        by_name = {stage["name"]: stage for stage in stages}
        assert (
            "groot_learning preflight-rigor"
            in by_name["access_capacity_preflight"]["run"]
        )
        assert (
            "groot_task_performance resolve-task-contract"
            in by_name["resolve_task_contract"]["run"]
        )
        assert (
            "groot_learning prepare-split" in by_name["prepare_retraining_split"]["run"]
        )
        assert "workbench groot finetune" in by_name["retrain_task_policy"]["run"]
        assert (
            "groot_learning select-offline-checkpoint"
            in by_name["select_offline_checkpoint"]["run"]
        )
        assert (
            "groot_task_performance analyze-paired-outcomes"
            in by_name["analyze_paired_outcomes"]["run"]
        )
        assert "groot_task_performance publish" in by_name["publish"]["run"]
        assert "gym-pusht==0.1.6" in by_name["evaluate_validation_baseline"]["setup"]
        assert (
            "transformers==4.57.3" in by_name["evaluate_validation_candidate"]["setup"]
        )
        assert "[viz]" in by_name["emit_rrd"]["setup"]
        assert by_name["evaluate_validation_baseline"]["config"]["kubernetes"][
            "pod_config"
        ]["spec"]["securityContext"] == {"runAsUser": 0, "runAsGroup": 0}
    finally:
        prepared.temp_dir.cleanup()


@pytest.mark.parametrize(
    ("gpu_count", "accumulation"),
    [(1, 128), (2, 64), (8, 16)],
)
def test_groot_workflow_gpu_count_matrix(
    gpu_count: int, accumulation: int
) -> None:
    spec = copy.deepcopy(load_spec(SPEC_PATH))
    spec.config.update(
        {
            "gpu_count": str(gpu_count),
            "per_device_batch_size": "1",
            "gradient_accumulation_steps": str(accumulation),
            "global_batch_size": "128",
        }
    )
    plan = build_plan(spec, run_id=f"groot-gpu-matrix-{gpu_count}")
    scheduler = build_scheduler_plan(
        spec, plan.steps, run_id=f"groot-gpu-matrix-{gpu_count}"
    )

    trainer = plan.steps[4]
    assert trainer.resources_profile["accelerators"] == f"RTXPRO6000:{gpu_count}"
    assert _option_value(trainer.argv, "--num-gpus") == str(gpu_count)
    assert _option_value(trainer.argv, "--global-batch-size") == "128"
    assert _option_value(trainer.argv, "--gradient-accumulation-steps") == str(
        accumulation
    )
    assert scheduler["tasks"][4]["resources"]["accelerators"] == (
        f"RTXPRO6000:{gpu_count}"
    )
