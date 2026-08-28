"""GR00T N1.7 operational data/train/evaluate/artifact/UI workflow coverage."""

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
from npa.orchestration.skypilot.image_bootstrap_contract import is_trusted_npa_image


REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = REPO_ROOT / "npa/workflows/workbench/npa-workflows/groot-1-7-finetune.yaml"

STATES = [
    "access_capacity_preflight",
    "prepare_deterministic_split",
    "baseline_inference_evaluation",
    "distributed_training",
    "trained_checkpoint_resolution",
    "post_training_inference_evaluation",
    "classify_learning_outcome",
    "generate_rrd",
    "generate_mcap",
    "publish_artifacts_run_summary",
    "agent_ui_load_viewer_verification",
]
TOOL_REFS = [
    "workflow.groot.preflight_rigor",
    "workflow.groot.prepare_split",
    "workbench.groot.baseline_eval",
    "workbench.groot.finetune",
    "workflow.groot.resolve_trained_checkpoint",
    "workbench.groot.posttrain_eval",
    "workflow.groot.compare_learning",
    "workflow.groot.emit_learning_rrd",
    "workflow.groot.emit_learning_mcap",
    "workflow.groot.publish_learning",
    "workflow.groot.verify_agent_ui",
]


def _option_value(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_groot_workflow_is_short_honest_operational_pipeline() -> None:
    spec = load_spec(SPEC_PATH)
    plan = build_plan(spec, run_id="groot-two-gpu-operational")

    assert spec.api_version == "npa.workflow/v0.0.1"
    assert [item.state for item in plan.steps] == STATES
    assert [item.tool_ref for item in plan.steps] == TOOL_REFS
    assert int(spec.config["gpu_count"]) == 2
    assert int(spec.config["max_steps"]) == 4
    assert 2 <= int(spec.config["max_steps"]) < 10000
    assert int(spec.config["logging_steps"]) == 1
    assert int(spec.config["save_steps"]) == int(spec.config["max_steps"])
    assert int(spec.config["global_batch_size"]) == (
        int(spec.config["gpu_count"])
        * int(spec.config["per_device_batch_size"])
        * int(spec.config["gradient_accumulation_steps"])
    )
    assert int(spec.config["train_episodes"]) > 0
    assert int(spec.config["heldout_episodes"]) > 0
    assert int(spec.config["final_episodes"]) == 0
    assert spec.config["action_representation"] == "ABSOLUTE"

    preflight = plan.steps[0]
    baseline = plan.steps[2]
    trainer = plan.steps[3]
    resolver = plan.steps[4]
    posttrain = plan.steps[5]
    assert _option_value(preflight.argv, "--max-steps") == "4"
    assert _option_value(preflight.argv, "--save-steps") == "4"
    assert _option_value(preflight.argv, "--save-total-limit") == "1"
    assert _option_value(baseline.argv, "--robot-embodiment") == "NEW_EMBODIMENT"
    assert trainer.resources_profile["accelerators"] == "B200:2"
    assert _option_value(trainer.argv, "--num-gpus") == "2"
    assert _option_value(trainer.argv, "--logging-steps") == "1"
    assert _option_value(trainer.argv, "--save-steps") == "4"
    assert _option_value(trainer.argv, "--checkpoint-s3-uri").endswith(
        "/checkpoints/candidate/"
    )
    assert _option_value(resolver.argv, "--expected-gpu-count") == "2"
    assert "--checkpoint-uri" not in posttrain.argv
    assert _option_value(posttrain.argv, "--robot-embodiment") == "NEW_EMBODIMENT"
    assert _option_value(posttrain.argv, "--checkpoint-ref-uri").endswith(
        "/reports/trained-checkpoint.json"
    )
    override_values = [
        trainer.argv[index + 1]
        for index, value in enumerate(trainer.argv[:-1])
        if value == "--override"
    ]
    assert {"tune-projector=true", "tune-diffusion-model=true"} <= set(
        override_values
    )
    assert all("denoising" not in value for step in plan.steps for value in step.argv)


def test_not_improved_outcome_does_not_bypass_diagnostic_or_ui_stages() -> None:
    spec = load_spec(SPEC_PATH)
    plan = build_plan(spec, run_id="groot-not-improved-diagnostics")
    compare_index = STATES.index("classify_learning_outcome")

    # The comparison command classifies learning but has no improvement-only
    # selector/gate. Its unconditional graph continues through both native
    # artifacts, publication, and the deployed-agent viewer handoff.
    assert "--selection-uri" not in plan.steps[compare_index].argv
    assert [item.state for item in plan.steps[compare_index + 1 :]] == [
        "generate_rrd",
        "generate_mcap",
        "publish_artifacts_run_summary",
        "agent_ui_load_viewer_verification",
    ]
    assert spec.states[STATES[-1]].terminal is True


def test_groot_workflow_reaches_plan_scheduler_and_vendor_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = "cr.ci.invalid/workbench"
    monkeypatch.setenv("NPA_REGISTRY", registry)
    monkeypatch.setenv(
        "NPA_PUBLIC_REGISTRY", "ghcr.io/nebius/nebius-physical-ai"
    )
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source/npa")
    prepared = prepare_npa_workflow_for_submit(
        SPEC_PATH,
        run_id="groot-operational-render",
        render_options=SkypilotRenderOptions(
            registry=registry,
            materialize_registry_secrets=False,
        ),
    )
    try:
        scheduler = build_scheduler_plan(
            prepared.spec, prepared.plan.steps, run_id="groot-operational-render"
        )
        assert [task["name"] for task in scheduler["tasks"]] == STATES
        assert scheduler["tasks"][3]["resources"]["accelerators"] == "B200:2"

        documents = [
            doc
            for doc in yaml.safe_load_all(prepared.skypilot_yaml_path.read_text())
            if doc
        ]
        assert len(documents) == len(STATES) + 1
        by_name = {stage["name"]: stage for stage in documents[1:]}
        assert "groot_learning preflight-rigor" in by_name[STATES[0]]["run"]
        assert "groot_learning prepare-split" in by_name[STATES[1]]["run"]
        assert "--robot-embodiment NEW_EMBODIMENT" in by_name[STATES[2]]["run"]
        assert "workbench groot finetune" in by_name[STATES[3]]["run"]
        assert "groot_learning posttrain-eval" in by_name[STATES[5]]["run"]
        assert "--robot-embodiment NEW_EMBODIMENT" in by_name[STATES[5]]["run"]
        assert "groot_learning compare-learning" in by_name[STATES[6]]["run"]
        assert "groot_learning emit-rrd" in by_name[STATES[7]]["run"]
        assert "groot_learning emit-mcap" in by_name[STATES[8]]["run"]
        assert "groot_learning publish" in by_name[STATES[9]]["run"]
        assert "groot_learning verify-agent-ui" in by_name[STATES[10]]["run"]
        assert "[viz]" in by_name[STATES[7]]["setup"]
        trusted_images = [
            stage["resources"]["image_id"]
            for stage in documents[1:]
            if stage.get("resources", {}).get("image_id")
            and is_trusted_npa_image(stage["resources"]["image_id"])
        ]
        assert trusted_images, "test must traverse the configured trusted-image guard"
        for state in (STATES[2], STATES[3], STATES[5]):
            assert "securityContext" not in yaml.safe_dump(
                by_name[state].get("config", {})
            )
        assert "runAsUser: 0" not in prepared.skypilot_yaml_path.read_text(
            encoding="utf-8"
        )
    finally:
        prepared.temp_dir.cleanup()


@pytest.mark.parametrize("gpu_count", [1, 2, 7, 8, 16])
def test_groot_workflow_gpu_count_matrix(gpu_count: int) -> None:
    spec = copy.deepcopy(load_spec(SPEC_PATH))
    spec.config.update(
        {
            "gpu_count": str(gpu_count),
            "per_device_batch_size": "1",
            "gradient_accumulation_steps": "1",
            "global_batch_size": str(gpu_count),
        }
    )
    run_id = f"groot-gpu-matrix-{gpu_count}"
    plan = build_plan(spec, run_id=run_id)
    scheduler = build_scheduler_plan(spec, plan.steps, run_id=run_id)

    trainer = plan.steps[3]
    assert trainer.resources_profile["accelerators"] == f"B200:{gpu_count}"
    assert _option_value(trainer.argv, "--num-gpus") == str(gpu_count)
    assert _option_value(trainer.argv, "--global-batch-size") == str(gpu_count)
    assert _option_value(trainer.argv, "--gradient-accumulation-steps") == "1"
    assert scheduler["tasks"][3]["resources"]["accelerators"] == (
        f"B200:{gpu_count}"
    )
