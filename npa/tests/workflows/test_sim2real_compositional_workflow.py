from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.detect import detect_submit_format
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "sim2real.yaml"


def test_canonical_is_one_standard_compositional_workflow() -> None:
    payload = yaml.safe_load(SPEC.read_text())
    assert payload["apiVersion"] == "npa.workflow/v0.0.1"
    assert payload["kind"] == "Workflow"
    assert detect_submit_format(SPEC) == "npa.workflow"
    assert not (ROOT / "npa" / "workflows" / "sim2real.yaml").exists()
    assert not (
        ROOT
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "sim2real-vlm-rl.yaml"
    ).exists()

    leaf_states = [state for state in payload["states"].values() if state.get("run")]
    assert len(leaf_states) >= 14
    rendered_commands = "\n".join(
        " ".join(state["run"].get("argv") or []) for state in leaf_states
    )
    for forbidden in (
        "run_preamble",
        "run_inner_loop",
        "run_single_outer_iteration",
        "run_finalize",
        "k8s_submit",
    ):
        assert forbidden not in rendered_commands


def test_reduced_plan_preserves_all_real_solution_boundaries() -> None:
    spec = load_spec(SPEC)
    spec.config.update(
        {
            "outer_iterations": "1",
            "inner_iterations": "1",
            "env_count": "12",
            "rollout_count": "1",
            "ppo_iterations": "5",
            "validation_count": "3",
            "gold_count": "3",
        }
    )
    plan = build_plan(spec, run_id="compose-1x1", assume_decision="loop_back")
    states = [step.state for step in plan.steps]
    expected = {
        "stage-01-trigger",
        "stage-02-assets",
        "stage-03-transfer",
        "stage-04-shard-0",
        "stage-04-shard-1",
        "stage-05-split",
        "stage-06-tokens",
        "stage-07-rollouts",
        "stage-08-reason2",
        "stage-08-reason3",
        "stage-09-ppo",
        "stage-10-gold",
        "stage-11-decision",
        "stage-12-external-seam",
        "stage-13-retrigger",
        "stage-14-visualize",
    }
    assert set(states) == expected
    assert states.count("stage-07-rollouts") == 1
    assert states.count("stage-09-ppo") == 1


def test_stage_adapters_do_not_submit_hidden_kubernetes_jobs() -> None:
    source = (
        ROOT / "npa" / "src" / "npa" / "workflows" / "sim2real" / "workflow_stage.py"
    ).read_text()
    assert "KubernetesJobClient" not in source
    assert "run_gpu_job_with_fallback" not in source
    assert "submit_sim2real" not in source
    assert "sim2real.engine" not in source
    assert "NPA_SIM2REAL_INLINE_TASK" in source


def test_exact_source_and_per_state_immutable_images_reach_rendered_tasks() -> None:
    spec = load_spec(SPEC)
    source_sha = "a" * 40
    image = "cr.example/npa/runtime@sha256:" + "b" * 64
    spec.config.update(
        {
            "source_sha": source_sha,
            "outer_iterations": "1",
            "inner_iterations": "1",
            "controller_image": image,
            "transfer_image": image,
            "envgen_image": image,
            "reason_image": image,
            "isaac_image": image,
            "viewer_image": image,
            "isaac_cache_pvc": "isaac-cache",
        }
    )
    plan = build_plan(spec, run_id="render-1x1", assume_decision="loop_back")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="render-1x1",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    documents = [item for item in yaml.safe_load_all(rendered) if item]
    tasks = [item for item in documents if item.get("envs")]
    assert tasks
    for task in tasks:
        assert task["envs"]["NPA_SIM2REAL_SOURCE_SHA"] == source_sha
        assert task["envs"]["NPA_TASK_IMAGE"] == image
        assert "immutable baked NPA runtime verified" in task["setup"]
        assert "pip install" not in task["setup"]
        assert "NPA_SRC_S3_URI" not in task["envs"]

    gpu_tasks = [task for task in tasks if task["resources"].get("accelerators")]
    assert gpu_tasks
    for task in gpu_tasks:
        pod_config = task["config"]["kubernetes"]["pod_config"]
        assert pod_config["metadata"]["labels"] == {
            "kueue.x-k8s.io/queue-name": "sim2real-gpu"
        }
        assert pod_config["spec"]["priorityClassName"] == "sim2real-production"
