from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.detect import detect_submit_format
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_setup_for_tool,
    render_skypilot_yaml,
)
from npa.workflows.sim2real.workflow_stage import _authoritative_scene_args


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


def test_environment_generation_and_split_share_stage_two_contract() -> None:
    root = "s3://bucket/runs/exact-run"
    expected = [
        "--scene-spec-uri",
        root + "/stage_02_assets/consumed_scene_spec.json",
    ]
    source = (
        ROOT / "npa" / "src" / "npa" / "workflows" / "sim2real" / "workflow_stage.py"
    ).read_text()

    assert _authoritative_scene_args(root) == expected
    assert source.count("*_authoritative_scene_args(root)") == 2


def test_stage_adapter_import_does_not_load_legacy_controller() -> None:
    script = """
import sys
import npa.workflows.sim2real.workflow_stage
for name in (
    'npa.workflows.sim2real.engine',
    'npa.workflows.sim2real.legacy_orchestration',
    'npa.workflows.sim2real.runner',
):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_baked_setup_executes_and_records_the_declared_interpreter(
    tmp_path: Path,
) -> None:
    record = tmp_path / "npa-python"
    setup = render_setup_for_tool(
        "run.shell",
        config={"require_baked_npa": "1"},
        options=SkypilotRenderOptions(),
    ).replace("/tmp/npa-python", str(record))
    source_sha = "a" * 40
    environment = {
        "PATH": "/usr/bin:/bin",
        "NPA_BAKED_PYTHON": sys.executable,
        "NPA_IMAGE_SOURCE_SHA": source_sha,
        "NPA_SIM2REAL_SOURCE_SHA": source_sha,
    }

    subprocess.run(["bash", "-c", setup], check=True, env=environment)

    assert record.read_text(encoding="utf-8").strip() == sys.executable
    environment["NPA_BAKED_PYTHON"] = "relative/python"
    failed = subprocess.run(
        ["bash", "-c", setup], check=False, capture_output=True, env=environment
    )
    assert failed.returncode == 68
    assert b"must be an absolute path" in failed.stderr


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
            "omni_kit_accept_eula": "YES",
            "isaacsim_accept_eula": "YES",
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
        assert "NPA_BAKED_PYTHON" in task["setup"]
        assert "/tmp/npa-python" in task["setup"]
        assert "baked NPA interpreter must be an absolute path" in task["setup"]
        assert "baked NPA interpreter is not executable" in task["setup"]
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

    isaac_env = spec.resources["isaac-gpu"]["kubernetes"]["pod_config"]["spec"][
        "containers"
    ][0]["env"]
    isaac_env_by_name = {item["name"]: item["value"] for item in isaac_env}
    assert isaac_env_by_name["NPA_BAKED_PYTHON"] == "/opt/npa/sim/venv/bin/python"
    rendered_isaac_tasks = []
    for task in tasks:
        containers = (
            task.get("config", {})
            .get("kubernetes", {})
            .get("pod_config", {})
            .get("spec", {})
            .get("containers", [])
        )
        if not containers:
            continue
        task_env = {
            item["name"]: item["value"] for item in containers[0].get("env", [])
        }
        if "NPA_BAKED_PYTHON" in task_env:
            rendered_isaac_tasks.append(task)
            assert task_env["OMNI_KIT_ACCEPT_EULA"] == "YES"
            assert task_env["ISAACSIM_ACCEPT_EULA"] == "YES"
    assert len(rendered_isaac_tasks) == 3


def test_canonical_isaac_eula_acceptance_is_operator_supplied_and_fail_closed() -> None:
    payload = yaml.safe_load(SPEC.read_text())
    assert payload["config"]["omni_kit_accept_eula"] == ""
    assert payload["config"]["isaacsim_accept_eula"] == ""
    env = payload["resources"]["isaac-gpu"]["kubernetes"]["pod_config"]["spec"][
        "containers"
    ][0]["env"]
    by_name = {item["name"]: item["value"] for item in env}
    assert by_name["OMNI_KIT_ACCEPT_EULA"] == "{{config.omni_kit_accept_eula}}"
    assert by_name["ISAACSIM_ACCEPT_EULA"] == "{{config.isaacsim_accept_eula}}"
