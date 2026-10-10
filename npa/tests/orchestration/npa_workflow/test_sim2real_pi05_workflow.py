from __future__ import annotations

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.scheduler import build_scheduler_task
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.workflows.sim2real.pi05_closed_loop import _validate_gold_scenarios
from npa.workflows.sim2real.pi05_collect_all import SPLITS
from npa.workflows.sim2real.pi05_isaac import support_surface_contract


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "workflows/testing/sim2real-pi05.yaml"
STATES = [
    "prepare-contract",
    "collect-expert",
    "export-dataset",
    "artifact-preflight",
    "start-base-service",
    "evaluate-base-gold",
    "cleanup-base-service",
    "train-lora",
    "reload-validation",
    "start-adapted-service",
    "evaluate-adapted-gold",
    "physical-robot-seam",
    "cleanup-adapted-service",
    "finalize",
]
_ISAAC_STATE_ENV = {
    "OMNI_USER_DIR": "omni-user",
    "OMNI_LOG_DIR": "logs",
    "XDG_RUNTIME_DIR": "xdg-runtime",
    "XDG_CACHE_HOME": "xdg-cache",
    "XDG_CONFIG_HOME": "xdg-config",
    "XDG_DATA_HOME": "xdg-data",
}


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_pi05_is_a_separate_static_declarative_variant() -> None:
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="pi05-contract")
    assert spec.api_version == "npa.workflow/v0.0.1"
    assert [step.state for step in plan.steps] == STATES
    assert all(not state.transitions for state in spec.states.values())
    assert spec.config["base_config_name"] == "pi05_droid_jointpos_polaris"
    assert "sim2real.yaml" not in str(spec.config)
    case = next(item for item in SUBMIT_LIVE_MATRIX if item.spec == SPEC.name)
    assert case.plan_only and case.runtime and case.tier == "multi"
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in case.secret_envs


def test_training_and_service_gpu_allocations_are_independent_and_single_host() -> None:
    spec = load_spec(SPEC)
    assert spec.resources["openpi-training"]["accelerators"] == (
        "{{config.training_gpu_type}}:{{config.training_gpu_count}}"
    )
    assert spec.resources["openpi-training"]["num_nodes"] == (
        "{{config.training_num_nodes}}"
    )
    plan = {step.state: step for step in build_plan(spec, run_id="gpu-split").steps}
    service = plan["start-base-service"].argv
    training = plan["train-lora"].argv
    assert _value(service, "--gpu-count") == spec.config["service_gpu_count"]
    assert (
        _value(training, "--expected-gpu-count") == (spec.config["training_gpu_count"])
    )
    assert spec.config["training_num_nodes"] == "1"
    assert _value(service, "--auth-secret") == spec.config["service_auth_secret"]
    isaac_env = spec.resources["isaac"]["kubernetes"]["pod_config"]["spec"][
        "containers"
    ][0]["env"]
    auth = next(
        item for item in isaac_env if item["name"] == "NPA_OPENPI_SERVICE_AUTH_TOKEN"
    )
    assert auth["valueFrom"]["secretKeyRef"] == {
        "name": "{{config.service_auth_secret}}",
        "key": "auth_token",
    }


def test_only_isaac_tasks_use_the_configured_graphics_runtime_class() -> None:
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="isaac-runtime-class")
    tasks = {
        step.state: build_scheduler_task(
            spec,
            step,
            run_id="isaac-runtime-class",
        )
        for step in plan.steps
    }

    assert spec.config["isaac_runtime_class"] == "nvidia"
    isaac_states = {
        "collect-expert",
        "evaluate-base-gold",
        "evaluate-adapted-gold",
    }
    for state in isaac_states:
        pod_spec = tasks[state]["resources"]["kubernetes"]["pod_config"]["spec"]
        _assert_isaac_state_contract(pod_spec)

    for state, task in tasks.items():
        if state in isaac_states:
            continue
        pod_spec = (
            task["resources"]
            .get("kubernetes", {})
            .get("pod_config", {})
            .get("spec", {})
        )
        assert "runtimeClassName" not in pod_spec
        assert not any(
            item.get("name") == "npa-isaac-job-state"
            for item in pod_spec.get("volumes", [])
        )


def _assert_isaac_state_contract(pod_spec: dict) -> None:
    assert pod_spec["runtimeClassName"] == "nvidia"
    container = pod_spec["containers"][0]
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    volumes = {item["name"]: item for item in pod_spec["volumes"]}
    env = {item["name"]: item.get("value") for item in container["env"]}
    assert mounts["npa-isaac-runtime-cache"]["readOnly"] is True
    assert mounts["npa-isaac-job-state"] == {
        "name": "npa-isaac-job-state",
        "mountPath": "/tmp/npa-isaac-state",
    }
    assert volumes["npa-isaac-job-state"] == {
        "name": "npa-isaac-job-state",
        "emptyDir": {},
    }
    assert env["NPA_ISAAC_STATE_DIR"] == "/tmp/npa-isaac-state"
    assert env["NPA_ISAAC_KIT_ARGS"] == "--portable-root /tmp/npa-isaac-state/kit"
    assert {name: env[name] for name in _ISAAC_STATE_ENV} == {
        name: f"/tmp/npa-isaac-state/{suffix}"
        for name, suffix in _ISAAC_STATE_ENV.items()
    }


def test_all_cpu_roles_share_one_exact_source_image() -> None:
    spec = load_spec(SPEC)
    cpu_image = "{{config.cpu_image}}"

    assert spec.config["cpu_image"] == ""
    assert spec.resources["cpu"]["image"] == cpu_image
    assert spec.resources["data"]["image"] == cpu_image
    assert spec.resources["service-control"]["image"] == cpu_image
    assert spec.resources["viewer"]["image"] == cpu_image
    assert "controller_image" not in spec.config
    assert "data_image" not in spec.config
    assert "viewer_image" not in spec.config


def test_gold_controller_states_share_exact_protocol_inputs() -> None:
    plan = build_plan(load_spec(SPEC), run_id="pi05-gold-contract")
    steps = {step.state: step for step in plan.steps}
    base = steps["evaluate-base-gold"].argv
    adapted = steps["evaluate-adapted-gold"].argv
    for flag in (
        "--gold-collection-uri",
        "--split",
        "--episodes",
        "--decisions",
        "--seed",
        "--wrist-ablation-modes",
        "--ablation-seed",
    ):
        assert _value(base, flag) == _value(adapted, flag)
    assert _value(base, "--checkpoint-role") == "base"
    assert _value(adapted, "--checkpoint-role") == "adapted"
    assert "--gold-collection-uri" in base
    assert _value(base, "--wrist-ablation-modes") == (
        "fresh,frozen,pixel_channel_shuffle"
    )


def test_default_split_seeds_match_collector_and_gold_validator() -> None:
    spec = load_spec(SPEC)
    plan = {step.state: step for step in build_plan(spec, run_id="split-seeds").steps}
    expert_seed = int(spec.config["expert_seed"])
    validation_seed = expert_seed + int(SPLITS["validation"]["seed_offset"])
    gold_seed = expert_seed + int(SPLITS["gold"]["seed_offset"])
    scenarios = [_gold_scenario(gold_seed + index) for index in range(8)]

    assert validation_seed == 20271007
    assert int(spec.config["gold_seed"]) == gold_seed == 20281007
    assert int(_value(plan["collect-expert"].argv, "--seed")) == expert_seed
    assert int(_value(plan["evaluate-base-gold"].argv, "--seed")) == gold_seed
    assert int(_value(plan["evaluate-adapted-gold"].argv, "--seed")) == gold_seed
    _validate_gold_scenarios(scenarios, gold_seed)
    with pytest.raises(RuntimeError, match="scenario seed"):
        _validate_gold_scenarios(scenarios, validation_seed)


def _gold_scenario(seed: int) -> dict:
    return {
        "scene_seed": seed,
        "initial_object_position_m": [0.5, -0.1, 0.024],
        "target_position_m": [0.5, 0.18, 0.024],
        "destination_marker": {
            "semantic_label": "pi05_destination_marker",
            "physics_collision": False,
        },
        "support_surface": support_surface_contract(),
    }


def test_declared_outputs_include_real_dataset_checkpoint_video_and_rrd() -> None:
    spec = load_spec(SPEC)
    outputs = {
        (state_name, artifact.schema, artifact.uri)
        for state_name, state in spec.states.items()
        for artifact in state.outputs
    }
    required_schemas = {
        "npa.sim2real.pi05.dense_dataset.v1",
        "npa.workbench.openpi.pi05-training.v1",
        "video/mp4",
        "application/vnd.rerun.rrd",
        "npa.sim2real.pi05.final_report.v1",
    }
    assert required_schemas <= {schema for _, schema, _ in outputs}
    final = spec.states["finalize"]
    assert final.terminal is True


def test_component_records_and_eval_report_roots_are_stage_consistent() -> None:
    spec = load_spec(SPEC)
    components = [
        artifact.uri
        for state in spec.states.values()
        for artifact in state.outputs
        if artifact.schema == "npa.sim2real.component_record.v1"
    ]
    assert sorted(uri.rsplit("_", 1)[-1] for uri in components) == [
        f"{stage:02d}.json" for stage in range(1, 15)
    ]
    plan = {step.state: step for step in build_plan(spec, run_id="roots").steps}
    for state in ("evaluate-base-gold", "evaluate-adapted-gold"):
        argv = plan[state].argv
        assert _value(argv, "--output-uri") == (
            _value(argv, "--artifact-root-uri").rstrip("/") + "/report.json"
        )
