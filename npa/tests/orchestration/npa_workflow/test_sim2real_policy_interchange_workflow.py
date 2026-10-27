from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.spec import load_spec


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "workflows/testing/sim2real-policy-interchange-smolvla.yaml"
NON_FRANKA_SPEC = ROOT / "workflows/testing/sim2real-policy-interchange-non-franka.yaml"
STATES = [
    "prepare-contract",
    "provision-runtime-prerequisites",
    "collect-expert",
    "export-dataset",
    "artifact-preflight",
    "start-base-service",
    "evaluate-base-gold",
    "cleanup-base-service",
    "train-native",
    "reload-validation",
    "start-adapted-service",
    "evaluate-adapted-gold",
    "physical-robot-seam",
    "cleanup-adapted-service",
    "finalize",
    "cleanup-runtime-prerequisites",
]


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_smolvla_reference_is_standard_static_clean_start_graph() -> None:
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="smolvla-interchange")
    assert [step.state for step in plan.steps] == STATES
    assert all(not state.transitions for state in spec.states.values())
    assert spec.config["base_checkpoint_uri"] == (
        "hf://lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
    )
    assert spec.states["collect-expert"].tool_ref == (
        "workbench.sim2real.policy_interchange_franka_collect"
    )
    assert spec.config["baked_image_source_sha_by_config_key"] == {
        "cpu_image": "cpu_image_source_sha",
        "isaac_image": "isaac_image_source_sha",
        "runtime_image": "runtime_image_source_sha",
    }


def test_native_train_reload_and_service_share_exact_checkpoint_lineage() -> None:
    spec = load_spec(SPEC)
    steps = {step.state: step for step in build_plan(spec, run_id="lineage").steps}
    train = steps["train-native"].argv
    reload = steps["reload-validation"].argv
    adapted = steps["start-adapted-service"].argv
    selected = _value(train, "--selected-checkpoint-s3-uri")
    assert _value(train, "--policy-type") == "smolvla"
    assert (
        _value(train, "--pretrained-policy-source")
        == spec.config["base_checkpoint_uri"]
    )
    assert _value(train, "--selected-checkpoint-s3-uri") == selected
    assert _value(reload, "--checkpoint-uri") == selected
    assert _value(adapted, "--checkpoint-uri") == selected
    assert _value(train, "--interchange-adapter-relative-path") == (
        "meta/npa_policy_adapter.json"
    )


def test_base_service_uses_zero_update_dataset_bound_checkpoint() -> None:
    spec = load_spec(SPEC)
    steps = {step.state: step for step in build_plan(spec, run_id="base-bind").steps}
    artifact = steps["artifact-preflight"].argv
    service = steps["start-base-service"].argv
    prepared = _value(artifact, "--prepared-checkpoint-uri")
    assert _value(artifact, "--checkpoint-uri") == spec.config["base_checkpoint_uri"]
    assert prepared.endswith("/stage_04_artifact/prepared-base/")
    assert _value(service, "--checkpoint-uri") == prepared
    assert "--bind-adapter-uri" not in service


def test_gold_arms_share_physics_protocol_but_not_fake_noise_contract() -> None:
    steps = {
        step.state: step
        for step in build_plan(load_spec(SPEC), run_id="same-gold").steps
    }
    base = steps["evaluate-base-gold"].argv
    adapted = steps["evaluate-adapted-gold"].argv
    for flag in (
        "--gold-collection-uri",
        "--episodes",
        "--decisions",
        "--seed",
        "--wrist-ablation-modes",
        "--ablation-seed",
        "--preparation-uri",
    ):
        assert _value(base, flag) == _value(adapted, flag)
    assert _value(base, "--policy-backend") == "lerobot_smolvla"
    assert _value(base, "--smolvla-action-selection") == ("uniform_temporal_ensemble")
    assert _value(adapted, "--smolvla-action-selection") == (
        "uniform_temporal_ensemble"
    )
    assert _value(base, "--recording-id") == "same-gold"
    assert _value(adapted, "--recording-id") == "same-gold"
    assert all("noise" not in item for item in base)


def test_every_stage_declares_explicit_interchange_component_record() -> None:
    spec = load_spec(SPEC)
    components = [
        artifact.uri
        for state in spec.states.values()
        for artifact in state.outputs
        if artifact.schema == "npa.sim2real.policy_interchange.component_record.v1"
    ]
    assert sorted(uri.rsplit("_", 1)[-1] for uri in components) == [
        f"{stage:02d}.json" for stage in range(1, 15)
    ]
    assert spec.states["finalize"].terminal is False
    assert spec.states["cleanup-runtime-prerequisites"].terminal is True


def test_reference_labels_its_bounded_franka_cube_scope() -> None:
    spec = load_spec(SPEC)
    description = str(spec.metadata["description"]).lower()
    collect = spec.states["collect-expert"].description.lower()
    assert "does not claim" in description
    assert "franka" in collect
    assert "cube" in collect


def test_reference_defaults_to_the_qualified_rtx_route() -> None:
    config = load_spec(SPEC).config
    assert config["training_gpu_type"] == "RTXPRO6000"
    assert config["isaac_gpu_type"] == "RTXPRO6000"
    assert config["service_gpu_node_selector_value"] == "RTXPRO6000"
    assert config["service_hardware_topology"] == "single-rtxpro6000"


def test_reference_renders_distinct_consumer_and_baked_image_sources() -> None:
    spec = load_spec(SPEC)
    images = {
        "cpu_image": "registry.example/npa/cpu@sha256:" + "1" * 64,
        "isaac_image": "registry.example/npa/isaac@sha256:" + "2" * 64,
        "runtime_image": "registry.example/npa/lerobot@sha256:" + "3" * 64,
    }
    sources = {
        "cpu_image_source_sha": "b" * 40,
        "isaac_image_source_sha": "c" * 40,
        "runtime_image_source_sha": "d" * 40,
    }
    spec.config.update(source_sha="a" * 40, **images, **sources)
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="mixed-image-sources"),
        run_id="mixed-image-sources",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    expected = {
        images["cpu_image"]: sources["cpu_image_source_sha"],
        images["isaac_image"]: sources["isaac_image_source_sha"],
        images["runtime_image"]: sources["runtime_image_source_sha"],
    }
    tasks = [doc for doc in yaml.safe_load_all(rendered) if doc and doc.get("envs")]
    assert len(tasks) == 16
    for task in tasks:
        assert task["envs"]["NPA_SIM2REAL_SOURCE_SHA"] == "a" * 40
        task_image = task["envs"]["NPA_TASK_IMAGE"]
        assert task["envs"]["NPA_EXPECTED_IMAGE_SOURCE_SHA"] == expected[task_image]


def test_isaac_client_mounts_the_smolvla_auth_variable() -> None:
    spec = load_spec(SPEC)
    env = spec.resources["isaac"]["kubernetes"]["pod_config"]["spec"]["containers"][0][
        "env"
    ]
    names = [item["name"] for item in env]
    assert "NPA_POLICY_AUTH_TOKEN" in names
    assert "NPA_OPENPI_SERVICE_AUTH_TOKEN" not in names


def test_runtime_fetch_isaac_stages_use_the_cache_bound_interpreter() -> None:
    for path, states in (
        (
            SPEC,
            ("collect-expert", "evaluate-base-gold", "evaluate-adapted-gold"),
        ),
        (
            NON_FRANKA_SPEC,
            ("collect-expert", "evaluate-base", "evaluate-adapted"),
        ),
    ):
        spec = load_spec(path)
        steps = {
            step.state: step for step in build_plan(spec, run_id="isaac-shim").steps
        }
        assert all(steps[state].argv[0] == "/isaac-sim/python.sh" for state in states)
        env = spec.resources["isaac"]["kubernetes"]["pod_config"]["spec"]["containers"][
            0
        ]["env"]
        by_name = {item["name"]: item["value"] for item in env if "value" in item}
        assert by_name["NPA_BAKED_PYTHON"] == "/isaac-sim/python.sh"


def test_isaac_default_leaves_standard_runtime_controller_headroom() -> None:
    cases = (
        (SPEC, {"collect-expert", "evaluate-base-gold", "evaluate-adapted-gold"}),
        (NON_FRANKA_SPEC, {"collect-expert", "evaluate-base", "evaluate-adapted"}),
    )
    for path, isaac_state_names in cases:
        spec = load_spec(path)
        spec.config["source_sha"] = "0123456789abcdef0123456789abcdef01234567"
        rendered = render_skypilot_yaml(
            spec,
            build_plan(spec, run_id="controller-headroom"),
            run_id="controller-headroom",
            options=SkypilotRenderOptions(
                image_overrides={
                    "*": "registry.example.invalid/npa@sha256:"
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                },
                materialize_registry_secrets=False,
            ),
        )
        tasks = [
            doc for doc in yaml.safe_load_all(rendered) if doc and doc.get("resources")
        ]
        isaac_tasks = [task for task in tasks if task["name"] in isaac_state_names]

        assert spec.config["isaac_cpus"] == "12"
        assert {task["name"] for task in isaac_tasks} == isaac_state_names
        assert all(task["resources"]["cpus"] == "12+" for task in isaac_tasks)


def test_non_franka_acceptance_uses_standard_runtime_and_real_native_seams() -> None:
    spec = load_spec(NON_FRANKA_SPEC)
    plan = build_plan(spec, run_id="ur10e-spool-smolvla")
    assert [step.state for step in plan.steps] == [
        "prepare-contract",
        "collect-expert",
        "export-dataset",
        "artifact-preflight",
        "start-base-service",
        "evaluate-base",
        "cleanup-base-service",
        "train-native",
        "reload-validation",
        "start-adapted-service",
        "evaluate-adapted",
        "physical-robot-seam",
        "cleanup-adapted-service",
        "finalize",
    ]
    steps = {step.state: step for step in plan.steps}
    collect = steps["collect-expert"]
    export = steps["export-dataset"]
    execute = steps["evaluate-adapted"]
    assert spec.states["evaluate-adapted"].tool_ref == (
        "workbench.sim2real.policy_interchange_non_franka_execute"
    )
    assert collect.argv[:3] == [
        "/isaac-sim/python.sh",
        "-m",
        "npa.workflows.sim2real.interchange_teacher_collect",
    ]
    assert _value(collect.argv, "--train-episodes") == "32"
    assert export.argv[:3] == [
        "python3",
        "-m",
        "npa.workflows.sim2real.interchange_teacher_export",
    ]
    assert execute.argv[:3] == [
        "/isaac-sim/python.sh",
        "-m",
        "npa.workflows.sim2real.interchange_isaac_entry",
    ]
    assert _value(execute.argv, "--preparation-uri").endswith(
        "/stage_01_prepare/preparation.json"
    )
    assert _value(execute.argv, "--service-receipt-uri").endswith(
        "/stage_10_adapted_service/start.json"
    )
    base = steps["evaluate-base"]
    assert _value(base.argv, "--seed") == _value(execute.argv, "--seed")
    assert _value(base.argv, "--decisions") == _value(execute.argv, "--decisions")
    components = [
        artifact.uri
        for state in spec.states.values()
        for artifact in state.outputs
        if artifact.schema == "npa.sim2real.policy_interchange.component_record.v1"
    ]
    assert sorted(uri.rsplit("_", 1)[-1] for uri in components) == [
        f"{stage:02d}.json" for stage in range(1, 15)
    ]
    description = str(spec.metadata["description"]).lower()
    assert "execution validity separate" in description
    assert "statistical improvement" in description
