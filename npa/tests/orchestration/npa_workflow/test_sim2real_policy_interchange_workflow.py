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
]


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_smolvla_reference_is_standard_static_fourteen_stage_graph() -> None:
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
    assert spec.states["finalize"].terminal is True


def test_reference_labels_its_bounded_franka_cube_scope() -> None:
    spec = load_spec(SPEC)
    description = str(spec.metadata["description"]).lower()
    collect = spec.states["collect-expert"].description.lower()
    assert "does not claim" in description
    assert "franka" in collect
    assert "cube" in collect


def test_isaac_client_mounts_the_smolvla_auth_variable() -> None:
    spec = load_spec(SPEC)
    env = spec.resources["isaac"]["kubernetes"]["pod_config"]["spec"]["containers"][0][
        "env"
    ]
    names = [item["name"] for item in env]
    assert "NPA_POLICY_AUTH_TOKEN" in names
    assert "NPA_OPENPI_SERVICE_AUTH_TOKEN" not in names


def test_isaac_default_leaves_standard_runtime_controller_headroom() -> None:
    spec = load_spec(SPEC)
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
    isaac_tasks = [task for task in tasks if task["name"] == "collect-expert"]

    assert spec.config["isaac_cpus"] == "12"
    assert len(isaac_tasks) == 1
    assert isaac_tasks[0]["resources"]["cpus"] == "12"


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
        "/opt/npa/sim/venv/bin/python",
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
        "/opt/npa/sim/venv/bin/python",
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
