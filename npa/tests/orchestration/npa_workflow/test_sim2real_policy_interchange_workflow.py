from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "workflows/testing/sim2real-policy-interchange-smolvla.yaml"
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
