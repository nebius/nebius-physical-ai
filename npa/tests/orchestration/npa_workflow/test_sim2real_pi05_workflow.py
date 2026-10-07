from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "workflows/testing/sim2real-pi05.yaml"
STATES = [
    "prepare-contract",
    "collect-expert",
    "export-dataset",
    "negative-terms-gate",
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
    ):
        assert _value(base, flag) == _value(adapted, flag)
    assert _value(base, "--checkpoint-role") == "base"
    assert _value(adapted, "--checkpoint-role") == "adapted"
    assert "--gold-collection-uri" in base


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
