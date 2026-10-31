from __future__ import annotations

import hashlib
import json
from pathlib import Path

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import merge_config_overrides


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "workflows/testing/sim2real-smolvla-franka-cube-eval.yaml"
READINESS = SPEC.with_suffix(".readiness.json")
STATES = [
    "prepare-contract",
    "provision-runtime-prerequisites",
    "start-adapted-service",
    "evaluate-adapted-gold",
    "cleanup-adapted-service",
    "finalize",
    "cleanup-runtime-prerequisites",
]


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_experimental_eval_is_static_checkpoint_bound_graph() -> None:
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="franka-cube-eval")

    assert [step.state for step in plan.steps] == STATES
    assert all(not state.transitions for state in spec.states.values())
    assert "collect" not in " ".join(STATES)
    assert "train" not in " ".join(STATES)
    assert spec.on_failure == "failure-cleanup-runtime-prerequisites"
    assert spec.states["cleanup-runtime-prerequisites"].terminal is True


def test_experimental_eval_readiness_is_hash_bound_and_honest() -> None:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))

    assert readiness["workflow_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest()
    assert readiness["prerequisites"]["target_runtime"]["status"] == "unverified"
    assert "did not meet" in readiness["prerequisites"]["target_runtime"]["reason"]


def test_experimental_eval_declares_uniform_ensemble_and_three_wrist_modes() -> None:
    spec = load_spec(SPEC)
    steps = {step.state: step for step in build_plan(spec, run_id="protocol").steps}
    evaluation = steps["evaluate-adapted-gold"].argv

    assert _value(evaluation, "--policy-backend") == "lerobot_smolvla"
    assert _value(evaluation, "--smolvla-action-selection") == (
        "uniform_temporal_ensemble"
    )
    assert _value(evaluation, "--wrist-ablation-modes") == (
        "fresh,frozen,pixel_channel_shuffle"
    )
    assert _value(evaluation, "--episodes") == "8"
    assert _value(evaluation, "--decisions") == "192"


def test_experimental_eval_owns_service_and_runtime_cleanup() -> None:
    spec = load_spec(SPEC)
    steps = {step.state: step for step in build_plan(spec, run_id="owned").steps}
    start = steps["start-adapted-service"].argv
    cleanup = steps["cleanup-adapted-service"].argv
    runtime_cleanup = steps["cleanup-runtime-prerequisites"].argv

    assert _value(start, "--checkpoint-uri") == spec.config["adapted_checkpoint_uri"]
    assert _value(start, "--auth-secret") == "npa-smolvla-owned-auth"
    assert _value(cleanup, "--checkpoint-uri") == spec.config["adapted_checkpoint_uri"]
    assert cleanup[0:3] == start[0:3]
    assert cleanup[3] == "cleanup"
    assert "cleanup" in cleanup
    assert _value(runtime_cleanup, "--run-id") == "owned"


def test_experimental_eval_finalizer_receives_every_native_artifact() -> None:
    spec = load_spec(SPEC)
    steps = {step.state: step for step in build_plan(spec, run_id="proof").steps}
    finalizer = steps["finalize"].argv

    for flag, key in (
        ("--adapted-uri", "adapted_gold_uri"),
        ("--checkpoint-uri", "adapted_checkpoint_uri"),
        ("--final-mp4-uri", "final_mp4_uri"),
        ("--final-witness-mp4-uri", "final_witness_mp4_uri"),
        ("--final-rrd-uri", "final_rrd_uri"),
        ("--final-policy-inputs-uri", "final_policy_inputs_uri"),
        ("--final-html-uri", "final_html_uri"),
    ):
        assert _value(finalizer, flag).endswith(str(spec.config[key]).split("}")[-1])


def test_experimental_eval_renders_production_skypilot_argv() -> None:
    spec = merge_config_overrides(
        load_spec(SPEC),
        {
            "source_sha": "c" * 40,
            "isaac_image": "ghcr.io/nebius/isaac@sha256:" + "b" * 64,
            "runtime_image": "ghcr.io/nebius/lerobot@sha256:" + "d" * 64,
        },
    )
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="rendered"),
        run_id="rendered",
        options=SkypilotRenderOptions(
            accept_eula=True,
            image_overrides={
                "workbench.sim2real": "ghcr.io/nebius/sim2real@sha256:" + "a" * 64,
                "workbench.lerobot": "ghcr.io/nebius/lerobot@sha256:" + "b" * 64,
            },
        ),
    )

    assert "experimental-eval-finalize" in rendered
    assert "uniform_temporal_ensemble" in rendered
    assert "runtime_prerequisites cleanup" in rendered
    assert "smolvla_service_lifecycle cleanup" in rendered
