from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.first_run_state import prepare_run
from npa.orchestration.npa_workflow.interpreter import _make_context, build_plan
from npa.orchestration.npa_workflow.run_resolution import validate_run_id
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.workflows.sim2real.runtime_prerequisites import resource_identities


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


def _spec():
    return merge_config_overrides(
        load_spec(SPEC),
        {
            "source_sha": "c" * 40,
            "isaac_image": "ghcr.io/nebius/isaac@sha256:" + "b" * 64,
            "runtime_image": "ghcr.io/nebius/lerobot@sha256:" + "d" * 64,
        },
    )


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_experimental_eval_is_static_checkpoint_bound_graph() -> None:
    spec = _spec()
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
    runtime = readiness["prerequisites"]["target_runtime"]
    assert runtime["status"] == "verified"
    assert "all seven stages" in runtime["reason"]
    assert any(
        "causal-wrist criterion remains unmet" in evidence
        for evidence in runtime["evidence"]
    )
    assert any(
        "does not claim the sixteen-stage" in evidence
        for evidence in runtime["evidence"]
    )


def test_experimental_eval_declares_uniform_ensemble_and_three_wrist_modes() -> None:
    spec = _spec()
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
    spec = _spec()
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


def test_experimental_eval_runtime_names_match_the_runtime_owner() -> None:
    run_id = "franka-cube-exp"
    spec = _spec()
    steps = {step.state: step for step in build_plan(spec, run_id=run_id).steps}
    prerequisites = steps["provision-runtime-prerequisites"].argv
    expected = resource_identities(run_id)

    for key, flag in (
        ("isaac_cache_pvc", "--isaac-cache-pvc"),
        ("policy_cache_pvc", "--policy-cache-pvc"),
        ("isaac_warm_job", "--isaac-warm-job"),
        ("auth_secret", "--auth-secret"),
        ("base_service_account", "--base-service-account"),
        ("adapted_service_account", "--adapted-service-account"),
    ):
        assert _value(prerequisites, flag) == expected[key]


def test_generated_run_identity_binds_safe_runtime_prerequisites() -> None:
    prepared = prepare_run(
        project="project-id",
        workflow_identity="sim2real-smolvla-franka-cube-eval",
        persist=False,
    )

    raw_run_id = validate_run_id(prepared.run_id)
    context = _make_context(_spec(), run_id=raw_run_id)
    identities = resource_identities(raw_run_id)

    assert re.fullmatch(
        r"sim2real-smolvla-franka-cube-eval-\d{8}T\d{12}Z-[0-9a-f]{6}",
        raw_run_id,
    )
    assert context.run["id"] == raw_run_id
    assert context.config["service_auth_secret"] == identities["auth_secret"]
    assert context.config["base_service_account"] == identities["base_service_account"]
    assert (
        context.config["adapted_service_account"]
        == identities["adapted_service_account"]
    )
    assert context.config["isaac_cache_pvc"] == identities["isaac_cache_pvc"]
    assert context.config["policy_cache_pvc"] == identities["policy_cache_pvc"]
    assert context.config["isaac_warm_job"] == identities["isaac_warm_job"]
    assert all(
        len(value) <= 63 and re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", value)
        for value in identities.values()
    )


def test_normalized_runtime_identity_names_remain_collision_safe() -> None:
    upper = "sim2real-smolvla-franka-cube-eval-20261101T120000000000Z-a1b2c3"
    lower = upper.lower()

    assert resource_identities(upper) != resource_identities(lower)


def test_workflow_owned_runtime_name_override_must_match_derived_identity() -> None:
    run_id = "explicit-owned-name"
    expected = resource_identities(run_id)
    matching = merge_config_overrides(
        _spec(), {"base_service_account": expected["base_service_account"]}
    )

    assert (
        _make_context(matching, run_id=run_id).config["base_service_account"]
        == expected["base_service_account"]
    )

    conflicting = merge_config_overrides(
        _spec(), {"base_service_account": "foreign-service-account"}
    )
    with pytest.raises(NpaWorkflowError, match="conflicts with workflow-owned runtime"):
        _make_context(conflicting, run_id=run_id)


def test_experimental_eval_finalizer_receives_every_native_artifact() -> None:
    spec = _spec()
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
    spec = _spec()
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="rendered"),
        run_id="rendered",
        options=SkypilotRenderOptions(accept_eula=True),
    )

    assert "experimental-eval-finalize" in rendered
    assert "uniform_temporal_ensemble" in rendered
    assert "runtime_prerequisites cleanup" in rendered
    assert "smolvla_service_lifecycle cleanup" in rendered
