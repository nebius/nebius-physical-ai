"""Regression tests for the sim2real genuine-signal RCA fixes.

Covers the degenerate-signal root causes:
- held-out scores collapsing to a flat ``1.0`` (no gradient),
- cross-rollout diversity diagnostics / anti-hollow gate,
- ``-genuine-`` image pull policy, and
- image-digest provenance captured from the running pod.
"""

from __future__ import annotations

import pytest

import npa.workflows.sim2real_loop as loop_module
from npa.workflows.sim2real_loop import (
    _apply_reference_adapter_heldout_gate,
    _heldout_env_score,
    _image_pull_policy,
    _inner_loop_progress_score,
    _signal_diversity_report,
)


def _signal(score: float, reward: float) -> dict[str, object]:
    return {
        "schema": loop_module.SCHEMA_RL_SIGNAL,
        "score": score,
        "per_step": [{"reward": reward}],
    }


def test_heldout_env_score_success_band_is_continuous_not_flat_one() -> None:
    # Two successful envs with different distance/reward must NOT both be 1.0.
    high = _heldout_env_score(1.0, 1.0, env_success=True)
    low = _heldout_env_score(0.2, 0.0, env_success=True)

    assert high == 1.0
    assert low < high
    assert 0.75 <= low <= 1.0


def test_heldout_env_score_success_outranks_failure() -> None:
    success = _heldout_env_score(0.5, 0.5, env_success=True)
    failure = _heldout_env_score(0.5, 0.5, env_success=False)

    assert success > failure
    assert 0.0 <= failure <= 0.6


def test_inner_loop_progress_score_uses_vlm_final_quality_and_reward_trend() -> None:
    score = _inner_loop_progress_score(
        {
            "reward_trend": [-0.2, 0.5],
            "final_quality": 0.52,
            "iterations": [{"sample_vlm_eval": {"score": 0.82}}],
        }
    )

    assert score == 0.82


def test_apply_reference_adapter_heldout_gate_preserves_sim_details() -> None:
    per_env = [
        {
            "env_id": "heldout-0000",
            "score": 0.11,
            "success": False,
            "details": {"source": "sim"},
        },
    ]
    envs = [{"env_id": "heldout-0000", "physics": {"friction": 0.5}}]

    _apply_reference_adapter_heldout_gate(
        per_env,
        envs,
        inner_evidence={
            "trainer_source": "reference",
            "iterations": [{"sample_vlm_eval": {"score": 0.8}}],
        },
        threshold=0.75,
    )

    assert per_env[0]["success"] is False
    assert per_env[0]["score"] == 0.11
    assert per_env[0]["details"]["sim_success"] is False
    assert per_env[0]["details"]["sim_score"] == 0.11
    assert per_env[0]["details"]["reference_adapter_score"] >= 0.75
    assert per_env[0]["details"]["reference_adapter_would_pass"] is True


def test_signal_diversity_report_flags_degenerate_batch() -> None:
    degenerate = [_signal(1.0, 0.5) for _ in range(6)]

    report = _signal_diversity_report(degenerate)

    assert report["total_rollouts"] == 6
    assert report["distinct_scores"] == 1
    assert report["coherent"] is False
    assert report["degenerate"] is True


def test_signal_diversity_report_accepts_varied_batch() -> None:
    varied = [_signal(0.2, -0.3), _signal(0.6, 0.1), _signal(0.9, 0.7)]

    report = _signal_diversity_report(varied)

    assert report["distinct_scores"] == 3
    assert report["distinct_mean_rewards"] == 3
    assert report["coherent"] is True
    assert report["degenerate"] is False


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        (
            "npa-cosmos3-reason:cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
            "IfNotPresent",
        ),
        (
            "npa-loop-eval:cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z",
            "IfNotPresent",
        ),
        ("npa-loop-eval:0.1.1", "IfNotPresent"),
        ("registry.example/team/npa-loop-eval:0.1.1", "IfNotPresent"),
        ("npa-loop-eval@sha256:" + "a" * 64, "IfNotPresent"),
    ],
)
def test_image_pull_policy(image: str, expected: str) -> None:
    assert _image_pull_policy(image) == expected


def test_image_pull_policy_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_IMAGE_PULL_POLICY", "Never")
    assert (
        _image_pull_policy(
            "npa-cosmos3-reason:cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
        )
        == "Never"
    )


def test_component_pod_info_captures_image_digests() -> None:
    from npa.workflows.sim2real.engine import _pod_info_from_snapshot
    from npa.workflows.sim2real.k8s_client import (
        ContainerSnapshot,
        JobSnapshot,
        KueueAdmission,
        PodSnapshot,
    )

    digest = "registry.example/npa-cosmos3-reason@sha256:" + "b" * 64
    container = ContainerSnapshot(
        name="component",
        image="npa-cosmos3-reason:exact",
        image_id=digest,
        restart_count=0,
        terminated_reason="Completed",
        exit_code=0,
    )
    pod = PodSnapshot(
        name="pod-1",
        uid="pod-uid",
        owner_uid="job-uid",
        phase="Succeeded",
        node_name="node-1",
        deletion_timestamp="",
        scheduled_status="True",
        scheduled_reason="",
        resource_requests={"nvidia.com/gpu": "1"},
        containers=(container,),
    )
    snapshot = JobSnapshot(
        name="job-1",
        namespace="default",
        uid="job-uid",
        resource_version="1",
        state="complete",
        active=0,
        succeeded=1,
        failed=0,
        deleting=False,
        condition_type="Complete",
        condition_reason="CompletionsReached",
        condition_message="",
        pods=(pod,),
        kueue=KueueAdmission(workload_name="job-1-workload", admitted=True),
    )
    info = _pod_info_from_snapshot(snapshot)

    assert info["image_digests"] == [digest]
    assert info["container_statuses"][0]["image_id"] == digest
