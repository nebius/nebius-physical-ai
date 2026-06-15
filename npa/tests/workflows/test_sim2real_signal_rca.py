"""Regression tests for the sim2real genuine-signal RCA fixes.

Covers the degenerate-signal root causes:
- held-out scores collapsing to a flat ``1.0`` (no gradient),
- cross-rollout diversity diagnostics / anti-hollow gate,
- ``-genuine-`` image pull policy, and
- image-digest provenance captured from the running pod.
"""

from __future__ import annotations

import json
import subprocess

import pytest

import npa.workflows.sim2real_loop as loop_module
from npa.workflows.sim2real_loop import (
    Sim2RealLoopConfig,
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
        {"env_id": "heldout-0000", "score": 0.11, "success": False, "details": {"source": "sim"}},
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

    assert per_env[0]["success"] is True
    assert per_env[0]["details"]["sim_success"] is False
    assert per_env[0]["details"]["sim_score"] == 0.11
    assert per_env[0]["details"]["reference_adapter_score"] >= 0.75


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
        ("npa-cosmos3-reason:3.0.1-genuine-sm120", "Always"),
        ("npa-sim2real-eval:0.1.1-genuine-sm120", "Always"),
        ("npa-sim2real-eval:0.1.1", "IfNotPresent"),
        ("registry.example/team/npa-sim2real-eval:0.1.1", "IfNotPresent"),
        ("npa-sim2real-eval@sha256:" + "a" * 64, "IfNotPresent"),
    ],
)
def test_image_pull_policy(image: str, expected: str) -> None:
    assert _image_pull_policy(image) == expected


def test_image_pull_policy_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_IMAGE_PULL_POLICY", "Never")
    assert _image_pull_policy("npa-cosmos3-reason:3.0.1-genuine-sm120") == "Never"


def test_component_pod_info_captures_image_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "registry.example/npa-cosmos3-reason@sha256:" + "b" * 64
    pod_payload = {
        "items": [
            {
                "metadata": {"name": "pod-1"},
                "spec": {"nodeName": "node-1", "containers": [{"resources": {}}]},
                "status": {
                    "phase": "Succeeded",
                    "containerStatuses": [
                        {
                            "name": "component",
                            "ready": False,
                            "restartCount": 0,
                            "image": "npa-cosmos3-reason:3.0.1-genuine-sm120",
                            "imageID": digest,
                            "state": {"terminated": {"exitCode": 0}},
                        }
                    ],
                },
            }
        ]
    }

    def fake_kubectl(config, args, **kwargs):  # noqa: ANN001 - test stub
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=json.dumps(pod_payload), stderr=""
        )

    monkeypatch.setattr(loop_module, "_kubectl", fake_kubectl)
    config = Sim2RealLoopConfig(run_id="rca-test")

    info = loop_module._component_pod_info(
        config, namespace="default", job_name="job-1"
    )

    assert info["image_digests"] == [digest]
    assert info["container_statuses"][0]["image_id"] == digest
