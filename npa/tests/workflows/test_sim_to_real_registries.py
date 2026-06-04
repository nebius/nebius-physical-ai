from __future__ import annotations

from pathlib import Path

import pytest

from npa.workflows.eval_backends import EvalMetric, RolloutContext, evaluate_backend, get_eval_backend
from npa.workflows.feedback import (
    ByoContainerFeedbackSource,
    FeedbackPayload,
    FeedbackRequest,
    FeedbackSourceError,
    FeedbackType,
    adapt_feedback_to_training_signal,
    collect_feedback,
)


def test_eval_backend_registry_selects_canonical_and_compatibility_aliases(tmp_path: Path) -> None:
    context = RolloutContext(
        rollout_path=tmp_path,
        task="pick",
        sim_backend="genesis",
        state={"pc_success": True},
        metrics={"action_mse": 3.0, "vlm_score": 0.7},
    )

    assert get_eval_backend("state-success").name == "state-success"
    assert get_eval_backend("genesis").name == "state-success"

    state_metric, state_status = evaluate_backend(
        "state-success",
        checkpoint_uri="s3://bucket/checkpoint/",
        context=context,
        threshold=0.75,
    )
    heldout_metric, _ = evaluate_backend(
        "heldout-metrics",
        checkpoint_uri="s3://bucket/checkpoint/",
        context=context,
        threshold=0.2,
    )
    vlm_metric, vlm_status = evaluate_backend(
        "vlm-frames",
        checkpoint_uri="s3://bucket/checkpoint/",
        context=context,
        threshold=0.5,
    )

    assert state_metric.score == 1.0
    assert state_status.name == "state_success_eval"
    assert heldout_metric.score == 0.25
    assert vlm_metric.score == 0.7
    assert vlm_status.tier == "PARTIAL"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (FeedbackPayload("test", FeedbackType.SCALAR, 0.8, score=0.8, success=True), {"scalar_reward": 0.8}),
        (
            FeedbackPayload("test", FeedbackType.DENSE_PER_STEP, [0.2, 0.6, 1.0], success=True),
            {"scalar_reward": 0.6, "dense_rewards": [0.2, 0.6, 1.0]},
        ),
        (FeedbackPayload("test", FeedbackType.PASS_FAIL, True, success=True), {"scalar_reward": 1.0}),
        (
            FeedbackPayload(
                "test",
                FeedbackType.CRITIQUE,
                {"score": 0.9, "critique": "stable"},
                score=0.9,
                success=True,
            ),
            {"scalar_reward": 0.9, "natural_language_critique": "stable"},
        ),
        (
            FeedbackPayload(
                "test",
                FeedbackType.PREFERENCE,
                {"chosen": "candidate", "rejected": "baseline", "score": 0.7},
                score=0.7,
                success=True,
            ),
            {"scalar_reward": 0.7, "preference": {"chosen": "candidate", "rejected": "baseline", "score": 0.7}},
        ),
    ],
)
def test_feedback_type_adapters_emit_training_signal(payload: FeedbackPayload, expected: dict) -> None:
    signal = adapt_feedback_to_training_signal(payload)

    for key, value in expected.items():
        assert signal[key] == value
    assert signal["schema"] == "npa.sim_to_real.training_signal.v1"
    assert signal["source"] == "test"
    assert signal["feedback_type"] == payload.feedback_type.value


def test_sim_env_feedback_source_adapts_eval_metric(tmp_path: Path) -> None:
    request = FeedbackRequest(
        rollout_path=tmp_path,
        output_path=tmp_path / "feedback",
        task="pick",
        checkpoint_uri="s3://bucket/checkpoint/",
        threshold=0.8,
        feedback_type=FeedbackType.PASS_FAIL,
        eval_metric=EvalMetric(name="state-success", score=0.9, passed=True),
    )

    payload, status = collect_feedback("sim-env", request)
    signal = adapt_feedback_to_training_signal(payload)

    assert payload.value is True
    assert status.name == "sim_env_feedback"
    assert signal["scalar_reward"] == 1.0


def test_byo_container_feedback_source_dispatches_provided_rollout_http(tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_post(url: str, payload: dict) -> dict:
        calls.append((url, payload))
        return {
            "feedback_type": "scalar",
            "value": 0.77,
            "score": 0.77,
            "success": True,
            "rationale": "accepted",
        }

    source = ByoContainerFeedbackSource(http_post=fake_post)
    payload, status = source.collect(
        FeedbackRequest(
            rollout_path=tmp_path / "rollout",
            output_path=tmp_path / "feedback",
            task="pick",
            checkpoint_uri="s3://bucket/checkpoint/",
            threshold=0.75,
            feedback_type=FeedbackType.SCALAR,
            byo_endpoint_url="https://feedback.invalid/score",
            byo_mode="provided-rollout",
        )
    )

    assert status.name == "byo_container_feedback"
    assert payload.score == 0.77
    assert calls[0][0] == "https://feedback.invalid/score"
    assert calls[0][1]["mode"] == "provided-rollout"
    assert calls[0][1]["rollout_path"].endswith("rollout")


def test_byo_container_feedback_source_dispatches_self_rollout_cli(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_command(command: list[str], payload: dict) -> dict:
        calls.append((command, payload))
        return {
            "feedback_type": "pass-fail",
            "value": True,
            "success": True,
            "rationale": "self rollout passed",
        }

    source = ByoContainerFeedbackSource(command_runner=fake_command)
    payload, status = source.collect(
        FeedbackRequest(
            rollout_path=tmp_path / "rollout",
            output_path=tmp_path / "feedback",
            task="pick",
            checkpoint_uri="s3://bucket/checkpoint/",
            threshold=0.75,
            feedback_type=FeedbackType.PASS_FAIL,
            byo_command="feedback-cli --json",
            byo_mode="self-rollout",
        )
    )

    assert status.evidence.endswith("self-rollout mode.")
    assert payload.value is True
    assert calls[0][0] == ["feedback-cli", "--json"]
    assert "rollout_path" not in calls[0][1]


def test_byo_container_feedback_source_rejects_mismatched_declared_type(tmp_path: Path) -> None:
    source = ByoContainerFeedbackSource(
        http_post=lambda _url, _payload: {"feedback_type": "critique", "value": {"score": 0.5}}
    )

    with pytest.raises(FeedbackSourceError, match="expected 'scalar'"):
        source.collect(
            FeedbackRequest(
                rollout_path=tmp_path / "rollout",
                output_path=tmp_path / "feedback",
                task="pick",
                checkpoint_uri="s3://bucket/checkpoint/",
                threshold=0.75,
                feedback_type=FeedbackType.SCALAR,
                byo_endpoint_url="https://feedback.invalid/score",
            )
        )
