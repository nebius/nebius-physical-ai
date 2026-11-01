from __future__ import annotations

from pathlib import Path

import pytest

from npa.workbench.lerobot.policy_container import LeRobotEvalResult
from npa.workflows.eval_backends import (
    EvalMetric,
    RolloutContext,
    evaluate_backend,
    get_eval_backend,
)
from npa.workflows.feedback import (
    ByoContainerFeedbackSource,
    FeedbackPayload,
    FeedbackRequest,
    FeedbackSourceError,
    FeedbackType,
    adapt_feedback_to_training_signal,
    collect_feedback,
    parse_source_payload,
)
from npa.workflows.sim_to_real import (
    SimToRealError,
    feedback_to_training_signal,
    parse_feedback_result,
)
from npa.workflows.sim2real import legacy_components, legacy_heldout
from npa.workflows.sim2real.models import Sim2RealLoopConfig, Sim2RealLoopError


def _legacy_normalizer_config() -> Sim2RealLoopConfig:
    image = "registry.example.invalid/workbench@sha256:" + "a" * 64
    return Sim2RealLoopConfig(
        run_id="strict-feedback",
        threshold=0.5,
        augment_image=image,
        envgen_image=image,
        policy_image=image,
        trainer_image=image,
        vlm_image=image,
        eval_image=image,
        isaac_image=image,
    )


def test_eval_backend_registry_selects_canonical_and_compatibility_aliases(
    tmp_path: Path,
) -> None:
    context = RolloutContext(
        rollout_path=tmp_path,
        task="pick",
        sim_backend="genesis",
        state={"pc_success": True},
        metrics={"action_mse": 3.0, "vlm_score": 0.7},
    )

    assert get_eval_backend("state-success").name == "state-success"
    assert get_eval_backend("genesis").name == "state-success"
    assert get_eval_backend("pusht").name == "state-success"
    assert get_eval_backend("lerobot-eval").name == "state-success"

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


def test_state_success_backend_adapts_lerobot_eval_pc_success(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.workflows import eval_backends as eval_backends_module

    eval_info = tmp_path / "eval" / "eval_info.json"

    def fake_run_lerobot_eval(**kwargs):
        assert kwargs["checkpoint_path"] == tmp_path / "checkpoint"
        assert kwargs["env_type"] == "pusht"
        eval_info.parent.mkdir(parents=True)
        eval_info.write_text("{}", encoding="utf-8")
        return LeRobotEvalResult(
            status="success",
            backend="pusht",
            command=["lerobot-eval"],
            output_dir=str(eval_info.parent),
            eval_info_path=str(eval_info),
            score=0.83,
            metric_name="pc_success",
            pc_success=0.83,
            avg_sum_reward=0.83,
            avg_max_reward=0.9,
            n_episodes=4,
            log_path=str(tmp_path / "eval.log"),
            duration_seconds=1.0,
            exit_code=0,
            raw_metrics={"overall": {"pc_success": 0.83}},
        )

    monkeypatch.setattr(eval_backends_module, "run_lerobot_eval", fake_run_lerobot_eval)

    metric, status = evaluate_backend(
        "lerobot-eval",
        checkpoint_uri=str(tmp_path / "checkpoint"),
        context=RolloutContext(
            rollout_path=tmp_path / "eval",
            task="pick",
            sim_backend="genesis",
            options={
                "lerobot_eval": {
                    "output_dir": tmp_path / "eval",
                    "env_type": "pusht",
                    "episodes": 4,
                    "device": "cuda",
                }
            },
        ),
        threshold=0.75,
    )

    assert metric.name == "state-success"
    assert metric.score == 0.83
    assert metric.passed is True
    assert metric.metadata["adapter"] == "lerobot-eval"
    assert metric.metadata["pc_success"] == 0.83
    assert status.tier == "WORKS"
    assert status.artifacts["eval_info"] == str(eval_info)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            FeedbackPayload("test", FeedbackType.SCALAR, 0.8, score=0.8, success=True),
            {"scalar_reward": 0.8},
        ),
        (
            FeedbackPayload(
                "test", FeedbackType.DENSE_PER_STEP, [0.2, 0.6, 1.0], success=True
            ),
            {"scalar_reward": 0.6, "dense_rewards": [0.2, 0.6, 1.0]},
        ),
        (
            FeedbackPayload("test", FeedbackType.PASS_FAIL, True, success=True),
            {"scalar_reward": 1.0},
        ),
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
            {
                "scalar_reward": 0.7,
                "preference": {
                    "chosen": "candidate",
                    "rejected": "baseline",
                    "score": 0.7,
                },
            },
        ),
    ],
)
def test_feedback_type_adapters_emit_training_signal(
    payload: FeedbackPayload, expected: dict
) -> None:
    signal = adapt_feedback_to_training_signal(payload)

    for key, value in expected.items():
        assert signal[key] == value
    assert signal["schema"] == "npa.sim_to_real.training_signal.v1"
    assert signal["source"] == "test"
    assert signal["feedback_type"] == payload.feedback_type.value


@pytest.mark.parametrize("score", [0, 1, 0.0, 0.2, 0.9, 1.0])
def test_feedback_score_boundaries_preserve_json_numbers(score: int | float) -> None:
    canonical = parse_source_payload(
        {"value": score, "success": False},
        source="byo-container",
        feedback_type=FeedbackType.SCALAR,
    )
    legacy = parse_feedback_result(
        {"success": False, "score": score, "rationale": "failed rollout"}
    )

    assert canonical.score == float(score)
    assert legacy.score == float(score)
    assert feedback_to_training_signal(legacy)["score"] == float(score)


@pytest.mark.parametrize(
    "score",
    [
        True,
        False,
        "0.9",
        None,
        [],
        {},
        float("nan"),
        float("inf"),
        float("-inf"),
        10**400,
    ],
)
def test_feedback_score_boundaries_reject_non_json_numbers(score: object) -> None:
    with pytest.raises(
        FeedbackSourceError, match="feedback score must be a finite JSON number"
    ):
        parse_source_payload(
            {"value": 0.2, "score": score, "success": False},
            source="byo-container",
            feedback_type=FeedbackType.SCALAR,
        )
    with pytest.raises(
        SimToRealError, match="feedback score must be a finite JSON number"
    ):
        parse_feedback_result(
            {"success": False, "score": score, "rationale": "failed rollout"}
        )


@pytest.mark.parametrize(
    ("payload", "feedback_type"),
    [
        ({"value": True}, FeedbackType.SCALAR),
        ({"value": "0.9"}, FeedbackType.SCALAR),
        ({"value": None}, FeedbackType.SCALAR),
        ({"value": [0.2, True]}, FeedbackType.DENSE_PER_STEP),
        ({"value": {"score": "0.9"}}, FeedbackType.CRITIQUE),
        ({"value": {"score": float("nan")}}, FeedbackType.PREFERENCE),
    ],
)
def test_feedback_type_values_reject_coerced_scores(
    payload: dict, feedback_type: FeedbackType
) -> None:
    with pytest.raises(
        FeedbackSourceError, match="feedback score must be a finite JSON number"
    ):
        parse_source_payload(
            {**payload, "success": False},
            source="byo-container",
            feedback_type=feedback_type,
        )


@pytest.mark.parametrize("success", ["false", "no", 0, 1, None, [], {}])
def test_legacy_feedback_parser_rejects_non_boolean_success(success: object) -> None:
    with pytest.raises(SimToRealError, match="feedback success must be a JSON boolean"):
        parse_feedback_result(
            {"success": success, "score": 0.9, "rationale": "failed rollout"}
        )


@pytest.mark.parametrize(
    "score",
    [True, False, "0.9", None, float("nan"), float("inf"), 10**400],
)
def test_legacy_sim2real_normalizers_reject_coerced_scores(score: object) -> None:
    config = _legacy_normalizer_config()
    with pytest.raises(Sim2RealLoopError, match="finite JSON number"):
        legacy_components._normalize_vlm_evaluation(
            {
                "score": score,
                "success": False,
                "per_step": [{"step": 0, "critique_text": "failed rollout"}],
            },
            manifest={},
            rollout_id="rollout-1",
            config=config,
            invocation={},
        )
    with pytest.raises(Sim2RealLoopError, match="finite JSON number"):
        legacy_heldout._normalize_heldout_report(
            {"per_env": [{"env_id": "env-1", "score": score, "success": False}]},
            config=config,
            outer_iteration=1,
            inner_evidence_uri="s3://bucket/inner.json",
            invocation={},
        )


def test_legacy_sim2real_normalizers_preserve_default_and_explicit_success() -> None:
    config = _legacy_normalizer_config()
    inferred = legacy_components._normalize_vlm_evaluation(
        {
            "score": 0.9,
            "per_step": [{"step": 0, "critique_text": "stable rollout"}],
        },
        manifest={},
        rollout_id="rollout-1",
        config=config,
        invocation={},
    )
    report = legacy_heldout._normalize_heldout_report(
        {
            "per_env": [
                {"env_id": "inferred", "score": 0.9},
                {"env_id": "explicit", "score": 0.9, "success": False},
            ]
        },
        config=config,
        outer_iteration=1,
        inner_evidence_uri="s3://bucket/inner.json",
        invocation={},
    )

    assert inferred["success"] is True
    assert [item["success"] for item in report["per_env"]] == [True, False]
    assert report["success_rate"] == 0.5


def test_reference_adapter_diagnostics_reject_coerced_sim_success() -> None:
    with pytest.raises(Sim2RealLoopError, match="success.*JSON boolean"):
        legacy_components._apply_reference_adapter_heldout_gate(
            [{"success": "false", "score": 0.1, "details": {}}],
            [{"env_id": "env-1", "physics": {}}],
            inner_evidence={"trainer_source": "reference"},
            threshold=0.5,
        )


def test_feedback_adapters_reject_manually_constructed_boolean_score() -> None:
    payload = FeedbackPayload(
        "test", FeedbackType.SCALAR, True, score=1.0, success=True
    )

    with pytest.raises(
        FeedbackSourceError, match="feedback score must be a finite JSON number"
    ):
        adapt_feedback_to_training_signal(payload)


@pytest.mark.parametrize("value", ["false", 0, 1, None, [], {}])
def test_pass_fail_adapter_rejects_manually_constructed_non_boolean_value(
    value: object,
) -> None:
    payload = FeedbackPayload(
        "test",
        FeedbackType.PASS_FAIL,
        value,
        score=0.0,
        success=False,
    )

    with pytest.raises(
        FeedbackSourceError,
        match="feedback pass-fail value must be a JSON boolean",
    ):
        adapt_feedback_to_training_signal(payload)


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

    payload, status = collect_feedback("rollout", request)
    signal = adapt_feedback_to_training_signal(payload)

    assert payload.value is True
    assert status.name == "sim_env_feedback"
    assert signal["scalar_reward"] == 1.0


@pytest.mark.parametrize("feedback_type", list(FeedbackType))
def test_sim_env_feedback_without_metric_emits_adaptable_zero_signal(
    tmp_path: Path,
    feedback_type: FeedbackType,
) -> None:
    request = FeedbackRequest(
        rollout_path=tmp_path,
        output_path=tmp_path / "feedback",
        task="pick",
        checkpoint_uri="s3://bucket/checkpoint/",
        threshold=0.8,
        feedback_type=feedback_type,
    )

    payload, status = collect_feedback("rollout", request)
    signal = adapt_feedback_to_training_signal(payload)

    assert status.tier == "SEAM"
    assert signal["scalar_reward"] == 0.0
    assert signal["success"] is False


@pytest.mark.parametrize("feedback_type", list(FeedbackType))
def test_blocked_byo_feedback_emits_adaptable_zero_signal(
    tmp_path: Path,
    feedback_type: FeedbackType,
) -> None:
    source = ByoContainerFeedbackSource()
    request = FeedbackRequest(
        rollout_path=tmp_path / "rollout",
        output_path=tmp_path / "feedback",
        task="pick",
        checkpoint_uri="s3://bucket/checkpoint/",
        threshold=0.75,
        feedback_type=feedback_type,
    )

    payload, status = source.collect(request)
    signal = adapt_feedback_to_training_signal(payload)

    assert status.tier == "BLOCKED"
    assert signal["scalar_reward"] == 0.0
    assert signal["success"] is False


def test_byo_container_feedback_source_dispatches_provided_rollout_http(
    tmp_path: Path,
) -> None:
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


def test_byo_container_feedback_source_dispatches_self_rollout_cli(
    tmp_path: Path,
) -> None:
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


def test_byo_container_feedback_source_rejects_mismatched_declared_type(
    tmp_path: Path,
) -> None:
    source = ByoContainerFeedbackSource(
        http_post=lambda _url, _payload: {
            "feedback_type": "critique",
            "value": {"score": 0.5},
        }
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


@pytest.mark.parametrize(
    ("source_payload", "expected_success"),
    [
        ({"value": 0.8}, True),
        ({"value": 0.2}, False),
        ({"value": 0.2, "success": True}, True),
        ({"value": 0.8, "success": False}, False),
    ],
)
def test_parse_source_payload_preserves_boolean_or_score_based_success(
    source_payload: dict, expected_success: bool
) -> None:
    payload = parse_source_payload(
        source_payload,
        source="byo-container",
        feedback_type=FeedbackType.SCALAR,
    )

    assert payload.success is expected_success


@pytest.mark.parametrize(
    "success", ["false", "true", 0, 1, None, [], [False], {}, {"value": False}]
)
def test_parse_source_payload_rejects_non_boolean_success(success: object) -> None:
    with pytest.raises(
        FeedbackSourceError,
        match="feedback source 'success' must be a JSON boolean",
    ):
        parse_source_payload(
            {"value": 0.9, "success": success},
            source="byo-container",
            feedback_type=FeedbackType.SCALAR,
        )


@pytest.mark.parametrize("route", ["http", "cli"])
@pytest.mark.parametrize(
    "response,parsed_success,training_success",
    [
        ({"value": True, "success": False}, False, False),
        ({"value": False, "success": True}, True, True),
        ({"value": True}, True, True),
        ({"value": False}, False, False),
        ({"value": False, "score": 0.8}, True, False),
        ({"value": True, "score": 0.2}, False, True),
        ({"success": False}, False, False),
    ],
)
def test_byo_pass_fail_training_signal_preserves_explicit_and_omitted_success(
    tmp_path: Path,
    route: str,
    response: dict,
    parsed_success: bool,
    training_success: bool,
) -> None:
    source = ByoContainerFeedbackSource(
        http_post=lambda *_args: response, command_runner=lambda *_args: response
    )
    payload, _status = source.collect(
        FeedbackRequest(
            rollout_path=tmp_path / "rollout",
            output_path=tmp_path / "feedback",
            task="pick",
            checkpoint_uri="s3://bucket/checkpoint/",
            threshold=0.75,
            feedback_type=FeedbackType.PASS_FAIL,
            byo_endpoint_url="https://feedback.invalid/score"
            if route == "http"
            else "",
            byo_command="feedback-cli --json" if route == "cli" else "",
        )
    )
    signal = adapt_feedback_to_training_signal(payload)
    assert payload.success is parsed_success
    assert signal["success"] is training_success
    assert signal["scalar_reward"] == float(training_success)


@pytest.mark.parametrize("route", ["http", "cli"])
@pytest.mark.parametrize("include_success", [False, True])
@pytest.mark.parametrize("include_score", [False, True])
@pytest.mark.parametrize("value", ["false", 0, 1, None, [], {}])
def test_byo_pass_fail_rejects_non_boolean_value_before_training_signal(
    tmp_path: Path,
    route: str,
    include_success: bool,
    include_score: bool,
    value: object,
) -> None:
    response = {"value": value}
    if include_success:
        response["success"] = False
    if include_score:
        response["score"] = 0.2
    source = ByoContainerFeedbackSource(
        http_post=lambda *_args: response,
        command_runner=lambda *_args: response,
    )
    request = FeedbackRequest(
        rollout_path=tmp_path / "rollout",
        output_path=tmp_path / "feedback",
        task="pick",
        checkpoint_uri="s3://bucket/checkpoint/",
        threshold=0.75,
        feedback_type=FeedbackType.PASS_FAIL,
        byo_endpoint_url="https://feedback.invalid/score" if route == "http" else "",
        byo_command="feedback-cli --json" if route == "cli" else "",
    )

    with pytest.raises(
        FeedbackSourceError,
        match="feedback pass-fail value must be a JSON boolean",
    ):
        source.collect(request)


@pytest.mark.parametrize("route", ["http", "cli"])
@pytest.mark.parametrize("success", ["false", 1, None])
def test_byo_source_rejects_malformed_success_before_training_signal(
    tmp_path: Path, route: str, success: object
) -> None:
    source = ByoContainerFeedbackSource(
        http_post=lambda *_args: {"value": True, "success": success},
        command_runner=lambda *_args: {"value": True, "success": success},
    )
    request = FeedbackRequest(
        rollout_path=tmp_path / "rollout",
        output_path=tmp_path / "feedback",
        task="pick",
        checkpoint_uri="s3://bucket/checkpoint/",
        threshold=0.75,
        feedback_type=FeedbackType.PASS_FAIL,
        byo_endpoint_url="https://feedback.invalid/score" if route == "http" else "",
        byo_command="feedback-cli --json" if route == "cli" else "",
    )
    with pytest.raises(FeedbackSourceError, match="'success' must be a JSON boolean"):
        source.collect(request)
