"""Verify model-enforcement disclosure across real and synthetic result paths."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
from PIL import Image

from npa.clients import token_factory
from npa.workbench import vlm_eval


@pytest.mark.parametrize("backend", ["api", "self-hosted"])
@pytest.mark.parametrize("model", ["MiniMaxAI/MiniMax-M3", "vendor/explicit-alias"])
def test_model_enforcement_follows_actual_backend_profile(monkeypatch, backend, model):
    completion = {
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": '{"success":true,"score":0.9,"rationale":"visible"}'
                },
            }
        ],
    }
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **_kwargs: "synthetic")
    monkeypatch.setattr(
        vlm_eval, "_post_with_readiness_retry", lambda **_kwargs: completion
    )
    result = vlm_eval._call_openai_compatible(
        backend=backend,
        model=model,
        endpoint_url="https://example.test/v1",
        api_key_env="TEST_KEY",
        prompt="Return JSON",
        frames=[],
        timeout_s=120,
    )
    assert result.served_model == model
    assert result.served_model_match_enforced is (backend == "api")


@pytest.mark.parametrize("enforced", [False, True])
def test_disclosure_follows_profile_behavior_not_a_constant(monkeypatch, enforced):
    # A hypothetical profile is a behavioral control, not a relaxation of any
    # current profile. Current advertised and custom-model profiles stay strict.
    profile = token_factory.TokenFactoryChatProfile(require_exact_model=enforced)
    monkeypatch.setattr(token_factory, "token_factory_chat_profile", lambda _: profile)
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **_: "")
    monkeypatch.setattr(
        vlm_eval,
        "_post_with_readiness_retry",
        lambda **_: {
            "model": "vendor/substituted",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"success":true,"score":0.9,"rationale":"control"}',
                    },
                }
            ],
        },
    )
    args = dict(
        backend="api",
        model="vendor/requested",
        endpoint_url="https://example.test/v1",
        api_key_env="TEST_KEY",
        prompt="Return JSON",
        frames=[],
        timeout_s=120,
    )
    if enforced:
        with pytest.raises(
            vlm_eval.VlmEvalError, match="does not match the requested model"
        ):
            vlm_eval._call_openai_compatible(**args)
    else:
        result = vlm_eval._call_openai_compatible(**args)
        assert result.served_model == "vendor/substituted"
        assert result.served_model_match_enforced is False


@pytest.mark.parametrize("backend", ["stub", "api", "self-hosted"])
def test_score_override_never_claims_identity_enforcement(backend, tmp_path):
    result = vlm_eval.evaluate_vlm(
        input_path="unused",
        output_path=str(tmp_path / "result.json"),
        backend=backend,
        score=0.9,
    )
    assert result.evidence is None
    assert result.served_model is None
    assert result.served_model_match_enforced is False


def test_loop_and_benchmark_persist_enforcement(monkeypatch, tmp_path):
    rollout = tmp_path / "rollouts" / "one"
    rollout.mkdir(parents=True)
    Image.new("RGB", (8, 8), "green").save(rollout / "frame.png")
    completion = {
        "model": "MiniMaxAI/MiniMax-M3",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": '{"success":true,"score":0.9,"rationale":"visible"}'
                },
            }
        ],
    }
    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **_kwargs: "synthetic")
    monkeypatch.setattr(
        vlm_eval, "_post_with_readiness_retry", lambda **_kwargs: completion
    )
    report = vlm_eval.evaluate_rollout_set(
        input_path=str(rollout.parent),
        output_path=str(tmp_path / "out"),
        backend="api",
        model="MiniMaxAI/MiniMax-M3",
    )
    row = report["rollouts"][0]
    assert row["requested_model"] == "MiniMaxAI/MiniMax-M3"
    assert row["served_model"] == row["requested_model"]
    assert row["served_model_match_enforced"] is True
    assert (
        json.loads((tmp_path / "out" / vlm_eval.LOOP_REPORT_FILENAME).read_text())[
            "rollouts"
        ]
        == report["rollouts"]
    )
    assert (
        json.loads(Path(row["result_uri"]).read_text())["served_model_match_enforced"]
        is True
    )
    benchmark = vlm_eval.benchmark_vlm_eval(backend="stub")
    assert all(
        case.served_model_match_enforced is False
        and case.served_model is None
        and asdict(case)["requested_model"] == benchmark.best_config.config.model
        for case in benchmark.best_config.results
    )


@pytest.mark.parametrize("backend", ["api", "self-hosted"])
@pytest.mark.parametrize("explicit_model", [False, True])
def test_aggregate_models_match_effective_requests(
    monkeypatch, tmp_path, backend, explicit_model
):
    monkeypatch.setenv(vlm_eval.SELF_HOSTED_MODEL_ENV, "vendor/environment-model")
    effective = (
        "vendor/explicit-model"
        if explicit_model
        else token_factory.DEFAULT_VISION_MODEL
        if backend == "api"
        else "vendor/environment-model"
    )
    requested = "vendor/explicit-model" if explicit_model else vlm_eval.DEFAULT_MODEL
    root = tmp_path / "rollouts"
    items = []
    for name, color, expected in (
        ("positive", "green", True),
        ("negative", "gray", False),
    ):
        directory = root / name
        directory.mkdir(parents=True)
        Image.new("RGB", (8, 8), color).save(directory / "frame.png")
        items.append(
            {"id": name, "rollout": str(directory), "expected_label": expected}
        )
    calls = []

    def post(**kwargs):
        calls.append(kwargs["request"])
        return {
            "model": kwargs["request"]["model"],
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"success":true,"score":0.9,"rationale":"synthetic control"}',
                    },
                }
            ],
        }

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **_: "")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    loop = vlm_eval.evaluate_rollout_set(
        input_path=str(root),
        output_path=str(tmp_path / "out"),
        backend=backend,
        model=requested,
        endpoint_url="https://example.test/v1",
    )
    assert loop["model"] == effective
    assert {row["requested_model"] for row in loop["rollouts"]} == {effective}
    assert {row["served_model"] for row in loop["rollouts"]} == {effective}
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(json.dumps({"items": items}))
    benchmark = vlm_eval.benchmark_vlm_eval(
        dataset=str(manifest),
        backend=backend,
        models=[requested],
        thresholds=[0.8],
        endpoint_url="https://example.test/v1",
    )
    assert benchmark.sweep["models"] == [effective]
    assert benchmark.best_config.config.model == effective
    assert {case.requested_model for case in benchmark.best_config.results} == {
        effective
    }
    assert {case.served_model for case in benchmark.best_config.results} == {effective}
    assert len(calls) == 4
    assert {request["model"] for request in calls} == {effective}


@pytest.mark.parametrize("backend", ["api", "self-hosted"])
def test_benchmark_deduplicates_resolved_models_in_caller_order(
    monkeypatch, tmp_path, backend
):
    monkeypatch.setenv(vlm_eval.SELF_HOSTED_MODEL_ENV, "vendor/environment-model")
    effective = (
        token_factory.DEFAULT_VISION_MODEL
        if backend == "api"
        else "vendor/environment-model"
    )
    Image.new("RGB", (8, 8), "green").save(tmp_path / "frame.png")
    Image.new("RGB", (8, 8), "red").save(tmp_path / "negative.png")
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "positive",
                        "rollout": str(tmp_path / "frame.png"),
                        "expected_label": True,
                    },
                    {
                        "id": "negative",
                        "rollout": str(tmp_path / "negative.png"),
                        "expected_label": False,
                    }
                ]
            }
        )
    )
    calls = []
    all_calls = []

    def post(**kwargs):
        all_calls.append(kwargs["request"])
        # Preserve the original per-configuration order/count oracle while
        # recording both required label classes for every configuration.
        if len(all_calls) % 2 == 1:
            calls.append(kwargs["request"])
        return {
            "model": kwargs["request"]["model"],
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"success":true,"score":0.9,"rationale":"synthetic deduplication control"}'
                    },
                }
            ],
        }

    monkeypatch.setattr(vlm_eval, "_resolve_api_key", lambda **_: "")
    monkeypatch.setattr(vlm_eval, "_post_with_readiness_retry", post)
    report = vlm_eval.benchmark_vlm_eval(
        dataset=str(manifest),
        backend=backend,
        models=[vlm_eval.DEFAULT_MODEL, effective, "vendor/second-model", effective],
        thresholds=[0.8],
        endpoint_url="https://example.test/v1",
    )
    assert report.sweep["models"] == [effective, "vendor/second-model"]
    assert len(report.ranked_configs) == len(calls) == 2
    assert [request["model"] for request in calls] == [effective, "vendor/second-model"]
    assert len(all_calls) == 2 * len(calls) == 4
    assert [request["model"] for request in all_calls] == [
        effective,
        effective,
        "vendor/second-model",
        "vendor/second-model",
    ]
    for config in report.ranked_configs:
        assert [case.expected_label for case in config.results] == [True, False]
        assert [case.score for case in config.results] == [0.9, 0.9]
        assert all(case.predicted_label for case in config.results)


def test_current_hosted_profiles_pin_exact_identity_for_promotion():
    assert token_factory._DEFAULT_CHAT_PROFILE.require_exact_model is True
    assert all(
        profile.require_exact_model is True
        for profile in token_factory._CHAT_PROFILES.values()
    )
