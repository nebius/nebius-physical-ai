"""Verify model-enforcement disclosure across real and synthetic result paths."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
from PIL import Image

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
        json.loads((tmp_path / "out" / "task_success_report.json").read_text())[
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
