"""Unit tests for the Gemini Robotics API-backed toolRef (issue #503).

All HTTP is mocked through httpx.MockTransport — no live API calls, no key.
"""

from __future__ import annotations

import json

import httpx
import pytest

from npa.cli.gemini_robotics import (
    API_KEY_ENV,
    DEFAULT_API_BASE_URL,
    GeminiRoboticsClient,
    GeminiRoboticsConfig,
    GeminiRoboticsError,
    resolve_config,
)


def _client(handler, **kwargs) -> GeminiRoboticsClient:
    config = GeminiRoboticsConfig(DEFAULT_API_BASE_URL, "test-key", 5.0)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiRoboticsClient(config, http_client=http, **kwargs)


def test_resolve_config_missing_key_fails_closed() -> None:
    with pytest.raises(GeminiRoboticsError, match=API_KEY_ENV):
        resolve_config(environ={})


def test_resolve_config_uses_env_key() -> None:
    config = resolve_config(environ={API_KEY_ENV: "  secret  "})
    assert config.api_key == "secret"
    assert config.base_url == DEFAULT_API_BASE_URL


def test_plan_formats_request_and_parses_response() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"text": "1. Approach the cup.\n2. Grasp gently."},
                                {
                                    "functionCall": {
                                        "name": "check_safety",
                                        "args": {"action": "grasp the cup"},
                                    }
                                },
                            ]
                        },
                    }
                ]
            },
            request=request,
        )

    result = _client(handler).plan(task="pick up the cup", model="test-model")

    assert seen["url"] == (
        f"{DEFAULT_API_BASE_URL}/v1beta/models/test-model:generateContent"
    )
    assert seen["headers"]["x-goog-api-key"] == "test-key"
    body = seen["body"]
    assert "systemInstruction" in body
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "check_safety"
    assert body["contents"][0]["parts"][0] == {"text": "pick up the cup"}
    assert result.text == "1. Approach the cup.\n2. Grasp gently."
    assert result.safety_calls == [
        {"name": "check_safety", "args": {"action": "grasp the cup"}}
    ]
    assert result.model == "test-model"
    assert result.finish_reason == "STOP"


def test_plan_auth_failure_mentions_key_env() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "API key not valid."}},
            request=request,
        )

    with pytest.raises(GeminiRoboticsError, match=API_KEY_ENV):
        _client(handler).plan(task="do something")


def test_plan_forbidden_failure_mentions_key_env() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"error": {"message": "denied"}}, request=request
        )

    with pytest.raises(GeminiRoboticsError, match=API_KEY_ENV):
        _client(handler).plan(task="do something")


def test_plan_retries_transient_errors() -> None:
    statuses = iter([429, 503, 200])
    sleeps: list[float] = []
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        status = next(statuses)
        if status == 200:
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "plan"}]}}]},
                request=request,
            )
        return httpx.Response(
            status, json={"error": {"message": "busy"}}, request=request
        )

    result = _client(handler, sleeper=sleeps.append).plan(task="x")
    assert result.text == "plan"
    assert calls == 3
    assert sleeps == [1.0, 2.0]


def test_plan_server_error_includes_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}}, request=request)

    # 500 is retryable; exhaust attempts then surface the error with detail.
    sleeps: list[float] = []
    with pytest.raises(GeminiRoboticsError, match="boom"):
        _client(handler, sleeper=sleeps.append).plan(task="x")
    assert len(sleeps) == 3


def test_submit_adaptation_posts_tuning_task() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200, json={"name": "tunedModels/abc/operations/op1"}, request=request
        )

    operation = _client(handler).submit_adaptation(
        display_name="adapt-1",
        base_model="base-m",
        examples=[{"input": "in", "output": "out"}],
    )
    assert operation == "tunedModels/abc/operations/op1"
    assert seen["url"] == f"{DEFAULT_API_BASE_URL}/v1beta/tunedModels"
    body = seen["body"]
    assert body["display_name"] == "adapt-1"
    assert body["base_model"] == "models/base-m"
    assert body["tuning_task"]["training_data"]["examples"]["examples"] == [
        {"text_input": "in", "output": "out"}
    ]


def test_wait_for_adaptation_polls_until_done() -> None:
    responses = iter(
        [
            {"done": False, "metadata": {"displayName": "adapt-1"}},
            {"done": True, "response": {"name": "tunedModels/abc"}},
        ]
    )
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        polls += 1
        return httpx.Response(200, json=next(responses), request=request)

    job = _client(handler, sleeper=lambda _: None).wait_for_adaptation(
        "tunedModels/abc/operations/op1", timeout_s=60, poll_interval_s=1
    )
    assert job.done is True
    assert job.tuned_model == "tunedModels/abc"
    assert job.error == ""
    assert polls == 2


def test_wait_for_adaptation_surfaces_job_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"done": True, "error": {"message": "quota exhausted"}},
            request=request,
        )

    with pytest.raises(GeminiRoboticsError, match="quota exhausted"):
        _client(handler, sleeper=lambda _: None).wait_for_adaptation("op")


def test_eval_plan_parses_json_scores() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '```json\n{"scores": {"safety": 9, "efficiency": 7}, '
                                    '"summary": "Solid plan."}\n```'
                                }
                            ]
                        }
                    }
                ]
            },
            request=request,
        )

    result = _client(handler).eval_plan(plan_text="plan", rubric="rubric")
    assert result.scores == {"safety": 9, "efficiency": 7}
    assert result.summary == "Solid plan."
    prompt = seen["body"]["contents"][0]["parts"][0]["text"]
    assert "rubric" in prompt and "plan" in prompt


def test_eval_plan_rejects_non_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "not json"}]}}]},
            request=request,
        )

    with pytest.raises(GeminiRoboticsError, match="JSON"):
        _client(handler).eval_plan(plan_text="plan", rubric="rubric")


def test_workbench_wrappers_point_at_cli_callbacks() -> None:
    from npa.workbench import gemini_robotics

    assert gemini_robotics.plan.__npa_cli_module__ == "npa.cli.gemini_robotics"
    assert gemini_robotics.plan.__npa_cli_callback__ == "plan_cmd"
    assert gemini_robotics.adapt.__npa_cli_callback__ == "adapt_cmd"
    assert gemini_robotics.eval.__npa_cli_callback__ == "eval_cmd"
    assert gemini_robotics.plan.__name__ == "plan"
    assert gemini_robotics.adapt.__name__ == "adapt"
    assert gemini_robotics.eval.__name__ == "eval"


def test_workbench_wrapper_delegates_to_callback(monkeypatch) -> None:
    import npa.cli.gemini_robotics as cli_module
    from npa.workbench import gemini_robotics

    captured: dict = {}

    def fake_plan_cmd(task: str):
        captured["task"] = task
        return {"ok": True}

    monkeypatch.setattr(cli_module, "plan_cmd", fake_plan_cmd)
    assert gemini_robotics.plan(task="pick up the cup") == {"ok": True}
    assert captured["task"] == "pick up the cup"
