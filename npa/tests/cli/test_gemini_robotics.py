"""Unit tests for the Gemini Robotics API-backed toolRef (issue #503).

All HTTP is mocked through httpx.MockTransport — no live API calls, no key.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
from typer.testing import CliRunner

from npa.clients.gemini_robotics import (
    API_KEY_ENV,
    BASE_URL_ENV,
    PROVISIONAL_API_BASE_URL,
    GeminiRoboticsClient,
    GeminiRoboticsConfig,
    GeminiRoboticsError,
    resolve_config,
)


def _client(handler, **kwargs) -> GeminiRoboticsClient:
    config = GeminiRoboticsConfig(PROVISIONAL_API_BASE_URL, "test-key", 5.0)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return GeminiRoboticsClient(config, http_client=http, **kwargs)


def _use_missing_credentials_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Keep missing-key tests independent of the operator's saved credentials."""
    import npa.clients.credentials as creds_mod

    monkeypatch.setattr(creds_mod, "CREDENTIALS_PATH", tmp_path / "missing.yaml")


def test_resolve_config_missing_key_fails_closed() -> None:
    with pytest.raises(GeminiRoboticsError, match=API_KEY_ENV):
        resolve_config(environ={})


def test_registered_cli_help_exposes_durable_handoff_flags() -> None:
    from npa.cli.main import app

    runner = CliRunner()
    plan = runner.invoke(app, ["workbench", "gemini-robotics", "plan", "--help"])
    eval_result = runner.invoke(app, ["workbench", "gemini-robotics", "eval", "--help"])
    assert plan.exit_code == 0, plan.output
    assert "--output-path" in plan.output
    assert "s3://" in plan.output
    assert eval_result.exit_code == 0, eval_result.output
    assert "--input-path" in eval_result.output
    assert "--output-path" in eval_result.output


def test_resolve_config_uses_env_key() -> None:
    config = resolve_config(
        environ={API_KEY_ENV: "  secret  ", BASE_URL_ENV: "https://example.test"}
    )
    assert config.api_key == "secret"
    assert config.base_url == "https://example.test"


def test_resolve_config_falls_back_to_credentials_file(tmp_path, monkeypatch) -> None:
    """tokens.GOOGLE_API_KEY in ~/.npa/credentials.yaml is used when env is absent."""
    import npa.clients.credentials as creds_mod

    creds_file = tmp_path / "credentials.yaml"
    creds_file.write_text("tokens:\n  GOOGLE_API_KEY: file-key\n", encoding="utf-8")
    monkeypatch.setattr(creds_mod, "CREDENTIALS_PATH", creds_file)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    config = resolve_config(base_url="https://example.test")
    assert config.api_key == "file-key"
    assert API_KEY_ENV not in os.environ


def test_resolve_config_env_beats_credentials_file(tmp_path, monkeypatch) -> None:
    """An exported GOOGLE_API_KEY takes precedence over the credentials file."""
    import npa.clients.credentials as creds_mod

    creds_file = tmp_path / "credentials.yaml"
    creds_file.write_text("tokens:\n  GOOGLE_API_KEY: file-key\n", encoding="utf-8")
    monkeypatch.setattr(creds_mod, "CREDENTIALS_PATH", creds_file)
    monkeypatch.setenv(API_KEY_ENV, "env-key")
    config = resolve_config(base_url="https://example.test")
    assert config.api_key == "env-key"


def test_resolve_config_hides_malformed_saved_credential_details(
    tmp_path, monkeypatch
) -> None:
    """A malformed saved store fails as a typed error without echoing its contents."""
    import npa.clients.credentials as creds_mod

    creds_file = tmp_path / "credentials.yaml"
    creds_file.write_text("tokens: [not-valid", encoding="utf-8")
    monkeypatch.setattr(creds_mod, "CREDENTIALS_PATH", creds_file)
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    with pytest.raises(GeminiRoboticsError, match="Could not read saved") as exc:
        resolve_config(base_url="https://example.test")
    assert "not-valid" not in str(exc.value)


def test_resolve_config_missing_base_url_fails_closed() -> None:
    """The provisional base URL is never used implicitly."""
    with pytest.raises(GeminiRoboticsError, match=BASE_URL_ENV):
        resolve_config(environ={API_KEY_ENV: "secret"})
    with pytest.raises(GeminiRoboticsError, match="api-base-url"):
        resolve_config(base_url="", environ={API_KEY_ENV: "secret"})


@pytest.mark.parametrize("command", ["plan", "eval"])
@pytest.mark.parametrize("missing", ["api_key", "api_base_url", "model"])
def test_cli_rejects_missing_config_before_storage_or_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path, command: str, missing: str
) -> None:
    """CLI preflight rejects incomplete Gemini settings before side effects."""
    from npa.cli.workbench import gemini_robotics as cli
    from npa.clients.storage import StorageClient

    attempts = {
        "storage_construct": 0,
        "storage_download": 0,
        "storage_read": 0,
        "storage_write": 0,
        "http": 0,
    }

    class StorageProbe:
        def download_file(self, *_args: object) -> None:
            attempts["storage_download"] += 1

        def read_bytes_with_etag(self, *_args: object) -> None:
            attempts["storage_read"] += 1
            return None

        def put_bytes_conditional(self, *_args: object, **_kwargs: object) -> str:
            attempts["storage_write"] += 1
            return "unexpected-write"

    def storage_factory(_cls: type[StorageClient]) -> StorageProbe:
        attempts["storage_construct"] += 1
        return StorageProbe()

    def http_request(*_args: object, **_kwargs: object) -> httpx.Response:
        attempts["http"] += 1
        raise AssertionError("incomplete config must not attempt HTTP")

    monkeypatch.setattr(StorageClient, "from_environment", classmethod(storage_factory))
    monkeypatch.setattr(httpx.Client, "request", http_request)
    _use_missing_credentials_file(monkeypatch, tmp_path)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    if missing != "api_key":
        monkeypatch.setenv(API_KEY_ENV, "test-key")

    args = [command]
    if command == "plan":
        args.append("inspect the scene")
    else:
        args.extend(["--input-path", "s3://example-bucket/inputs/eval.json"])
    args.extend(
        [
            "--model",
            "" if missing == "model" else "operator-selected-model",
            "--output-path",
            "s3://example-bucket/outputs/receipt.json",
        ]
    )
    if missing != "api_base_url":
        args.extend(["--api-base-url", "https://provider.example.invalid"])

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code != 0, result.output
    assert attempts == {
        "storage_construct": 0,
        "storage_download": 0,
        "storage_read": 0,
        "storage_write": 0,
        "http": 0,
    }


def test_provisional_constants_are_documented_as_unvalidated() -> None:
    assert PROVISIONAL_API_BASE_URL.startswith("https://")
    # The provisional guesses exist; entry points must not use them implicitly.
    with pytest.raises(GeminiRoboticsError):
        resolve_config(environ={API_KEY_ENV: "secret"})


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
        f"{PROVISIONAL_API_BASE_URL}/v1beta/models/test-model:generateContent"
    )
    assert seen["headers"]["x-goog-api-key"] == "test-key"
    body = seen["body"]
    assert "systemInstruction" in body
    assert "tools" not in body
    assert body["contents"][0]["parts"][0] == {"text": "pick up the cup"}
    assert result.text == "1. Approach the cup.\n2. Grasp gently."
    assert result.model_function_calls == [
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
        _client(handler).plan(task="do something", model="test-model")


def test_plan_forbidden_failure_mentions_key_env() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"error": {"message": "denied"}}, request=request
        )

    with pytest.raises(GeminiRoboticsError, match=API_KEY_ENV):
        _client(handler).plan(task="do something", model="test-model")


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

    result = _client(handler, sleeper=sleeps.append).plan(task="x", model="test-model")
    assert result.text == "plan"
    assert calls == 3
    assert sleeps == [1.0, 2.0]


def test_plan_server_error_includes_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}}, request=request)

    # 500 is retryable; exhaust attempts then surface the error with detail.
    sleeps: list[float] = []
    with pytest.raises(GeminiRoboticsError, match="boom"):
        _client(handler, sleeper=sleeps.append).plan(task="x", model="test-model")
    assert len(sleeps) == 3


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

    result = _client(handler).eval_plan(
        plan_text="plan", rubric="rubric", model="test-model"
    )
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
        _client(handler).eval_plan(
            plan_text="plan", rubric="rubric", model="test-model"
        )


def test_workbench_surface_is_pipeline_first() -> None:
    """The SDK surface re-exports the pipeline stages directly.

    New tools must not route through ``npa._sdk.make_cli_wrapper``: the
    workbench package is the primary surface and the CLI is a thin client.
    """
    from npa.workbench import gemini_robotics
    from npa.workflows.byof import gemini_robotics_pipeline as pipe

    assert gemini_robotics.plan is pipe.run_er_planning_stage
    assert gemini_robotics.eval is pipe.run_eval_stage
    assert gemini_robotics.run_er_planning_stage is pipe.run_er_planning_stage
    assert gemini_robotics.run_eval_stage is pipe.run_eval_stage
    assert not any(
        hasattr(getattr(gemini_robotics, name), "__npa_cli_module__")
        for name in ("plan", "eval")
    )
    assert set(gemini_robotics.__all__) == {
        "plan",
        "eval",
        "run_er_planning_stage",
        "run_eval_stage",
    }


def test_cli_and_pipeline_share_the_client_core() -> None:
    """The workflow must use the core client without importing the CLI adapter."""
    import inspect

    from npa.cli.workbench import gemini_robotics as cli
    from npa.clients import gemini_robotics as client_core
    from npa.workflows.byof import gemini_robotics_pipeline as pipe

    assert cli.GeminiRoboticsClient is client_core.GeminiRoboticsClient
    assert pipe.GeminiRoboticsClient is client_core.GeminiRoboticsClient
    assert inspect.getsourcefile(client_core.GeminiRoboticsClient) == inspect.getfile(
        client_core
    )
    assert "npa.cli.workbench.gemini_robotics" not in inspect.getsource(pipe)


def test_workbench_plan_runs_pipeline_stage(tmp_path) -> None:
    from npa.clients.gemini_robotics import PlanResult
    from npa.workbench import gemini_robotics
    from npa.workflows.byof.gemini_robotics_pipeline import (
        GeminiRoboticsPipelineConfig,
    )

    class FakeClient:
        def plan(self, **kwargs):
            return PlanResult(
                text="1. Approach.\n2. Grasp.",
                model_function_calls=[],
                model=kwargs.get("model", ""),
                finish_reason="STOP",
            )

    class FakeStorage:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def put_bytes_conditional(self, payload, uri, *, if_none_match, content_type):
            assert if_none_match is True
            self.objects[uri] = payload
            return "etag"

        def read_bytes_with_etag(self, uri):
            return None

    storage = FakeStorage()
    config = GeminiRoboticsPipelineConfig(
        task="pick up the cup",
        output_path="s3://example-bucket/receipts/plan.json",
        model="operator-selected-model",
    )
    receipt = gemini_robotics.plan(config, client=FakeClient(), storage=storage)
    assert receipt["task"] == "pick up the cup"
    assert receipt["plan_text"] == "1. Approach.\n2. Grasp."
    assert receipt["artifact_path"] in storage.objects
