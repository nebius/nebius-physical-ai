from __future__ import annotations

import json

from typer.testing import CliRunner

from npa.cli.agent import app

runner = CliRunner()


def test_agent_help_smoke() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "deploy" in result.output
    assert "verify-live" in result.output


def test_agent_status_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "203.0.113.50",
            "agent_url": "http://203.0.113.50:8088/",
            "rerun_url": "http://203.0.113.50:8088/rerun/",
            "auth_secret_path": "/tmp/agent-auth",
            "llm": {"provider": "token_factory", "model": "nvidia/Cosmos3-Super-Reasoner"},
        },
    )
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _: ("npa", "secret"))
    monkeypatch.setattr(
        "npa.cli.agent._health",
        lambda *_args, **_kwargs: (True, 200),
    )

    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["health"] is True
    assert payload["ui_status_code"] == 200
    assert payload["rerun_status_code"] == 200


def test_verify_live_runs_pytests(monkeypatch) -> None:
    class _Resp:
        def __init__(self, payload: dict[str, object]) -> None:
            self.status_code = 200
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    class _Proc:
        def __init__(self, code: int = 0) -> None:
            self.returncode = code

    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "203.0.113.50",
            "region": "us-central1",
            "agent_url": "http://203.0.113.50:8088/",
            "rerun_url": "http://203.0.113.50:8088/rerun/",
            "auth_secret_path": "/tmp/agent-auth",
        },
    )
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _: ("npa", "secret"))
    monkeypatch.setattr("npa.cli.agent._health", lambda *_args, **_kwargs: (True, 200))
    def _fake_http_get(url, *_args, **_kwargs):
        if str(url).endswith("/api/tools"):
            return _Resp({"tool_refs": [f"tool.{idx}" for idx in range(19)]})
        return _Resp({"ok": True, "tool_ref": "tool.0", "argv_template": ["echo", "ok"]})

    monkeypatch.setattr("npa.cli.agent.httpx.get", _fake_http_get)
    calls: list[list[str]] = []

    def _fake_run(args, **_kwargs):
        calls.append(list(args))
        return _Proc(0)

    monkeypatch.setattr("npa.cli.agent.subprocess.run", _fake_run)

    result = runner.invoke(app, ["verify-live"])
    assert result.exit_code == 0, result.output
    assert "verify-live: ok" in result.output
    assert calls == [
        ["npa/.venv/bin/python", "-m", "pytest", "npa/tests/cli/test_agent.py", "-q"],
        ["npa/.venv/bin/python", "-m", "pytest", "npa/tests/e2e/test_agent_live.py", "-q"],
    ]
