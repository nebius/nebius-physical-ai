from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from npa.cli.agent import app

runner = CliRunner()


def test_agent_help_smoke() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "deploy" in result.output
    assert "bootstrap" in result.output
    assert "verify-live" in result.output


def test_bootstrap_embeds_chat_endpoint() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert '@app.post("/chat")' in source
    assert "Workbench chat" in source
    assert "NEBIUS_TOKEN_FACTORY_KEY" in source
    assert "llm.env" in source


def test_resolve_deploy_llm_credentials_reads_credentials(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda: type("Creds", (), {"token_factory_api_key": "tf-test-key"})(),
    )
    from npa.cli.agent import _resolve_deploy_llm_credentials

    key, model = _resolve_deploy_llm_credentials()
    assert key == "tf-test-key"
    assert model == "nvidia/Cosmos3-Super-Reasoner"


def test_agent_status_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "8.8.8.8",
            "agent_url": "http://203.0.113.50:8088/",
            "rerun_url": "http://203.0.113.50:8088/rerun/",
            "sim_viz_url": "http://203.0.113.50:8088/rerun/",
            "sim_assets_url": "http://203.0.113.50:8088/",
            "cameras_api_url": "http://203.0.113.50:8088/api/sim-assets/cameras",
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
    assert payload["sim_viz_url"].endswith("/rerun/")
    assert payload["sim_assets_url"].endswith(":8088/")
    assert payload["cameras_api_url"].endswith("/api/sim-assets/cameras")


def test_verify_live_runs_pytests(monkeypatch) -> None:
    class _Resp:
        def __init__(self, payload: dict[str, object], *, status_code: int = 200) -> None:
            self.status_code = status_code
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
            "public_ip": "8.8.8.8",
            "region": "us-central1",
            "agent_url": "http://203.0.113.50:8088/",
            "rerun_url": "http://203.0.113.50:8088/rerun/",
            "sim_viz_url": "http://203.0.113.50:8088/rerun/",
            "sim_assets_url": "http://203.0.113.50:8088/",
            "cameras_api_url": "http://203.0.113.50:8088/api/sim-assets/cameras",
            "auth_secret_path": "/tmp/agent-auth",
        },
    )
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _: ("npa", "secret"))
    monkeypatch.setattr("npa.cli.agent._health", lambda *_args, **_kwargs: (True, 200))
    def _fake_http_get(url, *_args, **_kwargs):
        url_s = str(url)
        if str(url).endswith("/api/tools"):
            return _Resp({"tool_refs": [f"tool.{idx}" for idx in range(19)]})
        if url_s.endswith("/api/sim-assets"):
            return _Resp({"scene_spec": {"schema": "x"}, "robot_spec": {"schema": "y"}})
        if url_s.endswith("/api/sim-assets/cameras"):
            return _Resp({"cameras": [{"name": "workspace"}]})
        if url_s.endswith("/api/sim-assets/selection"):
            return _Resp({"scene_spec_uri": "stock://scene/default"})
        return _Resp({"ok": True, "tool_ref": "tool.0", "argv_template": ["echo", "ok"]})

    def _fake_http_post(url, *_args, **_kwargs):
        url_s = str(url)
        if url_s.endswith("/api/sim-assets/selection"):
            return _Resp({"ok": True, "selection": {"scene_spec_uri": "stock://scene/default"}})
        if url_s.endswith("/api/workflows/sim2real/submit"):
            return _Resp({"ok": True, "run_id": "agent-run-123"})
        return _Resp({"ok": True})

    monkeypatch.setattr("npa.cli.agent.httpx.get", _fake_http_get)
    monkeypatch.setattr("npa.cli.agent.httpx.post", _fake_http_post)
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
