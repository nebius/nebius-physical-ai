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
    assert "Workbench Chat" in source
    assert "NEBIUS_TOKEN_FACTORY_KEY" in source
    assert "llm.env" in source
    assert "renderInlineMarkdownLite" in source
    assert "showThinkingBubble" in source
    assert "thinking-dots" in source
    assert "font-family: Inter, system-ui" in source
    assert "font-family: monospace" not in source
    assert "quick-pill" in source
    assert "--brand: #5e43f3;" in source
    assert "--sidebar: #1e1f22;" in source
    assert "markdownLiteHtml" in source


def test_bootstrap_embeds_cameras_panel() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "cameras-panel" in source
    assert "Preview in Rerun" in source
    assert "cameraCards" in source
    assert '@app.get("/sim-assets/cameras")' in source
    assert '@app.post("/sim-viz/camera-preview")' in source
    assert "world/cameras/" in source
    assert "The **Cameras** panel is the center column below chat" in source
    assert "stock_workspace" in source
    assert "stock_ee_mounted" in source
    assert "frustumSvg" in source


def test_bootstrap_stock_camera_defaults_match_scene_assets() -> None:
    from npa.cli import agent as agent_module
    from npa.genesis.scene_assets import (
        CAMERA_PLACEMENT_STOCK_EE_MOUNTED,
        CAMERA_PLACEMENT_STOCK_WORKSPACE,
        DEFAULT_CAMERA_NAMES,
    )

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    for name in DEFAULT_CAMERA_NAMES:
        assert f'"name": "{name}"' in source
    assert CAMERA_PLACEMENT_STOCK_WORKSPACE in source
    assert CAMERA_PLACEMENT_STOCK_EE_MOUNTED in source
    assert '"pos": [1.0, 0.0, 0.8]' in source
    assert '"pos": [0.4, 0.0, 0.4]' in source


def test_bootstrap_embeds_franka_rerun_ux() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert '@app.post("/sim-viz/load-franka-demo")' in source
    assert "_wire_franka_demo" in source
    assert "_generate_franka_demo_rrd" in source
    assert "Load Franka in Rerun" in source
    assert "Open in Rerun" in source
    assert "robotPreset" in source
    assert "/rerun/?url=%2Fapi%2Fsim-viz%2Frrd" in source
    assert '"/rerun/?url=/api/sim-viz/rrd&camera=' in source
    assert "renderAssetsSummary" in source
    assert "selectionPayloadFromUi" in source


def test_bootstrap_ui_fetch_uses_credentials_include() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert 'credentials: "include"' in source
    assert 'credentials: "same-origin"' not in source
    assert "setChatBusy(true)" in source
    assert "setChatBusy(false)" in source
    assert "btn.disabled = Boolean(isBusy);" in source
    assert "input.disabled = Boolean(isBusy);" in source
    assert "JSON.stringify(assets.selection" not in source


def test_bootstrap_system_prompt_no_localhost() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "Never suggest localhost" in source
    assert "Load Franka in Rerun" in source
    assert "/api/sim-viz/load-franka-demo" in source
    assert "localhost:8080" not in source.split("_agent_system_prompt")[1].split("return")[0]


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
            return _Resp(
                {
                    "cameras": [
                        {"name": "workspace", "placement": "stock_workspace", "fov": 60.0},
                        {"name": "wrist", "placement": "stock_ee_mounted", "fov": 90.0},
                    ],
                    "selected": ["workspace"],
                }
            )
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
