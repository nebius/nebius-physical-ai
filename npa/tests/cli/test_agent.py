from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from npa.cli.agent import AGENT_UI_VERSION, app, build_agent_urls

runner = CliRunner()


def test_build_agent_urls_https_default() -> None:
    urls = build_agent_urls("203.0.113.50")
    assert urls["public_url"] == "https://203.0.113.50/"
    assert urls["agent_url"] == urls["public_url"]
    assert urls["rerun_url"] == "https://203.0.113.50/rerun/"
    assert urls["direct_url"] == "http://203.0.113.50:8088/"


def test_build_agent_urls_http_legacy() -> None:
    urls = build_agent_urls("203.0.113.50", public_https=False)
    assert urls["public_url"] == "http://203.0.113.50:8088/"
    assert urls["agent_url"] == urls["public_url"]


def test_bootstrap_enables_public_https_nginx() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "ssl_certificate /etc/nginx/ssl/npa-agent.crt" in source
    assert "DEFAULT_HTTPS_PORT" in source
    assert "Customer URL: use" in source
    assert "--no-public-https" in source


def test_bootstrap_nginx_serves_public_rerun_recording() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "location /rerun/recordings/" in source
    assert "auth_basic off" in source
    assert "alias /opt/npa-agent/recordings/" in source


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
    assert '@app.get("/session")' in source
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
    assert "Secure basic-auth session" in source
    assert "sparkle" in source
    bootstrap_split = f'        const lines = String(text || "").split(/\\r?\\n/);'
    assert "\r" not in bootstrap_split
    assert "\\r?\\n" in bootstrap_split
    assert "restoreSession" in source
    assert "bootPage()" in source
    assert "ensureFrankaRerunLoaded" in source
    assert "setTimeout(() =>" in source
    assert "startPeriodicRefresh" in source
    assert "fetchWithTimeout" in source
    assert "welcome.html" in source
    assert "login-help.html" in source
    assert "/welcome" in source
    assert "_agent_public_login_form_html" in source
    assert 'id="npa-sign-in"' in source
    assert "Sign in</button>" in source
    assert "encodeURIComponent(user)" in source
    assert "location.pathname === '/login-help.html'" in source or 'location.pathname === "/login-help.html"' in source
    assert "showRerunPlaceholder" in source
    assert "rerunIframeLoaded" in source
    assert "startApp()" in source
    assert "function bindClick(" in source
    assert "function wireUi()" in source
    assert "function showToast(" in source
    assert "id=\"statusBar\"" in source
    assert "id=\"toastHost\"" in source
    assert "DOMContentLoaded" in source
    assert "initNpaAgentUi" in source
    assert "AGENT_UI_VERSION" in source or "npa-ui-version" in source
    assert 'add_header Cache-Control "no-store, no-cache, must-revalidate"' in source


def test_bootstrap_public_login_form() -> None:
    from npa.cli import agent as agent_module

    html = agent_module._agent_public_login_form_html("npa")
    assert 'id="npa-sign-in"' in html
    assert 'type="submit">Sign in</button>' in html
    assert 'value="npa"' in html
    assert "encodeURIComponent(user)" in html
    assert "encodeURIComponent(pass)" in html
    assert "history.replaceState" in html
    assert 'location.pathname === "/login-help.html"' in html


def test_bootstrap_ui_button_wiring_patterns() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    for control_id in (
        "chatSend",
        "chatActionS3",
        "chatActionCosmos",
        "chatActionWatch",
        "loadFrankaRerun",
        "openRerun",
        "applySelection",
        "submitWorkflow",
        "workflowStatus",
    ):
        assert f'bindClick("{control_id}"' in source
    assert "await apiJson(\"/api/chat\"" in source
    assert "await apiJson(\"/api/sim-viz/load-franka-demo\"" in source
    assert "await apiJson(\"/api/sim-viz/camera-preview\"" in source
    assert "await apiJson(\"/api/sim-assets/selection\"" in source
    assert "setChatBusy(false)" in source
    assert "finally {" in source.split("async function sendChat")[1].split("async function")[0]


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
    assert "_log_franka_robot_geometry" in source
    assert "robot/franka/links" in source
    assert "Load Franka in Rerun" in source
    assert "Open in Rerun" in source
    assert "robotPreset" in source
    assert "rerunPlaceholder" in source
    assert 'id="rerunFrame" title="rerun" hidden' in source
    assert "RERUN_RECORDING_PATH" in source
    assert "location.origin + RERUN_RECORDING_PATH" in source
    assert "/rerun/recordings/sim2real.rrd" in source
    assert 'rel="preload" href="/rerun/re_viewer.js"' in source
    assert "waitForRerunReady" in source
    assert "mountRerunIframe" in source
    assert "mountRerunIframeUntilSuccess" in source
    assert "lastRerunBlobStatus" in source
    assert "lastRerunMountStatus" in source
    assert "RERUN_MOUNT_SUCCESS" in source
    assert "Rerun iframe mount missing SUCCESS blob/mount state" in source
    assert "resolveRerunRrdUrl" in source
    assert "RERUN_BLOB_SUCCESS" in source
    assert "/api/sim-viz/rrd-blob" in source
    assert "URL.createObjectURL" in source
    assert "apis_used" in source
    assert "format_live_context_block" in source
    assert "match_chat_intent" in source
    assert "renderAssetsSummary" in source
    assert "selectionPayloadFromUi" in source


def test_bootstrap_ui_strips_url_credentials() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert "location.username" in source
    assert "location.password" in source
    assert "history.replaceState" in source
    assert 'location.protocol + "//" + location.host + location.pathname' in source
    assert "_agent_strip_url_credentials_js" in source
    assert "stripUrlCredentials" in source


def test_bootstrap_ui_fetch_uses_credentials_include() -> None:
    from npa.cli import agent as agent_module

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    assert 'credentials: "include"' in source
    assert 'credentials: "same-origin"' not in source
    assert "setChatBusy(true)" in source
    assert "setChatBusy(false)" in source
    assert "btn.disabled = Boolean(isBusy);" in source
    assert "input.disabled = Boolean(isBusy);" in source
    assert "JSON.stringify(value)" in source
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
            "agent_url": "https://203.0.113.50/",
            "public_url": "https://203.0.113.50/",
            "public_https": True,
            "direct_url": "http://203.0.113.50:8088/",
            "rerun_url": "https://203.0.113.50/rerun/",
            "sim_viz_url": "https://203.0.113.50/rerun/",
            "sim_assets_url": "https://203.0.113.50/",
            "cameras_api_url": "https://203.0.113.50/api/sim-assets/cameras",
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
    assert payload["sim_assets_url"].endswith("203.0.113.50/")
    assert payload["cameras_api_url"].endswith("/api/sim-assets/cameras")


def test_verify_live_runs_pytests(monkeypatch) -> None:
    class _Resp:
        def __init__(self, payload: dict[str, object] | str | bytes, *, status_code: int = 200) -> None:
            self.status_code = status_code
            self._payload = payload
            if isinstance(payload, (bytes, str)):
                self.content = payload.encode("utf-8") if isinstance(payload, str) else payload
                self.text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
            else:
                self.content = b""
                self.text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            if isinstance(self._payload, dict):
                return self._payload
            return {"ok": True}

        @property
        def headers(self) -> dict[str, str]:
            if isinstance(self._payload, (bytes, str)):
                return {"content-type": "application/octet-stream"}
            return {"content-type": "application/json"}

    class _Proc:
        def __init__(self, code: int = 0) -> None:
            self.returncode = code

    monkeypatch.setattr(
        "npa.cli.agent._agent_record",
        lambda project, name: {
            "public_ip": "8.8.8.8",
            "region": "us-central1",
            "agent_url": "https://203.0.113.50/",
            "public_url": "https://203.0.113.50/",
            "public_https": True,
            "direct_url": "http://203.0.113.50:8088/",
            "rerun_url": "https://203.0.113.50/rerun/",
            "sim_viz_url": "https://203.0.113.50/rerun/",
            "sim_assets_url": "https://203.0.113.50/",
            "cameras_api_url": "https://203.0.113.50/api/sim-assets/cameras",
            "auth_secret_path": "/tmp/agent-auth",
        },
    )
    monkeypatch.setattr("npa.cli.agent._load_auth_secret", lambda _: ("npa", "secret"))
    monkeypatch.setattr("npa.cli.agent._health", lambda *_args, **_kwargs: (True, 200))
    def _fake_http_get(url, *_args, **_kwargs):
        url_s = str(url)
        if url_s.endswith("/api/tools"):
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
        if url_s.endswith("/api/session"):
            return _Resp({"chat_history": [], "selection": {}})
        if url_s.endswith("/api/sim-viz/status"):
            return _Resp({"rerun_ready": True, "rrd_uri": "/api/sim-viz/rrd", "stage": "demo"})
        if url_s.endswith("/api/sim-viz/rrd") or url_s.endswith("/api/sim-viz/rrd-blob"):
            return _Resp(b"RRD" * 32, status_code=200)
        if url_s.endswith("/api/health"):
            return _Resp({"ok": True})
        if url_s.endswith("/api/workflows/sim2real/status"):
            return _Resp({"latest_submit": {"run_id": "agent-run-123"}, "sim_viz": {"stage": "demo"}})
        if url_s.endswith("/welcome"):
            return _Resp("<html>NPA Agent is running</html>", status_code=200)
        if url_s.endswith("/healthz"):
            return _Resp('{"ok":true}', status_code=200)
        if "/rerun/" in url_s:
            return _Resp(b"console.log('rerun');", status_code=200)
        if url_s.rstrip("/").endswith(("203.0.113.50", ":8088")):
            html = (
                f'<html><head><meta name="npa-ui-version" content="{AGENT_UI_VERSION}"></head>'
                '<body><script>function wireUi(){} bindClick("chatSend"); initNpaAgentUi; '
                'history.replaceState(null, "", ""); location.username; location.password</script></body></html>'
            )
            return _Resp(html, status_code=200)
        return _Resp({"ok": True, "tool_ref": "tool.0", "argv_template": ["echo", "ok"]})

    def _fake_http_post(url, *_args, **_kwargs):
        url_s = str(url)
        if url_s.endswith("/api/chat"):
            return _Resp(
                {
                    "ok": True,
                    "grounded": True,
                    "reply": "**Sim2Real status**\n- **run_id**: `agent-run-123`\n- **stage**: `demo`",
                    "apis_used": ["sim-viz/status"],
                }
            )
        if url_s.endswith("/api/sim-assets/selection"):
            return _Resp({"ok": True, "selection": {"scene_spec_uri": "stock://scene/default"}})
        if url_s.endswith("/api/workflows/sim2real/submit"):
            return _Resp({"ok": True, "run_id": "agent-run-123"})
        if url_s.endswith("/api/sim-viz/load-franka-demo"):
            return _Resp({"ok": True, "sim_viz": {"rerun_ready": True, "rrd_uri": "/api/sim-viz/rrd"}})
        if url_s.endswith("/api/sim-viz/camera-preview"):
            return _Resp({"ok": True, "entity_path": "world/cameras/workspace"})
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
        [
            "npa/.venv/bin/python",
            "-m",
            "pytest",
            "npa/tests/smoke/test_agent_smoke.py",
            "npa/tests/smoke/test_agent_chat_smoke.py",
            "-q",
        ],
        ["npa/.venv/bin/python", "-m", "pytest", "npa/tests/cli/test_agent.py", "-q"],
        ["npa/.venv/bin/python", "-m", "pytest", "npa/tests/e2e/test_agent_live.py", "-q"],
    ]


def _sample_agent_state(*, rerun_ready: bool = True, stage: str = "demo") -> dict:
    return {
        "sim_viz": {
            "run_id": "agent-run-123",
            "stage": stage,
            "camera": "workspace",
            "rerun_ready": rerun_ready,
            "rrd_updated_at": "2025-06-25T12:00:00+00:00",
        },
        "selection": {
            "robot_preset": "franka",
            "sim_backend": "isaac",
            "scene_spec_uri": "stock://scene/default",
            "robot_spec_uri": "stock://robot/franka",
            "cameras_uri": "stock://cameras/default",
            "assets_uri": "",
            "props": ["cube"],
        },
        "latest_submit": {"run_id": "agent-run-123", "submitted_at": "2025-06-25T11:00:00+00:00"},
        "camera_selection": ["workspace"],
    }


def test_match_chat_intent_status_queries() -> None:
    from npa.cli.agent_chat import match_chat_intent

    assert match_chat_intent("what is the current sim2real status") == "sim2real_status"
    assert match_chat_intent("workflow status please") == "sim2real_status"
    assert match_chat_intent("watch the sim in rerun") == "watch_sim"
    assert match_chat_intent("tail the simulation timeline") == "watch_sim"
    assert match_chat_intent("open the rerun iframe and show latest timeline") == "watch_sim"
    assert match_chat_intent("show stage badge overlay for this run") == "watch_sim"
    assert match_chat_intent("poll sim-viz/status and refresh rerun iframe") == "watch_sim"
    assert match_chat_intent("rerun blob iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("blob+iframe until success") == "watch_sim"
    assert match_chat_intent("wait for RERUN_BLOB_SUCCESS and RERUN_MOUNT_SUCCESS") == "watch_sim"
    assert match_chat_intent("load franka in rerun") == "load_franka"
    assert match_chat_intent("show me the sim assets selection") == "sim_assets"
    assert match_chat_intent("list cameras") == "cameras"
    assert match_chat_intent("what tools can workbench do") == "tools_catalog"
    assert match_chat_intent("configure S3 bucket") == "configure_s3"
    assert match_chat_intent("setup cosmos3") == "cosmos3"
    assert match_chat_intent("camera angle inspector with frustum preview") == "cameras"
    assert match_chat_intent("specify scene robot cameras props selection") == "sim_assets"
    assert match_chat_intent("hello there") is None


def test_build_grounded_status_reply_unpacks_fields() -> None:
    from npa.cli.agent_chat import build_grounded_reply

    state = _sample_agent_state()
    reply = build_grounded_reply("sim2real_status", state, ["tool.a"], rerun_ready=True)
    assert "**run_id**" in reply
    assert "`agent-run-123`" in reply
    assert "**stage**" in reply
    assert "`demo`" in reply
    assert "GET /api" not in reply


def test_build_grounded_watch_sim_reply_mentions_status_polling_and_success() -> None:
    from npa.cli.agent_chat import build_grounded_reply

    state = _sample_agent_state()
    reply = build_grounded_reply("watch_sim", state, ["tool.a"], rerun_ready=True)
    assert "/api/sim-viz/status" in reply
    assert "rrd_uri" in reply
    assert "SUCCESS" in reply
    assert "**watch_stage**" in reply


def test_format_live_context_block_redacts_secrets() -> None:
    from npa.cli.agent_chat import format_live_context_block

    block = format_live_context_block(_sample_agent_state())
    assert "agent-run-123" in block
    assert "password" not in block.lower()
    assert "credentials" not in block.lower()


def test_apis_for_intent_includes_status_paths() -> None:
    from npa.cli.agent_chat import apis_for_intent

    apis = apis_for_intent("sim2real_status")
    assert "sim-viz/status" in apis
    assert "workflows/sim2real/status" in apis
    watch_apis = apis_for_intent("watch_sim")
    assert "sim-viz/rrd-blob" in watch_apis
