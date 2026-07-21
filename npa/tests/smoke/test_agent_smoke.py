from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

from npa.cli.agent import AGENT_MEDIA_PREVIEW_CONTRACT, AGENT_UI_VERSION, rendered_agent_ui_html

REPO_ROOT = Path(__file__).resolve().parents[3]
TMUX_SCRIPT = REPO_ROOT / "npa" / "scripts" / "start_agent_live_tmux.sh"
AGENT_MODULE = REPO_ROOT / "npa" / "src" / "npa" / "cli" / "agent.py"
AGENT_UI_MODULE = REPO_ROOT / "npa" / "src" / "npa" / "cli" / "agent_ui.html"

UI_BUTTON_IDS = (
    "chatActionS3",
    "chatActionCosmos",
    "chatActionWatch",
    "newChatSession",
    "loadFrankaRerun",
    "openRerun",
    "applySelection",
    "submitWorkflow",
    "workflowStatus",
)

UI_WIRING_MARKERS = (
    "function bindClick(",
    "function wireUi(",
    "function showToast(",
    "initNpaAgentUi",
    "DOMContentLoaded",
    'id="tabChat"',
    'id="tabRerun"',
    'id="stagesPanel"',
    "<h3>Stages</h3>",
)

RERUN_STATIC_CANDIDATES = (
    "/rerun/index.js",
    "/rerun/re_viewer.js",
    "/rerun/favicon.ico",
    "/rerun/version",
)


def test_agent_bootstrap_source_smoke() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui_source = AGENT_UI_MODULE.read_text(encoding="utf-8")
    ui = rendered_agent_ui_html()
    bundled = source + "\n" + ui_source + "\n" + ui
    assert '@app.get("/sim-viz/rrd")' in source
    assert '@app.post("/sim-viz/load-franka-demo")' in source
    assert '@app.post("/sim-viz/camera-preview")' in source
    assert '@app.get("/workflows/sim2real/status")' in source
    assert AGENT_UI_VERSION in source
    assert 'name="npa-ui-version" content="{AGENT_UI_VERSION}"' in ui_source
    for control_id in UI_BUTTON_IDS:
        assert f'bindClick("{control_id}"' in bundled
    for marker in UI_WIRING_MARKERS:
        assert marker in bundled, f"missing UI wiring marker: {marker!r}"
    for marker in AGENT_MEDIA_PREVIEW_CONTRACT:
        assert marker in bundled, f"missing media-preview contract marker: {marker!r}"
    assert 'id="chatSend"' in bundled
    assert 'id="chatForm"' in bundled
    assert 'id="chatSessionSelect"' in bundled
    assert 'chatForm.addEventListener("submit"' in bundled
    assert "/api/chat/sessions" in bundled
    assert 'add_header Cache-Control "no-store, no-cache, must-revalidate"' in source
    assert "media_type=artifact_media_type(safe_name)" in source


def test_agent_live_tmux_script_help() -> None:
    proc = subprocess.run(
        ["bash", str(TMUX_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--project" in proc.stdout
    assert "--bootstrap" in proc.stdout
    assert "--verify" in proc.stdout


@pytest.mark.skipif(
    subprocess.run(["bash", "-lc", "command -v tmux"], capture_output=True).returncode
    != 0,
    reason="tmux not installed",
)
def test_agent_live_tmux_script_dry_run() -> None:
    session = "npa-agent-live-smoke"
    proc = subprocess.run(
        [
            "bash",
            str(TMUX_SCRIPT),
            "--dry-run",
            "--session",
            session,
            "--project",
            "smoke-project",
            "--name",
            "smoke-agent",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"TMUX_SESSION={session}" in proc.stdout
    subprocess.run(["tmux", "kill-session", "-t", session], check=False)


def test_agent_rerun_static_candidate_list_nonempty() -> None:
    assert RERUN_STATIC_CANDIDATES
    for path in RERUN_STATIC_CANDIDATES:
        assert path.startswith("/rerun/")


def test_agent_verify_live_command_registered() -> None:
    from typer.testing import CliRunner

    from npa.cli.agent import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "verify-live" in result.output


def test_agent_health_client_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> dict[str, bool]:
            return {"ok": True}

    monkeypatch.setattr(httpx, "get", lambda *_args, **_kwargs: _Resp())
    from npa.cli.agent import _health

    ok, code = _health("http://203.0.113.50:8088/", user="npa", password="secret")
    assert ok is True
    assert code == 200
