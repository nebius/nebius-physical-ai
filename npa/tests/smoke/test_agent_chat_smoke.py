from __future__ import annotations

import stat
import subprocess
from pathlib import Path


from npa.cli import agent as agent_module
from npa.cli.agent import rendered_agent_ui_html
from npa.cli.agent_chat import (
    apis_for_intent,
    build_grounded_reply,
    format_cameras,
    format_sim_assets,
    match_chat_intent,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_MODULE = REPO_ROOT / "npa" / "src" / "npa" / "cli" / "agent.py"
AGENT_CHAT_MODULE = REPO_ROOT / "npa" / "src" / "npa" / "cli" / "agent_chat.py"
VERIFY_FRANKA_SCRIPT = REPO_ROOT / "npa" / "scripts" / "verify_agent_franka.sh"
VERIFY_CHAT_VIZ_SCRIPT = REPO_ROOT / "npa" / "scripts" / "verify_agent_chat_viz.sh"
VERIFY_RERUN_BUNDLE_SCRIPT = REPO_ROOT / "npa" / "scripts" / "verify_agent_rerun_bundle.sh"
NPA_AGENT_SKILL = REPO_ROOT / "skills" / "tools" / "npa-agent" / "SKILL.md"


def test_agent_bootstrap_chat_router_patterns() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    ui = rendered_agent_ui_html()
    bundled = source + "\n" + ui
    assert '@app.post("/chat")' in source
    assert "_agent_chat_with_tools" in source
    assert "_maybe_toolground_chat_reply" in source
    assert "ensureFrankaRerunLoaded" in bundled
    assert '"grounded": True' in source
    assert '"apis_used": apis_used' in source
    assert 'apiJson("/api/chat"' in bundled


def test_agent_chat_module_intent_patterns() -> None:
    source = AGENT_CHAT_MODULE.read_text(encoding="utf-8")
    assert "sim2real_status" in source
    assert "sim_assets" in source
    assert "cameras" in source
    assert "onboard_solution" in source
    assert "INTENT_APIS" in source
    assert match_chat_intent("what is the current sim2real status") == "sim2real_status"
    assert match_chat_intent("what sim assets are selected") == "sim_assets"
    assert match_chat_intent("list workspace cameras") == "cameras"
    assert match_chat_intent("onboard a github repo into the workbench with container smoke") == "onboard_solution"


def test_agent_chat_response_schema_in_bootstrap() -> None:
    source = AGENT_MODULE.read_text(encoding="utf-8")
    assert '"grounded": True' in source
    assert '"apis_used"' in source
    embedded = agent_module._embedded_agent_chat_source()
    assert "build_grounded_reply" in embedded
    assert "apis_for_intent" in embedded


def test_verify_agent_franka_script_exists_and_executable() -> None:
    assert VERIFY_FRANKA_SCRIPT.is_file()
    mode = VERIFY_FRANKA_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "verify_agent_franka.sh must be executable"


def test_verify_agent_chat_viz_script_exists_and_executable() -> None:
    assert VERIFY_CHAT_VIZ_SCRIPT.is_file()
    mode = VERIFY_CHAT_VIZ_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "verify_agent_chat_viz.sh must be executable"


def test_verify_agent_rerun_bundle_script_exists_and_executable() -> None:
    assert VERIFY_RERUN_BUNDLE_SCRIPT.is_file()
    mode = VERIFY_RERUN_BUNDLE_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "verify_agent_rerun_bundle.sh must be executable"
    text = VERIFY_RERUN_BUNDLE_SCRIPT.read_text(encoding="utf-8")
    assert "agent_rerun_bundle_check" in text
    assert VERIFY_CHAT_VIZ_SCRIPT.read_text(encoding="utf-8").count("verify_agent_rerun_bundle.sh") >= 1


def test_npa_agent_skill_documents_chat_maturity() -> None:
    skill = NPA_AGENT_SKILL.read_text(encoding="utf-8")
    assert "Chat Maturity Patterns" in skill
    assert "grounded" in skill.lower()
    assert "apis_used" in skill
    assert "sim2real_status" in skill


def test_match_chat_intent_sim_assets_and_cameras() -> None:
    assert match_chat_intent("what sim assets are selected") == "sim_assets"
    assert match_chat_intent("show robot_preset selection") == "sim_assets"
    assert match_chat_intent("list cameras") == "cameras"
    assert match_chat_intent("workspace camera frustum") == "cameras"


def test_apis_for_intent_status_and_assets() -> None:
    assert "sim-viz/status" in apis_for_intent("sim2real_status")
    assert "sim-assets" in apis_for_intent("sim_assets")
    assert apis_for_intent("cameras") == ["sim-assets/cameras"]


def test_format_sim_assets_with_mock_state() -> None:
    state = {
        "selection": {
            "robot_preset": "franka",
            "sim_backend": "isaac",
            "scene_spec_uri": "stock://scene/default",
            "robot_spec_uri": "stock://robot/franka",
            "props": ["cube"],
        }
    }
    reply = format_sim_assets(state)
    assert "franka" in reply
    assert "isaac" in reply
    assert "GET /api" not in reply


def test_format_cameras_with_mock_state() -> None:
    state = {"camera_selection": ["workspace", "wrist"]}
    reply = format_cameras(state)
    assert "workspace" in reply
    assert "wrist" in reply


def test_build_grounded_reply_sim_assets_mentions_selection() -> None:
    state = {
        "selection": {
            "robot_preset": "franka",
            "sim_backend": "isaac",
            "scene_spec_uri": "stock://scene/default",
        },
        "sim_viz": {},
        "latest_submit": {},
    }
    reply = build_grounded_reply("sim_assets", state, [])
    assert "Sim assets selection" in reply
    assert "franka" in reply
    assert "isaac" in reply


def test_verify_agent_chat_viz_script_shellcheck() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(VERIFY_CHAT_VIZ_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
