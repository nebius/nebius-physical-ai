from __future__ import annotations

from npa.cli import agent as agent_module
from npa.cli.agent_chat import (
    build_grounded_reply,
    format_sim2real_status,
    match_chat_intent,
)


def test_match_sim2real_status_intent() -> None:
    assert match_chat_intent("what is the current sim2real status") == "sim2real_status"
    assert match_chat_intent("What's the workflow status?") == "sim2real_status"


def test_format_sim2real_status_includes_run_id_and_stage() -> None:
    state = {
        "sim_viz": {
            "run_id": "agent-run-deadbeef",
            "stage": "demo",
            "camera": "workspace",
            "rerun_ready": True,
            "rrd_updated_at": "2026-06-25T00:00:00+00:00",
        },
        "latest_submit": {"run_id": "agent-run-deadbeef", "submitted_at": "2026-06-25T00:00:00+00:00"},
        "selection": {"robot_preset": "franka", "sim_backend": "isaac"},
    }
    reply = format_sim2real_status(state, rerun_ready=True)
    assert "run_id" in reply
    assert "agent-run-deadbeef" in reply
    assert "stage" in reply
    assert "demo" in reply
    assert "GET /api" not in reply


def test_build_grounded_reply_sim2real_status() -> None:
    state = {"sim_viz": {"run_id": "x", "stage": "idle"}, "selection": {}, "latest_submit": {}}
    reply = build_grounded_reply("sim2real_status", state, ["workbench.lerobot"], rerun_ready=False)
    assert "**stage**" in reply
    assert "GET /api" not in reply


def test_embedded_agent_chat_source_escapes_braces() -> None:
    source = agent_module._embedded_agent_chat_source()
    assert "{{" in source or "dict[str, Any]" in source
    assert "match_chat_intent" in source
    assert "{catalog_json}" not in source
