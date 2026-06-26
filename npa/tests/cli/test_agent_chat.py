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
    assert match_chat_intent("watch the sim") == "watch_sim"
    assert match_chat_intent("track the rerun timeline") == "watch_sim"
    assert match_chat_intent("keep me posted with live updates on the sim run") == "watch_sim"
    assert match_chat_intent("rerun blob iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("retry blob iframe until ready") == "watch_sim"
    assert match_chat_intent("watch sim and refresh when rrd lands") == "watch_sim"
    assert match_chat_intent("watch rerun blob+iframe until success") == "watch_sim"
    assert match_chat_intent("wait until both blob and iframe are SUCCESS") == "watch_sim"
    assert match_chat_intent("watch rerun blob iframe until consecutive success") == "watch_sim"
    assert match_chat_intent("keep rerun blob iframe green before finishing") == "watch_sim"
    assert match_chat_intent("mark rerun blob iframe passed before finishing") == "watch_sim"
    assert match_chat_intent("rerun blob-iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("rerun: blob/iframe; wait -> SUCCESS") == "watch_sim"
    assert match_chat_intent("watch until RERUN_BLOB_SUCCESS and RERUN_MOUNT_SUCCESS") == "watch_sim"
    assert match_chat_intent("load franka then rerun blob iframe until SUCCESS") == "watch_sim"
    assert match_chat_intent("camera angle inspector with top-down frustum preview") == "cameras"
    assert match_chat_intent("select scene robot props and cameras before submit") == "sim_assets"


def test_format_sim2real_status_includes_run_id_and_stage() -> None:
    state = {
        "sim_viz": {
            "run_id": "agent-run-deadbeef",
            "stage": "demo",
            "camera": "workspace",
            "rerun_ready": True,
            "rrd_updated_at": "2026-06-25T00:00:00+00:00",
            "rerun_iframe_url": "/rerun/?url=/api/sim-viz/rrd&camera=workspace",
        },
        "latest_submit": {"run_id": "agent-run-deadbeef", "submitted_at": "2026-06-25T00:00:00+00:00"},
        "selection": {"robot_preset": "franka", "sim_backend": "isaac"},
    }
    reply = format_sim2real_status(state, rerun_ready=True)
    assert "run_id" in reply
    assert "agent-run-deadbeef" in reply
    assert "stage" in reply
    assert "demo" in reply
    assert "rerun_iframe_url" in reply
    assert "/rerun/" in reply
    assert "GET /api" not in reply


def test_build_grounded_reply_sim2real_status() -> None:
    state = {"sim_viz": {"run_id": "x", "stage": "idle"}, "selection": {}, "latest_submit": {}}
    reply = build_grounded_reply("sim2real_status", state, ["workbench.lerobot"], rerun_ready=False)
    assert "**stage**" in reply
    assert "GET /api" not in reply


def test_build_grounded_reply_watch_sim_mentions_success() -> None:
    state = {"sim_viz": {"run_id": "x", "stage": "running"}, "selection": {}, "latest_submit": {}}
    reply = build_grounded_reply("watch_sim", state, ["workbench.lerobot"], rerun_ready=True)
    assert "SUCCESS" in reply
    assert "blob" in reply
    assert "iframe mount" in reply
    assert "Rerun blob iframe until SUCCESS" in reply
    assert "RERUN_BLOB_SUCCESS=SUCCESS" in reply
    assert "RERUN_MOUNT_SUCCESS=SUCCESS" in reply
    assert "consecutive SUCCESS confirmations" in reply
    assert "**rrd_uri**" in reply


def test_watch_sim_apis_include_rrd_paths() -> None:
    from npa.cli.agent_chat import apis_for_intent

    apis = apis_for_intent("watch_sim")
    assert "sim-viz/status" in apis
    assert "sim-viz/rrd" in apis
    assert "sim-viz/rrd-blob" in apis


def test_embedded_agent_chat_source_strips_future_import() -> None:
    source = agent_module._embedded_agent_chat_source()
    assert "from __future__ import annotations" not in source
    assert "match_chat_intent" in source
    assert "INTENT_APIS" in source
