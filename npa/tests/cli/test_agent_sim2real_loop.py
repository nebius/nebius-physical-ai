"""Tier-0 tests for the autonomous Sim2Real outer-loop orchestration.

All collaborators are fakes; no GPU, engine, or model is touched.
"""

from __future__ import annotations

from npa.cli import agent_chat
from npa.cli import agent_sim2real_loop as L


def test_drive_sim2real_intent_matches_paraphrases():
    for text in (
        "drive the sim2real loop",
        "autonomously run sim2real",
        "orchestrate the sim2real outer loop for me",
    ):
        assert agent_chat.match_chat_intent(text) == "drive_sim2real", text


def test_drive_sim2real_grounded_reply_mentions_gate_and_confirmation():
    reply = agent_chat.build_grounded_reply("drive_sim2real", {"sim_viz": {"run_id": "r1"}}, [])
    assert "promote_checkpoint" in reply
    assert "loop_back" in reply
    assert "confirmation" in reply.lower()
    assert "/api/agent/sim2real/drive" in reply


def _status_for(run_id: str, stage: str = "stage_10_eval_heldout") -> dict:
    return {"ok": True, "sim_viz": {"run_id": run_id, "stage": stage}, "run": {"run_id": run_id}}


def test_evaluate_gate_promotes_when_success_meets_threshold():
    ev = L.evaluate_gate({"success_rate": 0.9, "threshold": 0.8})
    assert ev["decision"] == L.DECISION_PROMOTE
    assert ev["promoted"] is True


def test_evaluate_gate_loops_back_below_threshold():
    ev = L.evaluate_gate({"success_rate": 0.3, "threshold": 0.8})
    assert ev["decision"] == L.DECISION_LOOP_BACK
    assert ev["promoted"] is False


def test_evaluate_gate_honors_explicit_engine_decision():
    ev = L.evaluate_gate({"decision": "loop_back_to_inner_loop", "success_rate": 0.99, "threshold": 0.1})
    assert ev["decision"] == L.DECISION_LOOP_BACK


def test_drive_requires_confirmation_before_any_launch():
    launched = {"count": 0}

    def _launch(cfg):  # pragma: no cover - must not run without confirmation
        launched["count"] += 1
        return {"run_id": "r"}

    result = L.drive_sim2real_loop(
        "drive the sim2real loop",
        config={"run_id": "r", "threshold": 0.8},
        launch=_launch,
        status=lambda rid: _status_for(rid),
        gate=lambda rid, it: {"success_rate": 1.0, "threshold": 0.8},
    )
    assert launched["count"] == 0
    assert result["needs_confirmation"] is True
    assert result["stopped_reason"] == L.STOP_NEEDS_CONFIRMATION
    assert result["proposed_action"]["action"] == "drive_sim2real"


def test_drive_promotes_on_first_pass_when_gate_met():
    calls = {"launch": 0}

    def _launch(cfg):
        calls["launch"] += 1
        return {"ok": True, "run_id": "run-1"}

    result = L.drive_sim2real_loop(
        "drive",
        config={"run_id": "run-1", "threshold": 0.8},
        launch=_launch,
        status=lambda rid: _status_for(rid),
        gate=lambda rid, it: {"success_rate": 0.95, "threshold": 0.8},
        confirm_token="tok",
        session_token="tok",
    )
    assert calls["launch"] == 1
    assert result["stopped_reason"] == L.STOP_PROMOTED
    assert result["decision"] == L.DECISION_PROMOTE
    assert result["final_run_id"] == "run-1"
    assert len(result["iterations"]) == 1


def test_drive_loops_back_then_promotes_with_adjust_and_diagnose():
    gate_scores = {1: 0.2, 2: 0.9}
    diag_calls = {"count": 0}
    adjust_calls = {"count": 0}

    def _gate(rid, it):
        return {"success_rate": gate_scores[it], "threshold": 0.8}

    def _diagnose(gate, status):
        diag_calls["count"] += 1
        return {"failure_mode": "low_success", "notes": "increase envs"}

    def _adjust(cfg, diagnosis):
        adjust_calls["count"] += 1
        new = dict(cfg)
        new["num_envs"] = cfg.get("num_envs", 1) * 2
        return new

    result = L.drive_sim2real_loop(
        "drive",
        config={"run_id": "run-x", "threshold": 0.8, "num_envs": 4},
        launch=lambda cfg: {"ok": True, "run_id": "run-x"},
        status=lambda rid: _status_for(rid),
        gate=_gate,
        diagnose=_diagnose,
        adjust=_adjust,
        confirm_token="t",
        session_token="t",
        max_iterations=3,
    )
    assert result["stopped_reason"] == L.STOP_PROMOTED
    assert len(result["iterations"]) == 2
    assert result["iterations"][0]["decision"] == L.DECISION_LOOP_BACK
    assert result["iterations"][0]["diagnosis"]["failure_mode"] == "low_success"
    assert result["iterations"][0]["adjusted_config"]["num_envs"] == 8
    assert diag_calls["count"] == 1
    assert adjust_calls["count"] == 1


def test_drive_stops_when_status_does_not_confirm_run():
    # status reports a different run id -> no fabricated progress.
    result = L.drive_sim2real_loop(
        "drive",
        config={"run_id": "run-a", "threshold": 0.8},
        launch=lambda cfg: {"ok": True, "run_id": "run-a"},
        status=lambda rid: _status_for("some-other-run"),
        gate=lambda rid, it: {"success_rate": 1.0, "threshold": 0.8},
        confirm_token="t",
        session_token="t",
    )
    assert result["stopped_reason"] == L.STOP_UNCONFIRMED_STATUS
    assert result["decision"] is None
    assert result["iterations"][0]["status_confirmed"] is False


def test_drive_exhausts_iterations_without_promotion():
    result = L.drive_sim2real_loop(
        "drive",
        config={"run_id": "run-e", "threshold": 0.99},
        launch=lambda cfg: {"ok": True, "run_id": "run-e"},
        status=lambda rid: _status_for(rid),
        gate=lambda rid, it: {"success_rate": 0.1, "threshold": 0.99},
        adjust=lambda cfg, d: cfg,
        confirm_token="t",
        session_token="t",
        max_iterations=2,
    )
    assert result["stopped_reason"] == L.STOP_EXHAUSTED
    assert len(result["iterations"]) == 2
    assert all(it["decision"] == L.DECISION_LOOP_BACK for it in result["iterations"])


def test_drive_surfaces_launch_error():
    def _boom(cfg):
        raise RuntimeError("gpu unavailable")

    result = L.drive_sim2real_loop(
        "drive",
        config={"run_id": "r", "threshold": 0.8},
        launch=_boom,
        status=lambda rid: _status_for(rid),
        gate=lambda rid, it: {},
        confirm_token="t",
        session_token="t",
    )
    assert result["stopped_reason"] == L.STOP_ERROR
    assert "gpu unavailable" in result["iterations"][0]["error"]
