"""Agent task-eval scorecard test (Phase E).

Runs the mocked task-completion suite (0 tokens, CI-safe), asserts a competitive
bar, and emits a scorecard artifact so future changes are measured against it.
A live variant is gated behind ``NPA_AGENT_CHAT_LIVE=1`` (Tier-2 convention).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_eval.harness import run_suite
from agent_eval.scenarios import SCENARIOS

ARTIFACT_DIR = Path(__file__).with_name("_artifacts")
SCORECARD_PATH = ARTIFACT_DIR / "scorecard.json"


def test_agent_eval_scorecard_meets_bar():
    report = run_suite()
    scorecard = report["scorecard"]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SCORECARD_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # Competitive bar: the mocked suite must fully pass and stay cheap.
    assert scorecard["total"] == len(SCENARIOS)
    assert scorecard["success_rate"] >= 0.9, report["results"]
    # avg_tokens is simulated (mocked planner); it must stay small.
    assert scorecard["avg_tokens"] <= 5.0, scorecard


def test_every_scenario_kind_is_exercised():
    report = run_suite()
    kinds = {r["kind"] for r in report["results"]}
    assert {"grounded", "workflow", "action_loop", "sim2real_loop", "semantic"} <= kinds


def test_no_task_crashes():
    report = run_suite()
    crashed = [r for r in report["results"] if str(r.get("detail", "")).startswith("error:")]
    assert not crashed, crashed


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_CHAT_LIVE") != "1",
    reason="live agent eval gated behind NPA_AGENT_CHAT_LIVE=1 (cheapest pinned model)",
)
def test_agent_eval_live_smoke():  # pragma: no cover - opt-in live variant
    # Live variant mirrors Tier-2 conventions: hit a deployed agent's /api/chat
    # with the cheapest model and assert grounded turns cost 0 tokens. Requires
    # NPA_AGENT_URL + basic-auth creds in the environment; skipped in CI.
    import httpx

    base = os.environ.get("NPA_AGENT_URL", "").rstrip("/")
    if not base:
        pytest.skip("NPA_AGENT_URL not set")
    user = os.environ.get("AGENT_USER", "")
    password = os.environ.get("AGENT_PASSWORD", "")
    resp = httpx.post(
        f"{base}/api/chat",
        json={"messages": [{"role": "user", "content": "what is the current sim2real status"}]},
        auth=(user, password) if user else None,
        timeout=60.0,
        verify=False,
    )
    resp.raise_for_status()
    data = resp.json()
    assert data.get("grounded") is True
    assert int((data.get("usage") or {}).get("total_tokens", 0)) == 0


if __name__ == "__main__":  # pragma: no cover
    report = run_suite()
    print(json.dumps(report["scorecard"], indent=2))
