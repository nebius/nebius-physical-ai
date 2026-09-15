"""Opt-in deployed improvement queue gates using private coordinator evidence.

The operator configures a dedicated queue and supplies a lifecycle bundle only
after real checks and an independently obtained review. No receipt is invented
by this test. All requests are grounded and use the existing private live-agent
credential helper.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import pytest

from npa.agent_backend.improvements import _read_private
from .agent_live_helpers import load_agent_live_context


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agent_live,
    pytest.mark.skipif(
        os.environ.get("NPA_AGENT_LIVE") != "1"
        or os.environ.get("NPA_INTEGRATION_E2E") != "1"
        or os.environ.get("NPA_AGENT_IMPROVEMENT_LIVE") != "1",
        reason="Enable live agent and dedicated improvement queue checks explicitly.",
    ),
]


def _result(response):
    assert response.status_code == 200, "deployed improvement request failed"
    payload = response.json()
    assert payload.get("grounded") is True
    assert payload.get("usage", {}).get("total_tokens") == 0
    assert payload.get("status") == "ready", "dedicated queue is not enabled"
    return payload["result"]


def test_deployed_observation_dedupe_claim_and_release():
    component = os.environ.get("NPA_AGENT_IMPROVEMENT_LIVE_COMPONENT", "")
    assert component, "supply an approved dedicated live component"
    ctx = load_agent_live_context()
    episode = "live-feedback-" + uuid.uuid4().hex
    payload = {"episode_id": episode, "result": {"ok": False, "steps": [{
        "phase": "call", "tool": component, "status": "error", "args": {},
        "observation": {"error": "Synthetic dedicated queue probe"},
    }]}}
    first = _result(ctx.post("/api/agent/improvements/reconcile", json=payload))
    assert len(first) == 1, "component must have a dedicated approved scope"
    second = _result(ctx.post("/api/agent/improvements/reconcile", json=payload))
    assert first[0]["id"] == second[0]["id"]
    assert first[0]["occurrences"] == second[0]["occurrences"]
    item = first[0]
    claim = _result(ctx.post(f"/api/agent/improvements/{item['id']}/claim", json={"owner": "live-probe", "version": item["version"]}))
    ownership = {key: claim[key] for key in ("owner", "generation", "claim_token")}
    try:
        state = _result(ctx.get(f"/api/agent/improvements/{item['id']}"))
        assert state["item"]["state"] == "claimed"
        forged = ctx.post(f"/api/agent/improvements/{item['id']}/review", json={"reviewer": "different-name", "passed": True})
        assert forged.status_code == 409
    finally:
        released = _result(ctx.post(f"/api/agent/improvements/{item['id']}/release", json=ownership))
        assert released["state"] == "observed"


def test_deployed_evidence_validation_review_and_lesson_retrieval():
    bundle_path = os.environ.get("NPA_AGENT_IMPROVEMENT_LIVE_BUNDLE", "")
    assert bundle_path, "supply owner-only bundle from real coordinator checks and independent review"
    bundle = json.loads(_read_private(Path(bundle_path)))
    ctx = load_agent_live_context()
    item_id = bundle["item_id"]
    # The review receipt was made locally from the same actual validations;
    # replaying those immutable references proves the HTTP adapter resolves them.
    for reference in bundle["validation_refs"]:
        state = _result(ctx.post(f"/api/agent/improvements/{item_id}/validation", json={
            **bundle["ownership"], "evidence_ref": reference,
        }))
    assert state["state"] == "ready_for_review"
    verified = _result(ctx.post(f"/api/agent/improvements/{item_id}/review", json={"evidence_ref": bundle["review_ref"]}))
    assert verified["state"] == "verified"
    lessons = _result(ctx.get("/api/agent/improvements/lessons?target=" + bundle["target"]))
    assert any(lesson["item_id"] == item_id for lesson in lessons)
    history = _result(ctx.get(f"/api/agent/improvements/{item_id}"))
    assert history["item"]["review"]["identity_provenance"] == "coordinator-attested-external-review"
