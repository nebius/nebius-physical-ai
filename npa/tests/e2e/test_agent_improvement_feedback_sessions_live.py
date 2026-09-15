"""Opt-in deployed session reconciliation using externally prepared receipts.

The dedicated lifecycle bundle extends NPA_AGENT_IMPROVEMENT_LIVE_BUNDLE with
``result``, ``episode_id``, and ``session_id`` from the original runtime failure.
A coordinator must prepare its validation_refs and independently obtained
review_ref against the exact deployed candidate before this test runs. It
preserves the verified lesson on replay, then deliberately invalidates that
lesson with a new session in the dedicated queue. It never creates receipts.
Run this file explicitly with its dedicated bundle, separately from other
improvement lifecycle tests that consume their own review receipts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import pytest

from npa.agent_backend.improvements import _digest, _read_private
from npa.cli.agent_deployment import build_deployment_manifest
from .agent_live_helpers import load_agent_live_context


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agent_live,
    pytest.mark.skipif(
        any(os.environ.get(name) != "1" for name in (
            "NPA_AGENT_LIVE", "NPA_INTEGRATION_E2E", "NPA_AGENT_IMPROVEMENT_LIVE",
        )),
        reason="Enable the dedicated deployed feedback-session lifecycle explicitly.",
    ),
]


def _result(response):
    assert response.status_code == 200, "deployed improvement request failed"
    body = response.json()
    assert body.get("grounded") is True
    assert body.get("usage", {}).get("total_tokens") == 0
    assert body.get("status") == "ready", "dedicated queue is not enabled"
    return body["result"]


def test_deployed_session_replay_preserves_review_until_distinct_recurrence():
    component = os.environ.get("NPA_AGENT_IMPROVEMENT_LIVE_COMPONENT", "")
    bundle_path = os.environ.get("NPA_AGENT_IMPROVEMENT_LIVE_BUNDLE", "")
    assert component and bundle_path, "supply the approved component and protected lifecycle bundle"
    bundle = json.loads(_read_private(Path(bundle_path)))
    assert bundle["episode_id"] and bundle["session_id"], "the original runtime context is required"
    assert bundle["validation_refs"] and bundle["review_ref"], "external receipts are required"
    assert bundle["result"]["ok"] is False
    assert any(step.get("tool") == component and step.get("status") == "error"
               for step in bundle["result"]["steps"])

    ctx = load_agent_live_context()
    repository = Path(__file__).resolve().parents[3]
    expected = build_deployment_manifest(project_alias=ctx.project, name=ctx.name, repo_root=repository)
    response = ctx.get("/api/deployment")
    assert response.status_code == 200, "deployed identity is unavailable"
    deployed = response.json()
    for field in ("deployment_id", "repository", "branch", "commit", "source_tree"):
        assert deployed.get(field) == expected[field], "deployment must match the clean local candidate"

    item_id = bundle["item_id"]
    item_path = f"/api/agent/improvements/{item_id}"
    initial = _result(ctx.get(item_path))
    assert initial["item"]["component"] == component
    assert initial["item"]["state"] == "ready_for_review", "prepare real validation and external review first"
    occurrences = initial["occurrences"]
    assert len(occurrences) == 1, "use an isolated queue with one original runtime failure"
    assert occurrences[0]["episode_ref"] == _digest(bundle["episode_id"])
    assert occurrences[0]["session_ref"] == _digest(bundle["session_id"])

    scope = initial["item"]["scope"]
    assert "npa/src/npa/agent_backend/improvement_routes.py" in scope["files"]
    files = []
    for name in scope["files"]:
        source = (repository / name).resolve()
        assert source.is_relative_to(repository), "candidate scope must remain in the source repository"
        files.append({"path": name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    snapshot = {"base_revision": scope["base_revision"], "files": files}
    assert initial["item"]["candidate_sha256"] == _digest(snapshot), "receipts must bind the current source bytes"

    for reference in bundle["validation_refs"]:
        validated = _result(ctx.post(item_path + "/validation", json={
            **bundle["ownership"], "evidence_ref": reference,
        }))
        assert validated["state"] == "ready_for_review"
    verified = _result(ctx.post(item_path + "/review", json={"evidence_ref": bundle["review_ref"]}))
    assert verified["state"] == "verified"
    before = _result(ctx.get(item_path))
    assert before["item"]["review"]["identity_provenance"] == "coordinator-attested-external-review"
    lessons_path = "/api/agent/improvements/lessons"
    lessons = _result(ctx.get(lessons_path, params={"target": component}))
    assert any(row["item_id"] == item_id for row in lessons)

    payload = {key: bundle[key] for key in ("result", "episode_id", "session_id")}
    for _ in range(2):
        replay = _result(ctx.post("/api/agent/improvements/reconcile", json=payload))
        assert len(replay) == 1 and replay[0] == verified
        assert _result(ctx.get(item_path)) == before
        assert _result(ctx.get(lessons_path, params={"target": component})) == lessons

    payload["session_id"] = "live-recurrence-" + uuid.uuid4().hex
    recurring = _result(ctx.post("/api/agent/improvements/reconcile", json=payload))
    assert len(recurring) == 1 and recurring[0]["id"] == item_id
    assert recurring[0]["state"] == "observed" and recurring[0]["occurrences"] == 2
    after = _result(ctx.get(item_path))
    assert len({row["session_ref"] for row in after["occurrences"]}) == 2
    assert len({row["episode_ref"] for row in after["occurrences"]}) == 1
    assert after["events"][-1]["event"] == "recurrence"
    assert not any(row["item_id"] == item_id for row in _result(ctx.get(lessons_path, params={"target": component})))
    assert _result(ctx.post("/api/agent/improvements/reconcile", json=payload)) == recurring
    assert _result(ctx.get(item_path)) == after
