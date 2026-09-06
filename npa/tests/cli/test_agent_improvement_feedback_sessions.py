"""Reconciliation preserves persisted session provenance and evidence gates."""

from __future__ import annotations

import subprocess
import sys

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from npa.agent_backend import improvement_routes
from npa.agent_backend.improvements import ImprovementScope, ImprovementStore


ACTION = {"ok": False, "steps": [{
    "phase": "call", "tool": "retrieval_search", "status": "error",
    "args": {"query": "synthetic regression"}, "error": "synthetic tool failure",
}]}
EPISODE = "synthetic-episode"
SESSION = "synthetic-session"


@pytest.fixture
def feedback(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / "candidate.py").write_text("result = 1\n")
    store = ImprovementStore(
        tmp_path / "queue", repository=repository,
        evidence_directory=tmp_path / "evidence",
        scopes=[ImprovementScope(
            scope_id="synthetic-feedback", component="retrieval_search",
            files=("candidate.py",), base_revision="a" * 40,
            required_checks=("reproducer",), lesson_keys=("inspect_failed_tool_evidence",),
        )], reviewers=("synthetic-reviewer",),
    )
    app = FastAPI()
    improvement_routes.register_improvement_routes(
        app, improvement_routes.ImprovementDeps(store=lambda: store), HTTPException,
    )
    with TestClient(app) as client:
        yield store, client


def _result(response):
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["grounded"] is True
    assert body["usage"] == {"total_tokens": 0}
    return body["result"]


def _reconcile(client, **context):
    return client.post("/agent/improvements/reconcile", json={
        "result": ACTION, "episode_id": EPISODE, **context,
    })


def _history(client, item_id):
    return _result(client.get(f"/agent/improvements/{item_id}"))


def _lessons(client):
    return _result(client.get("/agent/improvements/lessons", params={"target": "retrieval_search"}))


def _verify(store, client, item):
    claim = _result(client.post(f"/agent/improvements/{item['id']}/claim", json={
        "owner": "synthetic-builder", "version": item["version"],
    }))
    ownership = {key: claim[key] for key in ("owner", "generation", "claim_token")}
    candidate = store.begin_candidate(item["id"], changed_files=["candidate.py"], **ownership)
    completed = subprocess.run(
        [sys.executable, "-c", "from candidate import result; assert result == 1; print('synthetic check passed')"],
        cwd=store.repository, capture_output=True, check=True,
    )
    receipt = store.write_validation_receipt(
        candidate, check="reproducer", completed=completed, report=completed.stdout,
    )
    validated = _result(client.post(f"/agent/improvements/{item['id']}/validation", json={
        **ownership, "evidence_ref": receipt,
    }))
    assert validated["state"] == "ready_for_review"
    # Synthetic adapter evidence tests the receipt contract, not a live review.
    review = store.write_review_receipt(
        item["id"], reviewer="synthetic-reviewer", accepted=True,
        lesson_key="inspect_failed_tool_evidence", report=b"Synthetic unit review fixture.\n",
    )
    verified = _result(client.post(f"/agent/improvements/{item['id']}/review", json={"evidence_ref": review}))
    assert verified["state"] == "verified"
    assert len(_lessons(client)) == 1
    return verified


def test_runtime_failure_replay_preserves_verified_lesson(feedback, monkeypatch):
    store, client = feedback
    monkeypatch.setattr(improvement_routes, "current_episode_id", lambda: EPISODE)
    monkeypatch.setattr(improvement_routes, "current_session_id", lambda: SESSION)
    runtime = improvement_routes.ImprovementRuntime(lambda: store)
    recorded = runtime.record(ACTION)
    assert recorded["status"] == "recorded"
    item = _history(client, recorded["item_ids"][0])["item"]
    verified = _verify(store, client, item)
    before = _history(client, item["id"])
    lessons = _lessons(client)

    for _ in range(2):
        replay = _result(_reconcile(client, session_id=SESSION))[0]
        assert replay == verified
        assert _history(client, item["id"]) == before
        assert _lessons(client) == lessons

    # A new store instance reads the same persisted occurrence and review.
    reopened = ImprovementStore(
        store.directory, repository=store.repository, evidence_directory=store.evidence_directory,
        scopes=list(store.scopes.values()), reviewers=store.reviewers,
    )
    assert reopened.history(item["id"]) == before
    occurrence = before["occurrences"][0]
    assert occurrence["session_ref"] and occurrence["session_ref"] != SESSION
    assert occurrence["episode_ref"] and occurrence["episode_ref"] != EPISODE


@pytest.mark.parametrize("verify_first", [False, True])
def test_same_episode_in_distinct_sessions_records_recurrence(feedback, verify_first):
    store, client = feedback
    first = _result(_reconcile(client, session_id=SESSION))[0]
    if verify_first:
        _verify(store, client, first)
    second = _result(_reconcile(client, session_id="another-synthetic-session"))[0]
    assert second["id"] == first["id"]
    assert second["occurrences"] == 2
    assert second["state"] == "observed"
    history = _history(client, first["id"])
    assert len({row["session_ref"] for row in history["occurrences"]}) == 2
    assert len({row["episode_ref"] for row in history["occurrences"]}) == 1
    assert history["events"][-1]["event"] == ("recurrence" if verify_first else "observed")
    assert _lessons(client) == []
    assert _result(_reconcile(client, session_id="another-synthetic-session"))[0] == second
    assert _history(client, first["id"]) == history


def test_omitted_session_matches_legacy_recording_and_empty_session(feedback):
    store, client = feedback
    first = store.observe_action(ACTION, episode_id=EPISODE)[0]
    verified = _verify(store, client, first)
    before = _history(client, first["id"])
    assert _result(_reconcile(client))[0] == verified
    assert _result(_reconcile(client, session_id=""))[0] == verified
    assert _history(client, first["id"]) == before
    assert before["occurrences"][0]["session_ref"] == ""
    recurring = _result(_reconcile(client, episode_id="another-synthetic-episode"))[0]
    assert recurring["occurrences"] == 2
    assert recurring["state"] == "observed"
    assert _lessons(client) == []


def test_same_session_replay_does_not_restore_stale_source_evidence(feedback):
    store, client = feedback
    first = _result(_reconcile(client, session_id=SESSION))[0]
    verified = _verify(store, client, first)
    before = _history(client, first["id"])
    (store.repository / "candidate.py").write_text("result = 2\n")
    assert _lessons(client) == []
    assert _result(_reconcile(client, session_id=SESSION))[0] == verified
    assert _history(client, first["id"]) == before
    assert _lessons(client) == []


@pytest.mark.parametrize("missing", ["result", "episode_id"])
def test_missing_required_context_returns_sanitized_error_without_recording(feedback, missing):
    store, client = feedback
    payload = {"result": ACTION, "episode_id": EPISODE, "session_id": SESSION}
    del payload[missing]
    response = client.post("/agent/improvements/reconcile", json=payload)
    assert response.status_code == 409
    assert response.json() == {"detail": "improvement scope, ownership or evidence check failed"}
    assert store.list_items() == []


def test_reconcile_storage_failure_returns_sanitized_error(feedback):
    store, client = feedback
    store.path.write_bytes(b"synthetic invalid SQLite database")
    response = _reconcile(client, session_id=SESSION)
    assert response.status_code == 503
    assert response.json() == {"detail": "improvement storage unavailable"}


@pytest.mark.parametrize("operation", ["validation", "review"])
def test_reconciliation_does_not_accept_asserted_receipts(feedback, operation):
    _, client = feedback
    first = _result(_reconcile(client, session_id=SESSION))[0]
    claim = _result(client.post(f"/agent/improvements/{first['id']}/claim", json={
        "owner": "synthetic-builder", "version": first["version"],
    }))
    before = _history(client, first["id"])
    ownership = {key: claim[key] for key in ("owner", "generation", "claim_token")}
    response = client.post(f"/agent/improvements/{first['id']}/{operation}", json={
        **ownership, "passed": True, "reviewer": "synthetic-reviewer",
    })
    assert response.status_code == 409
    assert _result(_reconcile(client, session_id=SESSION))[0]["state"] == "claimed"
    assert _history(client, first["id"]) == before
    assert _lessons(client) == []
