"""Adversarial HTTP replay regression; every artifact stays under pytest tmp_path."""

import hashlib
import json
import subprocess
import sys

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from npa.agent_backend import improvement_routes
from npa.agent_backend.actions import run_action_loop
from npa.agent_backend.improvements import ImprovementScope, ImprovementStore
from npa.agent_backend.improvement_routes import (
    ImprovementDeps,
    ImprovementRuntime,
    register_improvement_routes,
)


@pytest.fixture
def setup_store(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "adapter.py").write_text("candidate = 1\n")
    scope = ImprovementScope(
        scope_id="synthetic-feedback",
        component="retrieval_search",
        files=("adapter.py",),
        base_revision="b" * 40,
        required_checks=("behavior",),
        lesson_keys=("inspect_failed_tool_evidence",),
    )
    store = ImprovementStore(
        tmp_path / "queue",
        repository=repo,
        evidence_directory=tmp_path / "evidence",
        scopes=[scope],
        reviewers=["independent-reviewer"],
    )
    app = FastAPI()
    register_improvement_routes(app, ImprovementDeps(lambda: store), HTTPException)
    return store, TestClient(app)


def _record_runtime(store, monkeypatch):
    monkeypatch.setattr(
        improvement_routes, "current_episode_id", lambda: "original-episode"
    )
    monkeypatch.setattr(
        improvement_routes, "current_session_id", lambda: "original-session"
    )
    result = {
        "ok": False,
        "steps": [
            {
                "phase": "call",
                "tool": "retrieval_search",
                "status": "error",
                "args": {"query": "adapter fields"},
                "error": "synthetic backend unavailable",
            }
        ],
    }
    runtime = ImprovementRuntime(lambda: store)
    recorded = runtime.record(result, runtime.prepare(["retrieval_search"]))
    assert recorded["status"] == "recorded"
    return result, recorded["item_ids"][0]


def _verify(store, item_id):
    item = store.history(item_id)["item"]
    claim = store.claim(item_id, owner="builder", version=item["version"])
    ownership = {key: claim[key] for key in ("owner", "generation", "claim_token")}
    candidate = store.begin_candidate(
        item_id, changed_files=["adapter.py"], **ownership
    )
    completed = subprocess.run(
        [sys.executable, "-c", "print('isolated behavior check passed')"],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    receipt = store.write_validation_receipt(
        candidate, check="behavior", completed=completed, report=completed.stdout
    )
    assert (
        store.record_validation(item_id, evidence_ref=receipt, **ownership)["state"]
        == "ready_for_review"
    )
    review = store.write_review_receipt(
        item_id,
        reviewer="independent-reviewer",
        lesson_key="inspect_failed_tool_evidence",
        accepted=True,
        report=b"Separate reviewer confirmed synthetic lifecycle fixture.\n",
    )
    assert store.review(item_id, evidence_ref=review)["state"] == "verified"


def test_reconcile_original_failure_preserves_verified_lesson(setup_store, monkeypatch):
    store, client = setup_store
    result, item_id = _record_runtime(store, monkeypatch)
    _verify(store, item_id)
    before = store.history(item_id)
    response = client.post(
        "/agent/improvements/reconcile",
        json={
            "result": result,
            "episode_id": "original-episode",
            "session_id": "original-session",
        },
    )
    assert response.status_code == 200
    after = store.history(item_id)
    assert len(after["occurrences"]) == len(before["occurrences"]) == 1, (
        "same-session replay created duplicate failure occurrence"
    )
    assert after["item"]["state"] == "verified", (
        "replayed historical failure incorrectly revoked reviewed lesson"
    )
    assert store.matching_verified_lessons(["retrieval_search"])


def test_reconcile_retains_distinct_session_provenance(setup_store):
    store, client = setup_store
    result = {
        "ok": False,
        "steps": [
            {
                "tool": "retrieval_search",
                "status": "error",
                "error": "synthetic backend unavailable",
            }
        ],
    }
    for session in ("session-one", "session-two"):
        response = client.post(
            "/agent/improvements/reconcile",
            json={
                "result": result,
                "episode_id": "caller-episode",
                "session_id": session,
            },
        )
        assert response.status_code == 200
    item = store.list_items()[0]
    occurrences = store.history(item["id"])["occurrences"]
    assert len(occurrences) == 2, (
        "distinct parent sessions were collapsed during reconciliation"
    )
    expected_refs = {
        hashlib.sha256(
            json.dumps(
                s, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        for s in ("session-one", "session-two")
    }
    assert {entry["session_ref"] for entry in occurrences} == expected_refs


@pytest.mark.parametrize("failure", ["provider_unavailable", "malformed_plan"])
def test_action_planner_failure_reaches_only_action_loop_scope(tmp_path, failure):
    repository = tmp_path / "source"
    repository.mkdir()
    components = ("action-loop", "sim2real-drive")
    scopes = [
        ImprovementScope(
            scope_id=component,
            component=component,
            files=(component + ".py",),
            base_revision="a" * 40,
            required_checks=("reproducer",),
        )
        for component in components
    ]
    store = ImprovementStore(
        tmp_path / "queue",
        repository=repository,
        evidence_directory=tmp_path / "evidence",
        scopes=scopes,
        reviewers=(),
    )
    planner_calls = []
    tool_calls = []

    def planner(*args, **kwargs):
        planner_calls.append(1)
        if failure == "provider_unavailable":
            raise RuntimeError("synthetic planner service unavailable")
        return {"choices": [{"message": {"content": "not a JSON action"}}]}

    def tool(args):
        tool_calls.append(args)
        return {"ok": True}

    result = run_action_loop(
        "inspect the available retrieval evidence",
        tools={"retrieval_search": tool},
        model_call=planner,
    )
    assert result["ok"] is False
    assert result["stopped_reason"] == (
        "error" if failure == "provider_unavailable" else "no_plan"
    )
    assert len(planner_calls) == (1 if failure == "provider_unavailable" else 2)
    assert tool_calls == []
    runtime = ImprovementRuntime(lambda: store)
    feedback = runtime.record(
        result, {"request_id": "planner-failure-episode", "lessons": []}
    )
    assert feedback["status"] == "recorded"
    items = store.list_items()
    assert [item["component"] for item in items] == ["action-loop"]
    occurrence = store.history(items[0]["id"])["occurrences"][0]
    assert occurrence["evidence"] == result["steps"][0]
    assert occurrence["evidence"]["phase"] == "plan"
