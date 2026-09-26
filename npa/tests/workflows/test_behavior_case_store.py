"""Prove evaluation recovery cannot restart a begun or completed rollout."""

import json

import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.behavior_challenge.case_store import CaseAlreadyStarted, CaseStore


class ConditionalStorage:
    def __init__(self):
        self.objects = {}
        self.sequence = 0

    def read_bytes_with_etag(self, uri):
        return self.objects.get(uri)

    def put_bytes_conditional(
        self, payload, uri, *, if_match="", if_none_match=False, **_
    ):
        current = self.objects.get(uri)
        if (if_none_match and current) or (
            if_match and (not current or current[1] != if_match)
        ):
            raise StoragePreconditionFailed("conflict")
        self.sequence += 1
        etag = f'"revision-{self.sequence}"'
        self.objects[uri] = payload, etag
        return etag


@pytest.fixture
def state():
    storage = ConditionalStorage()
    store = CaseStore(storage, "s3://example-bucket/evaluation", "a" * 64)
    case = {
        "task": "picking_up_trash",
        "index": 10,
        "instance_id": 311,
        "rollout_id": 0,
        "policy_port": None,
    }
    return storage, store, case


def test_replaced_prestart_worker_cannot_invoke_evaluator(state):
    _, store, case = state
    first = store.claim(case, "worker-a")
    resumed = store.claim(case, "worker-b")
    assert resumed.record["previous_claims"][0]["claim_id"] == first.record["claim_id"]
    with pytest.raises(StoragePreconditionFailed):
        store.start(first)
    assert store.start(resumed).record["state"] == "started"


def test_started_rollout_blocks_automatic_reclaim(state):
    _, store, case = state
    original = store.start(store.claim(case, "worker-a"))
    with pytest.raises(CaseAlreadyStarted, match="Recover original evidence"):
        store.claim(case, "worker-b")
    assert store.read(case) == original


def test_completed_rollout_is_reused_without_another_start(state):
    _, store, case = state
    started = store.start(store.claim(case, "worker-a"))
    result = {
        **case,
        "q_score": 0.0,
        "success": False,
        "steps": 400,
        "video_frames": 400,
        "files": {},
    }
    completed = store.complete(started, result)
    assert store.claim(case, "worker-b") == completed
    assert store.artifact_prefix(completed) == store.artifact_prefix(started)
    with pytest.raises(ValueError, match="Only a claimed"):
        store.start(completed)
    with pytest.raises(StoragePreconditionFailed):
        store.complete(started, {**result, "q_score": 1.0})


def test_completion_cannot_cross_cases_or_precede_start(state):
    _, store, case = state
    claim = store.claim(case, "worker-a")
    with pytest.raises(ValueError, match="original started"):
        store.complete(claim, case)
    started = store.start(claim)
    with pytest.raises(ValueError, match="prescribed case"):
        store.complete(started, {**case, "instance_id": 312})


@pytest.mark.parametrize(
    "field,value",
    [("panel_id", "b" * 64), ("claim_id", "../../elsewhere"), ("state", "retryable")],
)
def test_conflicting_stored_evidence_is_rejected(state, field, value):
    storage, store, case = state
    claim = store.claim(case, "worker-a")
    tampered = {**claim.record, field: value}
    storage.objects[store.uri(case)] = json.dumps(tampered).encode(), claim.etag
    with pytest.raises(ValueError, match="identity or state"):
        store.read(case)


def test_storage_failure_is_not_treated_as_absence(state, monkeypatch):
    storage, store, case = state

    def denied(_):
        raise PermissionError("denied")

    monkeypatch.setattr(storage, "read_bytes_with_etag", denied)
    with pytest.raises(PermissionError):
        store.claim(case, "worker-a")
    assert storage.objects == {}


@pytest.mark.parametrize(
    "field,value",
    [("worker_id", ""), ("claimed_at", "yesterday"), ("previous_claims", [{}])],
)
def test_malformed_ownership_record_fails_at_read_boundary(state, field, value):
    storage, store, case = state
    version = store.claim(case, "worker-a")
    storage.objects[store.uri(case)] = (
        json.dumps({**version.record, field: value}).encode(),
        version.etag,
    )
    with pytest.raises(ValueError, match="identity or state"):
        store.read(case)


def test_complete_state_requires_its_original_rollout(state):
    storage, store, case = state
    version = store.start(store.claim(case, "worker-a"))
    malformed = {
        **version.record,
        "state": "complete",
        "completed_at": version.record["started_at"],
    }
    storage.objects[store.uri(case)] = json.dumps(malformed).encode(), version.etag
    with pytest.raises(ValueError, match="identity or state"):
        store.read(case)
