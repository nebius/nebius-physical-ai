from __future__ import annotations

import json

import pytest

from npa.orchestration.npa_workflow.decisions import (
    DECISION_LOOP_BACK,
    DECISION_PROMOTE,
    decision_from_payload,
    load_decision,
    normalize_decision,
    refresh_context_decision,
    write_decision,
)
from npa.orchestration.npa_workflow.errors import NpaWorkflowError


def test_normalize_decision_aliases() -> None:
    assert normalize_decision("promote") == DECISION_PROMOTE
    assert normalize_decision("loop_back") == DECISION_LOOP_BACK


def test_load_and_write_decision_roundtrip() -> None:
    store: dict[tuple[str, str], bytes] = {}

    def writer(bucket: str, key: str, body: bytes) -> None:
        store[(bucket, key)] = body

    def reader(bucket: str, key: str) -> str:
        return store[(bucket, key)].decode("utf-8")

    uri = "s3://bucket/run/decision.json"
    write_decision(uri, "promote", writer=writer)
    assert load_decision(uri, reader=reader) == DECISION_PROMOTE


def test_refresh_context_decision_prefers_s3_uri() -> None:
    payload = json.dumps({"decision": "loop_back"})
    context = {
        "last_decision": DECISION_PROMOTE,
        "config": {"decision_uri": "s3://bucket/run/decision.json"},
    }
    decision = refresh_context_decision(
        context,
        reader=lambda _bucket, _key: payload,
    )
    assert decision == DECISION_LOOP_BACK


def test_decision_from_payload_requires_field() -> None:
    with pytest.raises(NpaWorkflowError):
        decision_from_payload({})
