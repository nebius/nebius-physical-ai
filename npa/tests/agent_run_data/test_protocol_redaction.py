"""Protocol integrity must coexist with data redaction and lossless key renames."""

from __future__ import annotations

import copy
import hashlib
import io
import json

import pytest

from npa.agent_backend import trajectory as emitter


class Storage:
    def __init__(self, *, fail: bool = False):
        self.s3 = self
        self.fail = fail
        self.objects = {}
        self.calls = 0

    def head_bucket(self, **kwargs):
        self.calls += 1

    def put_object(self, *, Key, Body, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic unavailable destination")
        if Key in self.objects:
            raise RuntimeError("precondition failed")
        self.objects[Key] = Body

    def get_object(self, *, Key, **kwargs):
        self.calls += 1
        return {"Body": io.BytesIO(self.objects[Key])}

    def delete_object(self, *, Key, **kwargs):
        self.objects.pop(Key, None)


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("NPA_AGENT_DATASET_TENANT_ID", "tenant-test")
    monkeypatch.setenv("NPA_AGENT_DATASET_URI", "s3://test-bucket/dataset")
    monkeypatch.setenv("NPA_AGENT_DATASET_OUTBOX", str(tmp_path / "outbox"))
    monkeypatch.delenv("NPA_AGENT_DATASET_REDACTION_FILE", raising=False)
    return emitter.resolve_dataset_config(active_tenant_id="tenant-test", active_bucket="test-bucket")


def emit(storage, **changes):
    args = {
        "episode_id": "episode-test", "session_id": "session-test",
        "request_content": "inspect", "intent": "inspect", "initial_state": {},
        "trajectory": [{"sequence": 0, "phase": "tool", "tool": "inspect", "arguments": {},
                        "observation": {}, "status": "ok"}],
        "outcome": {"status": "succeeded", "verified": True, "verified_by": ["synthetic check"]},
        "routing": {"grounded": True, "model": "", "input_tokens": 0, "output_tokens": 0},
        "versions": {"agent": "test", "tools": {}},
        "started_at": "2026-08-30T00:00:00+00:00", "ended_at": "2026-08-30T00:01:00+00:00",
        "storage": storage, "active_tenant_id": "tenant-test", "active_bucket": "test-bucket",
    }
    args.update(changes)
    return emitter.emit_trajectory(**args)


def raw_record(storage):
    return next(json.loads(body) for key, body in storage.objects.items() if "/episodes/" in key)


@pytest.fixture
def payload(configured):
    storage = Storage()
    assert emit(storage)[0] == emitter.CollectionStatus.COLLECTED
    return raw_record(storage)


def private_policy(monkeypatch, tmp_path, literals):
    path = tmp_path / "redaction.json"
    path.write_text(json.dumps({"literals": literals}))
    path.chmod(0o600)
    monkeypatch.setenv("NPA_AGENT_DATASET_REDACTION_FILE", str(path))


@pytest.mark.parametrize("literal", [
    "agent", "scope", "schema_version", "status", "pending", "succeeded", "tool",
    "plan", "grounded", "input_tokens", "agent-finetuning-raw", "redaction", "inline-data",
])
@pytest.mark.parametrize("pending", [False, True])
def test_protocol_collisions_preserve_structure_and_redact_data(
    configured, monkeypatch, tmp_path, literal, pending,
):
    private_policy(monkeypatch, tmp_path, [literal])
    storage = Storage(fail=pending)
    status, _ = emit(storage, request_content=literal, versions={"agent": literal, "tools": {}},
                     initial_state={literal: "preserved-value", "nested": {"schema_version": literal}})
    assert status == ("pending" if pending else "collected")
    row = (json.loads(next((tmp_path / "outbox").glob("*.json")).read_bytes())["payload"]
           if pending else raw_record(storage))
    assert row["schema_version"] == "npa.agent.trajectory.v1"
    assert row["scope"]["dataset_role"] == "agent-finetuning-raw"
    assert row["collection"]["status"] == "pending"
    assert row["trajectory"][0]["phase"] == "tool"
    assert row["outcome"]["status"] == "succeeded"
    assert row["routing"]["input_tokens"] == 0
    assert row["request"]["content"] == "<private-ref>"
    assert row["versions"]["agent"] == "<private-ref>"
    assert literal not in row["initial_state"]
    assert "preserved-value" in row["initial_state"].values()
    assert emitter._sanitize_payload(row, configured) == row
    assert emitter._validated_body(configured, row) == emitter._canonical_json(row).encode()


@pytest.mark.parametrize("field", ["episode_id", "session_id"])
def test_caller_identity_is_not_a_protocol_exception(configured, monkeypatch, tmp_path, field):
    private_policy(monkeypatch, tmp_path, ["caller-private-identity"])
    storage = Storage()
    with pytest.raises(emitter.AgentRunDataError, match="safe stable identifiers"):
        emit(storage, **{field: "caller-private-identity"})
    assert storage.calls == 0
    assert not (tmp_path / "outbox").exists()


def test_redacted_timing_fails_before_destination_probe(configured, monkeypatch, tmp_path):
    private_policy(monkeypatch, tmp_path, ["2026"])
    storage = Storage()
    with pytest.raises(emitter.AgentRunDataError, match="timestamps must remain valid") as caught:
        emit(storage)
    assert caught.value.__context__ is None
    assert storage.calls == 0
    assert not (tmp_path / "outbox").exists()


@pytest.mark.parametrize("input_tokens,output_tokens", [(None, None), (None, 0), (0, None)])
@pytest.mark.parametrize("pending", [False, True])
def test_unknown_token_usage_remains_null(configured, tmp_path, input_tokens, output_tokens, pending):
    storage = Storage(fail=pending)
    status, _ = emit(storage, routing={"grounded": False, "input_tokens": input_tokens,
                                      "output_tokens": output_tokens})
    assert status == ("pending" if pending else "collected")
    row = (json.loads(next((tmp_path / "outbox").glob("*.json")).read_bytes())["payload"]
           if pending else raw_record(storage))
    assert row["routing"]["input_tokens"] is input_tokens
    assert row["routing"]["output_tokens"] is output_tokens
    assert emitter._sanitize_payload(row, configured) == row
    assert emitter._validated_body(configured, row) == emitter._canonical_json(row).encode()


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("value", [True, "0", 1.0])
def test_nullable_usage_still_rejects_other_types(configured, tmp_path, field, value):
    storage = Storage()
    with pytest.raises(emitter.AgentRunDataError, match="protocol field type"):
        emit(storage, routing={field: value})
    assert storage.calls == 0
    assert not (tmp_path / "outbox").exists()


def test_data_strings_and_hashes_remain_private(configured, payload, monkeypatch, tmp_path):
    private_policy(monkeypatch, tmp_path, ["2026", "model-choice", "tool-choice", "abcd"])
    payload["routing"]["model"] = "model-choice"
    payload["trajectory"][0]["tool"] = "tool-choice"
    payload["collection"]["content_sha256"] = "abcd" * 16
    payload["versions"]["tools"] = {"artifact_hash": "abcd" * 16}
    row = emitter._sanitize_payload(payload, configured)
    assert "2026" not in row["timing"]["started_at"]
    assert row["routing"]["model"] == row["trajectory"][0]["tool"] == "<private-ref>"
    assert "abcd" not in row["collection"]["content_sha256"]
    assert "abcd" not in row["versions"]["tools"]["artifact_hash"]


@pytest.mark.parametrize("path", [
    ("initial_state",), ("trajectory", 0, "arguments"), ("trajectory", 0, "observation"),
    ("outcome", "extra"), ("versions", "tools"), ("routing", "extra"),
])
def test_lookalike_schema_in_data_is_not_authoritative(configured, payload, monkeypatch, tmp_path, path):
    private_policy(monkeypatch, tmp_path, ["scope", "status", "schema_version", "agent"])
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = {"schema_version": "npa.agent.trajectory.v1", "scope": {"status": "agent"}}
    row = emitter._sanitize_payload(payload, configured)
    current = row
    for key in path:
        current = current[key]
    assert "scope" not in current and "schema_version" not in current
    assert "npa.agent.trajectory.v1" not in json.dumps(current)
    assert "agent" not in json.dumps(current)


@pytest.mark.parametrize("path,value", [
    (("scope",), []), (("scope", "dataset_role"), "forged-value"),
    (("schema_version",), "forged-value"), (("collection", "status"), "collected"),
    (("trajectory",), ["forged-event"]), (("trajectory", 0, "phase"), "forged-value"),
    (("trajectory", 0, "status"), "forged-value"), (("trajectory", 0, "sequence"), True),
    (("outcome", "verified"), "true"), (("routing", "input_tokens"), True),
    (("versions", "agent"), {}), (("request", "content"), []),
    (("redaction", "applied"), 1), (("redaction", "fields_removed"), ["forged-value"]),
])
def test_forged_protocol_shape_refuses_before_persistence(configured, payload, monkeypatch, path, value):
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    calls = []
    monkeypatch.setattr(emitter, "_secure_outbox", lambda: calls.append("outbox"))
    with pytest.raises(emitter.AgentRunDataError, match="trajectory"):
        emitter._write_outbox(configured, payload)
    assert not calls


@pytest.mark.parametrize("path", [("scope", "tenant_id"), ("timing", "started_at"), ("collection", "status")])
def test_missing_required_internal_fields_fail_safely(configured, payload, path):
    del payload[path[0]][path[1]]
    with pytest.raises(emitter.AgentRunDataError, match="protocol structure"):
        emitter._validated_body(configured, payload)


def test_partial_caller_metadata_and_unknown_fields_survive(configured, monkeypatch, tmp_path):
    private_policy(monkeypatch, tmp_path, ["scope"])
    storage = Storage()
    assert emit(storage, trajectory=[{}, {"observation": "scope"}, {"scope": "kept"}],
                outcome={"note": "scope"}, routing={"usage_status": "unknown"}, versions={})[0] == "collected"
    row = raw_record(storage)
    assert row["trajectory"][0] == {}
    assert row["trajectory"][1]["observation"] == "<private-ref>"
    assert "kept" in row["trajectory"][2].values()
    assert row["outcome"] == {"note": "<private-ref>"}
    assert row["routing"] == {"usage_status": "unknown"}
    assert row["versions"] == {}


@pytest.mark.parametrize("redactor", ["patterns", "literals"])
def test_mapping_key_collision_never_discards_a_value(redactor):
    original = "https://example.invalid/item" if redactor == "patterns" else "synthetic-private-name"
    marker = "<uri-ref>" if redactor == "patterns" else "<private-ref>"
    lookalike = marker + "-" + hashlib.sha256(original.encode()).hexdigest()[:12]
    mapping = {original: "first", lookalike: "second", lookalike + "<private-ref>": "third"}
    sanitize = emitter.redact if redactor == "patterns" else lambda value: emitter._redact_identifiers(value, {original: marker})
    safe = sanitize(mapping)
    assert len(safe) == 3
    assert set(safe.values()) == {"first", "second", "third"}
    assert sanitize(safe) == safe
    assert sanitize(dict(reversed(list(mapping.items())))) == safe


def test_new_markers_and_generated_suffixes_are_not_rescanned_as_private_data():
    replacements = {"synthetic-private-name": "<private-ref>", "private": "<private-ref>", "a": "<private-ref>"}
    original = {"synthetic-private-name": "synthetic-private-name", "a": "safe"}
    safe = emitter._redact_identifiers(original, replacements)
    assert len(safe) == 2
    assert emitter._redact_identifiers(safe, replacements) == safe
    assert "synthetic-private-name" not in json.dumps(safe)
    # The generated digest is scanned too, rather than exempting hash-shaped strings.
    for key in safe:
        assert "a" not in emitter._REDACTION_MARKER_RE.sub("", key)


@pytest.mark.parametrize("destination", ["s3", "outbox"])
def test_prewrite_revalidation_rejects_changed_private_policy(configured, payload, monkeypatch, tmp_path, destination):
    payload["request"]["content"] = "newly-private-value"
    payload["collection"]["content_sha256"] = emitter._content_sha256(payload)
    private_policy(monkeypatch, tmp_path, ["newly-private-value"])
    storage = Storage()
    with pytest.raises(emitter.AgentRunDataError, match="pre-write privacy"):
        if destination == "s3":
            emitter._put_and_verify(configured, payload, storage)
        else:
            emitter._write_outbox(configured, payload)
    assert storage.calls == 0
    assert not (tmp_path / "outbox").exists()


def test_tenant_restoration_does_not_accept_a_forged_destination(configured, payload):
    payload["scope"]["tenant_id"] = "other-tenant"
    payload["collection"]["content_sha256"] = emitter._content_sha256(payload)
    with pytest.raises(emitter.AgentRunDataError, match="tenant does not match"):
        emitter._validated_body(configured, payload)
    assert payload["scope"]["tenant_id"] == "other-tenant"


def test_sanitization_does_not_mutate_input(configured, payload, monkeypatch, tmp_path):
    private_policy(monkeypatch, tmp_path, ["agent"])
    before = copy.deepcopy(payload)
    emitter._sanitize_payload(payload, configured)
    assert payload == before
