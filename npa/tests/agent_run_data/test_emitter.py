"""Verification matrix for goal-level agent trajectory collection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import threading
from pathlib import Path

import pytest

from npa.agent_run_data.emitter import (
    AgentRunDataError,
    CollectionStatus,
    DatasetConfig,
    emit_trajectory,
    flush_outbox,
    goal_episode_boundary,
    redact,
    resolve_dataset_config,
    verify_destination,
)


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_writes = False
        self.fail_reads = False
        self.lock = threading.Lock()
        self.put_count = 0

    def head_bucket(self, **kwargs: object) -> None:
        return None

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs: object) -> None:
        if self.fail_writes:
            raise RuntimeError("synthetic write failure")
        with self.lock:
            if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
                raise RuntimeError("precondition failed")
            self.objects[Key] = Body
            self.put_count += 1

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if self.fail_reads:
            raise RuntimeError("synthetic read failure")
        with self.lock:
            return {"Body": _BytesReader(self.objects[Key])}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        with self.lock:
            self.objects.pop(Key, None)


class FakeStorage:
    def __init__(self, s3: FakeS3) -> None:
        self.s3 = s3


@pytest.fixture
def dataset_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NPA_AGENT_DATASET_TENANT_ID", "tenant-test")
    monkeypatch.setenv("NPA_AGENT_DATASET_URI", "s3://test-bucket/agent-dataset")
    monkeypatch.setenv("NPA_AGENT_DATASET_OUTBOX", str(tmp_path / "outbox"))
    monkeypatch.delenv("NPA_AGENT_DATASET_REDACTION_FILE", raising=False)


def _trajectory() -> list[dict]:
    return [
        {
            "sequence": 0,
            "phase": "tool",
            "tool": "workbench.robocasa.random_rollout",
            "arguments": {"output": "s3://private-bucket/run"},
            "observation": {"ok": True},
            "status": "ok",
        }
    ]


def _outcome(status: str = "succeeded") -> dict:
    return {
        "status": status,
        "verified": status == "succeeded",
        "verified_by": ["pytest"] if status == "succeeded" else [],
        "artifact_uris": [],
        "operator_interventions": [],
        "preference_pairs": [],
    }


def _emit(storage: FakeStorage, *, episode_id: str = "ep-1", request: str = "run", events: list | None = None):
    return emit_trajectory(
        episode_id=episode_id,
        session_id="session-1",
        request_content=request,
        intent="run",
        trajectory=_trajectory() if events is None else events,
        outcome=_outcome(),
        routing={"grounded": False, "tier": "cheap", "model": "test"},
        versions={"agent": "test", "tools": {}},
        storage=storage,
        started_at="2026-08-30T00:00:00+00:00",
        ended_at="2026-08-30T00:01:00+00:00",
        active_tenant_id="tenant-test",
        active_bucket="test-bucket",
    )


def _episode_payloads(s3: FakeS3) -> list[dict]:
    return [
        json.loads(body)
        for key, body in sorted(s3.objects.items())
        if "/episodes/" in key
    ]


def test_resolve_dataset_config_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_AGENT_DATASET_TENANT_ID", raising=False)
    monkeypatch.delenv("NPA_AGENT_DATASET_URI", raising=False)
    assert resolve_dataset_config() is None


def test_tenant_and_bucket_mismatch_fail_before_collection(dataset_env: None) -> None:
    with pytest.raises(AgentRunDataError, match="tenant does not match"):
        resolve_dataset_config(active_tenant_id="other-tenant", active_bucket="test-bucket")
    with pytest.raises(AgentRunDataError, match="bucket does not match"):
        resolve_dataset_config(active_tenant_id="tenant-test", active_bucket="other-bucket")


def test_partial_or_signed_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_AGENT_DATASET_TENANT_ID", "tenant-test")
    monkeypatch.delenv("NPA_AGENT_DATASET_URI", raising=False)
    with pytest.raises(AgentRunDataError, match="set together"):
        resolve_dataset_config()
    monkeypatch.setenv("NPA_AGENT_DATASET_URI", "s3://bucket/path?signature=secret")
    with pytest.raises(AgentRunDataError, match="unsigned"):
        resolve_dataset_config()


def test_redaction_removes_secrets_infrastructure_and_signed_urls() -> None:
    redacted = redact(
        {
            "token": "Bearer synthetic-token",
            "nested": {"password": "synthetic-password", "ok": "fine"},
            "text": (
                "authorization=synthetic-auth s3://private-bucket/path "
                "https://private.invalid/object?signature=synthetic"
            ),
        }
    )
    serialized = json.dumps(redacted)
    for forbidden in (
        "synthetic-token",
        "synthetic-password",
        "synthetic-auth",
        "private-bucket",
        "private.invalid",
        "signature",
    ):
        assert forbidden not in serialized
    assert redacted["nested"]["ok"] == "fine"


def test_success_upload_is_collected_only_with_read_after_write(dataset_env: None) -> None:
    s3 = FakeS3()
    status, episode_id = _emit(FakeStorage(s3))
    assert (status, episode_id) == (CollectionStatus.COLLECTED, "ep-1")
    payload = _episode_payloads(s3)[0]
    assert payload["collection"]["status"] == CollectionStatus.PENDING
    receipts = [json.loads(body) for key, body in s3.objects.items() if "/receipts/" in key]
    assert receipts[0]["status"] == CollectionStatus.COLLECTED
    assert receipts[0]["content_sha256"] == payload["collection"]["content_sha256"]
    assert payload["collection"]["content_sha256"]
    assert payload["timing"]["latency_ms"] == 60_000
    assert "private-bucket" not in json.dumps(payload)


def test_deterministic_idempotent_key_uses_episode_start_date(dataset_env: None) -> None:
    s3 = FakeS3()
    storage = FakeStorage(s3)
    _emit(storage, episode_id="same")
    first_keys = [key for key in s3.objects if "/episodes/" in key]
    _emit(storage, episode_id="same")
    second_keys = [key for key in s3.objects if "/episodes/" in key]
    assert first_keys == second_keys
    assert "/episodes/2026/08/30/" in first_keys[0]


def test_concurrent_episodes_produce_distinct_immutable_objects(dataset_env: None) -> None:
    s3 = FakeS3()
    storage = FakeStorage(s3)
    threads = [
        threading.Thread(target=_emit, args=(storage,), kwargs={"episode_id": f"ep-{i}"})
        for i in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    keys = [key for key in s3.objects if "/episodes/" in key]
    assert len(keys) == 20
    assert len(set(keys)) == 20


def test_s3_failure_outbox_then_flush_has_read_after_write_proof(
    dataset_env: None, tmp_path: Path
) -> None:
    s3 = FakeS3()
    s3.fail_writes = True
    status, _ = _emit(
        FakeStorage(s3),
        episode_id="pending",
        request="token=synthetic-secret s3://private-bucket/path",
    )
    assert status == CollectionStatus.PENDING
    pending = list((tmp_path / "outbox").glob("*.json"))
    assert len(pending) == 1
    body = pending[0].read_text()
    assert "synthetic-secret" not in body
    assert "private-bucket" not in body
    envelope = json.loads(body)
    assert envelope["payload"]["collection"]["status"] == CollectionStatus.PENDING
    expected_body = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":")).encode()

    s3.fail_writes = False
    assert flush_outbox(storage=FakeStorage(s3), active_tenant_id="tenant-test", active_bucket="test-bucket") == ["pending"]
    assert not list((tmp_path / "outbox").glob("*.json"))
    uploaded = _episode_payloads(s3)[0]
    assert uploaded["collection"]["status"] == CollectionStatus.PENDING
    uploaded_body = next(body for key, body in s3.objects.items() if "/episodes/" in key)
    assert uploaded_body == expected_body
    assert hashlib.sha256(uploaded_body).hexdigest() == envelope["payload_sha256"]


def test_existing_loose_outbox_permissions_are_repaired(
    dataset_env: None, tmp_path: Path
) -> None:
    outbox = tmp_path / "outbox"
    outbox.mkdir(mode=0o777)
    outbox.chmod(0o777)
    s3 = FakeS3()
    s3.fail_writes = True
    _emit(FakeStorage(s3), episode_id="permissions")
    assert outbox.stat().st_mode & 0o777 == 0o700
    path = next(outbox.glob("*.json"))
    assert path.stat().st_mode & 0o777 == 0o600


def test_verify_destination_requires_read_after_write(dataset_env: None) -> None:
    s3 = FakeS3()
    s3.fail_reads = True
    config = DatasetConfig(
        tenant_id="tenant-test",
        dataset_uri="s3://test-bucket/agent-dataset",
        bucket="test-bucket",
        prefix="agent-dataset",
    )
    with pytest.raises(AgentRunDataError, match="not writable"):
        verify_destination(config, storage=FakeStorage(s3))


@pytest.mark.parametrize(
    ("response", "expected_status"),
    [
        ({"ok": True, "grounded": False, "usage": {"prompt_tokens": 2, "completion_tokens": 3}}, "succeeded"),
        ({"ok": False, "steps": [{"tool": "demo", "ok": False}]}, "failed"),
        ({"ok": False, "refused": True}, "refused"),
        ({"ok": True, "grounded": True, "usage": {"total_tokens": 0}}, "succeeded"),
    ],
)
def test_goal_boundary_records_success_tool_failure_refusal_and_grounded_zero_token(
    dataset_env: None, response: dict, expected_status: str
) -> None:
    s3 = FakeS3()

    @goal_episode_boundary(
        storage_factory=lambda: FakeStorage(s3),
        active_tenant_id=lambda: "tenant-test", active_bucket=lambda: "test-bucket",
    )
    def endpoint(payload: dict) -> dict:
        return dict(response)

    result = endpoint({"session_id": "session-1", "messages": [{"role": "user", "content": "goal"}]})
    assert result == response
    payload = _episode_payloads(s3)[0]
    assert payload["outcome"]["status"] == expected_status
    if response.get("grounded"):
        assert payload["routing"]["input_tokens"] == 0
        assert payload["routing"]["output_tokens"] == 0


def test_goal_boundary_records_cancellation(dataset_env: None) -> None:
    s3 = FakeS3()

    @goal_episode_boundary(
        storage_factory=lambda: FakeStorage(s3),
        active_tenant_id=lambda: "tenant-test", active_bucket=lambda: "test-bucket",
    )
    def endpoint(payload: dict) -> dict:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        endpoint({"messages": [{"role": "user", "content": "cancel"}]})
    assert _episode_payloads(s3)[0]["outcome"]["status"] == "cancelled"


class _StagingSSH:
    def __init__(self) -> None:
        self.staged: list[str] = []

    def upload_private_text(self, content: str, _remote_path: str) -> None:
        self.staged.append(content)

    def run_or_raise(self, command: str, **kwargs: object) -> str:
        return ""

    def run(self, command: str) -> str:
        return ""


def _stage_dataset_env(monkeypatch: pytest.MonkeyPatch, *, tenant: str, uri: str) -> list[str]:
    from npa.cli.agent_env_files import _write_agent_nebius_env

    monkeypatch.setenv("NPA_AGENT_DATASET_TENANT_ID", tenant)
    monkeypatch.setenv("NPA_AGENT_DATASET_URI", uri)
    ssh = _StagingSSH()
    _write_agent_nebius_env(
        ssh,
        project_alias="test",
        agent_name="agent",
        project_id="project-test",
        tenant_id="tenant-test",
        region="test-region",
        service_account_id="service-account-test",
        bucket="test-bucket",
        endpoint="https://storage.invalid",
        access_key="synthetic-access-key",
        secret_key="synthetic-secret-key",
    )
    return ssh.staged


def test_dataset_configuration_is_staged_in_owner_only_agent_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = _stage_dataset_env(
        monkeypatch,
        tenant="tenant-test",
        uri="s3://test-bucket/agent-dataset",
    )
    assert len(staged) == 1
    assert "NPA_AGENT_DATASET_TENANT_ID=tenant-test" in staged[0]
    assert "NPA_AGENT_DATASET_URI=s3://test-bucket/agent-dataset" in staged[0]


@pytest.mark.parametrize(
    ("tenant", "uri", "match"),
    [
        ("other-tenant", "s3://test-bucket/agent-dataset", "tenant does not match"),
        ("tenant-test", "s3://other-bucket/agent-dataset", "deployment's unsigned S3 bucket"),
    ],
)
def test_dataset_configuration_mismatch_is_not_staged(
    monkeypatch: pytest.MonkeyPatch, tenant: str, uri: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _stage_dataset_env(monkeypatch, tenant=tenant, uri=uri)


@pytest.mark.parametrize("scope", [{}, {"active_tenant_id": "tenant-test"}, {"active_bucket": "test-bucket"}])
def test_missing_active_identity_fails_closed(dataset_env: None, scope: dict) -> None:
    with pytest.raises(AgentRunDataError, match="verified active deployment"):
        resolve_dataset_config(**scope)


@pytest.mark.parametrize("pending", [False, True])
def test_private_payload_classes_removed_before_every_write(
    dataset_env: None, tmp_path: Path, pending: bool,
) -> None:
    secret = "synthetic-secret-material"
    resource_id = "project-" + "z" * 20
    data = {
        "AWS_SECRET_ACCESS_KEY": secret,
        "env": {"otherwise_ordinary": "synthetic-environment-secret"},
        "image_data": "synthetic-inline-image",
        "text": (
            "-----BEGIN PRIVATE KEY-----\nsynthetic-private-material\n-----END PRIVATE KEY----- "
            "data:image/png;base64,c3ludGhldGlj "
            "s3://customer-bucket/customer-sensitive-name/asset.usd "
            "https://operator:password@private.invalid/customer-sensitive-name?secret=value "
            "203.0.113.50 2001:db8::1 " + resource_id
        ),
        resource_id: "safe-observation",
    }
    s3 = FakeS3()
    s3.fail_writes = pending
    status, _ = _emit(FakeStorage(s3), events=[{"sequence": 0, "arguments": data}])
    assert status == (CollectionStatus.PENDING if pending else CollectionStatus.COLLECTED)
    if pending:
        serialized = next((tmp_path / "outbox").glob("*.json")).read_text()
    else:
        serialized = next(body.decode() for key, body in s3.objects.items() if "/episodes/" in key)
    for forbidden in (
        secret, "synthetic-environment-secret", "synthetic-inline-image", "synthetic-private-material",
        "c3ludGhldGlj", "customer-bucket", "customer-sensitive-name", "private.invalid",
        "203.0.113.50", "2001:db8::1", resource_id,
    ):
        assert forbidden not in serialized
    assert "safe-observation" in serialized


def test_owner_only_literal_redaction_applies_to_keys_and_values(
    dataset_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "redaction.json"
    path.write_text(json.dumps({"literals": ["synthetic-customer-name"]}))
    path.chmod(0o600)
    monkeypatch.setenv("NPA_AGENT_DATASET_REDACTION_FILE", str(path))
    s3 = FakeS3()
    _emit(FakeStorage(s3), request="synthetic-customer-name", events=[{"synthetic-customer-name": "metadata"}])
    assert "synthetic-customer-name" not in b"".join(s3.objects.values()).decode()
    path.chmod(0o644)
    with pytest.raises(AgentRunDataError, match="owner-only"):
        _emit(FakeStorage(s3))


def test_wrapped_inline_image_cannot_leave_payload_fragments() -> None:
    assert redact("data:image/png;base64,first-line\nsecond-line\nlast-line") == "<inline-data-ref>"


@pytest.mark.parametrize("length", [40, 64])
@pytest.mark.parametrize("uppercase", [False, True])
def test_hex_digests_that_resemble_resource_ids_are_preserved(length: int, uppercase: bool) -> None:
    digest = "e00" + "a" * (length - 3)
    if uppercase:
        digest = digest.upper()
    assert redact({"sha256": digest, "version": digest}) == {"sha256": digest, "version": digest}
    non_hex_identifier = "u00" + "a" * (length - 3)
    assert non_hex_identifier not in redact(non_hex_identifier)


@pytest.mark.parametrize("dimension", ["tenant", "bucket", "prefix"])
def test_outbox_never_crosses_original_destination(
    dataset_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dimension: str,
) -> None:
    s3 = FakeS3()
    s3.fail_writes = True
    _emit(FakeStorage(s3))
    pending_path = next((tmp_path / "outbox").glob("*.json"))
    original = pending_path.read_bytes()
    tenant = "tenant-other" if dimension == "tenant" else "tenant-test"
    bucket = "other-bucket" if dimension == "bucket" else "test-bucket"
    prefix = "other-prefix" if dimension == "prefix" else "agent-dataset"
    monkeypatch.setenv("NPA_AGENT_DATASET_TENANT_ID", tenant)
    monkeypatch.setenv("NPA_AGENT_DATASET_URI", f"s3://{bucket}/{prefix}")
    s3.fail_writes = False
    assert flush_outbox(storage=s3, active_tenant_id=tenant, active_bucket=bucket) == []
    assert pending_path.read_bytes() == original
    assert s3.put_count == 0


@pytest.mark.parametrize("corruption", ["payload", "digest", "filename", "tenant", "legacy"])
def test_corrupt_or_unbound_outbox_is_retained_without_writes(
    dataset_env: None, tmp_path: Path, corruption: str,
) -> None:
    s3 = FakeS3()
    s3.fail_writes = True
    _emit(FakeStorage(s3))
    path = next((tmp_path / "outbox").glob("*.json"))
    envelope = json.loads(path.read_bytes())
    if corruption == "payload":
        envelope["payload"]["request"]["content"] = "changed after finalization"
    elif corruption == "digest":
        envelope["payload_sha256"] = "0" * 64
    elif corruption == "tenant":
        envelope["payload"]["scope"]["tenant_id"] = "tenant-other"
    elif corruption == "legacy":
        envelope = envelope["payload"]
    path.write_text(json.dumps(envelope))
    if corruption == "filename":
        changed = path.with_name("changed.json")
        path.rename(changed)
        path = changed
    original = path.read_bytes()
    s3.fail_writes = False
    assert flush_outbox(storage=s3, active_tenant_id="tenant-test", active_bucket="test-bucket") == []
    assert path.read_bytes() == original
    assert s3.put_count == 0


def test_uncertain_upload_retries_identical_finalized_bytes(dataset_env: None, tmp_path: Path) -> None:
    class UncertainS3(FakeS3):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        def get_object(self, *, Bucket, Key):
            if "/episodes/" in Key and Key in self.objects and self.fail_once:
                self.fail_once = False
                raise RuntimeError("synthetic lost read response")
            return super().get_object(Bucket=Bucket, Key=Key)

    s3 = UncertainS3()
    assert _emit(FakeStorage(s3))[0] == CollectionStatus.PENDING
    first = {key: body for key, body in s3.objects.items() if "/episodes/" in key}
    envelope = json.loads(next((tmp_path / "outbox").glob("*.json")).read_bytes())
    assert next(iter(first.values())) == json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":")).encode()
    assert not any("/receipts/" in key for key in s3.objects)
    assert flush_outbox(storage=s3, active_tenant_id="tenant-test", active_bucket="test-bucket") == ["ep-1"]
    assert {key: body for key, body in s3.objects.items() if "/episodes/" in key} == first
    assert len([key for key in s3.objects if "/receipts/" in key]) == 1


def test_divergent_episode_preserved_and_reported_as_conflict(
    dataset_env: None, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    s3 = FakeS3()
    assert _emit(FakeStorage(s3), request="first")[0] == CollectionStatus.COLLECTED
    first = dict(s3.objects)
    assert _emit(FakeStorage(s3), request="different")[0] == CollectionStatus.PENDING
    assert all(s3.objects[key] == body for key, body in first.items())
    assert len(_episode_payloads(s3)) == 2
    assert len([key for key in s3.objects if "/receipts/" in key]) == 1
    assert "content conflict" in caplog.text
    envelope = json.loads(next((tmp_path / "outbox").glob("*.json")).read_bytes())
    assert envelope["failure"] == "episode_conflict"


def test_conditional_write_is_never_downgraded(dataset_env: None) -> None:
    class UnsupportedConditionalS3(FakeS3):
        def __init__(self):
            super().__init__()
            self.unconditional_episode_writes = 0

        def put_object(self, *, Bucket, Key, Body, **kwargs):
            if "/episodes/" in Key:
                if "IfNoneMatch" in kwargs:
                    raise TypeError("synthetic unsupported conditional put")
                self.unconditional_episode_writes += 1
            return super().put_object(Bucket=Bucket, Key=Key, Body=Body, **kwargs)

    s3 = UnsupportedConditionalS3()
    assert _emit(FakeStorage(s3))[0] == CollectionStatus.PENDING
    assert s3.unconditional_episode_writes == 0
    assert not _episode_payloads(s3)


@pytest.mark.parametrize("value", [b"synthetic-image-bytes", object(), float("nan")])
def test_unsupported_values_never_reach_storage(dataset_env: None, value: object) -> None:
    s3 = FakeS3()
    with pytest.raises((AgentRunDataError, ValueError)):
        _emit(FakeStorage(s3), events=[{"value": value}])
    assert s3.put_count == 0


@pytest.mark.parametrize("episode_id", ["../escape", "/absolute", "", "contains/slash"])
def test_episode_ids_cannot_escape_outbox(dataset_env: None, episode_id: str) -> None:
    s3 = FakeS3()
    with pytest.raises(AgentRunDataError, match="safe stable identifiers"):
        _emit(FakeStorage(s3), episode_id=episode_id)
    assert s3.put_count == 0


def test_storage_factory_failure_preserves_product_and_pending_record(dataset_env: None, tmp_path: Path) -> None:
    calls = []

    def broken_factory():
        raise RuntimeError("synthetic factory error")

    @goal_episode_boundary(
        storage_factory=broken_factory,
        active_tenant_id=lambda: "tenant-test", active_bucket=lambda: "test-bucket",
    )
    def endpoint(payload):
        calls.append("executed")
        return {"ok": True}

    assert endpoint({}) == {"ok": True}
    assert calls == ["executed"]
    assert len(list((tmp_path / "outbox").glob("*.json"))) == 1


def test_malformed_usage_is_unknown_and_does_not_mask_product(dataset_env: None) -> None:
    s3 = FakeS3()

    @goal_episode_boundary(
        storage_factory=lambda: s3,
        active_tenant_id=lambda: "tenant-test", active_bucket=lambda: "test-bucket",
    )
    def endpoint(payload):
        return {"ok": True, "usage": {"prompt_tokens": None, "completion_tokens": "unknown"}}

    assert endpoint({})["ok"]
    routing = _episode_payloads(s3)[0]["routing"]
    assert "input_tokens" not in routing
    assert "output_tokens" not in routing


def test_record_assembly_failure_preserves_original_exception(dataset_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.agent_backend import trajectory

    def broken_record(**kwargs):
        raise RuntimeError("synthetic telemetry assembly failure")

    monkeypatch.setattr(trajectory, "_episode_record", broken_record)

    @goal_episode_boundary(storage_factory=FakeS3)
    def endpoint(payload):
        raise ValueError("original product failure")

    with pytest.raises(ValueError, match="original product failure"):
        endpoint({})


@pytest.mark.parametrize("same_payload", [True, False])
def test_concurrent_same_episode_claim_is_atomic(dataset_env: None, same_payload: bool) -> None:
    s3 = FakeS3()
    barrier = threading.Barrier(2)
    statuses = []
    errors = []

    def write(index):
        try:
            barrier.wait()
            statuses.append(_emit(FakeStorage(s3), request="same" if same_payload else str(index))[0])
        except Exception as exc:
            errors.append(type(exc).__name__)

    workers = [threading.Thread(target=write, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert not errors
    assert sorted(statuses) == ([CollectionStatus.COLLECTED] * 2 if same_payload else [CollectionStatus.COLLECTED, CollectionStatus.PENDING])
    assert len(_episode_payloads(s3)) == (1 if same_payload else 2)
    assert len([key for key in s3.objects if "/receipts/" in key]) == 1


def test_receipt_readback_failure_cannot_report_collected(dataset_env: None, tmp_path: Path) -> None:
    class ReceiptReadFailure(FakeS3):
        def __init__(self):
            super().__init__()
            self.fail_receipt = True

        def get_object(self, *, Bucket, Key):
            if "/receipts/" in Key and Key in self.objects and self.fail_receipt:
                raise RuntimeError("synthetic receipt read failure")
            return super().get_object(Bucket=Bucket, Key=Key)

    s3 = ReceiptReadFailure()
    assert _emit(FakeStorage(s3))[0] == CollectionStatus.PENDING
    assert list((tmp_path / "outbox").glob("*.json"))
    original = dict(s3.objects)
    s3.fail_receipt = False
    assert flush_outbox(storage=s3, active_tenant_id="tenant-test", active_bucket="test-bucket") == ["ep-1"]
    assert s3.objects == original


def test_tenant_survives_only_authorized_scope_field(dataset_env: None) -> None:
    s3 = FakeS3()
    _emit(FakeStorage(s3), request="tenant-test test-bucket", events=[{"tenant-test": "tenant-test"}])
    row = _episode_payloads(s3)[0]
    assert row["scope"]["tenant_id"] == "tenant-test"
    serialized = json.dumps(row)
    assert serialized.count("tenant-test") == 1
    assert "test-bucket" not in serialized


def test_outbox_publication_cannot_replace_concurrent_destination(
    dataset_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.agent_backend import trajectory

    def competing_writer(source, destination, **kwargs):
        envelope = json.loads(Path(source).read_bytes())
        envelope["destination_sha256"] = "f" * 64
        Path(destination).write_text(json.dumps(envelope))
        raise FileExistsError("synthetic competing process published first")

    monkeypatch.setattr(trajectory.os, "link", competing_writer)
    s3 = FakeS3()
    s3.fail_writes = True
    with pytest.raises(AgentRunDataError, match="concurrent outbox key conflicts"):
        _emit(FakeStorage(s3))
    row = json.loads(next((tmp_path / "outbox").glob("*.json")).read_bytes())
    assert row["destination_sha256"] == "f" * 64
    assert not list((tmp_path / "outbox").glob("*.tmp"))


@pytest.mark.parametrize("pending", [False, True])
def test_secret_aliases_and_bracketed_private_literals_never_serialize(
    dataset_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pending: bool,
) -> None:
    private_literal = "synthetic-customer-northern"
    marker_suffix_literal = "abc" * 4
    path = tmp_path / "redaction.json"
    path.write_text(json.dumps({"literals": [private_literal, marker_suffix_literal]}))
    path.chmod(0o600)
    monkeypatch.setenv("NPA_AGENT_DATASET_REDACTION_FILE", str(path))
    aliases = {
        "secret": "synthetic-secret-value",
        "client_secret": "synthetic-client-value",
        "accessToken": "synthetic-access-value",
        "refreshToken": "synthetic-refresh-value",
    }
    s3 = FakeS3()
    s3.fail_writes = pending
    _emit(
        FakeStorage(s3), request=f"<{private_literal}> <private-ref>-{marker_suffix_literal}",
        events=[{"arguments": aliases, "observation": "client_secret=synthetic-assigned-secret"}],
    )
    body = (
        next((tmp_path / "outbox").glob("*.json")).read_text()
        if pending else b"".join(s3.objects.values()).decode()
    )
    for forbidden in [private_literal, marker_suffix_literal, "synthetic-assigned-secret", *aliases.values()]:
        assert forbidden not in body


@pytest.mark.parametrize("parent_link", [False, True])
def test_outbox_rejects_symlink_components_before_creating_files(
    dataset_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent_link: bool,
) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("NPA_AGENT_DATASET_OUTBOX", str(link / "outbox" if parent_link else link))
    s3 = FakeS3()
    s3.fail_writes = True
    with pytest.raises(AgentRunDataError, match="symlink"):
        _emit(FakeStorage(s3))
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("ok", [False, True])
def test_endpoint_response_cannot_assert_objective_goal_verification(dataset_env: None, ok: bool) -> None:
    s3 = FakeS3()

    @goal_episode_boundary(
        storage_factory=lambda: s3,
        active_tenant_id=lambda: "tenant-test", active_bucket=lambda: "test-bucket",
    )
    def endpoint(payload):
        return {"ok": ok, "verified_by": ["untrusted self-reported success"], "verified": True}

    endpoint({})
    outcome = _episode_payloads(s3)[0]["outcome"]
    assert outcome["verified"] is False
    assert outcome["verified_by"] == []


def test_observed_endpoint_exception_is_objective_failure_evidence(dataset_env: None) -> None:
    s3 = FakeS3()

    @goal_episode_boundary(
        storage_factory=lambda: s3,
        active_tenant_id=lambda: "tenant-test", active_bucket=lambda: "test-bucket",
    )
    def endpoint(payload):
        raise ValueError("synthetic observed endpoint failure")

    with pytest.raises(ValueError):
        endpoint({})
    outcome = _episode_payloads(s3)[0]["outcome"]
    assert outcome == {
        "status": "failed", "verified": True, "verified_by": ["agent endpoint exception"],
        "artifact_uris": [], "operator_interventions": [], "preference_pairs": [],
    }


def test_flush_skips_fifo_without_attempting_a_read(
    dataset_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    fifo = outbox / "unexpected.json"
    os.mkfifo(fifo)
    read_attempts = []
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == fifo:
            read_attempts.append(path)
            raise AssertionError("FIFO must not be read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    assert flush_outbox(storage=FakeS3(), active_tenant_id="tenant-test", active_bucket="test-bucket") == []
    assert not read_attempts
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


def test_existing_outbox_fifo_is_rejected_without_reading(
    dataset_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = FakeS3()
    s3.fail_writes = True
    _emit(FakeStorage(s3))
    fifo = next((tmp_path / "outbox").glob("*.json"))
    fifo.unlink()
    os.mkfifo(fifo)
    read_attempts = []
    original = Path.read_bytes

    def guarded_read(path):
        if path == fifo:
            read_attempts.append(path)
            raise AssertionError("FIFO must not be read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(AgentRunDataError, match="regular file"):
        _emit(FakeStorage(s3))
    assert not read_attempts
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


@pytest.mark.parametrize("pending", [False, True])
def test_recognizable_unlabelled_tokens_never_serialize(dataset_env: None, tmp_path: Path, pending: bool) -> None:
    # Assemble clearly synthetic shapes; no credential is copied from runtime.
    tokens = [
        "eyJ" + "hbGciOiJub25lIn0" + "." + "eyJzdWIiOiJzeW50aGV0aWMifQ" + "." + "synthetic-signature",
        "xoxb" + "-" + "1" * 12 + "-" + "2" * 12 + "-" + "synthetic" * 4,
        "sk" + "_live_" + "synthetic" * 4,
        "AIza" + "synthetic" * 4,
        "nvapi" + "-" + "synthetic" * 8,
        "ASIA" + "A" * 16,
        "AKIA" + "B" * 16,
    ]
    s3 = FakeS3()
    s3.fail_writes = pending
    _emit(FakeStorage(s3), request=" ".join(tokens), events=[{"observation": {"text": " ".join(tokens)}}])
    body = (
        next((tmp_path / "outbox").glob("*.json")).read_text()
        if pending else b"".join(s3.objects.values()).decode()
    )
    assert all(token not in body for token in tokens)
