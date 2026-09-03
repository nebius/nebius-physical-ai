"""Verification matrix for goal-level agent trajectory collection."""

from __future__ import annotations

import asyncio
import json
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


def _emit(storage: FakeStorage, *, episode_id: str = "ep-1", request: str = "run"):
    return emit_trajectory(
        episode_id=episode_id,
        session_id="session-1",
        request_content=request,
        intent="run",
        trajectory=_trajectory(),
        outcome=_outcome(),
        routing={"grounded": False, "tier": "cheap", "model": "test"},
        versions={"agent": "test", "tools": {}},
        storage=storage,
        started_at="2026-08-30T00:00:00+00:00",
        ended_at="2026-08-30T00:01:00+00:00",
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
        resolve_dataset_config(active_tenant_id="other-tenant")
    with pytest.raises(AgentRunDataError, match="bucket does not match"):
        resolve_dataset_config(active_bucket="other-bucket")


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
    assert payload["collection"]["status"] == CollectionStatus.COLLECTED
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
    assert json.loads(body)["collection"]["status"] == CollectionStatus.PENDING

    s3.fail_writes = False
    assert flush_outbox(storage=FakeStorage(s3)) == ["pending"]
    assert not list((tmp_path / "outbox").glob("*.json"))
    uploaded = _episode_payloads(s3)[0]
    assert uploaded["collection"]["status"] == CollectionStatus.COLLECTED


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

    @goal_episode_boundary(storage_factory=lambda: FakeStorage(s3))
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

    @goal_episode_boundary(storage_factory=lambda: FakeStorage(s3))
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
