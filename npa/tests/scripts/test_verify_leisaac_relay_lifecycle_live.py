from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from npa.workbench.leisaac import session_attestation


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "verify_leisaac_relay_lifecycle_live.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_leisaac_relay_lifecycle_live", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rotated_manifest_stays_nonsecret_and_changes_attestation() -> None:
    module = _load_module()
    nonce = "b" * 64
    body = module._rotated_manifest(
        {
            "schema": "npa.leisaac.session.v2",
            "run_id": "existing-run",
            "session_attestation": "a" * 64,
            "expires_at": "2026-08-16T00:00:00Z",
        },
        nonce,
    )
    payload = json.loads(body)
    assert "session_nonce" not in payload
    assert payload["session_attestation"] == session_attestation(nonce)
    assert payload["expires_at"] == ""
    original_uri = "s3://bucket/prefix/existing-run/reports/leisaac-session.json"
    rotated_uri = module._rotated_manifest_uri(original_uri, nonce)
    assert rotated_uri != original_uri
    assert rotated_uri.startswith("s3://bucket/prefix/existing-run/credential-")
    assert rotated_uri.endswith("/reports/leisaac-session.json")
    root_rotated = module._rotated_manifest_uri(
        "s3://bucket/reports/leisaac-session.json", nonce
    )
    assert root_rotated.startswith("s3://bucket/credential-")
    assert "bucket//" not in root_rotated


def test_evidence_failure_records_type_without_exception_text() -> None:
    module = _load_module()
    provider_details = "ssh failed at user@203.0.113.10 in tenant-id"

    failure = module._evidence_failure(
        RuntimeError(provider_details), "operational_error"
    )

    assert failure == {"code": "operational_error", "type": "RuntimeError"}
    assert provider_details not in json.dumps(failure)


def test_owner_only_evidence_is_atomically_written(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "private" / "evidence.json"
    module._write_evidence(path, {"outcome": "success"})
    assert json.loads(path.read_text()) == {"outcome": "success"}
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_evidence_does_not_chmod_an_existing_caller_directory(tmp_path: Path) -> None:
    module = _load_module()
    parent = tmp_path / "caller-owned"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    path = parent / "evidence.json"

    module._write_evidence(path, {"outcome": "success"})

    assert parent.stat().st_mode & 0o777 == 0o755
    assert path.stat().st_mode & 0o777 == 0o600


def test_rotated_capability_write_is_create_only(monkeypatch) -> None:
    module = _load_module()
    writes = []

    class FakeS3:
        def put_object(self, **kwargs) -> None:
            writes.append(kwargs)

    monkeypatch.setattr(module.boto3, "client", lambda *_args, **_kwargs: FakeS3())
    module._put_manifest(
        storage={
            "bucket": "bucket",
            "endpoint": "https://storage.example",
            "access_key": "access",
            "secret_key": "secret",
            "region": "region",
        },
        manifest_uri="s3://bucket/run/credential-new/reports/leisaac-session.json",
        body=b"{}",
    )

    assert writes[0]["IfNoneMatch"] == "*"


def test_stale_backhaul_timeout_is_not_misreported_as_denial(monkeypatch) -> None:
    module = _load_module()

    class FakeSocket:
        def settimeout(self, _timeout: float) -> None:
            return None

    class FakeWebSocket:
        connection = FakeSocket()

        def send(self, _payload: bytes) -> None:
            return None

        def receive(self):
            raise socket.timeout

        def close(self) -> None:
            return None

    monkeypatch.setattr(module._WebSocket, "connect", lambda **_kwargs: FakeWebSocket())
    assert (
        module._stale_backhaul_denied(
            host="203.0.113.10",
            user="agent",
            password="password",
            stale_nonce="a" * 64,
            certificate_sha256="b" * 64,
            relay_connected=lambda: False,
        )
        is False
    )


def test_wait_closed_does_not_misreport_a_read_timeout_as_disconnect() -> None:
    module = _load_module()
    receives = iter((socket.timeout(), EOFError("closed")))

    class FakeWebSocket:
        calls = 0

        def receive(self, *, deadline):
            assert deadline > 0
            self.calls += 1
            raise next(receives)

    connection = FakeWebSocket()

    assert module._wait_closed(connection, timeout=1.0) >= 0
    assert connection.calls == 2


def test_wait_closed_polls_lock_health_during_a_silent_connection(monkeypatch) -> None:
    module = _load_module()
    checks = 0

    class FakeWebSocket:
        calls = 0

        def receive(self, *, deadline):
            self.calls += 1
            raise TimeoutError(f"silent until {deadline}")

    def check_lock() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("lifecycle lock renewal failed")

    connection = FakeWebSocket()
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)

    with pytest.raises(RuntimeError, match="lock renewal failed"):
        module._wait_closed(
            connection,
            timeout=140.0,
            progress_check=check_lock,
        )

    assert connection.calls == 1
    assert checks == 2


def test_baseline_video_failure_closes_control_for_safety_release(monkeypatch) -> None:
    module = _load_module()

    class FakeWebSocket:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    control = FakeWebSocket()
    video = FakeWebSocket()
    monkeypatch.setattr(module, "_press", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        module,
        "_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TimeoutError("baseline frame timed out")
        ),
    )

    with pytest.raises(TimeoutError, match="baseline frame timed out"):
        module._press_and_read_frame(
            control,
            video,
            run_id="existing-run",
            client_id="browser",
            sequence=1,
        )

    assert control.closed is True
    assert video.closed is True


def test_backhaul_probe_sends_rolling_upgrade_compatibility_hello(monkeypatch) -> None:
    module = _load_module()
    sent = []

    class FakeWebSocket:
        def send(self, payload: bytes) -> None:
            sent.append(payload)

    monkeypatch.setattr(
        module._WebSocket,
        "connect",
        lambda **_kwargs: FakeWebSocket(),
    )
    module._open_backhaul_probe(
        host="203.0.113.10",
        user="agent",
        password="password",
        nonce="a" * 64,
        certificate_sha256="b" * 64,
    )

    kind, stream_id, size = module.HEADER.unpack(sent[0][: module.HEADER.size])
    hello = json.loads(sent[0][module.HEADER.size :])
    assert (kind, stream_id, size) == (module.HELLO, 0, len(sent[0]) - 9)
    assert hello == {"nonce": "a" * 64, "peer_public_ip": "0.0.0.0"}


def test_stale_backhaul_websocket_close_is_a_denial(monkeypatch) -> None:
    module = _load_module()

    class FakeSocket:
        def settimeout(self, _timeout: float) -> None:
            return None

    class FakeWebSocket:
        connection = FakeSocket()

        def send(self, _payload: bytes) -> None:
            return None

        def receive(self):
            return 8, b"policy rejection"

        def close(self) -> None:
            return None

    monkeypatch.setattr(module._WebSocket, "connect", lambda **_kwargs: FakeWebSocket())
    assert module._stale_backhaul_denied(
        host="203.0.113.10",
        user="agent",
        password="password",
        stale_nonce="a" * 64,
        certificate_sha256="b" * 64,
        relay_connected=lambda: False,
    )


def test_stale_backhaul_drop_requires_reachable_disconnected_relay(
    monkeypatch,
) -> None:
    module = _load_module()
    relay_states = iter((False, RuntimeError("relay unavailable")))

    class FakeSocket:
        def settimeout(self, _timeout: float) -> None:
            return None

    class FakeWebSocket:
        connection = FakeSocket()

        def receive(self):
            raise EOFError("socket dropped")

        def close(self) -> None:
            return None

    def relay_connected() -> bool:
        state = next(relay_states)
        if isinstance(state, Exception):
            raise state
        return state

    monkeypatch.setattr(
        module,
        "_open_backhaul_probe",
        lambda **_kwargs: FakeWebSocket(),
    )
    with pytest.raises(RuntimeError, match="relay unavailable"):
        module._stale_backhaul_denied(
            host="203.0.113.10",
            user="agent",
            password="password",
            stale_nonce="a" * 64,
            certificate_sha256="b" * 64,
            relay_connected=relay_connected,
        )


def test_live_transport_requires_the_pinned_agent_certificate() -> None:
    module = _load_module()

    class FakeTLS:
        closed = False

        def getpeercert(self, *, binary_form: bool):
            assert binary_form is True
            return b"agent-certificate"

        def close(self) -> None:
            self.closed = True

    connection = FakeTLS()

    expected = hashlib.sha256(b"agent-certificate").hexdigest()
    module._verify_certificate(connection, expected)
    assert connection.closed is False

    with pytest.raises(RuntimeError, match="fingerprint changed"):
        module._verify_certificate(connection, "0" * 64)
    assert connection.closed is True


def test_pinned_https_body_read_recomputes_absolute_deadline(monkeypatch) -> None:
    module = _load_module()
    moments = iter((10.0, 10.0, 10.0, 10.0, 10.4, 10.6))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(moments))

    class FakeSocket:
        def __init__(self) -> None:
            self.timeouts = []

        def settimeout(self, timeout: float) -> None:
            self.timeouts.append(timeout)

        def getpeercert(self, *, binary_form: bool):
            assert binary_form is True
            return b"agent-certificate"

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.reads = 0

        def read1(self, _size: int) -> bytes:
            self.reads += 1
            return b"chunk"

        def getheaders(self):
            return []

    response = FakeResponse()

    class FakeHTTPSConnection:
        def __init__(self, *_args, **_kwargs) -> None:
            self.sock = FakeSocket()

        def connect(self) -> None:
            return None

        def request(self, *_args, **_kwargs) -> None:
            return None

        def getresponse(self):
            return response

        def close(self) -> None:
            return None

    monkeypatch.setattr(module.http.client, "HTTPSConnection", FakeHTTPSConnection)

    with pytest.raises(TimeoutError, match="request deadline expired"):
        module._pinned_https_request(
            host="203.0.113.10",
            path="/api/health",
            method="GET",
            user="agent",
            password="password",
            certificate_sha256=hashlib.sha256(b"agent-certificate").hexdigest(),
            timeout=0.5,
        )

    assert response.reads == 1


def test_websocket_upgrade_eof_fails_without_spinning(monkeypatch) -> None:
    module = _load_module()
    certificate = b"agent-certificate"

    class FakeTLS:
        closed = False

        def getpeercert(self, *, binary_form: bool):
            assert binary_form is True
            return certificate

        def settimeout(self, _timeout: float) -> None:
            return None

        def sendall(self, _payload: bytes) -> None:
            return None

        def recv(self, _size: int) -> bytes:
            return b""

        def close(self) -> None:
            self.closed = True

    connection = FakeTLS()

    class FakeRaw:
        def settimeout(self, _timeout: float) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeContext:
        check_hostname = True
        verify_mode = None

        def wrap_socket(self, _raw, *, server_hostname: str):
            assert server_hostname == "203.0.113.10"
            return connection

    monkeypatch.setattr(module.ssl, "create_default_context", FakeContext)
    monkeypatch.setattr(
        module.socket,
        "create_connection",
        lambda *_args, **_kwargs: FakeRaw(),
    )

    with pytest.raises(EOFError, match="HTTP upgrade"):
        module._WebSocket.connect(
            host="203.0.113.10",
            path="/api/leisaac/transport/control",
            subprotocol="npa.leisaac.control.v1",
            authorization="basic",
            certificate_sha256=hashlib.sha256(certificate).hexdigest(),
            origin="https://203.0.113.10",
        )
    assert connection.closed is True


def test_websocket_absolute_deadline_is_not_extended_by_heartbeats(monkeypatch) -> None:
    module = _load_module()
    moments = iter((10.0, 10.6))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(moments))

    class FakeTLS:
        def __init__(self) -> None:
            self.timeouts = []
            self.sent = []

        def settimeout(self, timeout: float) -> None:
            self.timeouts.append(timeout)

        def recv(self, _size: int) -> bytes:
            return b"\x89\x00"  # unmasked server ping

        def sendall(self, payload: bytes) -> None:
            self.sent.append(payload)

    connection = FakeTLS()
    websocket = module._WebSocket(connection)

    with pytest.raises(TimeoutError, match="operation deadline expired"):
        websocket.receive(deadline=10.5)

    assert connection.timeouts == [pytest.approx(0.5)]
    assert connection.sent  # the first ping was answered before the deadline


def test_scale_to_zero_waits_until_no_deployment_replica_is_active(monkeypatch) -> None:
    module = _load_module()
    results = iter(
        (
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "spec": {"replicas": 0},
                        "status": {"replicas": 1, "readyReplicas": 1},
                    }
                ),
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"spec": {"replicas": 0}, "status": {}}),
                stderr="",
            ),
        )
    )
    calls = []
    monkeypatch.setattr(
        module,
        "_kubectl",
        lambda *args: calls.append(args) or next(results),
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    module._scale_deployment("cluster", "namespace", "deployment", 0)

    assert calls[0][-1][-1] == "--replicas=0"
    assert len(calls) == 3


def test_missing_relay_process_does_not_count_as_disconnected(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_relay_connection_state",
        lambda _ssh: (_ for _ in ()).throw(RuntimeError("relay stopped")),
    )

    with pytest.raises(RuntimeError, match="last state: None"):
        module._wait_relay_connection(object(), connected=False, timeout=0.0)


def test_best_effort_cleanup_runs_later_actions_after_failure() -> None:
    module = _load_module()
    called: list[str] = []

    def fail_restore() -> None:
        called.append("restore")
        raise RuntimeError("partial restore")

    failures = module._best_effort_cleanup(
        (
            ("restore credential", fail_restore),
            ("scale deployment", lambda: called.append("scale")),
            ("verify relay", lambda: called.append("verify")),
        )
    )

    assert called == ["restore", "scale", "verify"]
    assert [(label, str(exc)) for label, exc in failures] == [
        ("restore credential", "partial restore")
    ]


def test_browser_connections_close_before_rollback_and_continue_after_error() -> None:
    module = _load_module()
    calls: list[str] = []

    class Connection:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            calls.append(f"close-{self.name}")
            if self.fail:
                raise RuntimeError(f"{self.name} close failed")

    connections = [Connection("control"), Connection("video", fail=True)]
    failures = module._close_browser_connections(connections)
    calls.append("rollback")

    assert calls == ["close-video", "close-control", "rollback"]
    assert connections == []
    assert [(label, type(exc).__name__) for label, exc in failures] == [
        ("close browser connection 1", "RuntimeError")
    ]


def test_scale_down_failure_still_requires_replica_restoration(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def scale(_context, _namespace, _deployment, replicas, **_kwargs):
        calls.append(replicas)
        if replicas == 0:
            raise RuntimeError("scale observation failed")

    monkeypatch.setattr(module, "_scale_deployment", scale)
    restoration = module._ReplicaRestoration("cluster", "namespace", "deployment")

    with pytest.raises(RuntimeError, match="scale observation failed"):
        restoration.scale_down()
    assert restoration.required is True

    restoration.scale_up()
    assert calls == [0, 1]
    assert restoration.required is False


def test_cleanup_replica_restoration_bypasses_failed_lease_check(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def scale(
        _context,
        _namespace,
        _deployment,
        replicas,
        *,
        progress_check=None,
    ) -> None:
        calls.append((replicas, progress_check))

    def failed_lease() -> None:
        raise RuntimeError("lifecycle lease lost")

    monkeypatch.setattr(module, "_scale_deployment", scale)
    restoration = module._ReplicaRestoration(
        "cluster",
        "namespace",
        "deployment",
        progress_check=failed_lease,
    )
    restoration.required = True

    restoration.scale_up(enforce_lease=False)

    assert calls == [(1, None)]
    assert restoration.required is False


def test_evidence_error_does_not_replace_active_operational_failure(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_write_evidence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    module._publish_evidence(
        tmp_path / "evidence.json",
        {"outcome": "failure"},
        active_failure=RuntimeError("relay failed"),
    )
    with pytest.raises(OSError, match="disk full"):
        module._publish_evidence(
            tmp_path / "evidence.json",
            {"outcome": "success"},
            active_failure=None,
        )


def test_lifecycle_lock_contention_fails_before_mutation(monkeypatch) -> None:
    module = _load_module()
    calls = []
    results = iter(
        (
            SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Error from server (AlreadyExists)",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "metadata": {
                            "resourceVersion": "7",
                            "annotations": {
                                "npa.nebius.com/lifecycle-holder": "holder-b",
                                "npa.nebius.com/lifecycle-acquired-epoch": "999",
                                "npa.nebius.com/lifecycle-renewed-epoch": "999",
                            },
                        },
                    }
                ),
                stderr="",
            ),
        )
    )
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(
        module,
        "_kubectl",
        lambda *args, **kwargs: calls.append((args, kwargs)) or next(results),
    )

    with pytest.raises(RuntimeError, match="already holds"):
        module._acquire_lifecycle_lock("cluster", "namespace", "deployment", "holder-a")

    assert calls[0][0][2] == ["create", "-f", "-"]
    created = json.loads(calls[0][1]["stdin"])
    assert created["kind"] == "Secret"
    assert created["metadata"]["name"] == module._lifecycle_lock_name("deployment")


def test_main_locks_before_reading_mutable_relay_state(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    calls: list[str] = []
    service = {
        "metadata": {
            "annotations": {
                "npa.nebius.com/agent-project": "project",
                "npa.nebius.com/agent-name": "agent",
            }
        }
    }
    monkeypatch.setattr(
        module,
        "_kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(service),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        module,
        "_acquire_lifecycle_lock",
        lambda *_args: calls.append("lock") or "lifecycle-lock",
    )
    monkeypatch.setattr(
        module,
        "_require_lifecycle_lock_permissions",
        lambda *_args: calls.append("rbac"),
    )

    class FakeHeartbeat:
        def __init__(self, *_args) -> None:
            return None

        def start(self) -> None:
            calls.append("heartbeat-start")

        def stop(self) -> None:
            calls.append("heartbeat-stop")

        def assert_healthy(self) -> None:
            return None

    monkeypatch.setattr(module, "_LifecycleLockHeartbeat", FakeHeartbeat)

    def fail_context(*_args):
        calls.append("relay-context")
        raise RuntimeError("context unavailable")

    monkeypatch.setattr(module, "_agent_relay_context", fail_context)
    monkeypatch.setattr(
        module,
        "_release_lifecycle_lock",
        lambda *_args: calls.append("release"),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT),
            "--project",
            "project",
            "--name",
            "agent",
            "--context",
            "cluster",
            "--namespace",
            "namespace",
            "--run-id",
            "existing-run",
            "--evidence",
            str(tmp_path / "evidence.json"),
        ],
    )

    with pytest.raises(RuntimeError, match="context unavailable"):
        module.main()

    assert calls == [
        "rbac",
        "lock",
        "heartbeat-start",
        "relay-context",
        "heartbeat-stop",
        "release",
    ]


def test_stale_lifecycle_lock_is_reclaimed_with_resource_version(monkeypatch) -> None:
    module = _load_module()
    calls = []
    results = iter(
        (
            SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Error from server (AlreadyExists)",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "metadata": {
                            "resourceVersion": "7",
                            "annotations": {
                                "npa.nebius.com/lifecycle-holder": "abandoned",
                                "npa.nebius.com/lifecycle-acquired-epoch": "100",
                                "npa.nebius.com/lifecycle-renewed-epoch": "100",
                            },
                        },
                    }
                ),
                stderr="",
            ),
            SimpleNamespace(returncode=0, stdout="replaced", stderr=""),
        )
    )
    monkeypatch.setattr(
        module.time,
        "time",
        lambda: 100.0 + module._LIFECYCLE_LOCK_STALE_SECONDS + 1.0,
    )
    monkeypatch.setattr(
        module,
        "_kubectl",
        lambda *args, **kwargs: calls.append((args, kwargs)) or next(results),
    )

    assert module._acquire_lifecycle_lock(
        "cluster", "namespace", "deployment", "holder-a"
    ) == module._lifecycle_lock_name("deployment")
    replacement = json.loads(calls[2][1]["stdin"])
    assert calls[2][0][2] == ["replace", "-f", "-"]
    assert replacement["metadata"]["resourceVersion"] == "7"
    assert replacement["kind"] == "Secret"
    assert (
        replacement["metadata"]["annotations"]["npa.nebius.com/lifecycle-holder"]
        == "holder-a"
    )


def test_lifecycle_lock_release_is_an_atomic_reclaimable_replace(monkeypatch) -> None:
    module = _load_module()
    results = iter(
        (
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "metadata": {
                            "resourceVersion": "8",
                            "uid": "lock-uid",
                            "annotations": {
                                "npa.nebius.com/lifecycle-holder": "holder-a"
                            },
                        },
                    }
                ),
                stderr="",
            ),
            SimpleNamespace(returncode=0, stdout="replaced", stderr=""),
            SimpleNamespace(returncode=0, stdout="deleted", stderr=""),
        )
    )
    calls = []
    monkeypatch.setattr(
        module,
        "_kubectl",
        lambda *args, **kwargs: calls.append((args, kwargs)) or next(results),
    )

    module._release_lifecycle_lock(
        "cluster", "namespace", "deployment-lifecycle-lock", "holder-a"
    )

    assert calls[0][0][2][:3] == ["get", "secret", "deployment-lifecycle-lock"]
    assert calls[1][0][2] == ["replace", "-f", "-"]
    assert calls[2][0][2] == [
        "delete",
        "secret",
        "deployment-lifecycle-lock",
        "--ignore-not-found=true",
    ]
    released = json.loads(calls[1][1]["stdin"])
    assert released["kind"] == "Secret"
    assert released["type"] == "Opaque"
    assert released["metadata"]["name"] == "deployment-lifecycle-lock"
    assert released["metadata"]["namespace"] == "namespace"
    assert released["metadata"]["resourceVersion"] == "8"
    assert released["metadata"]["uid"] == "lock-uid"
    annotations = released["metadata"]["annotations"]
    assert annotations["npa.nebius.com/lifecycle-holder"] == ""
    assert float(annotations["npa.nebius.com/lifecycle-acquired-epoch"]) > 0
    assert (
        annotations["npa.nebius.com/lifecycle-renewed-epoch"]
        == annotations["npa.nebius.com/lifecycle-acquired-epoch"]
    )


def test_lifecycle_lock_names_retain_full_deployment_identity() -> None:
    module = _load_module()
    prefix = "leisaac-" + "a" * 80
    first = module._lifecycle_lock_name(prefix + "-first")
    second = module._lifecycle_lock_name(prefix + "-second")

    assert first != second
    assert len(first) <= 63
    assert len(second) <= 63


def test_lifecycle_lock_heartbeat_renews_until_stopped(monkeypatch) -> None:
    module = _load_module()
    renewals = []

    class FakeEvent:
        calls = 0

        def wait(self, _seconds: float) -> bool:
            self.calls += 1
            return self.calls > 1

    heartbeat = module._LifecycleLockHeartbeat("cluster", "namespace", "lock", "holder")
    heartbeat.stop_event = FakeEvent()
    monkeypatch.setattr(
        module,
        "_renew_lifecycle_lock",
        lambda *args: renewals.append(args),
    )

    heartbeat._run()

    assert renewals == [("cluster", "namespace", "lock", "holder")]
    heartbeat.assert_healthy()


def test_lifecycle_lock_heartbeat_latches_a_renewal_failure(monkeypatch) -> None:
    module = _load_module()

    class FakeEvent:
        def wait(self, _seconds: float) -> bool:
            return False

    heartbeat = module._LifecycleLockHeartbeat("cluster", "namespace", "lock", "holder")
    heartbeat.stop_event = FakeEvent()
    monkeypatch.setattr(
        module,
        "_renew_lifecycle_lock",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("API unavailable")),
    )

    heartbeat._run()

    with pytest.raises(RuntimeError, match="lock renewal failed"):
        heartbeat.assert_healthy()


def test_lock_cleanup_failure_changes_success_evidence_to_failure() -> None:
    module = _load_module()
    evidence = {"outcome": "success"}

    module._record_lock_cleanup_failures(
        evidence,
        [("release lifecycle lock", RuntimeError("delete failed"))],
    )

    assert evidence["outcome"] == "failure"
    assert evidence["failure"] == "lifecycle lock cleanup failed"
    assert evidence["lock_cleanup_failure"] == [
        {
            "operation": "release lifecycle lock",
            "code": "lock_cleanup_error",
            "type": "RuntimeError",
        }
    ]
    assert "delete failed" not in json.dumps(evidence)


def test_lock_cleanup_failure_preserves_primary_failure_evidence() -> None:
    module = _load_module()
    evidence = {"outcome": "failure", "failure": "primary relay failure"}

    module._record_lock_cleanup_failures(
        evidence,
        [("release lifecycle lock", RuntimeError("delete failed"))],
    )

    assert evidence["failure"] == "primary relay failure"
    assert evidence["lock_cleanup_failure"] == [
        {
            "operation": "release lifecycle lock",
            "code": "lock_cleanup_error",
            "type": "RuntimeError",
        }
    ]
    assert "delete failed" not in json.dumps(evidence)


def test_relay_wait_stops_immediately_when_lock_heartbeat_fails(monkeypatch) -> None:
    module = _load_module()
    relay_checks = []
    monkeypatch.setattr(
        module,
        "_relay_connection_state",
        lambda _ssh: relay_checks.append(True) or False,
    )

    with pytest.raises(RuntimeError, match="lock renewal failed"):
        module._wait_relay_connection(
            object(),
            connected=True,
            progress_check=lambda: (_ for _ in ()).throw(
                RuntimeError("lock renewal failed")
            ),
        )

    assert relay_checks == []


def test_browser_socket_wait_retries_through_negative_manifest_cache(
    monkeypatch,
) -> None:
    module = _load_module()
    expected = (object(), object(), {"available": True})
    attempts = iter(
        (
            RuntimeError("browser transport authorization returned HTTP 404"),
            RuntimeError("browser transport authorization returned HTTP 404"),
            expected,
        )
    )

    def browser_sockets(**_kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(module, "_browser_sockets", browser_sockets)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    assert (
        module._wait_browser_sockets(
            host="203.0.113.10",
            run_id="existing-run",
            user="agent",
            password="password",
            certificate_sha256="b" * 64,
        )
        == expected
    )


def test_browser_socket_wait_preserves_last_error_on_timeout(monkeypatch) -> None:
    module = _load_module()
    observed = iter((10.0, 10.0, 10.6))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(observed))
    monkeypatch.setattr(
        module,
        "_browser_sockets",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("browser transport authorization returned HTTP 404")
        ),
    )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        module._wait_browser_sockets(
            host="203.0.113.10",
            run_id="existing-run",
            user="agent",
            password="password",
            certificate_sha256="b" * 64,
            timeout=0.5,
            poll_interval=0.0,
        )


def test_live_proof_requires_a_healthy_idle_recorder() -> None:
    module = _load_module()
    module._require_healthy_idle(
        health_http=200,
        health={"ok": True},
        status={"available": True, "recorder": {"state": "idle"}},
    )

    for health_http, health, status in (
        (503, {"ok": True}, {"available": True, "recorder": {"state": "idle"}}),
        (200, {"ok": False}, {"available": True, "recorder": {"state": "idle"}}),
        (200, {"ok": True}, {"available": False, "recorder": {"state": "idle"}}),
        (200, {"ok": True}, {"available": True, "recorder": {"state": "recording"}}),
    ):
        with pytest.raises(RuntimeError, match="recorder is idle"):
            module._require_healthy_idle(
                health_http=health_http,
                health=health,
                status=status,
            )


def test_forced_release_requires_empty_keys_and_an_advanced_sequence() -> None:
    module = _load_module()
    assert (
        module._forced_release_count(
            {"keys_down": [], "next_seq": 13},
            pressed_sequence=10,
            phase="credential expiry",
        )
        == 2
    )
    with pytest.raises(RuntimeError, match="did not durably release"):
        module._forced_release_count(
            {"keys_down": ["W"], "next_seq": 13},
            pressed_sequence=10,
            phase="credential expiry",
        )
    with pytest.raises(RuntimeError, match="did not durably release"):
        module._forced_release_count(
            {"keys_down": [], "next_seq": 11},
            pressed_sequence=10,
            phase="credential expiry",
        )


def test_disconnect_release_wait_observes_status_without_claiming_owner(
    monkeypatch,
) -> None:
    module = _load_module()
    statuses = iter(
        (
            {"input_events": 10, "applied_inputs": 10},
            {"input_events": 11, "applied_inputs": 10},
            {"input_events": 11, "applied_inputs": 11},
        )
    )
    calls = []

    def browser_status(**kwargs):
        calls.append(kwargs)
        return next(statuses)

    monkeypatch.setattr(module, "_browser_status", browser_status)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    elapsed = module._wait_controller_release(
        host="203.0.113.10",
        run_id="existing-run",
        user="agent",
        password="password",
        certificate_sha256="a" * 64,
        after_input_events=10,
    )

    assert elapsed >= 0
    assert len(calls) == 3
    assert all(call["timeout"] > 0 for call in calls)


def test_disconnect_release_wait_rejects_a_late_status_success(monkeypatch) -> None:
    module = _load_module()
    observed = iter((10.0, 10.0, 10.6, 10.6))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(observed))
    monkeypatch.setattr(
        module,
        "_browser_status",
        lambda **_kwargs: {"input_events": 11, "applied_inputs": 11},
    )

    with pytest.raises(RuntimeError, match="did not settle within 0.5s"):
        module._wait_controller_release(
            host="203.0.113.10",
            run_id="existing-run",
            user="agent",
            password="password",
            certificate_sha256="a" * 64,
            after_input_events=10,
            timeout=0.5,
            poll_interval=0.0,
        )


def test_relay_recovery_and_release_share_the_disconnect_deadline(monkeypatch) -> None:
    module = _load_module()
    moments = iter((10.0, 12.0, 15.0, 20.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(moments))
    calls = []
    monkeypatch.setattr(
        module,
        "_wait_relay_ready",
        lambda *_args, **kwargs: calls.append(("relay", kwargs["timeout"])),
    )
    monkeypatch.setattr(
        module,
        "_wait_controller_release",
        lambda **kwargs: calls.append(("release", kwargs["timeout"])),
    )

    module._wait_relay_and_release(
        object(),
        "a" * 64,
        release_deadline=40.0,
        host="203.0.113.10",
        run_id="existing-run",
        user="agent",
        password="password",
        certificate_sha256="b" * 64,
        after_input_events=10,
        phase="relay restart",
    )

    assert calls == [("relay", 30.0), ("release", 25.0)]


def test_relay_and_release_rejects_success_after_original_deadline(monkeypatch) -> None:
    module = _load_module()
    moments = iter((10.0, 12.0, 15.0, 41.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(module, "_wait_relay_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_wait_controller_release", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="within 30s of disconnect"):
        module._wait_relay_and_release(
            object(),
            "a" * 64,
            release_deadline=40.0,
            host="203.0.113.10",
            run_id="existing-run",
            user="agent",
            password="password",
            certificate_sha256="b" * 64,
            after_input_events=10,
            phase="relay restart",
        )


def test_restart_safety_deadline_starts_before_relay_mutation(monkeypatch) -> None:
    module = _load_module()
    moments = iter((10.0, 25.0, 26.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(moments))
    calls = []
    monkeypatch.setattr(
        module,
        "_install_agent_relay",
        lambda *_args, **_kwargs: calls.append("restart"),
    )
    monkeypatch.setattr(
        module,
        "_wait_closed",
        lambda *_args, **kwargs: (
            calls.append(("disconnect", kwargs["timeout"])) or 0.25
        ),
    )

    disconnected, started, deadline = module._restart_relay_for_release_proof(
        object(),
        object(),
        run_id="existing-run",
        session_nonce="a" * 64,
        manifest_uri="s3://bucket/run/reports/leisaac-session.json",
    )

    assert calls == ["restart", ("disconnect", 15.0)]
    assert (disconnected, started, deadline) == (0.25, 10.0, 40.0)


def test_restart_safety_deadline_rejects_a_stalled_restart(monkeypatch) -> None:
    module = _load_module()
    moments = iter((10.0, 41.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(moments))
    waited = []
    monkeypatch.setattr(module, "_install_agent_relay", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module, "_wait_closed", lambda *_args, **_kwargs: waited.append(True)
    )

    with pytest.raises(RuntimeError, match="within 30s of disconnect"):
        module._restart_relay_for_release_proof(
            object(),
            object(),
            run_id="existing-run",
            session_nonce="a" * 64,
            manifest_uri="s3://bucket/run/reports/leisaac-session.json",
        )

    assert waited == []


def test_wait_relay_ready_rejects_ready_observed_after_deadline(monkeypatch) -> None:
    module = _load_module()
    moments = iter((10.0, 10.0, 10.6))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(
        module,
        "_relay_status",
        lambda *_args, **_kwargs: {"state": "ready"},
    )

    with pytest.raises(RuntimeError, match="ready after deadline"):
        module._wait_relay_ready(object(), "a" * 64, timeout=0.5)


def test_release_deadline_fails_before_another_recovery_attempt(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module.time, "monotonic", lambda: 40.1)
    called = []
    monkeypatch.setattr(
        module,
        "_wait_relay_ready",
        lambda *_args, **_kwargs: called.append(True),
    )

    with pytest.raises(RuntimeError, match="within 30s of disconnect"):
        module._wait_relay_and_release(
            object(),
            "a" * 64,
            release_deadline=40.0,
            host="203.0.113.10",
            run_id="existing-run",
            user="agent",
            password="password",
            certificate_sha256="b" * 64,
            after_input_events=10,
            phase="relay restart",
        )

    assert called == []


def test_resume_wait_closes_busy_attempt_before_returning_live_pair(
    monkeypatch,
) -> None:
    module = _load_module()
    responses = iter(
        (
            {"type": "error", "code": "controller_busy"},
            {"type": "resumed", "lease_id": "new", "next_seq": 3},
        )
    )
    pairs = []

    class FakeSocket:
        def __init__(self, response=None) -> None:
            self.response = response
            self.closed = False
            self.connection = self
            self.timeouts = []

        def settimeout(self, timeout) -> None:
            self.timeouts.append(timeout)

        def send(self, _payload) -> None:
            return None

        def receive_json(self, **_kwargs):
            return self.response

        def close(self) -> None:
            self.closed = True

    def browser_sockets(**_kwargs):
        pair = (FakeSocket(next(responses)), FakeSocket())
        pairs.append(pair)
        return *pair, {"available": True}

    monkeypatch.setattr(module, "_browser_sockets", browser_sockets)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    control, video, status, resumed = module._wait_resumed_browser_sockets(
        host="203.0.113.10",
        run_id="existing-run",
        user="agent",
        password="password",
        certificate_sha256="a" * 64,
        client_id="browser",
        last_acked_seq=1,
        lease_id="old",
    )

    assert pairs[0][0].closed and pairs[0][1].closed
    assert control is pairs[1][0] and video is pairs[1][1]
    assert control.closed is False and video.closed is False
    assert control.timeouts[-1] == module._WEBSOCKET_OPERATION_TIMEOUT_SECONDS
    assert video.timeouts[-1] == module._WEBSOCKET_OPERATION_TIMEOUT_SECONDS
    assert status == {"available": True}
    assert resumed["next_seq"] == 3


def test_resume_wait_rejects_late_success_and_closes_pair(monkeypatch) -> None:
    module = _load_module()
    observed = iter((10.0, 10.0, 10.0, 10.6, 10.6))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(observed))

    class FakeConnection:
        def settimeout(self, _timeout) -> None:
            return None

    class FakeSocket:
        def __init__(self) -> None:
            self.connection = FakeConnection()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    control = FakeSocket()
    video = FakeSocket()
    monkeypatch.setattr(
        module,
        "_browser_sockets",
        lambda **_kwargs: (control, video, {"available": True}),
    )
    monkeypatch.setattr(
        module,
        "_resume",
        lambda *_args, **_kwargs: {"type": "resumed", "lease_id": "b" * 64},
    )

    with pytest.raises(RuntimeError, match="did not resume within 0.5s"):
        module._wait_resumed_browser_sockets(
            host="203.0.113.10",
            run_id="existing-run",
            user="agent",
            password="password",
            certificate_sha256="a" * 64,
            client_id="browser",
            last_acked_seq=1,
            lease_id="a" * 64,
            timeout=0.5,
            poll_interval=0.0,
        )

    assert control.closed and video.closed


def test_rotated_lease_rejects_reused_or_invalid_capability() -> None:
    module = _load_module()
    old = "a" * 64
    new = "b" * 64

    assert (
        module._require_rotated_lease(
            {"lease_id": new}, prior_lease_id=old, phase="relay restart"
        )
        == new
    )
    for candidate in (old, "not-a-lease", ""):
        with pytest.raises(RuntimeError, match="did not rotate"):
            module._require_rotated_lease(
                {"lease_id": candidate},
                prior_lease_id=old,
                phase="relay restart",
            )
