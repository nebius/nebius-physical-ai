from __future__ import annotations

import base64
import hashlib
import json
import re
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.clients.ssh import SSHTimeoutError
from npa.cli.workbench.leisaac import (
    _TransientRelayStatusError,
    _agent_artifact_storage,
    _delete_resources,
    _existing_relay_contract,
    _external_ip,
    _install_agent_relay,
    _node_internal_ip,
    _kubectl,
    _load_manifest,
    _put_manifest,
    _relay_media_server,
    _relay_status,
    _require_lifecycle_lock_permissions,
    _release_lifecycle_lock,
    _remove_agent_relay,
    _select_agent_leisaac_run,
    _wait_timeout,
    _wait_ready,
    _wait_relay_status,
    app,
)
from npa.agent_backend.leisaac_registry import DEFAULT_TASK, REGISTRY_FINGERPRINT


IMAGE = "registry.example/npa-leisaac@sha256:" + "1" * 64
runner = CliRunner()


@pytest.mark.parametrize(
    "labels",
    [
        {"nvidia.com/gpu.product": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"},
        {"nebius.com/gpu-name": "RTX6000"},
    ],
)
def test_node_internal_ip_accepts_either_verified_rtx6000_label(
    monkeypatch, labels
) -> None:
    def kubectl(_context, _namespace, args, **_kwargs):
        assert args == ["get", "nodes", "-o", "json"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"labels": labels},
                            "status": {
                                "conditions": [{"type": "Ready", "status": "True"}],
                                "addresses": [
                                    {"type": "InternalIP", "address": "10.0.0.8"}
                                ],
                            },
                        }
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("npa.cli.workbench.leisaac._kubectl", kubectl)

    assert _node_internal_ip("cluster", "namespace") == "10.0.0.8"


class _FakeLifecycleLease:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events

    def assert_healthy(self) -> None:
        if self.events is not None:
            self.events.append("checked")

    def close(self) -> None:
        if self.events is not None:
            self.events.append("closed")


def test_kubectl_preserves_unbounded_mutation_contract(monkeypatch) -> None:
    calls = []

    def run_kubectl(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("npa.cli.workbench.leisaac.run_kubectl", run_kubectl)

    _kubectl("cluster", "namespace", ["apply", "-f", "-"], stdin="manifest")

    assert calls == [
        (
            ["--namespace", "namespace", "apply", "-f", "-"],
            {"context": "cluster", "stdin": "manifest", "timeout": None},
        )
    ]


def test_lifecycle_lock_permissions_are_preflighted_before_mutation(
    monkeypatch,
) -> None:
    calls = []

    def kubectl(_context, _namespace, args, **_kwargs):
        calls.append(args)
        allowed = args[2] not in {"create", "update"}
        return SimpleNamespace(
            returncode=0,
            stdout="yes\n" if allowed else "no\n",
            stderr="",
        )

    monkeypatch.setattr("npa.cli.workbench.leisaac._kubectl", kubectl)

    with pytest.raises(RuntimeError, match="missing or unverified: create, update"):
        _require_lifecycle_lock_permissions("cluster", "namespace")

    assert calls == [
        ["auth", "can-i", verb, "secrets"]
        for verb in ("get", "create", "update", "delete")
    ]


def test_lifecycle_lock_release_is_an_atomic_reclaimable_replace(monkeypatch) -> None:
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
        "npa.cli.workbench.leisaac._kubectl",
        lambda *args, **kwargs: calls.append((args, kwargs)) or next(results),
    )

    _release_lifecycle_lock(
        "cluster", "namespace", "deployment-lifecycle-lock", "holder-a"
    )

    assert calls[0][0][2][:3] == [
        "get",
        "secret",
        "deployment-lifecycle-lock",
    ]
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


def test_agent_artifact_storage_uses_owner_only_project_credentials(
    monkeypatch,
) -> None:
    record = {"region": "us-central1"}
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._agent_record", lambda *_args: record
    )
    monkeypatch.setattr(
        "npa.cli.agent._resolve_agent_storage_credentials",
        lambda project, selected: (
            "bucket",
            "agent-prefix",
            "https://storage.example",
            "access",
            "secret",
            "service-account",
        ),
    )

    storage = _agent_artifact_storage("rtxpro", "opendreamer")

    assert storage == {
        "bucket": "bucket",
        "prefix": "agent-prefix",
        "endpoint": "https://storage.example",
        "access_key": "access",
        "secret_key": "secret",
        "region": "us-central1",
    }
    assert "credentials" not in record


def test_select_agent_leisaac_run_pins_tls_before_sending_credentials(
    monkeypatch,
) -> None:
    certificate = b"agent-certificate"
    requests = []
    response_timeouts = []

    class FakeTLS:
        def getpeercert(self, *, binary_form=False):
            assert binary_form is True
            return certificate

        def settimeout(self, seconds):
            response_timeouts.append(seconds)

        def close(self):
            pass

    class FakeContext:
        check_hostname = True
        verify_mode = None

        def wrap_socket(self, _raw, *, server_hostname):
            assert server_hostname == "8.8.4.4"
            return FakeTLS()

    class FakeResponse:
        status = 200

        def read(self, limit):
            assert limit == 131073
            return b'{"selected":true,"run_id":"live-relay"}'

    class FakeConnection:
        def __init__(self, host, port, *, timeout):
            assert (host, port, timeout) == ("8.8.4.4", 443, 10)
            self.sock = None

        def request(self, method, path, *, body, headers):
            requests.append((method, path, body, headers))

        def getresponse(self):
            return FakeResponse()

        def close(self):
            self.sock.close()

    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.ssl.create_default_context", FakeContext
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.socket.create_connection",
        lambda address, *, timeout: object(),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.http.client.HTTPConnection", FakeConnection
    )

    fingerprint = hashlib.sha256(certificate).hexdigest()
    _select_agent_leisaac_run(
        "8.8.4.4",
        auth_user="npa",
        auth_password="secret",
        run_id="live-relay",
        certificate_sha256=fingerprint,
    )

    assert len(requests) == 1
    assert response_timeouts == [60]
    method, path, body, headers = requests[0]
    assert (method, path) == ("POST", "/api/leisaac/select")
    assert json.loads(body) == {"run_id": "live-relay"}
    assert headers["Authorization"].startswith("Basic ")
    assert headers["X-NPA-LeIsaac-Control"] == "1"

    with pytest.raises(RuntimeError, match="fingerprint changed"):
        _select_agent_leisaac_run(
            "8.8.4.4",
            auth_user="npa",
            auth_password="secret",
            run_id="live-relay",
            certificate_sha256="0" * 64,
        )
    assert len(requests) == 1


def test_select_agent_leisaac_run_retries_transient_backhaul_unavailability(
    monkeypatch,
) -> None:
    certificate = b"agent-certificate"
    statuses = iter((503, 200))
    requests = []
    sleeps = []

    class FakeTLS:
        def getpeercert(self, *, binary_form=False):
            assert binary_form is True
            return certificate

        def settimeout(self, _seconds):
            pass

        def close(self):
            pass

    class FakeContext:
        check_hostname = True
        verify_mode = None

        def wrap_socket(self, _raw, *, server_hostname):
            assert server_hostname == "8.8.4.4"
            return FakeTLS()

    class FakeResponse:
        def __init__(self):
            self.status = next(statuses)

        def read(self, _limit):
            if self.status == 503:
                return b'{"detail":"LeIsaac service is unavailable"}'
            return b'{"selected":true,"run_id":"live-relay"}'

    class FakeConnection:
        def __init__(self, *_args, **_kwargs):
            self.sock = None

        def request(self, method, path, *, body, headers):
            requests.append((method, path, body, headers))

        def getresponse(self):
            return FakeResponse()

        def close(self):
            self.sock.close()

    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.ssl.create_default_context", FakeContext
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.socket.create_connection",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.http.client.HTTPConnection", FakeConnection
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.time.sleep", lambda seconds: sleeps.append(seconds)
    )

    _select_agent_leisaac_run(
        "8.8.4.4",
        auth_user="npa",
        auth_password="secret",
        run_id="live-relay",
        certificate_sha256=hashlib.sha256(certificate).hexdigest(),
    )

    assert len(requests) == 2
    assert sleeps == [2]


def test_select_agent_leisaac_run_bounds_transient_unavailability(
    monkeypatch,
) -> None:
    certificate = b"agent-certificate"

    class FakeTLS:
        def getpeercert(self, *, binary_form=False):
            assert binary_form is True
            return certificate

        def settimeout(self, _seconds):
            return None

        def close(self):
            return None

    class FakeContext:
        check_hostname = True
        verify_mode = None

        def wrap_socket(self, _raw, *, server_hostname):
            assert server_hostname == "8.8.4.4"
            return FakeTLS()

    class FakeResponse:
        status = 503

        def read(self, _limit):
            return b'{"detail":"LeIsaac service is unavailable"}'

    class FakeConnection:
        def __init__(self, *_args, **_kwargs):
            self.sock = None

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return FakeResponse()

        def close(self):
            self.sock.close()

    monotonic = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.ssl.create_default_context", FakeContext
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.socket.create_connection",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.http.client.HTTPConnection", FakeConnection
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.time.monotonic", lambda: next(monotonic)
    )

    with pytest.raises(TimeoutError, match="agent run selection"):
        _select_agent_leisaac_run(
            "8.8.4.4",
            auth_user="npa",
            auth_password="secret",
            run_id="live-relay",
            certificate_sha256=hashlib.sha256(certificate).hexdigest(),
            timeout_seconds=1.0,
        )


def test_select_agent_leisaac_run_aborts_retry_when_lifecycle_lease_is_lost(
    monkeypatch,
) -> None:
    certificate = b"agent-certificate"
    requests = []
    checks = 0

    class FakeTLS:
        def getpeercert(self, *, binary_form=False):
            assert binary_form is True
            return certificate

        def settimeout(self, _seconds):
            return None

        def close(self):
            return None

    class FakeContext:
        check_hostname = True
        verify_mode = None

        def wrap_socket(self, _raw, *, server_hostname):
            assert server_hostname == "8.8.4.4"
            return FakeTLS()

    class FakeResponse:
        status = 503

        def read(self, _limit):
            return b'{"detail":"LeIsaac service is unavailable"}'

    class FakeConnection:
        def __init__(self, *_args, **_kwargs):
            self.sock = None

        def request(self, *_args, **_kwargs):
            requests.append(True)

        def getresponse(self):
            return FakeResponse()

        def close(self):
            self.sock.close()

    def assert_lease() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("selected run lifecycle lock renewal failed")

    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.ssl.create_default_context", FakeContext
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.socket.create_connection",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.http.client.HTTPConnection", FakeConnection
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.time.sleep",
        lambda *_args: pytest.fail("lease loss must abort before a retry sleep"),
    )

    with pytest.raises(RuntimeError, match="lock renewal failed"):
        _select_agent_leisaac_run(
            "8.8.4.4",
            auth_user="npa",
            auth_password="secret",
            run_id="live-relay",
            certificate_sha256=hashlib.sha256(certificate).hexdigest(),
            progress_check=assert_lease,
        )

    assert len(requests) == 1


def test_wait_ready_rejects_old_ready_replica_during_rollout(monkeypatch) -> None:
    old_replica = {
        "metadata": {"generation": 2},
        "spec": {"replicas": 1},
        "status": {
            "observedGeneration": 2,
            "replicas": 2,
            "updatedReplicas": 1,
            "readyReplicas": 1,
            "availableReplicas": 1,
            "unavailableReplicas": 1,
        },
    }
    current_replica = {
        "metadata": {"generation": 2},
        "spec": {"replicas": 1},
        "status": {
            "observedGeneration": 2,
            "replicas": 1,
            "updatedReplicas": 1,
            "readyReplicas": 1,
            "availableReplicas": 1,
        },
    }
    responses = iter((old_replica, current_replica))
    calls = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(next(responses)), stderr=""
        ),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.time.sleep", lambda seconds: calls.append(seconds)
    )

    _wait_ready("cluster", "leisaac", "leisaac-live")

    assert calls == [5]


def test_wait_ready_checks_external_progress_before_polling(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: pytest.fail("readiness poll should not run"),
    )

    with pytest.raises(RuntimeError, match="lifecycle lease lost"):
        _wait_ready(
            "cluster",
            "namespace",
            "deployment",
            progress_check=lambda: (_ for _ in ()).throw(
                RuntimeError("lifecycle lease lost")
            ),
        )


def test_external_ip_timeout_reports_last_provider_observation(monkeypatch) -> None:
    times = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr("npa.cli.workbench.leisaac.time.monotonic", lambda: next(times))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="provider still allocating"
        ),
    )
    with pytest.raises(TimeoutError, match="provider still allocating") as failure:
        _external_ip(
            "cluster",
            "leisaac",
            "leisaac-live",
            timeout_seconds=1.0,
            poll_interval_seconds=0.0,
        )
    assert "Service leisaac/leisaac-live" in str(failure.value)


def test_external_ip_rejects_address_observed_after_deadline(monkeypatch) -> None:
    times = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr("npa.cli.workbench.leisaac.time.monotonic", lambda: next(times))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="203.0.113.10\n", stderr=""
        ),
    )

    with pytest.raises(TimeoutError, match="external IPv4 was assigned"):
        _external_ip(
            "cluster",
            "leisaac",
            "leisaac-live",
            timeout_seconds=1.0,
            poll_interval_seconds=0.0,
        )


def test_wait_timeout_accepts_operator_deadline_beyond_one_day(monkeypatch) -> None:
    monkeypatch.setenv("NPA_LEISAAC_TEST_TIMEOUT_SECONDS", "172800")

    assert _wait_timeout("NPA_LEISAAC_TEST_TIMEOUT_SECONDS", 1.0) == 172800.0


def test_wait_ready_timeout_reports_rollout_counters(monkeypatch) -> None:
    times = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr("npa.cli.workbench.leisaac.time.monotonic", lambda: next(times))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "metadata": {"generation": 3},
                    "spec": {"replicas": 1},
                    "status": {
                        "observedGeneration": 2,
                        "readyReplicas": 0,
                        "unavailableReplicas": 1,
                    },
                }
            ),
            stderr="",
        ),
    )
    with pytest.raises(TimeoutError, match="generation=3") as failure:
        _wait_ready(
            "cluster",
            "leisaac",
            "leisaac-live",
            timeout_seconds=1.0,
            poll_interval_seconds=0.0,
        )
    assert "unavailable=1" in str(failure.value)


def test_wait_ready_rejects_readiness_observed_after_deadline(monkeypatch) -> None:
    times = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr("npa.cli.workbench.leisaac.time.monotonic", lambda: next(times))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "metadata": {"generation": 1},
                    "spec": {"replicas": 1},
                    "status": {
                        "observedGeneration": 1,
                        "updatedReplicas": 1,
                        "readyReplicas": 1,
                        "availableReplicas": 1,
                    },
                }
            ),
            stderr="",
        ),
    )

    with pytest.raises(TimeoutError, match="ready deployment observed"):
        _wait_ready(
            "cluster",
            "leisaac",
            "leisaac-live",
            timeout_seconds=1.0,
            poll_interval_seconds=0.0,
        )


def test_wait_progress_uses_stderr_without_polluting_json_output(monkeypatch) -> None:
    messages = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.typer.echo",
        lambda message, **kwargs: messages.append((message, kwargs)),
    )
    monotonic = iter((0.0,) * 8)
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.time.monotonic", lambda: next(monotonic)
    )
    external_results = iter(
        (
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="203.0.113.20", stderr=""),
        )
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: next(external_results),
    )
    monkeypatch.setattr("npa.cli.workbench.leisaac.time.sleep", lambda _seconds: None)

    assert _external_ip("cluster", "namespace", "service") == "203.0.113.20"

    not_ready = {
        "metadata": {"generation": 2},
        "spec": {"replicas": 1},
        "status": {"observedGeneration": 1, "unavailableReplicas": 1},
    }
    ready = {
        "metadata": {"generation": 2},
        "spec": {"replicas": 1},
        "status": {
            "observedGeneration": 2,
            "updatedReplicas": 1,
            "readyReplicas": 1,
            "availableReplicas": 1,
        },
    }
    monotonic = iter((0.0,) * 8)
    ready_results = iter((not_ready, ready))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.time.monotonic", lambda: next(monotonic)
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(next(ready_results)), stderr=""
        ),
    )

    _wait_ready("cluster", "namespace", "deployment")

    assert len(messages) == 2
    assert all(kwargs.get("err") is True for _message, kwargs in messages)


def test_readiness_poll_timeout_is_bounded_by_remaining_deadline(monkeypatch) -> None:
    calls = []
    monotonic = iter((10.0, 11.0, 11.0))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.time.monotonic", lambda: next(monotonic)
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *args, **kwargs: (
            calls.append((args, kwargs))
            or SimpleNamespace(returncode=0, stdout="203.0.113.20", stderr="")
        ),
    )

    assert (
        _external_ip(
            "cluster",
            "namespace",
            "service",
            timeout_seconds=5.0,
        )
        == "203.0.113.20"
    )
    assert calls[0][1]["timeout"] == 4.0


def test_delete_resources_addresses_each_kubernetes_kind_explicitly(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *args, **kwargs: (
            calls.append((args, kwargs))
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )

    _delete_resources("cluster", "leisaac", "leisaac-live")

    assert calls[0][0][2] == [
        "delete",
        "deployment/leisaac-live",
        "service/leisaac-live-tcp",
        "service/leisaac-live-media",
        "service/leisaac-live-relay",
        "secret/leisaac-live-relay-client",
        "secret/leisaac-live-recorder",
        "--ignore-not-found=true",
    ]


def test_install_relay_creates_required_agent_directories() -> None:
    class CaptureSSH:
        command = ""

        def run_or_raise(self, command, **_kwargs):
            self.command = command
            return 0, "b" * 64 + "\n", ""

    ssh = CaptureSSH()
    _install_agent_relay(
        ssh,
        run_id="live-relay",
        session_nonce="a" * 64,
        expires_at="2099-01-01T00:00:00Z",
        manifest_uri="s3://bucket/leisaac/live-relay/reports/leisaac-session.json",
    )

    assert "sudo install -d -m 0755 /etc/npa /opt/npa-agent" in ssh.command
    assert "net.core.rmem_max=8388608" in ssh.command
    assert "net.core.wmem_max=8388608" in ssh.command
    assert "net.core.netdev_max_backlog=5000" in ssh.command
    assert "systemctl is-active --quiet coturn.service" in ssh.command
    assert "leisaac-relay.restore-coturn" in ssh.command
    assert "systemctl stop coturn.service" in ssh.command
    assert "DynamicUser=yes" not in ssh.command  # unit is base64-encoded in transit
    assert "openssl req -x509" not in ssh.command

    encoded_payloads = re.findall(r"echo ([A-Za-z0-9+/=]+) \| base64 -d", ssh.command)
    unit = base64.b64decode(encoded_payloads[-1]).decode("utf-8")
    assert "${CREDENTIALS_DIRECTORY}/leisaac.json" in unit
    config = json.loads(base64.b64decode(encoded_payloads[-2]).decode("utf-8"))
    assert config["manifest_uri"] == (
        "s3://bucket/leisaac/live-relay/reports/leisaac-session.json"
    )


def test_remove_relay_restores_only_its_recorded_baseline_coturn() -> None:
    class CaptureSSH:
        command = ""

        def run_or_raise(self, command, **_kwargs):
            self.command = command
            return 0, "", ""

    ssh = CaptureSSH()
    _remove_agent_relay(ssh, run_id="live-relay")

    assert "leisaac-relay.restore-coturn" in ssh.command
    assert 'if [ "$existing" != live-relay ]; then exit 0; fi' in ssh.command
    assert "systemctl start coturn.service" in ssh.command


def test_relay_status_bounds_each_remote_health_probe() -> None:
    class CaptureSSH:
        command = ""
        kwargs = {}

        def run(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            return 0, '{"state":"ready"}\nNPA_HTTP_STATUS:200\n', ""

    ssh = CaptureSSH()

    assert _relay_status(ssh, session_nonce="a" * 64) == {"state": "ready"}
    assert "--connect-timeout 5" in ssh.command
    assert "--max-time 10" in ssh.command
    assert "--fail" not in ssh.command
    assert ssh.kwargs == {"timeout": 10.0}


def test_wait_relay_status_retries_an_ssh_watchdog_timeout(monkeypatch) -> None:
    attempts = 0

    class TimeoutThenReadySSH:
        def run(self, _command, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise SSHTimeoutError("SSH command timed out after 10s")
            return 0, '{"state":"ready"}\nNPA_HTTP_STATUS:200\n', ""

    monkeypatch.setattr("npa.cli.workbench.leisaac.time.sleep", lambda _value: None)

    status = _wait_relay_status(
        TimeoutThenReadySSH(),
        session_nonce="a" * 64,
        timeout_seconds=60.0,
    )

    assert status == {"state": "ready"}
    assert attempts == 2


def test_wait_relay_status_retries_a_bounded_startup_probe(monkeypatch) -> None:
    attempts = []

    def relay_status(_ssh, *, session_nonce, timeout_seconds):
        attempts.append((session_nonce, timeout_seconds))
        if len(attempts) == 1:
            raise _TransientRelayStatusError("new backhaul has not answered")
        return {"state": "ready", "task": DEFAULT_TASK}

    monkeypatch.setattr("npa.cli.workbench.leisaac._relay_status", relay_status)
    monkeypatch.setattr("npa.cli.workbench.leisaac.time.sleep", lambda _value: None)

    status = _wait_relay_status(
        object(),
        session_nonce="a" * 64,
        timeout_seconds=60.0,
    )

    assert status["state"] == "ready"
    assert len(attempts) == 2
    assert all(0 < timeout <= 10 for _nonce, timeout in attempts)


def test_wait_relay_status_reports_last_failure_at_deadline(monkeypatch) -> None:
    moments = iter((10.0, 10.0, 10.0, 10.6, 10.6))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.time.monotonic", lambda: next(moments)
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._relay_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            _TransientRelayStatusError("half-open status stream")
        ),
    )

    with pytest.raises(TimeoutError, match="half-open status stream"):
        _wait_relay_status(
            object(),
            session_nonce="a" * 64,
            timeout_seconds=0.5,
            poll_interval_seconds=0.0,
        )


def test_wait_relay_status_rejects_ready_observed_after_deadline(monkeypatch) -> None:
    moments = iter((10.0, 10.0, 10.6))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.time.monotonic", lambda: next(moments)
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._relay_status",
        lambda *_args, **_kwargs: {"state": "ready"},
    )

    with pytest.raises(TimeoutError, match="last observation: state=ready"):
        _wait_relay_status(
            object(),
            session_nonce="a" * 64,
            timeout_seconds=0.5,
            poll_interval_seconds=0.0,
        )


def test_wait_relay_status_propagates_terminal_failure_without_retry(
    monkeypatch,
) -> None:
    attempts = []

    def terminal(*_args, **_kwargs):
        attempts.append(True)
        raise RuntimeError("LeIsaac relay status returned HTTP 403")

    monkeypatch.setattr("npa.cli.workbench.leisaac._relay_status", terminal)

    with pytest.raises(RuntimeError, match="HTTP 403"):
        _wait_relay_status(
            object(),
            session_nonce="a" * 64,
            timeout_seconds=14_400.0,
        )

    assert attempts == [True]


@pytest.mark.parametrize(
    ("exit_code", "http_status", "body", "error_type"),
    (
        (28, "000", "", _TransientRelayStatusError),
        (0, "503", '{"state":"starting"}', _TransientRelayStatusError),
        (0, "503", '{"state":"failed"}', RuntimeError),
        (0, "403", '{"detail":"forbidden"}', RuntimeError),
        (0, "200", "not-json", RuntimeError),
    ),
)
def test_relay_status_classifies_transient_and_terminal_failures(
    exit_code, http_status, body, error_type
) -> None:
    class CaptureSSH:
        def run(self, _command, **_kwargs):
            return exit_code, f"{body}\nNPA_HTTP_STATUS:{http_status}\n", ""

    with pytest.raises(error_type):
        _relay_status(CaptureSSH(), session_nonce="a" * 64)


def test_relay_media_server_is_the_stable_private_service_host(monkeypatch) -> None:
    service = {"spec": {"clusterIP": "10.101.249.14"}}
    calls = []

    def kubectl(*args):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=json.dumps(service), stderr="")

    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        kubectl,
    )

    assert _relay_media_server("cluster", "leisaac", "deployment") == "10.101.249.14"
    assert calls[0][2] == [
        "get",
        "service",
        "deployment-relay",
        "-o",
        "json",
    ]


def _args() -> list[str]:
    return [
        "launch",
        "--run-id",
        "live-relay",
        "--image",
        IMAGE,
        "--context",
        "cluster",
        "--namespace",
        "leisaac",
        "--source-range",
        "8.8.8.8/32",
        "--artifact-uri",
        "s3://bucket/checkpoints",
        "--output-path",
        "s3://bucket/checkpoints/datasets/leisaac",
        "--transport",
        "agent-relay",
        "--agent-project",
        "rtxpro",
        "--agent-name",
        "agent",
    ]


def _patch_launch(monkeypatch):
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._acquire_run_lifecycle_lease",
        lambda *_args: _FakeLifecycleLease(),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="secret/x", stderr=""
        ),
    )
    applied = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._apply",
        lambda _context, _namespace, documents: applied.extend(documents),
    )
    ssh = object()
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._agent_relay_context",
        lambda *_args: ("vm-agent", "8.8.4.4", ssh, "npa", "secret"),
    )
    storage = {
        "bucket": "bucket",
        "prefix": "checkpoints",
        "endpoint": "https://storage.example",
        "access_key": "access",
        "secret_key": "secret",
        "region": "region",
    }
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._agent_artifact_storage",
        lambda *_args: storage,
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._existing_turn_peer_source",
        lambda *_args: "",
    )
    ingress_calls = []

    def ensure(**kwargs):
        ingress_calls.append(kwargs)
        return SimpleNamespace(changed=True)

    monkeypatch.setattr("npa.cli.workbench.leisaac.ensure_ingress", ensure)
    install_calls = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._install_agent_relay",
        lambda *args, **kwargs: install_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._agent_certificate_sha256", lambda _ip: "f" * 64
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._wait_ready",
        lambda *_args, **_kwargs: None,
    )

    def resolve_relay_media_server(*_args):
        # Regression: the Service must already have been applied, while the
        # Deployment must not yet exist when its stable ClusterIP is resolved.
        assert any(item.get("kind") == "Service" for item in applied)
        assert not any(item.get("kind") == "Deployment" for item in applied)
        return "10.96.34.22"

    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._relay_media_server",
        resolve_relay_media_server,
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._remove_agent_turn",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._relay_status",
        lambda *_args, **_kwargs: {
            "state": "ready",
            "task": DEFAULT_TASK,
            "source_commit": "1651c321e9b0c1bb54233211fc7b3cd70d8373d5",
            "session_nonce": "nonce-filled-later",
            "gpu": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        },
    )
    manifests = []

    def put(_uri, manifest, *, storage):
        assert storage["endpoint"] == "https://storage.example"
        manifests.append(manifest)
        return "s3://bucket/checkpoints/live-relay/reports/leisaac-session.json"

    monkeypatch.setattr("npa.cli.workbench.leisaac._put_manifest", put)
    selections = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._select_agent_leisaac_run",
        lambda *args, **kwargs: selections.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.secrets.token_hex", lambda _n: "a" * 64
    )
    # Make the attestation nonce exactly match the deterministic nonce above.
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._relay_status",
        lambda *_args, **_kwargs: {
            "state": "ready",
            "task": DEFAULT_TASK,
            "source_commit": "1651c321e9b0c1bb54233211fc7b3cd70d8373d5",
            "task_registry_fingerprint": REGISTRY_FINGERPRINT,
            "session_attestation": hashlib.sha256(
                ("npa-leisaac-session:" + "a" * 64).encode()
            ).hexdigest(),
            "environment_id": "operator-0",
            "environment_index": 0,
            "seed": 42,
            "gpu": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        },
    )
    return (
        applied,
        ingress_calls,
        install_calls,
        manifests,
        ssh,
        selections,
    )


def test_agent_relay_manifest_uses_selected_agent_storage_not_shell_endpoint(
    monkeypatch,
) -> None:
    calls = []

    class S3:
        def put_object(self, **kwargs):
            calls.append(kwargs)

    client_calls = []

    def client(service, **kwargs):
        client_calls.append((service, kwargs))
        return S3()

    monkeypatch.setattr("boto3.client", client)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://wrong-region.example")
    manifest = {"run_id": "live-relay", "session_nonce": "a" * 64}
    storage = {
        "bucket": "bucket",
        "prefix": "checkpoints",
        "endpoint": "https://agent-region.example",
        "access_key": "agent-access",
        "secret_key": "agent-secret",
        "region": "agent-region",
    }

    uri = _put_manifest("s3://bucket/checkpoints", manifest, storage=storage)

    assert uri == "s3://bucket/checkpoints/live-relay/reports/leisaac-session.json"
    assert client_calls[0][1]["endpoint_url"] == "https://agent-region.example"
    assert client_calls[0][1]["aws_access_key_id"] == "agent-access"
    assert client_calls[0][1]["aws_secret_access_key"] == "agent-secret"
    assert calls[0]["Bucket"] == "bucket"
    assert "session_nonce" not in json.loads(calls[0]["Body"])


def test_agent_relay_manifest_rejects_storage_scope_mismatch() -> None:
    storage = {
        "bucket": "agent-bucket",
        "prefix": "checkpoints",
        "endpoint": "https://agent-region.example",
        "access_key": "agent-access",
        "secret_key": "agent-secret",
        "region": "agent-region",
    }

    for uri in ("s3://other/checkpoints", "s3://agent-bucket/outside"):
        try:
            _put_manifest(uri, {"run_id": "live-relay"}, storage=storage)
        except Exception as exc:
            assert "agent-relay artifact URI" in str(exc)
        else:
            raise AssertionError(f"storage scope mismatch was accepted: {uri}")


def test_put_manifest_treats_an_explicit_manifest_leaf_as_a_leaf(monkeypatch) -> None:
    calls = []

    class S3:
        def put_object(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("boto3.client", lambda *_args, **_kwargs: S3())
    uri = _put_manifest(
        "s3://bucket/checkpoints/live/reports/leisaac-session.json",
        {"run_id": "live"},
    )
    assert uri == "s3://bucket/checkpoints/live/reports/leisaac-session.json"
    assert calls[0]["Key"] == "checkpoints/live/reports/leisaac-session.json"


def test_load_manifest_is_exact_bounded_and_agent_storage_scoped(monkeypatch) -> None:
    payload = {
        "schema": "npa.leisaac.session.v2",
        "run_id": "live-relay",
        "transport": "agent-relay",
        "signal_host": "127.0.0.1",
    }
    calls = []

    class Body:
        def read(self, limit):
            assert limit == 1_048_577
            return json.dumps(payload).encode()

    class S3:
        def get_object(self, **kwargs):
            calls.append(kwargs)
            return {"ContentLength": 128, "Body": Body()}

    monkeypatch.setattr("boto3.client", lambda *_args, **_kwargs: S3())
    storage = {
        "bucket": "bucket",
        "prefix": "checkpoints",
        "endpoint": "https://agent-region.example",
        "access_key": "agent-access",
        "secret_key": "agent-secret",
        "region": "agent-region",
    }
    uri = "s3://bucket/checkpoints/live-relay/reports/leisaac-session.json"

    assert _load_manifest(uri, run_id="live-relay", storage=storage) == payload
    assert calls == [
        {
            "Bucket": "bucket",
            "Key": "checkpoints/live-relay/reports/leisaac-session.json",
        }
    ]
    with pytest.raises(Exception, match="selected agent's bucket"):
        _load_manifest(
            "s3://other/checkpoints/live-relay/reports/leisaac-session.json",
            run_id="live-relay",
            storage=storage,
        )


def test_existing_relay_contract_refuses_public_service(monkeypatch) -> None:
    labels = {
        "app.kubernetes.io/name": "leisaac",
        "app.kubernetes.io/instance": "leisaac-live-relay",
        "app.kubernetes.io/managed-by": "npa",
    }
    service = {
        "metadata": {
            "name": "leisaac-live-relay-relay",
            "namespace": "leisaac",
            "labels": labels,
            "annotations": {
                "npa.nebius.com/agent-project": "rtxpro",
                "npa.nebius.com/agent-name": "agent",
                "npa.nebius.com/source-ranges": "8.8.8.8/32",
            },
        },
        "spec": {"type": "LoadBalancer", "clusterIP": "10.96.34.22"},
    }
    deployment = {
        "metadata": {
            "name": "leisaac-live-relay",
            "namespace": "leisaac",
            "labels": labels,
        }
    }
    responses = iter((service, deployment))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(next(responses)), stderr=""
        ),
    )

    with pytest.raises(Exception, match="not private"):
        _existing_relay_contract(
            "cluster",
            "leisaac",
            run_id="live-relay",
            agent_project="rtxpro",
            agent_name="agent",
        )


def test_existing_relay_contract_accepts_secret_referenced_nonce(monkeypatch) -> None:
    labels = {
        "app.kubernetes.io/name": "leisaac",
        "app.kubernetes.io/instance": "leisaac-live-relay",
        "app.kubernetes.io/managed-by": "npa",
    }
    service = {
        "metadata": {
            "name": "leisaac-live-relay-relay",
            "namespace": "leisaac",
            "labels": labels,
            "annotations": {
                "npa.nebius.com/agent-project": "rtxpro",
                "npa.nebius.com/agent-name": "agent",
                "npa.nebius.com/source-ranges": "8.8.8.8/32",
            },
        },
        "spec": {"type": "ClusterIP", "clusterIP": "10.96.34.22"},
    }
    deployment = {
        "metadata": {
            "name": "leisaac-live-relay",
            "namespace": "leisaac",
            "labels": labels,
        },
        "spec": {
            "replicas": 1,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "leisaac",
                            "env": [
                                {"name": "NPA_LEISAAC_RUN_ID", "value": "live-relay"},
                                {"name": "ACCEPT_EULA", "value": "Y"},
                                {
                                    "name": "NPA_LEISAAC_SESSION_NONCE",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "leisaac-live-relay-relay-client",
                                            "key": "NPA_LEISAAC_SESSION_NONCE",
                                        }
                                    },
                                },
                            ],
                        },
                        {"name": "agent-relay-client"},
                        {"name": "turn"},
                    ],
                    "volumes": [
                        {
                            "name": "relay-client",
                            "secret": {"secretName": "leisaac-live-relay-relay-client"},
                        }
                    ],
                }
            },
        },
    }
    responses = iter((service, deployment))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(next(responses)), stderr=""
        ),
    )

    media_server, source_ranges = _existing_relay_contract(
        "cluster",
        "leisaac",
        run_id="live-relay",
        agent_project="rtxpro",
        agent_name="agent",
    )

    assert media_server == "10.96.34.22"
    assert source_ranges == ["8.8.8.8/32"]


def test_list_tasks_json_is_machine_readable_and_parallel_launch_is_rejected(
    monkeypatch,
) -> None:
    listed = runner.invoke(app, ["list-tasks", "--output", "json"])
    assert listed.exit_code == 0
    payload = json.loads(listed.output)
    assert {item["task"] for item in payload["tasks"]} == {
        "LeIsaac-SO101-PickOrange-v0",
        "LeIsaac-SO101-LiftCube-v0",
    }
    monkeypatch.delenv("ACCEPT_EULA", raising=False)
    rejected = runner.invoke(app, [*_args(), "--num-envs", "2"])
    assert rejected.exit_code == 1
    assert "exactly one active environment" in rejected.output


def test_launch_defaults_to_agent_relay_and_rejects_undiscoverable_public_lb(
    monkeypatch,
) -> None:
    (
        _applied,
        _ingress,
        _install,
        manifests,
        _ssh,
        _selections,
    ) = _patch_launch(monkeypatch)
    default_args = _args()
    transport_index = default_args.index("--transport")
    del default_args[transport_index : transport_index + 2]

    defaulted = runner.invoke(app, default_args)

    assert defaulted.exit_code == 0, defaulted.output
    assert manifests[0]["transport"] == "agent-relay"

    insecure_args = _args()
    insecure_args[insecure_args.index("agent-relay")] = "public-load-balancer"
    rejected = runner.invoke(app, insecure_args)
    assert rejected.exit_code == 1
    assert "cannot securely provision browser credentials" in rejected.output
    assert "use agent-relay" in rejected.output
    assert len(manifests) == 1


@pytest.mark.parametrize("value", ["Y", "YES", "yes", "1", "TRUE"])
def test_launch_normalizes_affirmative_accept_eula(monkeypatch, value) -> None:
    monkeypatch.setenv("ACCEPT_EULA", value)
    applied, *_rest = _patch_launch(monkeypatch)

    result = runner.invoke(app, _args())

    assert result.exit_code == 0, result.output
    deployment = next(item for item in applied if item["kind"] == "Deployment")
    environment = {
        item["name"]: item.get("value")
        for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
        if "value" in item
    }
    assert environment["ACCEPT_EULA"] == "Y"
    assert "OMNI_KIT_ACCEPT_EULA" not in environment
    assert "ISAACSIM_ACCEPT_EULA" not in environment


@pytest.mark.parametrize("value", ["", "N", "no", "0", "FALSE"])
def test_launch_refuses_accept_eula_opt_out_before_mutation(monkeypatch, value) -> None:
    monkeypatch.setenv("ACCEPT_EULA", value)
    mutations: list[str] = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._acquire_run_lifecycle_lease",
        lambda *_args, **_kwargs: mutations.append("lock"),
    )
    result = runner.invoke(app, _args())

    assert result.exit_code == 1
    assert "explicitly disabled" in result.output
    assert "No expensive action has begun" in result.output
    assert mutations == []


def test_launch_rejects_invalid_accept_eula_distinctly_before_mutation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ACCEPT_EULA", "maybe")
    mutations: list[str] = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._acquire_run_lifecycle_lease",
        lambda *_args, **_kwargs: mutations.append("lock"),
    )

    result = runner.invoke(app, _args())

    assert result.exit_code == 1
    assert "Invalid ACCEPT_EULA" in result.output
    assert "No expensive action has begun" in result.output
    assert mutations == []


def test_launch_agent_relay_wires_private_cluster_public_agent_and_manifest(
    monkeypatch,
) -> None:
    (
        applied,
        ingress_calls,
        install_calls,
        manifests,
        ssh,
        selections,
    ) = _patch_launch(monkeypatch)

    result = runner.invoke(app, _args())

    assert result.exit_code == 0, result.output
    assert "transport: agent-relay" in result.output
    assert "public_agent_url: https://8.8.4.4/" in result.output
    deployment = next(item for item in applied if item.get("kind") == "Deployment")
    assert "imagePullSecrets" not in deployment["spec"]["template"]["spec"]
    services = [item for item in applied if item.get("kind") == "Service"]
    assert len(services) == 1
    assert services[0]["spec"]["type"] == "ClusterIP"
    assert applied[1]["kind"] == "Secret"
    assert applied[2]["kind"] == "Secret"
    assert applied[2]["metadata"]["name"].endswith("-recorder")
    deployment = next(item for item in applied if item["kind"] == "Deployment")
    deployment_env = {
        item["name"]: item
        for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert deployment_env["ACCEPT_EULA"] == {
        "name": "ACCEPT_EULA",
        "value": "Y",
    }
    assert deployment_env["NPA_LEISAAC_MEDIA_HOST"]["valueFrom"] == {
        "fieldRef": {"fieldPath": "status.podIP"}
    }
    media_port = next(
        item
        for item in deployment["spec"]["template"]["spec"]["containers"][0]["ports"]
        if item["name"] == "media"
    )
    assert "hostPort" not in media_port
    assert (
        "npa.nebius.com/turn-peer-source" not in services[0]["metadata"]["annotations"]
    )
    assert ingress_calls == [
        {
            "vm_id": "vm-agent",
            "ports": (3478,),
            "source": "8.8.8.8/32",
            "tool": "leisaac-turn-control",
            "protocol": "UDP",
        },
        {
            "vm_id": "vm-agent",
            "ports": (3478,),
            "source": "8.8.8.8/32",
            "tool": "leisaac-turn-control-tcp",
            "protocol": "TCP",
        },
    ]
    assert install_calls[0][0] == (ssh,)
    assert install_calls[0][1]["session_nonce"] == "a" * 64
    assert install_calls[0][1].get("media_target_host", "") == ""
    assert len(install_calls) == 1
    assert manifests[0]["transport"] == "agent-relay"
    assert manifests[0]["signal_host"] == "127.0.0.1"
    assert manifests[0]["media_host"] == "8.8.4.4"
    assert manifests[0]["media_server"] == "10.96.34.22"
    assert "expires_at" not in manifests[0]
    assert len(selections) == 1
    selection_args, selection_kwargs = selections[0]
    assert selection_args == ("8.8.4.4",)
    assert selection_kwargs.pop("timeout_seconds") > 0
    progress_check = selection_kwargs.pop("progress_check")
    progress_check()
    assert selection_kwargs == {
        "auth_user": "npa",
        "auth_password": "secret",
        "run_id": "live-relay",
        "certificate_sha256": "f" * 64,
    }


def test_reconnect_agent_rotates_only_existing_relay_contract(monkeypatch) -> None:
    monkeypatch.delenv("ACCEPT_EULA", raising=False)
    lease_events: list[str] = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._acquire_run_lifecycle_lease",
        lambda *_args: _FakeLifecycleLease(lease_events),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._existing_relay_contract",
        lambda *_args, **_kwargs: ("10.96.34.22", ["8.8.8.8/32"]),
    )
    storage = {
        "bucket": "bucket",
        "prefix": "checkpoints",
        "endpoint": "https://storage.example",
        "access_key": "access",
        "secret_key": "secret",
        "region": "region",
    }
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._agent_artifact_storage",
        lambda *_args: storage,
    )
    manifest = {
        "schema": "npa.leisaac.session.v2",
        "run_id": "live-relay",
        "transport": "agent-relay",
        "signal_host": "127.0.0.1",
        "media_host": "8.8.8.8",
        "media_server": "10.96.34.22",
        "task": DEFAULT_TASK,
        "source_commit": "source-commit",
        "task_registry_fingerprint": REGISTRY_FINGERPRINT,
        "dataset": {"output_path": "s3://bucket/checkpoints/dataset"},
        "image": IMAGE,
    }
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._load_manifest",
        lambda *_args, **_kwargs: manifest,
    )
    ssh = object()
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._agent_relay_context",
        lambda *_args: ("replacement-vm", "8.8.4.4", ssh, "npa", "password"),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._agent_certificate_sha256", lambda _ip: "f" * 64
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.secrets.token_hex", lambda _size: "a" * 64
    )
    ingress = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.ensure_ingress",
        lambda **kwargs: ingress.append(kwargs) or SimpleNamespace(changed=True),
    )
    installs = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._install_agent_relay",
        lambda *args, **kwargs: installs.append((args, kwargs)),
    )
    applied = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._apply",
        lambda _context, _namespace, documents: applied.extend(documents),
    )
    kubectl_calls = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *args, **_kwargs: (
            kubectl_calls.append(args)
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._wait_ready", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._wait_relay_status",
        lambda *_args, **_kwargs: {
            "state": "ready",
            "task": DEFAULT_TASK,
            "source_commit": "source-commit",
            "task_registry_fingerprint": REGISTRY_FINGERPRINT,
            "session_attestation": hashlib.sha256(
                ("npa-leisaac-session:" + "a" * 64).encode()
            ).hexdigest(),
        },
    )
    published = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._put_manifest",
        lambda *args, **kwargs: published.append((args, kwargs)) or args[0],
    )
    selections = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._select_agent_leisaac_run",
        lambda *args, **kwargs: selections.append((args, kwargs)),
    )
    uri = "s3://bucket/checkpoints/live-relay/reports/leisaac-session.json"

    result = runner.invoke(
        app,
        [
            "reconnect-agent",
            "--run-id",
            "live-relay",
            "--manifest-uri",
            uri,
            "--agent-project",
            "rtxpro",
            "--agent-name",
            "agent",
            "--context",
            "cluster",
            "--namespace",
            "leisaac",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "status: reconnected" in result.output
    assert len(applied) == 1
    assert applied[0]["kind"] == "Secret"
    assert not any(item.get("kind") == "Deployment" for item in applied)
    assert kubectl_calls == [
        (
            "cluster",
            "leisaac",
            [
                "patch",
                "deployment/leisaac-live-relay",
                "--type=strategic",
                "--patch",
                kubectl_calls[0][2][-1],
            ],
        )
    ]
    rotation_patch = json.loads(kubectl_calls[0][2][-1])
    assert rotation_patch["spec"]["strategy"] == {
        "type": "Recreate",
        "rollingUpdate": None,
    }
    nonce_env = rotation_patch["spec"]["template"]["spec"]["containers"][0]["env"][0]
    assert nonce_env == {
        "name": "NPA_LEISAAC_SESSION_NONCE",
        "value": None,
        "valueFrom": {
            "secretKeyRef": {
                "name": "leisaac-live-relay-relay-client",
                "key": "NPA_LEISAAC_SESSION_NONCE",
            }
        },
    }
    assert "a" * 64 not in kubectl_calls[0][2][-1]
    assert {item["protocol"] for item in ingress} == {"UDP", "TCP"}
    assert installs[0][0] == (ssh,)
    assert installs[0][1]["manifest_uri"] == uri
    rotated_manifest = published[0][0][1]
    assert rotated_manifest["media_host"] == "8.8.4.4"
    assert rotated_manifest["dataset"] == manifest["dataset"]
    assert rotated_manifest["image"] == IMAGE
    assert len(selections) == 1
    assert lease_events[-1] == "closed"


def test_successful_launch_warns_when_only_lifecycle_release_fails(monkeypatch) -> None:
    _patch_launch(monkeypatch)
    deleted = []

    class ReleaseFailureLease(_FakeLifecycleLease):
        def close(self) -> None:
            raise RuntimeError("delete precondition failed")

    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._acquire_run_lifecycle_lease",
        lambda *_args: ReleaseFailureLease(),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._delete_resources",
        lambda *args: deleted.append(args),
    )

    result = runner.invoke(app, [*_args(), "--output", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert "lifecycle lock cleanup failed" in payload["warning"]
    assert "delete precondition failed" in payload["warning"]
    assert deleted == []


def test_launch_fails_closed_when_explicit_byof_pull_secret_is_missing(
    monkeypatch,
) -> None:
    applied, *_rest = _patch_launch(monkeypatch)
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="not found"
        ),
    )

    result = runner.invoke(
        app, [*_args(), "--image-pull-secret", "customer-registry"]
    )

    assert result.exit_code == 1
    assert "image pull secret 'customer-registry' is missing" in result.output
    assert applied == []


def test_launch_rejects_invalid_readiness_timeout_before_lock_or_mutation(
    monkeypatch,
) -> None:
    applied, *_middle, _selections = _patch_launch(monkeypatch)
    lease_calls = []
    deleted = []
    monkeypatch.setenv("NPA_LEISAAC_READY_TIMEOUT_SECONDS", "not-a-duration")
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._acquire_run_lifecycle_lease",
        lambda *args: lease_calls.append(args) or _FakeLifecycleLease(),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._delete_resources",
        lambda *args: deleted.append(args),
    )

    result = runner.invoke(app, _args())

    assert result.exit_code == 1
    assert "must be a positive number of seconds" in result.output
    assert lease_calls == []
    assert applied == []
    assert deleted == []


def test_launch_refuses_a_same_run_lifecycle_lock_before_any_mutation(
    monkeypatch,
) -> None:
    applied, *_middle, _selections = _patch_launch(monkeypatch)
    deleted = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._acquire_run_lifecycle_lease",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError(
                "another LeIsaac lifecycle operation already holds the selected run lock"
            )
        ),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._delete_resources",
        lambda *args: deleted.append(args),
    )

    result = runner.invoke(app, _args())

    assert result.exit_code == 1
    assert "already holds the selected run lock" in result.output
    assert applied == []
    assert deleted == []


def test_interrupted_launch_rolls_back_and_always_releases_lifecycle_lease(
    monkeypatch,
) -> None:
    _patch_launch(monkeypatch)
    lease_events: list[str] = []
    deleted = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._acquire_run_lifecycle_lease",
        lambda *_args: _FakeLifecycleLease(lease_events),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._wait_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._delete_resources",
        lambda *args: deleted.append(args),
    )

    result = runner.invoke(app, _args())

    assert result.exit_code == 1
    assert "LeIsaac launch interrupted" in result.output
    assert deleted == [("cluster", "leisaac", "leisaac-live-relay")]
    assert lease_events[-1] == "closed"


def test_failed_agent_relay_launch_removes_partial_relay_ingress_and_kubernetes(
    monkeypatch,
) -> None:
    (
        _applied,
        _ingress,
        _install,
        _manifests,
        _ssh,
        _selections,
    ) = _patch_launch(monkeypatch)
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._put_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("publish failed")),
    )
    removed_relay = []
    removed_turn = []
    removed_ingress = []
    removed_kubernetes = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._remove_agent_relay",
        lambda *args, **kwargs: removed_relay.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._remove_agent_turn",
        lambda *args, **kwargs: removed_turn.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.remove_exact_npa_ingress_for_instance",
        lambda *args, **kwargs: removed_ingress.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._delete_resources",
        lambda *args: removed_kubernetes.append(args),
    )

    result = runner.invoke(app, _args())

    assert result.exit_code == 1
    assert "publish failed" in result.output
    assert removed_relay
    assert removed_turn
    assert removed_ingress
    assert removed_kubernetes == [("cluster", "leisaac", "leisaac-live-relay")]


def test_launch_rollback_attempts_every_cleanup_when_teardown_steps_raise(
    monkeypatch,
) -> None:
    _patch_launch(monkeypatch)
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._put_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("primary publish failure")
        ),
    )
    attempted: list[str] = []

    def fail(label: str):
        def operation(*_args, **_kwargs):
            attempted.append(label)
            raise RuntimeError(f"{label} teardown failure")

        return operation

    turn_calls = 0

    def fail_turn_cleanup(*_args, **_kwargs):
        nonlocal turn_calls
        turn_calls += 1
        if turn_calls == 1:
            return None  # legacy migration cleanup during the successful launch path
        attempted.append("TURN")
        raise RuntimeError("TURN teardown failure")

    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._remove_agent_turn", fail_turn_cleanup
    )
    monkeypatch.setattr("npa.cli.workbench.leisaac._remove_agent_relay", fail("relay"))
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.remove_exact_npa_ingress_for_instance",
        fail("ingress"),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._delete_resources", fail("Kubernetes")
    )

    result = runner.invoke(app, _args())

    assert result.exit_code == 1
    assert attempted == ["TURN", "relay", "ingress", "ingress", "Kubernetes"]
    assert "primary publish failure" in result.output
    for label in attempted:
        assert f"{label} cleanup" in result.output


def test_relaunch_migrates_and_removes_only_the_prior_gpu_egress_rule(
    monkeypatch,
) -> None:
    (
        applied,
        _ingress,
        _relay,
        _manifests,
        _ssh,
        _selections,
    ) = _patch_launch(monkeypatch)
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._existing_turn_peer_source",
        lambda *_args: "4.4.4.0/24",
    )
    removals = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.remove_exact_npa_ingress_for_instance",
        lambda *args, **kwargs: removals.append((args, kwargs)),
    )

    result = runner.invoke(app, _args())

    assert result.exit_code == 0, result.output
    services = [item for item in applied if item.get("kind") == "Service"]
    assert len(services) == 1
    assert (
        "npa.nebius.com/turn-peer-source" not in services[0]["metadata"]["annotations"]
    )
    assert removals == [
        (
            ("vm-agent",),
            {
                "ports": (47999,),
                "source": "4.4.4.0/24",
                "tool": "leisaac-turn-media",
                "protocol": "UDP",
            },
        )
    ]


def test_destroy_uses_service_metadata_to_remove_only_its_agent_relay(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._acquire_run_lifecycle_lease",
        lambda *_args: _FakeLifecycleLease(),
    )
    relay_service = {
        "metadata": {
            "annotations": {
                "npa.nebius.com/agent-project": "rtxpro",
                "npa.nebius.com/agent-name": "agent",
                "npa.nebius.com/source-ranges": "8.8.8.8/32",
                "npa.nebius.com/turn-peer-source": "9.9.8.0/22",
            }
        }
    }
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(relay_service), stderr=""
        ),
    )
    ssh = object()
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._agent_relay_context",
        lambda *_args: ("vm-agent", "8.8.4.4", ssh, "npa", "secret"),
    )
    relay_removals = []
    turn_removals = []
    ingress_removals = []
    k8s_removals = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._remove_agent_relay",
        lambda *args, **kwargs: relay_removals.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._remove_agent_turn",
        lambda *args, **kwargs: turn_removals.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.remove_exact_npa_ingress_for_instance",
        lambda *args, **kwargs: ingress_removals.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._delete_resources",
        lambda *args: k8s_removals.append(args),
    )

    result = runner.invoke(
        app,
        [
            "destroy",
            "--run-id",
            "live-relay",
            "--context",
            "cluster",
            "--namespace",
            "leisaac",
        ],
    )

    assert result.exit_code == 0, result.output
    assert relay_removals == [((ssh,), {"run_id": "live-relay"})]
    assert turn_removals == [((ssh,), {"run_id": "live-relay"})]
    assert [item[1]["source"] for item in ingress_removals] == [
        "8.8.8.8/32",
        "8.8.8.8/32",
        "9.9.8.0/22",
    ]
    assert [item[1]["ports"] for item in ingress_removals] == [
        (3478,),
        (3478,),
        (47999,),
    ]
    assert all(item[0] == ("vm-agent",) for item in ingress_removals)
    assert [item[1]["protocol"] for item in ingress_removals] == [
        "UDP",
        "TCP",
        "UDP",
    ]
    assert k8s_removals == [("cluster", "leisaac", "leisaac-live-relay")]


def test_destroy_refuses_a_same_run_lifecycle_lock_before_any_mutation(
    monkeypatch,
) -> None:
    kubectl_calls = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._acquire_run_lifecycle_lease",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError(
                "another LeIsaac lifecycle operation already holds the selected run lock"
            )
        ),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *args, **kwargs: kubectl_calls.append((args, kwargs)),
    )

    result = runner.invoke(
        app,
        [
            "destroy",
            "--run-id",
            "live-relay",
            "--context",
            "cluster",
            "--namespace",
            "leisaac",
        ],
    )

    assert result.exit_code == 1
    assert "already holds the selected run lock" in result.output
    assert kubectl_calls == []
