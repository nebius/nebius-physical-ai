from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from npa.cli import agent_quota
from npa.cli import agent_setup_convergence
from npa.provisioning_journal import ProvisioningOperation, operation_heartbeats


def _remote_payload(**overrides):
    payload = {
        "schema_version": "npa.agent.setup.identity.v1",
        "project_alias": "live",
        "agent_name": "agent-live-id",
        "project_id": "project-exact",
        "endpoint": "192.0.2.20",
        "phase": "remote_health_ready",
        "service_fingerprint": "service-sha",
        "credential_fingerprint": "credential-sha",
        "credential_fingerprint_files": ["llm.env", "s3.env", "nebius.env"],
    }
    payload.update(overrides)
    return payload


def _reconcile(monkeypatch: pytest.MonkeyPatch, payload, *, http_status=200):
    class SSH:
        def __init__(self, **_kwargs):
            pass

        def run(self, _command):
            if payload is None:
                return 1, "", "missing"
            return 0, json.dumps(payload), ""

    monkeypatch.setattr(agent_setup_convergence, "SSHClient", SSH)
    monkeypatch.setattr(
        agent_setup_convergence,
        "resolve_ssh_config",
        lambda **_kwargs: SimpleNamespace(ssh=object()),
    )
    monkeypatch.setattr(
        agent_setup_convergence.httpx,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=http_status),
    )
    return agent_setup_convergence.reconcile_agent_setup(
        host="192.0.2.20",
        ssh_user="ubuntu",
        ssh_key_path="/tmp/key",
        project_alias="live",
        agent_name="agent-live-id",
        project_id="project-exact",
        auth_user="npa",
        auth_password="never-journal-this",
        agent_port=8088,
        public_https=True,
    )


def test_healthy_remote_is_adopted_after_missing_final_local_write(monkeypatch) -> None:
    result = _reconcile(monkeypatch, _remote_payload())
    assert result["state"] == "healthy"
    assert result["models_healthy"] is True
    assert "never-journal-this" not in json.dumps(result)


@pytest.mark.parametrize(
    ("payload", "state", "first_incomplete"),
    [
        (None, "incomplete", "remote_service_deployment"),
        (
            _remote_payload(credential_fingerprint_files=["s3.env"]),
            "incomplete",
            "credentials_staging",
        ),
        (
            _remote_payload(phase="services_deployed"),
            "incomplete",
            "health_verification",
        ),
    ],
)
def test_setup_resumes_from_first_incomplete_phase(
    monkeypatch, payload, state, first_incomplete
) -> None:
    result = _reconcile(monkeypatch, payload)
    assert result["state"] == state
    assert result["first_incomplete_phase"] == first_incomplete


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "project-other"),
        ("agent_name", "wrong-agent"),
        ("endpoint", "192.0.2.99"),
    ],
)
def test_stale_endpoint_or_wrong_identity_is_never_adopted(
    monkeypatch, field, value
) -> None:
    result = _reconcile(monkeypatch, _remote_payload(**{field: value}))
    assert result["state"] == "identity_mismatch"
    assert field in result["mismatch_fields"]


def test_health_transport_failure_remains_indeterminate(monkeypatch) -> None:
    class SSH:
        def __init__(self, **_kwargs):
            pass

        def run(self, _command):
            return 0, json.dumps(_remote_payload()), ""

    monkeypatch.setattr(agent_setup_convergence, "SSHClient", SSH)
    monkeypatch.setattr(
        agent_setup_convergence,
        "resolve_ssh_config",
        lambda **_kwargs: SimpleNamespace(ssh=object()),
    )
    monkeypatch.setattr(
        agent_setup_convergence.httpx,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(httpx.ReadTimeout("lost")),
    )
    result = agent_setup_convergence.reconcile_agent_setup(
        host="192.0.2.20",
        ssh_user="ubuntu",
        ssh_key_path="/tmp/key",
        project_alias="live",
        agent_name="agent-live-id",
        project_id="project-exact",
        auth_user="npa",
        auth_password="secret",
        agent_port=8088,
        public_https=True,
    )
    assert result["state"] == "indeterminate"
    assert result["error_category"] == "ReadTimeout"


def test_blocking_calls_emit_structured_secret_free_heartbeats(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    operation = ProvisioningOperation.prepare(
        command="npa agent bootstrap",
        project_alias="live",
        project_id="project-exact",
        resource_type="agent",
        requested_name="agent-live-id",
        resume_command="",
    )
    events = []
    periodic_heartbeat = threading.Event()

    def emit(event) -> None:
        events.append(event)
        if event["heartbeat_sequence"] > 0:
            periodic_heartbeat.set()

    with operation_heartbeats(
        operation,
        phase="remote_bootstrap",
        interval_seconds=0.01,
        emit=emit,
    ):
        assert periodic_heartbeat.wait(timeout=1.0)
    assert len(events) >= 2
    assert all(event["operation_phase"] == "remote_bootstrap" for event in events)
    journal = operation.read()
    rendered = json.dumps(journal)
    assert "heartbeat_sequence" in rendered
    assert "secret" not in rendered.casefold()


def test_capacity_reuses_one_exact_journal_owned_cluster(monkeypatch) -> None:
    owned = SimpleNamespace(
        read=lambda: {
            "phase": "committed",
            "requested_name": "k8s-live-id",
            "resources": [
                {
                    "resource_type": "managed_kubernetes_cluster",
                    "ownership": "created_by_this_operation",
                    "project_id": "project-exact",
                    "requested_name": "k8s-live-id",
                    "provider_id": "mk8scluster-exact",
                }
            ],
        }
    )
    monkeypatch.setattr(
        "npa.provisioning_journal.list_operations", lambda **_kwargs: [owned]
    )
    assert (
        agent_quota._exact_owned_cluster_name("project-exact", "npa-cluster")
        == "k8s-live-id"
    )


def test_capacity_does_not_choose_between_multiple_owned_clusters(monkeypatch) -> None:
    def operation(name: str, provider_id: str):
        return SimpleNamespace(
            read=lambda: {
                "phase": "committed",
                "requested_name": name,
                "resources": [
                    {
                        "resource_type": "managed_kubernetes_cluster",
                        "ownership": "created_by_this_operation",
                        "project_id": "project-exact",
                        "requested_name": name,
                        "provider_id": provider_id,
                    }
                ],
            }
        )

    monkeypatch.setattr(
        "npa.provisioning_journal.list_operations",
        lambda **_kwargs: [operation("one", "id-one"), operation("two", "id-two")],
    )
    assert (
        agent_quota._exact_owned_cluster_name("project-exact", "npa-cluster")
        == "npa-cluster"
    )
