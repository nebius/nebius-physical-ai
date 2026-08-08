from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.workbench.leisaac import (
    _delete_resources,
    _install_agent_relay,
    _put_manifest,
    _relay_media_server,
    _select_agent_leisaac_run,
    _wait_ready,
    app,
)
from npa.agent_backend.leisaac_registry import DEFAULT_TASK, REGISTRY_FINGERPRINT


IMAGE = "registry.example/npa-leisaac@sha256:" + "1" * 64
runner = CliRunner()


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
        lambda *args: SimpleNamespace(
            returncode=0, stdout=json.dumps(next(responses)), stderr=""
        ),
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.time.sleep", lambda seconds: calls.append(seconds)
    )

    _wait_ready("cluster", "leisaac", "leisaac-live")

    assert calls == [5]


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
    )

    assert "sudo install -d -m 0755 /etc/npa /opt/npa-agent" in ssh.command
    assert "DynamicUser=yes" not in ssh.command  # unit is base64-encoded in transit
    assert "openssl req -x509" not in ssh.command


def test_relay_media_server_is_the_single_ready_pod_host(monkeypatch) -> None:
    pod_list = {
        "items": [
            {
                "metadata": {"name": "old", "deletionTimestamp": "now"},
                "status": {
                    "phase": "Running",
                    "podIP": "10.96.34.1",
                    "containerStatuses": [{"name": "leisaac", "ready": True}],
                },
            },
            {
                "metadata": {"name": "current"},
                "status": {
                    "phase": "Running",
                    "podIP": "10.96.34.22",
                    "containerStatuses": [
                        {"name": "agent-relay-client", "ready": True},
                        {"name": "leisaac", "ready": True},
                    ],
                },
            },
        ]
    }
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._kubectl",
        lambda *_args: SimpleNamespace(
            returncode=0, stdout=json.dumps(pod_list), stderr=""
        ),
    )

    assert _relay_media_server("cluster", "leisaac", "deployment") == "10.96.34.22"


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
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setenv("ISAACSIM_ACCEPT_EULA", "YES")
    registry_refreshes = []
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.ensure_registry_pull_secret_for_images",
        lambda *args, **kwargs: registry_refreshes.append((args, kwargs)),
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
    monkeypatch.setattr("npa.cli.workbench.leisaac._wait_ready", lambda *_args: None)
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._relay_media_server",
        lambda *_args: "10.96.34.22",
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._remove_agent_turn",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac._relay_status",
        lambda *_args: {
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
        lambda *_args: {
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
        registry_refreshes,
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
    manifest = {"run_id": "live-relay"}
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
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setenv("ISAACSIM_ACCEPT_EULA", "YES")
    rejected = runner.invoke(app, [*_args(), "--num-envs", "2"])
    assert rejected.exit_code == 1
    assert "exactly one active environment" in rejected.output


def test_launch_agent_relay_wires_private_cluster_public_agent_and_manifest(
    monkeypatch,
) -> None:
    (
        applied,
        ingress_calls,
        install_calls,
        manifests,
        ssh,
        registry_refreshes,
        selections,
    ) = _patch_launch(monkeypatch)

    result = runner.invoke(app, _args())

    assert result.exit_code == 0, result.output
    assert "transport: agent-relay" in result.output
    assert "public_agent_url: https://8.8.4.4/" in result.output
    assert registry_refreshes == [
        (
            (IMAGE,),
            {
                "secret_name": "npa-registry",
                "namespace": "leisaac",
                "k8s_context": "cluster",
            },
        )
    ]
    assert applied[0]["spec"]["type"] == "ClusterIP"
    assert applied[1]["kind"] == "Secret"
    assert applied[2]["kind"] == "Secret"
    assert applied[2]["metadata"]["name"].endswith("-recorder")
    deployment = next(item for item in applied if item["kind"] == "Deployment")
    deployment_env = {
        item["name"]: item
        for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
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
        "npa.nebius.com/turn-peer-source" not in applied[-1]["metadata"]["annotations"]
    )
    assert ingress_calls == [
        {
            "vm_id": "vm-agent",
            "ports": (3478,),
            "source": "8.8.8.8/32",
            "tool": "leisaac-turn-control",
            "protocol": "UDP",
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
    assert selections == [
        (
            ("8.8.4.4",),
            {
                "auth_user": "npa",
                "auth_password": "secret",
                "run_id": "live-relay",
                "certificate_sha256": "f" * 64,
            },
        )
    ]


def test_launch_fails_closed_before_deployment_when_registry_refresh_fails(
    monkeypatch,
) -> None:
    applied, *_rest = _patch_launch(monkeypatch)
    monkeypatch.setattr(
        "npa.cli.workbench.leisaac.ensure_registry_pull_secret_for_images",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("registry credential refresh failed")
        ),
    )

    result = runner.invoke(app, _args())

    assert result.exit_code == 1
    assert "registry credential refresh failed" in result.output
    assert applied == []


def test_failed_agent_relay_launch_removes_partial_relay_ingress_and_kubernetes(
    monkeypatch,
) -> None:
    (
        _applied,
        _ingress,
        _install,
        _manifests,
        _ssh,
        _registry,
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
    assert attempted == ["TURN", "relay", "ingress", "Kubernetes"]
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
        _registry,
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
    assert (
        "npa.nebius.com/turn-peer-source" not in applied[0]["metadata"]["annotations"]
    )
    assert (
        "npa.nebius.com/turn-peer-source" not in applied[-1]["metadata"]["annotations"]
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
        "9.9.8.0/22",
    ]
    assert [item[1]["ports"] for item in ingress_removals] == [(3478,), (47999,)]
    assert all(item[0] == ("vm-agent",) for item in ingress_removals)
    assert all(item[1]["protocol"] == "UDP" for item in ingress_removals)
    assert k8s_removals == [("cluster", "leisaac", "leisaac-live-relay")]
