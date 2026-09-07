"""Provider-bound host-key discovery and real SSH authentication regressions."""

from __future__ import annotations

import io
import json
import socket
import threading
import time

import paramiko
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from npa.clients.config import SSHConfig
from npa.clients.ssh import SSHClient, SSHHostKeyError
from npa.deploy import ssh_trust


def _key():
    private = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    ).decode()
    key = paramiko.Ed25519Key.from_private_key(io.StringIO(private))
    return key, private


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr(ssh_trust, "NPA_CONFIG_DIR", tmp_path)
    key, _private = _key()
    state = {
        "args": {"project_id": "project-fixture", "instance_id": "instance-fixture",
                 "host": "192.0.2.20", "nonce": "a" * 64},
        "instance": {
            "metadata": {"id": "instance-fixture", "parent_id": "project-fixture"},
            "spec": {"cloud_init_user_data": "# NPA_SSH_HOST_KEY_NONCE=" + "a" * 64 + "\n"},
            "status": {"network_interfaces": [{"public_ip_address": {"address": "192.0.2.20/32"}}]},
        },
        "row": {
            "labels": {"sp_resource_id": "instance-fixture", "__bucket__": "sp_serial"},
            "message": f"NPA_SSH_HOST_KEY {'a' * 64} ssh-ed25519 {key.get_base64()} guest",
            "timestamp": "2026-01-01T00:00:00Z", "level": "INFO",
        },
        "queries": [],
    }
    monkeypatch.setattr(ssh_trust.nebius, "_run_json", lambda _args: state["instance"])

    def query(args):
        state["queries"].append(args)
        return state.get("raw", json.dumps(state["row"]))

    monkeypatch.setattr(ssh_trust.nebius, "_run", query)
    return state


def test_verified_instance_pins_key_privately_and_reuses_identical_evidence(provider):
    path = ssh_trust.verify_host_key(**provider["args"])
    expected = provider["row"]["message"].split(" ", 2)[2].removesuffix(" guest")
    assert path.read_text() == "192.0.2.20 " + expected + "\n"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert ssh_trust.verify_host_key(**provider["args"]) == path
    query = provider["queries"][0]
    assert query[2] == '{sp_resource_id="instance-fixture"} |= "NPA_SSH_HOST_KEY ' + "a" * 64 + ' "'
    assert query[query.index("--project-id") + 1] == "project-fixture"
    assert query[query.index("--bucket") + 1] == "sp_serial"


@pytest.mark.parametrize("field,value", [("id", "other-instance"), ("parent_id", "other-project")])
def test_provider_identity_mismatch_prevents_log_read(provider, field, value):
    provider["instance"]["metadata"][field] = value
    with pytest.raises(ssh_trust.HostTrustError, match="instance identity"):
        ssh_trust.verify_host_key(**provider["args"])
    assert not provider["queries"]


def test_wrong_ip_and_nonce_stop_before_log_read(provider):
    for override in ({"host": "192.0.2.21"}, {"nonce": "b" * 64}):
        with pytest.raises(ssh_trust.HostTrustError):
            ssh_trust.verify_host_key(**{**provider["args"], **override})
    assert not provider["queries"]


@pytest.mark.parametrize("field,value", [("sp_resource_id", "other-instance"), ("__bucket__", "default")])
def test_log_identity_mismatch_is_terminal(provider, field, value):
    provider["row"]["labels"][field] = value
    with pytest.raises(ssh_trust.HostTrustError, match="log identity"):
        ssh_trust.verify_host_key(**provider["args"])
    assert not ssh_trust.known_hosts_path(provider["args"]["host"]).exists()


@pytest.mark.parametrize("raw", ["", "\n"])
def test_empty_logs_are_pending_without_accepting_unknown_host(provider, raw):
    provider["raw"] = raw
    with pytest.raises(ssh_trust.HostKeyPending):
        ssh_trust.verify_host_key(**provider["args"])
    assert not ssh_trust.known_hosts_path(provider["args"]["host"]).exists()


@pytest.mark.parametrize("raw", ["not-json", "[]", "{}", '{"labels":{}}'])
def test_malformed_provider_records_fail_closed(provider, raw):
    provider["raw"] = raw
    with pytest.raises(ssh_trust.HostTrustError):
        ssh_trust.verify_host_key(**provider["args"])


def test_full_jsonl_is_validated_before_any_host_key_is_written(provider):
    other = {**provider["row"], "labels": {"sp_resource_id": "other-instance", "__bucket__": "sp_serial"}}
    provider["raw"] = json.dumps(provider["row"]) + "\n" + json.dumps(other)
    with pytest.raises(ssh_trust.HostTrustError):
        ssh_trust.verify_host_key(**provider["args"])
    assert not ssh_trust.known_hosts_path(provider["args"]["host"]).exists()


def test_changed_and_conflicting_keys_are_rejected(provider):
    path = ssh_trust.verify_host_key(**provider["args"])
    original = path.read_bytes()
    second, _ = _key()
    changed = {**provider["row"], "message": f"NPA_SSH_HOST_KEY {'a' * 64} ssh-ed25519 {second.get_base64()}"}
    path.write_text(f"192.0.2.20 ssh-ed25519 {second.get_base64()}\n")
    provider["raw"] = json.dumps(changed)
    with pytest.raises(ssh_trust.HostTrustError, match="changed"):
        ssh_trust.verify_host_key(**provider["args"])
    path.write_bytes(original)
    path.with_suffix(".json").unlink()
    provider["raw"] = json.dumps(provider["row"]) + "\n" + json.dumps(changed)
    with pytest.raises(ssh_trust.HostTrustError, match="Conflicting"):
        ssh_trust.verify_host_key(**provider["args"])
    assert path.read_bytes() == original


def test_authenticated_pin_survives_expired_serial_logs(provider):
    path = ssh_trust.verify_host_key(**provider["args"])
    provider["raw"] = ""
    assert ssh_trust.verify_host_key(**provider["args"]) == path
    assert len(provider["queries"]) == 1


def test_older_instance_requires_existing_operator_verified_key(provider, monkeypatch, tmp_path):
    path = tmp_path / "operator-known-hosts"
    key, _ = _key()
    path.write_text(f"192.0.2.20 ssh-ed25519 {key.get_base64()}\n")
    monkeypatch.setenv("NPA_SSH_KNOWN_HOSTS", str(path))
    assert ssh_trust.verify_host_key(**{**provider["args"], "nonce": ""}) == path
    path.write_text(f"192.0.2.21 ssh-ed25519 {key.get_base64()}\n")
    with pytest.raises(ssh_trust.HostTrustError, match="no independently verified"):
        ssh_trust.verify_host_key(**{**provider["args"], "nonce": ""})
    assert not provider["queries"]


@pytest.mark.parametrize("matching", [True, False])
def test_real_ssh_handshake_authenticates_only_the_verified_host(tmp_path, monkeypatch, matching):
    """Run a real loopback SSH exchange, including public-key user auth."""
    monkeypatch.setattr(ssh_trust, "NPA_CONFIG_DIR", tmp_path / "config")
    monkeypatch.delenv("NPA_SSH_KNOWN_HOSTS", raising=False)
    server_key, _ = _key()
    expected = server_key if matching else _key()[0]
    user_key, user_private = _key()
    private_file = tmp_path / "user-key"
    private_file.write_text(user_private)
    private_file.chmod(0o600)
    pin = ssh_trust.known_hosts_path("127.0.0.1")
    ssh_trust._atomic_private_file(pin, f"127.0.0.1 ssh-ed25519 {expected.get_base64()}\n")
    auth_attempts = []

    class Server(paramiko.ServerInterface):
        def check_auth_publickey(self, username, key):
            auth_attempts.append(username)
            return paramiko.AUTH_SUCCESSFUL if key == user_key else paramiko.AUTH_FAILED

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve():
        connection, _ = listener.accept()
        with paramiko.Transport(connection) as transport:
            transport.add_server_key(server_key)
            transport.start_server(server=Server())
            while transport.is_active():
                time.sleep(0.01)

    thread = threading.Thread(target=serve)
    thread.start()
    connect = paramiko.SSHClient.connect

    def local_connect(self, **kwargs):
        return connect(self, sock=socket.create_connection(listener.getsockname()), **kwargs)

    monkeypatch.setattr(paramiko.SSHClient, "connect", local_connect)
    try:
        client = SSHClient(SSHConfig(host="127.0.0.1", user="fixture", key_path=str(private_file)))
        if matching:
            client._connect().close()
            assert auth_attempts == ["fixture"]
        else:
            with pytest.raises(SSHHostKeyError):
                client._connect()
            assert auth_attempts == []
    finally:
        thread.join()
        listener.close()
