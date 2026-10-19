"""Exercise agent certificate trust against real HTTPS listeners with synthetic auth."""

import ipaddress
import json
import ssl
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import typer
from typer.testing import CliRunner
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from npa.cli import (
    agent,
    agent_deployment,
    agent_setup_convergence,
    agent_tls,
    agent_verification,
)
from npa.clients.ssh import SSHHostKeyError
from npa.clients.network import NetworkIngressError
from npa.clients.nebius import NebiusError
from npa.orchestration.npa_workflow import artifact_load


def _certificate(tmp_path, name, *, host="127.0.0.1", expired=False):
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=2))
        .not_valid_after(now + timedelta(days=-1 if expired else 1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(host))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    pem = certificate.public_bytes(serialization.Encoding.PEM).decode()
    cert_path, key_path = tmp_path / f"{name}.crt", tmp_path / f"{name}.key"
    cert_path.write_text(pem)
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    return SimpleNamespace(pem=pem, cert=cert_path, key=key_path)


@contextmanager
def _listener(certificate):
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate.cert, certificate.key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield SimpleNamespace(
            url=f"https://127.0.0.1:{server.server_port}/", observed=observed
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def _trusted_ssh(certificate):
    return SimpleNamespace(run=Mock(return_value=(0, certificate.pem, "")))


def test_real_agent_certificate_and_hostname_are_verified(tmp_path):
    certificate = _certificate(tmp_path, "agent")
    ssh = _trusted_ssh(certificate)
    context = agent_tls.trusted_agent_tls_context(ssh)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    assert len(context.get_ca_certs(binary_form=True)) <= 1
    with _listener(certificate) as listener:
        assert agent._health(
            listener.url, user="synthetic", password="test-only", verify=context
        ) == (True, 200)
        assert len(listener.observed) == 1
        assert listener.observed[0].startswith("Basic ")
    ssh.run.assert_called_once_with(
        "openssl x509 -in /etc/nginx/ssl/npa-agent.crt -outform PEM"
    )


@pytest.mark.parametrize("failure", ["substituted-certificate", "hostname", "expired"])
def test_bad_tls_identity_receives_no_http_credentials(tmp_path, failure):
    certificate = _certificate(
        tmp_path,
        "listener",
        host="127.0.0.2" if failure == "hostname" else "127.0.0.1",
        expired=failure == "expired",
    )
    trusted = (
        _certificate(tmp_path, "real-agent")
        if failure == "substituted-certificate"
        else certificate
    )
    context = agent_tls.trusted_agent_tls_context(_trusted_ssh(trusted))
    with _listener(certificate) as listener:
        assert agent._health(
            listener.url, user="synthetic", password="test-only", verify=context
        ) == (False, 0)
        assert listener.observed == []


@pytest.mark.parametrize(
    "output",
    [
        "",
        "invalid",
        "-----BEGIN CERTIFICATE-----\ninvalid\n-----END CERTIFICATE-----",
        "two-certificates",
    ],
)
def test_malformed_or_multiple_ssh_certificates_fail_closed(tmp_path, output):
    if output == "two-certificates":
        output = _certificate(tmp_path, "agent").pem * 2
    ssh = SimpleNamespace(run=Mock(return_value=(0, output, "")))
    with pytest.raises(ValueError, match="invalid agent TLS certificate"):
        agent_tls.trusted_agent_tls_context(ssh)


def test_failed_certificate_read_does_not_expose_remote_diagnostics():
    ssh = SimpleNamespace(run=Mock(return_value=(1, "", "private diagnostic")))
    with pytest.raises(ValueError, match="verified SSH") as error:
        agent_tls.trusted_agent_tls_context(ssh)
    assert "private diagnostic" not in str(error.value)


def test_untrusted_ssh_key_is_not_used_to_establish_https_trust():
    ssh = SimpleNamespace(run=Mock(side_effect=SSHHostKeyError("unknown SSH key")))
    with pytest.raises(SSHHostKeyError):
        agent_tls.trusted_agent_tls_context(ssh)


@pytest.fixture
def recorded_agent(monkeypatch, tmp_path):
    certificate = _certificate(tmp_path, "recorded-agent")
    resolve = Mock(return_value=SimpleNamespace(public_ip="127.0.0.1/32"))
    connect = Mock(return_value=_trusted_ssh(certificate))
    monkeypatch.setattr(agent_tls, "resolve_instance_network_context", resolve)
    monkeypatch.setattr(agent_tls, "SSHClient", connect)
    record = {
        "public_https": True,
        "instance_id": "instance-test",
        "public_ip": "127.0.0.1",
        "ssh_user": "fixture-user",
        "ssh_key_path": str(certificate.key),
    }
    return SimpleNamespace(
        record=record, cert=certificate, resolve=resolve, connect=connect
    )


def test_recorded_instance_and_ssh_identity_precede_https(recorded_agent):
    context = agent_deployment.record_tls_verify(recorded_agent.record)
    with _listener(recorded_agent.cert) as listener:
        assert agent._health(
            listener.url, user="synthetic", password="test-only", verify=context
        ) == (True, 200)
    recorded_agent.resolve.assert_called_once_with("instance-test")
    config = recorded_agent.connect.call_args.args[0]
    assert (config.host, config.user, config.key_path) == (
        "127.0.0.1",
        "fixture-user",
        recorded_agent.record["ssh_key_path"],
    )


@pytest.mark.parametrize("field", ["instance_id", "ssh_key_path", "public_ip"])
def test_incomplete_record_does_not_fall_back_to_unverified_tls(recorded_agent, field):
    recorded_agent.record.pop(field)
    with pytest.raises(ValueError):
        agent_deployment.record_tls_verify(recorded_agent.record)
    recorded_agent.connect.assert_not_called()


def test_provider_address_change_rejected_before_ssh(recorded_agent):
    recorded_agent.resolve.return_value.public_ip = "127.0.0.2"
    with pytest.raises(ValueError, match="differs from provider"):
        agent_deployment.record_tls_verify(recorded_agent.record)
    recorded_agent.connect.assert_not_called()


def test_non_public_mode_keeps_system_ca_verification():
    assert agent_deployment.record_tls_verify({"public_https": False}) is True


def test_verification_uses_one_trust_identity_throughout_run(recorded_agent):
    context = agent_verification._VerificationContext(
        recorded_agent.record, ("synthetic", "test-only"), "test", Mock(), Mock()
    )
    assert context.tls_verify is context.tls_verify
    recorded_agent.connect.assert_called_once()


@pytest.mark.parametrize(
    "error",
    [
        ValueError("private diagnostic"),
        SSHHostKeyError("private diagnostic"),
        NetworkIngressError("private diagnostic"),
        NebiusError("private diagnostic"),
        PermissionError("private diagnostic"),
    ],
)
def test_status_trust_failure_emits_safe_json_without_http(monkeypatch, error):
    record = {"agent_url": "https://agent.invalid/", "public_https": True}
    monkeypatch.setattr(agent, "_resolve_project_alias", lambda _: "fixture")
    monkeypatch.setattr(agent, "_agent_record", lambda *_: record)
    monkeypatch.setattr(
        agent, "_load_auth_secret", lambda _: ("synthetic", "test-only")
    )
    monkeypatch.setattr(agent, "_record_tls_verify", Mock(side_effect=error))
    request = Mock(side_effect=AssertionError("untrusted HTTP must not be sent"))
    monkeypatch.setattr(agent.httpx, "get", request)
    result = CliRunner().invoke(agent.app, ["status", "--project", "fixture", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["health"] is False
    assert payload["endpoint_disclosure_allowed"] is False
    assert payload["basic_auth_enforced"] is False
    assert payload["error_category"] == type(error).__name__
    for key in (
        "public_ip",
        "public_url",
        "direct_url",
        "ui_url",
        "rerun_url",
        "sim_viz_url",
        "sim_assets_url",
        "cameras_api_url",
    ):
        assert payload[key] == ""
    assert "private diagnostic" not in result.output
    assert "agent.invalid" not in result.output
    request.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        ValueError("private diagnostic"),
        SSHHostKeyError("private diagnostic"),
        NetworkIngressError("private diagnostic"),
        NebiusError("private diagnostic"),
        PermissionError("private diagnostic"),
    ],
)
def test_verification_reports_trust_failure_before_http(monkeypatch, error):
    monkeypatch.setattr(
        agent_verification, "record_tls_verify", Mock(side_effect=error)
    )
    request = Mock(side_effect=AssertionError("untrusted HTTP must not be sent"))
    monkeypatch.setattr(agent_verification.httpx, "get", request)
    failed = Mock(side_effect=typer.Exit(1))
    with pytest.raises(typer.Exit):
        agent_verification.verify_agent_endpoints(
            {"agent_url": "https://agent.invalid/", "public_https": False},
            auth=("synthetic", "test-only"),
            ui_version="test",
            health_check=Mock(),
            fail=failed,
        )
    failed.assert_called_once()
    message = failed.call_args.args[0]
    assert type(error).__name__ in message
    assert "private diagnostic" not in message
    request.assert_not_called()


def test_status_rejects_substituted_listener_before_auth(
    monkeypatch, recorded_agent, tmp_path, capsys
):
    substitute = _certificate(tmp_path, "substituted")
    with _listener(substitute) as listener:
        recorded_agent.record.update(agent_url=listener.url, sim_viz_url=listener.url)
        monkeypatch.setattr(agent, "_resolve_project_alias", lambda _project: "fixture")
        monkeypatch.setattr(
            agent, "_agent_record", lambda *_args: recorded_agent.record
        )
        monkeypatch.setattr(
            agent, "_load_auth_secret", lambda _path: ("synthetic", "test-only")
        )
        agent.status_cmd(project="fixture", name="agent", output_json=True)
        result = json.loads(capsys.readouterr().out)
        assert result["health"] is False
        assert result["endpoint_disclosure_allowed"] is False
        assert listener.observed == []


def test_artifact_handoff_rejects_substituted_listener(recorded_agent, tmp_path):
    substitute = _certificate(tmp_path, "substituted")
    runtime = SimpleNamespace(record_tls_verify=agent_deployment.record_tls_verify)
    with _listener(substitute) as listener:
        connection = artifact_load._agent_connection(
            runtime,
            recorded_agent.record,
            listener.url.rstrip("/"),
            httpx.request,
            ("synthetic", "test-only"),
        )
        with pytest.raises(artifact_load.ArtifactLoadError):
            artifact_load._sim_viz_status(connection, "test")
        assert listener.observed == []


def _setup_state():
    return {
        "project_alias": "fixture",
        "agent_name": "agent",
        "project_id": "project-test",
        "endpoint": "127.0.0.1",
        "phase": "remote_health_ready",
        "service_fingerprint": "service-sha",
        "credential_fingerprint": "credential-sha",
        "credential_fingerprint_files": [
            "llm.env",
            "s3.env",
            "artifact-sources.env",
            "nebius.env",
        ],
    }


@pytest.mark.parametrize("substituted", [False, True])
def test_bootstrap_recovery_authenticates_tls_before_health(
    monkeypatch, tmp_path, substituted
):
    certificate = _certificate(tmp_path, "actual-agent")
    listener_certificate = (
        _certificate(tmp_path, "foreign") if substituted else certificate
    )
    payload = _setup_state()
    ssh = SimpleNamespace(
        run=Mock(side_effect=[(0, json.dumps(payload), ""), (0, certificate.pem, "")])
    )
    monkeypatch.setattr(agent_setup_convergence, "SSHClient", lambda **_kwargs: ssh)
    monkeypatch.setattr(
        agent_setup_convergence,
        "resolve_ssh_config",
        lambda **_kwargs: SimpleNamespace(ssh=None),
    )
    actual_get = httpx.get
    with _listener(listener_certificate) as listener:

        def local_get(url, **kwargs):
            assert url == "https://127.0.0.1/api/models"
            return actual_get(listener.url, **kwargs)

        monkeypatch.setattr(agent_setup_convergence.httpx, "get", local_get)
        result = agent_setup_convergence.reconcile_agent_setup(
            host="127.0.0.1",
            ssh_user="fixture",
            ssh_key_path=str(certificate.key),
            project_alias="fixture",
            agent_name="agent",
            project_id="project-test",
            auth_user="synthetic",
            auth_password="test-only",
            agent_port=8088,
            public_https=True,
        )
        assert result["state"] == ("indeterminate" if substituted else "healthy")
        assert result["models_healthy"] is not substituted
        assert len(listener.observed) == (0 if substituted else 1)
        assert "test-only" not in json.dumps(result)
