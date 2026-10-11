"""Verify real TLS transport and reject public scheduler or certificate mistakes."""

import base64
import ssl
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from npa.workbench.team import endpoint_health
from npa.workbench.team.endpoint_certificate import verify_certificate
from npa.workbench.team.errors import BackendError


@pytest.fixture
def certificate(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "team.example.test")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("team.example.test")]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    cert_path, key_path = tmp_path / "certificate.pem", tmp_path / "private.pem"
    cert_path.write_bytes(pem)
    key_path.write_bytes(private)
    key_path.chmod(0o600)
    return SimpleNamespace(
        cert=cert_path,
        key=key_path,
        secret={
            "data": {
                "tls.crt": base64.b64encode(pem).decode(),
                "tls.key": base64.b64encode(private).decode(),
            }
        },
    )


@pytest.fixture
def https_server(certificate):
    observed = []
    statuses = {
        "/health": 200,
        "/v1/me": 401,
        "/api/health": 404,
        "/api/status": 404,
        "/status": 404,
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append(dict(self.headers))
            self.send_response(statuses[self.path])
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate.cert, certificate.key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            address=server.server_address, headers=observed, statuses=statuses
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _connect_to_test_server(monkeypatch, server):
    real_connect = endpoint_health.socket.create_connection
    real_resolve = endpoint_health.socket.getaddrinfo

    def connect(address, **kwargs):
        assert address == ("203.0.113.8", 443)
        return real_connect(server.address, **kwargs)

    monkeypatch.setattr(endpoint_health.socket, "create_connection", connect)

    def resolve(*args, **kwargs):
        if args[0] == "team.example.test":
            return [(2, 1, 6, "", ("203.0.113.8", 443))]
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(endpoint_health.socket, "getaddrinfo", resolve)


def test_actual_tls_verifies_sni_and_boundary_without_a_personal_credential(
    certificate, https_server, monkeypatch
):
    _connect_to_test_server(monkeypatch, https_server)
    request = SimpleNamespace(
        endpoint="https://team.example.test", ca_file=certificate.cert
    )
    proof = endpoint_health.verify_endpoint(request, "203.0.113.8")
    assert proof == {
        "dns_points_to_lb": True,
        "boundary_statuses": https_server.statuses,
    }
    assert len(https_server.headers) == 5
    assert all("Authorization" not in headers for headers in https_server.headers)


def test_untrusted_certificate_is_not_treated_as_a_connectivity_success(
    certificate, https_server, monkeypatch
):
    _connect_to_test_server(monkeypatch, https_server)
    request = SimpleNamespace(endpoint="https://team.example.test", ca_file=None)
    with pytest.raises(ssl.SSLCertVerificationError):
        endpoint_health.verify_endpoint(request, "203.0.113.8")
    assert https_server.headers == []


def test_exposed_native_scheduler_fails_setup_boundary(
    certificate, https_server, monkeypatch
):
    _connect_to_test_server(monkeypatch, https_server)
    https_server.statuses["/api/health"] = 200
    request = SimpleNamespace(
        endpoint="https://team.example.test", ca_file=certificate.cert
    )
    with pytest.raises(BackendError, match="authentication boundary"):
        endpoint_health.verify_endpoint(request, "203.0.113.8")


def test_certificate_preflight_accepts_matching_key_and_hostname(certificate):
    verify_certificate(
        SimpleNamespace(endpoint="https://team.example.test"), certificate.secret
    )


def test_certificate_preflight_rejects_wrong_hostname_without_exposing_key(certificate):
    with pytest.raises(
        BackendError, match="certificate or private key is invalid"
    ) as error:
        verify_certificate(
            SimpleNamespace(endpoint="https://different.example.test"),
            certificate.secret,
        )
    assert "BEGIN" not in str(error.value)


def test_certificate_preflight_rejects_mismatched_private_key(certificate):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    certificate.secret["data"]["tls.key"] = base64.b64encode(key).decode()
    with pytest.raises(BackendError):
        verify_certificate(
            SimpleNamespace(endpoint="https://team.example.test"), certificate.secret
        )
