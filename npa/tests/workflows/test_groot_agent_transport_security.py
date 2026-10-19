"""Exercise agent credential boundaries through real HTTP and TLS listeners."""

import ipaddress
import ssl
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from npa.workflows import groot_learning as learning


@pytest.fixture
def agent_servers(tmp_path):
    servers = []

    def start(*, tls=False, redirect="", matching_hostname=True):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(dict(self.headers))
                self.send_response(302 if redirect else 200)
                if redirect:
                    self.send_header("Location", redirect)
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        ca_file = None
        if tls:
            key = ec.generate_private_key(ec.SECP256R1())
            subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "agent.test")])
            now = datetime.now(timezone.utc)
            san = (
                x509.IPAddress(ipaddress.ip_address("127.0.0.1"))
                if matching_hostname
                else x509.DNSName("wrong-agent.test")
            )
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(subject)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=1))
                .not_valid_after(now + timedelta(days=1))
                .add_extension(x509.SubjectAlternativeName([san]), critical=False)
                .sign(key, hashes.SHA256())
            )
            ca_file = tmp_path / f"agent-{server.server_port}.crt"
            key_file = ca_file.with_suffix(".key")
            ca_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
            key_file.write_bytes(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(ca_file, key_file)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return (
            f"{'https' if tls else 'http'}://127.0.0.1:{server.server_port}",
            ca_file,
            received,
        )

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize("matching_hostname", [False, True])
def test_tls_rejects_untrusted_or_wrong_identity_before_auth(
    agent_servers, monkeypatch, trusted, matching_hostname
):
    origin, ca_file, received = agent_servers(
        tls=True, matching_hostname=matching_hostname
    )
    monkeypatch.delenv("NPA_AGENT_TLS_VERIFY", raising=False)
    monkeypatch.delenv("NPA_AGENT_CA_FILE", raising=False)
    if trusted:
        monkeypatch.setenv("NPA_AGENT_CA_FILE", str(ca_file))
    opener = learning._agent_http_opener(origin)
    request = urllib.request.Request(
        origin + "/api/health", headers={"Authorization": "Basic synthetic"}
    )
    if trusted and matching_hostname:
        with opener.open(request) as response:
            assert response.status == 200
        assert received[0]["Authorization"] == "Basic synthetic"
    else:
        with pytest.raises(urllib.error.URLError, match="CERTIFICATE_VERIFY_FAILED"):
            opener.open(request)
        assert received == []


def test_redirect_cannot_forward_agent_auth_to_another_listener(agent_servers):
    target, _, target_requests = agent_servers()
    origin, _, agent_requests = agent_servers(redirect=target + "/steal")
    opener = learning._agent_http_opener(origin)
    request = urllib.request.Request(
        origin + "/api/health", headers={"Authorization": "Basic synthetic"}
    )
    with pytest.raises(urllib.error.HTTPError, match="302"):
        opener.open(request)
    assert agent_requests[0]["Authorization"] == "Basic synthetic"
    assert target_requests == []


@pytest.mark.parametrize(
    "origin",
    [
        "http://agent.example",
        "https://user:password@agent.example",
        "https://agent.example/path",
        "https://agent.example?query=1",
        "https://agent.example#fragment",
        "file:///etc/passwd",
    ],
)
def test_invalid_origins_are_rejected(origin):
    with pytest.raises(learning.GrootVisualizationError, match="origin"):
        learning._agent_http_opener(origin)


def test_tls_verification_cannot_be_disabled(monkeypatch):
    monkeypatch.setenv("NPA_AGENT_TLS_VERIFY", "0")
    with pytest.raises(learning.GrootVisualizationError, match="NPA_AGENT_CA_FILE"):
        learning._agent_http_opener("https://agent.example")
