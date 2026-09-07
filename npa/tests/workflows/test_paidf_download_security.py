"""Real HTTPS redirects never disclose source credentials to another origin."""
from __future__ import annotations

import datetime
import hashlib
import ipaddress
import ssl
import threading
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from urllib.request import HTTPSHandler, ProxyHandler

from npa.workflows import data_factory_input as dfi


@pytest.fixture
def https_origins(tmp_path, monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    key_path.chmod(0o600)
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    client_context = ssl.create_default_context(cafile=str(cert_path))
    state = {"source": [], "cdn": [], "redirect": "", "body": b"synthetic-media"}

    class Source(BaseHTTPRequestHandler):
        def do_GET(self):
            state["source"].append(self.headers.get("Authorization"))
            self.send_response(302)
            self.send_header("Location", "/same" if self.path == "/start" else state["redirect"])
            self.end_headers()

        def log_message(self, *_args):
            pass

    class CDN(BaseHTTPRequestHandler):
        def do_GET(self):
            state["cdn"].append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *_args):
            pass

    with ExitStack() as stack:
        origins = []
        for handler in [Source, CDN]:
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            server.socket = server_context.wrap_socket(server.socket, server_side=True)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            stack.callback(thread.join)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            origins.append(f"https://127.0.0.1:{server.server_port}")
        state["url"] = origins[0] + "/start"
        state["redirect"] = origins[1] + "/media"
        original = dfi.build_opener
        monkeypatch.setattr(dfi, "build_opener", lambda handler: original(ProxyHandler({}), HTTPSHandler(context=client_context), handler))
        yield state


@pytest.mark.parametrize("authenticated", [False, True])
def test_real_https_download_scopes_token_to_source_origin(https_origins, tmp_path, monkeypatch, authenticated):
    state = https_origins
    monkeypatch.setenv("HF_TOKEN", "synthetic-private-token")
    contract = {
        "asset_id": "fixture",
        "integrity": {"sha256": hashlib.sha256(state["body"]).hexdigest(), "byte_size": len(state["body"])},
        "source": {"asset_url": state["url"]},
        "license": {"authentication_required": authenticated},
    }
    result, status = dfi._fetch_starter(contract, cache_dir=tmp_path / "cache", offline=False, reporter=lambda _message: None)
    assert status == "verified_fetch"
    assert result.read_bytes() == state["body"]
    expected = "Bearer synthetic-private-token" if authenticated else None
    assert state["source"] == [expected, expected]
    assert state["cdn"] == [None]


def test_real_https_download_refuses_redirect_to_plaintext(https_origins):
    state = https_origins
    state["redirect"] = state["redirect"].replace("https:", "http:")
    with pytest.raises(dfi.PaidfInputError, match="HTTPS"):
        dfi._open_starter_url(state["url"], "synthetic-private-token")
    assert state["source"] == ["Bearer synthetic-private-token"] * 2
    assert state["cdn"] == []


@pytest.mark.parametrize("url", ["http://example.test/asset", "file:///etc/passwd", "https://user:password@example.test/asset", "https://example.test:invalid/asset"])
def test_invalid_initial_download_url_fails_before_network(url, monkeypatch):
    monkeypatch.setattr(dfi, "build_opener", lambda *_args: pytest.fail("network opener created"))
    with pytest.raises(dfi.PaidfInputError, match="HTTPS"):
        dfi._open_starter_url(url, "synthetic-private-token")
