"""Offline CPU regression evidence for the AnyIO 4.14.2 security fixes.

Covers GHSA-82r6-8w77-94w6 (TLS IDNA 2003/2008 hostname certificate confusion)
and GHSA-5p39-cfhj-2xmp (undrained worker-process stderr blocks ``run_sync``).
Upstream: https://github.com/agronholm/anyio/releases/tag/4.14.2,
https://github.com/advisories/GHSA-82r6-8w77-94w6,
https://github.com/advisories/GHSA-5p39-cfhj-2xmp. All network activity here
is loopback-only (127.0.0.1); no external host is contacted.
"""

from __future__ import annotations

import datetime
import os
import signal
import socket
import ssl
import subprocess
import sys
import threading
from importlib import metadata

import anyio
import pytest
from anyio.streams.tls import TLSStream
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from packaging.version import Version

# "faß.de" encodes differently under the two IDNA standards: IDNA2003 (RFC
# 3490, what the stdlib "idna" codec and pre-4.14.2 AnyIO used) maps ß -> ss,
# while IDNA2008/UTS46 (what idna.encode(uts46=True) and patched AnyIO use)
# keeps it as its own code point.
_UNICODE_HOSTNAME = "faß.de"
_IDNA2008_NAME = "xn--fa-hia.de"
_IDNA2003_NAME = "fass.de"
_HANDSHAKE_TIMEOUT_S = 10
_SOCKET_WATCHDOG_S = 10


def _generate_ca() -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Generate an ephemeral self-signed CA key and certificate."""
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "anyio-test-ca")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(minutes=10))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    return ca_key, ca_cert


def _issue_leaf_certificate(
    ca_key, ca_cert, dns_name: str
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Issue a leaf certificate for ``dns_name`` signed by the ephemeral CA."""
    now = datetime.datetime.now(datetime.timezone.utc)
    leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, dns_name)]))
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(minutes=10))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(dns_name)]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    return leaf_key, leaf_cert


def _write_leaf_pem_files(tmp_path, leaf_key, leaf_cert) -> tuple[str, str]:
    """Write the leaf cert/key to PEM files; the key file is chmod 0600."""
    leaf_pem = tmp_path / "leaf.pem"
    key_pem = tmp_path / "leaf_key.pem"
    leaf_pem.write_bytes(leaf_cert.public_bytes(serialization.Encoding.PEM))
    key_pem.write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    os.chmod(key_pem, 0o600)
    return str(leaf_pem), str(key_pem)


def _start_loopback_tls_server(
    leaf_pem: str, key_pem: str
) -> tuple[socket.socket, threading.Thread, int]:
    """Serve one TLS connection on 127.0.0.1 presenting the given leaf cert.

    Both the accept() and the handshake carry a socket-level watchdog timeout
    so a stuck client cannot leave the serving thread running forever; the
    caller must join() the returned thread to confirm it actually exited.
    """
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.load_cert_chain(leaf_pem, key_pem)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(_SOCKET_WATCHDOG_S)
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve_once() -> None:
        try:
            conn, _ = listener.accept()
        except OSError:
            return
        conn.settimeout(_SOCKET_WATCHDOG_S)
        try:
            server_ctx.wrap_socket(conn, server_side=True).close()
        except OSError:
            pass
        finally:
            conn.close()

    thread = threading.Thread(target=serve_once, daemon=True)
    thread.start()
    return listener, thread, port


async def _tls_handshake(port: int, ca_path) -> None:
    """Complete a client TLS handshake to 127.0.0.1:port for the Unicode hostname."""
    client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    client_ctx.load_verify_locations(cafile=str(ca_path))
    with anyio.fail_after(_HANDSHAKE_TIMEOUT_S):
        sock = await anyio.connect_tcp("127.0.0.1", port)
        try:
            await TLSStream.wrap(
                sock, hostname=_UNICODE_HOSTNAME, ssl_context=client_ctx
            )
        finally:
            await sock.aclose()


def _run_tls_case(tmp_path, dns_name: str) -> None:
    ca_key, ca_cert = _generate_ca()
    leaf_key, leaf_cert = _issue_leaf_certificate(ca_key, ca_cert, dns_name)
    ca_path = tmp_path / "ca.pem"
    ca_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    leaf_pem, key_pem = _write_leaf_pem_files(tmp_path, leaf_key, leaf_cert)
    listener, thread, port = _start_loopback_tls_server(leaf_pem, key_pem)
    try:
        anyio.run(_tls_handshake, port, ca_path)
    finally:
        listener.close()
        thread.join(timeout=_SOCKET_WATCHDOG_S + 5)
        assert not thread.is_alive(), "TLS server thread did not exit"


def test_tls_rejects_idna2003_spoofed_certificate_for_unicode_hostname(tmp_path):
    """GHSA-82r6-8w77-94w6: a cert obtained for the IDNA2003 spelling of a
    Unicode hostname must not validate against that hostname."""
    with pytest.raises(ssl.SSLCertVerificationError, match="Hostname mismatch"):
        _run_tls_case(tmp_path, _IDNA2003_NAME)


def test_tls_accepts_idna2008_certificate_for_unicode_hostname(tmp_path):
    """GHSA-82r6-8w77-94w6: the correctly IDNA2008/UTS46-encoded certificate
    for the same Unicode hostname must validate."""
    _run_tls_case(tmp_path, _IDNA2008_NAME)  # must not raise


_TO_PROCESS_DRIVER = """\
import sys
import anyio
import anyio.to_process


def _write_large_stderr_payload():
    sys.stderr.write("x" * 2_000_000)
    sys.stderr.flush()
    return "worker-done"


async def _main():
    result = await anyio.to_process.run_sync(_write_large_stderr_payload)
    sys.stdout.write(result)
    sys.stdout.flush()


if __name__ == "__main__":
    anyio.run(_main)
"""


def _reap_process_group(proc: subprocess.Popen) -> bytes:
    """Kill proc's process group if still running, then reap it and close its
    pipes. Never reads stderr — draining it would defeat the regression this
    test is checking for."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)
    try:
        return proc.stdout.read()
    finally:
        proc.stdout.close()
        proc.stderr.close()


def test_to_process_worker_stderr_never_blocks_run_sync(tmp_path):
    """GHSA-5p39-cfhj-2xmp: a worker writing 2MB to stderr must not deadlock
    ``run_sync`` even when the caller never drains the worker's stderr pipe.

    The driver's own stderr is piped and intentionally left unread, which is
    the real trigger: pre-fix, the worker inherits that pipe and a large
    write blocks forever. The pytest timeout below is a correctness watchdog
    for that deadlock, not a cost limit; ``_reap_process_group`` always kills
    the driver's process group afterward so no process is left running,
    whether it exited on its own or timed out.
    """
    driver_path = tmp_path / "driver.py"
    driver_path.write_text(_TO_PROCESS_DRIVER)
    proc = subprocess.Popen(
        [sys.executable, str(driver_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        output = _reap_process_group(proc)
    if timed_out:
        pytest.fail("run_sync worker stderr write blocked past the watchdog timeout")
    assert proc.returncode == 0
    assert output == b"worker-done"


def test_anyio_version_meets_security_floor():
    """Evidence only: confirms the installed AnyIO satisfies the >=4.14.2
    floor pinned in npa/pyproject.toml. The TLS and to_process tests above
    prove the fix behavior directly; this does not by itself."""
    assert Version(metadata.version("anyio")) >= Version("4.14.2")
