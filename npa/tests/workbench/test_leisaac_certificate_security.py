"""Anchor LeIsaac's TLS pin in authenticated SSH before using agent credentials."""

import shlex
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from npa.cli.workbench.leisaac import _agent_certificate_sha256
from npa.clients.ssh import SSHHostKeyError
from npa.workbench.leisaac import LeIsaacConfigError


@pytest.fixture(autouse=True)
def forbid_public_certificate_discovery(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("The public network cannot establish the trusted TLS pin")

    monkeypatch.setattr("npa.cli.workbench.leisaac.socket.create_connection", forbidden)


@pytest.mark.parametrize("algorithm", ["SHA256", "sha256"])
def test_certificate_pin_comes_from_selected_agents_trusted_ssh(algorithm):
    fingerprint = ":".join(["AB"] * 32)
    ssh = SimpleNamespace(
        run=Mock(return_value=(0, f"{algorithm} Fingerprint={fingerprint}\n", ""))
    )

    assert _agent_certificate_sha256(ssh) == "ab" * 32

    ssh.run.assert_called_once_with(
        "openssl x509 -in /etc/nginx/ssl/npa-agent.crt -noout -fingerprint -sha256"
    )


@pytest.mark.parametrize(
    "output",
    [
        "",
        "ab" * 32,
        "SHA256 Fingerprint=ab:cd",
        "MD5 Fingerprint=" + ":".join(["AB"] * 32),
        "SHA256 Fingerprint=" + ":".join(["AB"] * 32) + "\nunexpected output",
    ],
)
def test_malformed_ssh_fingerprint_is_rejected(output):
    ssh = SimpleNamespace(run=Mock(return_value=(0, output, "")))

    with pytest.raises(LeIsaacConfigError, match="fingerprint"):
        _agent_certificate_sha256(ssh)


def test_certificate_command_failure_never_falls_back_to_public_network():
    ssh = SimpleNamespace(run=Mock(return_value=(1, "", "private diagnostics")))

    with pytest.raises(RuntimeError, match="trusted SSH") as error:
        _agent_certificate_sha256(ssh)

    assert "private diagnostics" not in str(error.value)


def test_unknown_ssh_host_key_never_falls_back_to_public_network():
    ssh = SimpleNamespace(run=Mock(side_effect=SSHHostKeyError("untrusted host key")))

    with pytest.raises(SSHHostKeyError):
        _agent_certificate_sha256(ssh)


def _certificate():
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "agent.example.test")])
    now = datetime.now(timezone.utc)
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )


def test_real_openssl_pin_matches_certificate_der_sha256(tmp_path):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("Requires the agent's OpenSSL command")
    certificate = _certificate()
    path = tmp_path / "agent.crt"
    path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))

    def run(command):
        arguments = shlex.split(command)
        assert arguments == [
            "openssl",
            "x509",
            "-in",
            "/etc/nginx/ssl/npa-agent.crt",
            "-noout",
            "-fingerprint",
            "-sha256",
        ]
        arguments[0], arguments[3] = openssl, str(path)
        result = subprocess.run(arguments, capture_output=True, text=True, check=False)
        return result.returncode, result.stdout, result.stderr

    assert (
        _agent_certificate_sha256(SimpleNamespace(run=run))
        == certificate.fingerprint(hashes.SHA256()).hex()
    )
