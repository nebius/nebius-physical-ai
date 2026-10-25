"""Reject invalid operator TLS material before creating public endpoint resources."""

import base64
from datetime import datetime, timezone
from ipaddress import ip_address
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    Encoding,
    PublicFormat,
)

from .errors import BackendError


def verify_certificate(request, secret):
    """Check certificate validity, hostname and key without exposing secret bytes.

    Args:
        request: Operator's selected HTTPS origin.
        secret: Private TLS Secret returned by the selected Kubernetes API.
    Returns:
        None after validating the TLS material.
    Raises:
        BackendError: The certificate, key, validity period or hostname is invalid.
    """
    try:
        certificate = x509.load_pem_x509_certificate(
            base64.b64decode(secret["data"]["tls.crt"], validate=True)
        )
        key = load_pem_private_key(
            base64.b64decode(secret["data"]["tls.key"], validate=True), password=None
        )
        now = datetime.now(timezone.utc)
        if (
            _public_bytes(certificate.public_key()) != _public_bytes(key.public_key())
            or not certificate.not_valid_before_utc
            <= now
            < certificate.not_valid_after_utc
        ):
            raise ValueError("certificate or key is invalid")
        names = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        if not _matches(urlsplit(request.endpoint).hostname, names):
            raise ValueError("certificate hostname is invalid")
    except (ValueError, KeyError, TypeError, x509.ExtensionNotFound):
        raise BackendError(
            "operator TLS certificate or private key is invalid for this endpoint"
        ) from None


def _public_bytes(key):
    return key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)


def _matches(host, names):
    try:
        address = ip_address(host)
    except ValueError:
        host = host.lower().rstrip(".")
        return any(
            _dns_match(host, name.lower().rstrip("."))
            for name in names.get_values_for_type(x509.DNSName)
        )
    return address in names.get_values_for_type(x509.IPAddress)


def _dns_match(host, name):
    if "*" not in name:
        return host == name
    return (
        name.startswith("*.")
        and name.count("*") == 1
        and host.count(".") == name.count(".")
        and host.endswith(name[1:])
    )
