from __future__ import annotations

import logging

from botocore.exceptions import ClientError, NoCredentialsError
import pytest

from npa.clients.scoped_credentials import run_with_host_credential_fallback
from npa.errors import ScopedCredentialError


def _access_denied(message: str = "AccessDenied") -> ClientError:
    return ClientError(
        {"Error": {"Code": "AccessDenied", "Message": message}},
        "PutObject",
    )


def test_scoped_credentials_succeed_without_warning(caplog) -> None:
    calls: list[str] = []
    logger = logging.getLogger("npa.tests.credential_fallback")

    with caplog.at_level(logging.WARNING):
        result = run_with_host_credential_fallback(
            lambda: calls.append("scoped") or "scoped-ok",
            lambda: calls.append("host") or "host-ok",
            bucket="bucket",
            operation="test upload",
            allow_host_creds=False,
            logger=logger,
        )

    assert result == "scoped-ok"
    assert calls == ["scoped"]
    assert not caplog.records


def test_scoped_credentials_access_denied_without_flag_raises() -> None:
    with pytest.raises(ScopedCredentialError, match="bucket") as exc_info:
        run_with_host_credential_fallback(
            lambda: (_ for _ in ()).throw(_access_denied()),
            lambda: "host-ok",
            bucket="bucket",
            operation="test upload",
            allow_host_creds=False,
        )
    assert exc_info.value.source_project is None
    assert exc_info.value.target_project is None
    assert exc_info.value.failed_project is None


def test_scoped_credentials_access_denied_with_flag_warns_and_falls_back(
    caplog,
) -> None:
    calls: list[str] = []
    logger = logging.getLogger("npa.tests.credential_fallback")

    with caplog.at_level(logging.WARNING):
        result = run_with_host_credential_fallback(
            lambda: (_ for _ in ()).throw(_access_denied("scoped denied")),
            lambda: calls.append("host") or "host-ok",
            bucket="bucket",
            operation="test upload",
            allow_host_creds=True,
            logger=logger,
            on_fallback=lambda exc: calls.append(str(exc)),
        )

    assert result == "host-ok"
    assert calls[0].startswith("An error occurred (AccessDenied)")
    assert calls[-1] == "host"
    assert "bucket" in caplog.text
    assert "test upload" in caplog.text


def test_scoped_credentials_missing_without_flag_raises() -> None:
    with pytest.raises(ScopedCredentialError, match="bucket"):
        run_with_host_credential_fallback(
            lambda: (_ for _ in ()).throw(NoCredentialsError()),
            lambda: "host-ok",
            bucket="bucket",
            operation="test upload",
            allow_host_creds=False,
        )


def test_scoped_credentials_generic_exception_propagates() -> None:
    with pytest.raises(RuntimeError, match="network exploded"):
        run_with_host_credential_fallback(
            lambda: (_ for _ in ()).throw(RuntimeError("network exploded")),
            lambda: "host-ok",
            bucket="bucket",
            operation="test upload",
            allow_host_creds=True,
        )
