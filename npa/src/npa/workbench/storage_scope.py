"""Request-scoped URI authorization for workbench storage services."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import unquote, urlparse


class StorageAuthorizationError(ValueError):
    """Raised before a storage operation crosses its configured URI boundary."""


@dataclass(frozen=True)
class S3Root:
    bucket: str
    prefix: str


@dataclass(frozen=True)
class AuthorizedUri:
    kind: str
    original: str
    local_path: Path | None = None
    bucket: str = ""
    key: str = ""


@dataclass(frozen=True)
class StorageScope:
    """Canonical local sandboxes and S3 bucket/prefix roots allowed to a request."""

    s3_roots: tuple[S3Root, ...] = ()
    local_roots: tuple[Path, ...] = ()

    @classmethod
    def from_config(
        cls,
        *,
        s3_roots: Sequence[str] = (),
        local_roots: Sequence[str | Path] = (),
    ) -> "StorageScope":
        parsed_s3 = tuple(
            _parse_s3_root(value) for value in s3_roots if str(value).strip()
        )
        parsed_local = tuple(
            Path(value).expanduser().resolve(strict=False)
            for value in local_roots
            if str(value).strip()
        )
        return cls(s3_roots=parsed_s3, local_roots=parsed_local)

    @classmethod
    def from_env(cls, prefix: str) -> "StorageScope":
        s3_values = _split_config(os.environ.get(f"{prefix}_ALLOWED_S3_ROOTS", ""), ",")
        local_values = _split_config(
            os.environ.get(f"{prefix}_ALLOWED_LOCAL_ROOTS", ""), os.pathsep
        )
        return cls.from_config(s3_roots=s3_values, local_roots=local_values)

    def authorize(self, uri: str, *, operation: str) -> AuthorizedUri:
        target = _parse_uri(uri, operation=operation)
        if target.kind == "s3":
            return self._authorize_s3(target, operation=operation)
        return self._authorize_local(target, operation=operation)

    def _authorize_s3(
        self, target: AuthorizedUri, *, operation: str
    ) -> AuthorizedUri:
        bucket = target.bucket
        key = target.key
        if not any(
            bucket == root.bucket
            and (
                not root.prefix
                or key == root.prefix
                or key.startswith(root.prefix + "/")
            )
            for root in self.s3_roots
        ):
            raise StorageAuthorizationError(
                f"{operation} S3 URI is outside the configured bucket/prefix roots"
            )
        return target

    def _authorize_local(
        self, target: AuthorizedUri, *, operation: str
    ) -> AuthorizedUri:
        assert target.local_path is not None
        candidate = target.local_path
        if not any(
            candidate == root or candidate.is_relative_to(root)
            for root in self.local_roots
        ):
            raise StorageAuthorizationError(
                f"{operation} local path is outside the configured sandbox roots"
            )
        return target


def _parse_uri(uri: str, *, operation: str) -> AuthorizedUri:
    """Parse a URI without applying service-only containment policy."""

    value = str(uri or "").strip()
    if not value:
        raise StorageAuthorizationError(f"{operation} URI must not be empty")
    parsed = urlparse(value)
    if parsed.scheme == "s3":
        return _parse_s3_uri(value, parsed, operation=operation)
    if parsed.scheme not in {"", "file"}:
        raise StorageAuthorizationError(
            f"{operation} URI scheme {parsed.scheme!r} is not allowed; use s3://, file://, or a local path"
        )
    return _parse_local_uri(value, parsed, operation=operation)


def _parse_s3_uri(value: str, parsed, *, operation: str) -> AuthorizedUri:
    if (
        not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise StorageAuthorizationError(f"{operation} S3 URI is malformed")
    return AuthorizedUri(
        kind="s3",
        original=value,
        bucket=parsed.hostname or "",
        key=_canonical_s3_key(parsed.path),
    )


def _parse_local_uri(value: str, parsed, *, operation: str) -> AuthorizedUri:
    if parsed.scheme == "file":
        if parsed.netloc or parsed.query or parsed.fragment:
            raise StorageAuthorizationError(
                f"{operation} file URI hosts, queries, and fragments are not allowed"
            )
        raw_path = unquote(parsed.path)
    else:
        raw_path = value
    return AuthorizedUri(
        kind="local",
        original=value,
        local_path=Path(raw_path).expanduser().resolve(strict=False),
    )


_ACTIVE_SCOPE: ContextVar[StorageScope | None] = ContextVar(
    "npa_workbench_storage_scope", default=None
)


@contextmanager
def use_storage_scope(scope: StorageScope) -> Iterator[None]:
    """Apply one request/test scope without mutating process-global configuration."""
    token = _ACTIVE_SCOPE.set(scope)
    try:
        yield
    finally:
        _ACTIVE_SCOPE.reset(token)


def authorize_uri(uri: str, *, operation: str) -> AuthorizedUri:
    """Authorize inside an active service request; otherwise preserve embedded I/O."""

    scope = _ACTIVE_SCOPE.get()
    if scope is None:
        return _parse_uri(uri, operation=operation)
    return scope.authorize(uri, operation=operation)


def _parse_s3_root(value: str) -> S3Root:
    parsed = urlparse(str(value).strip())
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise StorageAuthorizationError(
            "allowed S3 roots must be canonical s3://bucket[/prefix] URIs"
        )
    return S3Root(bucket=parsed.hostname or "", prefix=_canonical_s3_key(parsed.path))


def _canonical_s3_key(path: str) -> str:
    decoded = unquote(path).lstrip("/")
    if "\\" in decoded:
        raise StorageAuthorizationError("S3 URI keys must not contain backslashes")
    segments = decoded.split("/") if decoded else []
    if any(segment in {"", ".", ".."} for segment in segments):
        raise StorageAuthorizationError(
            "S3 URI keys must be canonical and traversal-free"
        )
    return "/".join(segments)


def _split_config(value: str, separator: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(separator) if item.strip())
