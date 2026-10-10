"""Access personal object prefixes using explicit credentials and auditable operations."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .errors import AuthorizationError, BackendError, TeamError
from .models import StorageGrant


def workload_secrets(allocation):
    """Read explicitly granted model-provider tokens without inheriting server secrets.

    Args:
        allocation: Personal allocation with an optional private JSON credential file.
    Returns:
        Allowed model-provider environment variables and their nonempty values.
    Raises:
        BackendError: File permissions, keys, or values are invalid.
    """
    path = allocation.workload_secrets_file
    if path is None:
        return {}
    allowed = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NGC_API_KEY", "NVIDIA_API_KEY"}
    try:
        if path.stat().st_mode & 0o077:
            raise ValueError("workload credential file is not private")
        values = json.loads(path.read_text())
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError("unsupported workload credential")
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise ValueError("invalid workload credential value")
        return values
    except (OSError, ValueError, TypeError) as exc:
        raise BackendError(
            "allocated model-provider credentials are unavailable"
        ) from exc


def storage_credentials(grant: StorageGrant) -> dict[str, str]:
    """Load the selected storage principal without ambient credential fallback.

    Args:
        grant: Administrator-owned storage allocation.
    Returns:
        Explicit S3 access key and secret, with an optional session token.
    Raises:
        BackendError: The private credential file is invalid or missing.
    """
    try:
        if grant.credentials_file.stat().st_mode & 0o077:
            raise ValueError("credential file is not private")
        values = json.loads(grant.credentials_file.read_text())
        if not isinstance(values, dict):
            raise ValueError("credential file must contain an object")
        required = ("aws_access_key_id", "aws_secret_access_key")
        if any(not isinstance(values.get(k), str) or not values[k] for k in required):
            raise ValueError("credential pair is incomplete")
        if values.get("aws_session_token") and not isinstance(
            values["aws_session_token"], str
        ):
            raise ValueError("invalid session token")
        return {k: values[k] for k in (*required, "aws_session_token") if values.get(k)}
    except (OSError, ValueError, TypeError) as exc:
        raise BackendError("the allocated storage credential is unavailable") from exc


def object_key(prefix: str, relative: str) -> str:
    """Resolve an object name strictly beneath an authorized prefix.

    Args:
        prefix: Administrator-selected object prefix.
        relative: Untrusted relative object name.
    Returns:
        A scoped object key.
    Raises:
        TeamError: The path is absolute, ambiguous, or traverses its boundary.
    """
    parts = relative.split("/")
    if not relative or any(p in ("", ".", "..") for p in parts):
        raise TeamError("artifact name must be a nonempty relative object path")
    if any(c in relative for c in ("\\", "%", "\x00", "?", "#")):
        raise TeamError("artifact name contains unsupported characters")
    return str(PurePosixPath(prefix) / relative)


def authorize_uri(uri: str, grant: StorageGrant, shared: tuple[str, ...] = ()) -> None:
    """Check a declared workflow URI against private output or shared input scope.

    Args:
        uri: Fully resolved S3 object URI.
        grant: Personal storage allocation.
        shared: Explicit read-only input prefixes.
    Returns:
        None.
    Raises:
        AuthorizationError: The URI is outside all approved scopes.
    """
    allowed = (f"s3://{grant.bucket}/{grant.prefix.strip('/')}/", *shared)
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or parsed.query or parsed.fragment:
        raise AuthorizationError("workflow artifact must use an authorized S3 location")
    if not any(uri.startswith(scope.rstrip("/") + "/") for scope in allowed):
        raise AuthorizationError("workflow artifact is outside its allocated storage")
    if any(p in (".", "..") for p in parsed.path.split("/")) or "%" in parsed.path:
        raise AuthorizationError("workflow artifact path is ambiguous")


class PersonalStorage:
    """Perform object operations with one explicit workload principal.

    Args:
        grant: Explicit personal storage scope and credential file.
        client: Optional object client for deterministic storage tests.
    Returns:
        A PersonalStorage instance.
    Raises:
        TeamError: The private credential file is invalid.
    """

    def __init__(self, grant: StorageGrant, *, client=None):
        """Bind the object client to a personal allocation.

        Args:
            grant: Explicit bucket, prefix, endpoint and credential reference.
            client: Injectable S3 client for offline acceptance tests.
        Returns:
            None.
        Raises:
            BackendError: Credentials cannot be loaded.
        """
        import boto3

        self.grant = grant
        self.client = client or boto3.client(
            "s3", endpoint_url=grant.endpoint, **storage_credentials(grant)
        )

    def list(self, prefix: str) -> list[dict]:
        """List objects within a previously authorized run prefix.

        Args:
            prefix: Run prefix beneath the personal allocation.
        Returns:
            Relative object names and sizes, without credentials or signed URLs.
        Raises:
            AuthorizationError: Prefix escapes the allocation.
        """
        self._check(prefix)
        result = []
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.grant.bucket, Prefix=prefix.rstrip("/") + "/"
        )
        for page in pages:
            for item in page.get("Contents", []):
                result.append(
                    {
                        "name": item["Key"][len(prefix.rstrip("/")) + 1 :],
                        "size": item["Size"],
                    }
                )
        return result

    def read(self, prefix: str, relative: str):
        """Open one authorized object as a streaming body.

        Args:
            prefix, relative: Authorized run prefix and relative object name.
        Returns:
            The S3 streaming response body.
        Raises:
            TeamError: The object path is invalid or outside the allocation.
        """
        key = object_key(prefix, relative)
        self._check(key)
        return self.client.get_object(Bucket=self.grant.bucket, Key=key)["Body"]

    def _check(self, key):
        authorize_uri(f"s3://{self.grant.bucket}/{key}", self.grant)


class PrivateRunFiles:
    """Keep authoritative workflow runtime records outside writable workload storage.

    Args:
        root, prefix: Private runtime directory and corresponding S3 prefix.
    Returns:
        A PrivateRunFiles instance.
    Raises:
        OSError: Private runtime storage cannot be created.
    """

    def __init__(self, root: Path, prefix: str):
        """Bind a run's authoritative records to a server-owned directory.

        Args:
            root: Private per-run server directory.
            prefix: Workflow storage prefix used by RunStateStore.
        Returns:
            None.
        Raises:
            OSError: Private storage cannot be created.
        """
        self.root = root
        self.prefix = prefix.rstrip("/") + "/"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def read(self, bucket: str, key: str) -> bytes:
        """Read an authoritative workflow record.

        Args:
            bucket: Interface-compatible bucket name, never used for placement.
            key: RunStateStore record key.
        Returns:
            Stored bytes.
        Raises:
            FileNotFoundError: Record has not been written.
            TeamError: Key escapes the run boundary.
        """
        return self._path(key).read_bytes()

    def write(self, bucket: str, key: str, body: bytes) -> None:
        """Atomically persist a server-owned record.

        Args:
            bucket, key, body: RunStateStore-compatible record arguments.
        Returns:
            None.
        Raises:
            OSError: Durable storage is unavailable.
            TeamError: Key escapes the run boundary.
        """
        import os
        import tempfile

        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _path(self, key):
        if not key.startswith(self.prefix):
            raise TeamError("runtime record is outside its run")
        relative = key[len(self.prefix) :]
        object_key("records", relative)
        return self.root / relative
