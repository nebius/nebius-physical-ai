"""Authenticated HTTPS and verified artifact transport for Nebius Stream."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx


class StreamError(RuntimeError):
    """A hosted request or its delivery evidence failed validation."""


def read_api_key(path: str, environ: Mapping[str, str]) -> str:
    """Read a private key file or NPA's explicitly supplied secret binding.

    Args:
        path: Optional regular mode-0600 credential file.
        environ: Worker environment carrying a file path or NPA secret binding.
    Returns:
        The bearer secret, which must never enter reports or request identity.
    Raises:
        StreamError: The credential is missing, unsafe, or malformed.
    """
    location = path or environ.get("WMH_API_KEY_FILE", "")
    if location:
        try:
            fd = os.open(location, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                    raise StreamError("Stream key file must be a private regular file")
                if info.st_uid != os.geteuid() or info.st_size > 512:
                    raise StreamError("Stream key file has invalid ownership or size")
                key = stream.read().decode("utf-8").strip()
                key = key.removeprefix("WMH_API_KEY=")
        except (OSError, UnicodeError):
            raise StreamError("Cannot read the private Stream key file") from None
    else:
        key = environ.get("WMH_API_KEY", "").strip()
    if not key or len(key) > 512 or any(character.isspace() for character in key):
        raise StreamError("A private Stream API key or NPA secret binding is required")
    return key


def validate_origin(value: str) -> str:
    """Require one HTTPS origin without embedded credentials or URL parameters.

    Args:
        value: Operator-supplied service origin.
    Returns:
        The origin without a trailing slash.
    Raises:
        StreamError: The supplied value is not an HTTPS origin.
    """
    if (
        not value
        or any(c.isspace() for c in value)
        or any(c in value for c in ("\\", "%"))
    ):
        raise StreamError("Stream endpoint must be an unambiguous HTTPS origin")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError:
        raise StreamError("Stream endpoint has an invalid host or port") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise StreamError("Stream endpoint must be one HTTPS origin")
    return value.rstrip("/")


def document_sha256(value: Mapping[str, Any]) -> str:
    """Hash a canonical request or receipt without accepting non-finite values.

    Args:
        value: JSON-compatible evidence or request.
    Returns:
        Lowercase SHA-256 digest.
    Raises:
        ValueError: The document contains invalid JSON numbers.
    """
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


class StreamClient:
    """Use one authenticated origin with TLS and redirects strictly enforced.

    Args:
        origin: Operator-supplied HTTPS API origin.
        key: Customer-scoped bearer credential.
        transport: Optional injected transport for hermetic tests.
    Returns:
        A reusable, thread-safe HTTP client.
    Raises:
        StreamError: The origin is invalid.
    """

    def __init__(self, origin: str, key: str, *, transport: Any = None):
        self.origin = validate_origin(origin)
        self._client = httpx.Client(
            base_url=self.origin,
            headers={"Authorization": "Bearer " + key},
            verify=True,
            trust_env=False,
            follow_redirects=False,
            timeout=60,
            transport=transport,
        )

    def close(self) -> None:
        """Close pooled transports.

        Args:
            None.
        Returns:
            None.
        Raises:
            None.
        """
        self._client.close()

    def request(self, method: str, route: str, *, body=None, identity="") -> dict:
        """Send JSON without automatically replaying an ambiguous submission.

        Args:
            method: GET or POST.
            route: Relative API path on this exact origin.
            body: Optional JSON body containing no authentication secret.
            identity: Durable idempotency key for this exact logical request.
        Returns:
            A JSON object from a successful response.
        Raises:
            StreamError: Transport, status, redirect, or JSON validation failed.
        """
        if not route.startswith("/v1/") or any(
            x in route for x in ("..", "?", "#", "\\")
        ):
            raise StreamError("Invalid Stream API route")
        headers = {"Idempotency-Key": identity} if identity else {}
        while True:
            try:
                response = self._client.request(
                    method, route, json=body, headers=headers
                )
            except httpx.HTTPError:
                raise StreamError(
                    "Stream transport failed; retain the request identity"
                ) from None
            if response.status_code != 429 or method != "GET":
                break
            # Reads are safe to retry. Keep the admitted job and honor backpressure.
            try:
                delay = float(response.headers.get("Retry-After", "5"))
            except ValueError:
                delay = 5.0
            time.sleep(delay if math.isfinite(delay) and delay >= 1 else 5.0)
        if not 200 <= response.status_code < 300:
            raise StreamError(f"Stream API returned HTTP {response.status_code}")
        try:
            value = response.json()
        except ValueError:
            raise StreamError("Stream returned invalid JSON") from None
        if not isinstance(value, dict):
            raise StreamError("Stream returned an unexpected JSON shape")
        return value

    def wait(self, job_id: str, *, observe=None) -> dict:
        """Poll a retained job without generating a new request identity.

        Args:
            job_id: Previously admitted job identifier.
            observe: Optional callback receiving each durable status response.
        Returns:
            The successfully published job.
        Raises:
            StreamError: The job failed, is ambiguous, or requires recovery.
        """
        _identifier(job_id)
        active = {
            "queued",
            "staging",
            "running",
            "publishing",
            "publication_queued",
            "submitted",
            "dispatching",
            "cancel_requested",
        }
        while True:
            job = self.request("GET", "/v1/jobs/" + job_id)
            if observe:
                observe(job)
            if job.get("status") in {"succeeded", "completed"}:
                return job
            if job.get("status") not in active:
                raise StreamError(
                    "Stream job needs inspection; retain its ID and artifacts"
                )
            time.sleep(5)

    def download(
        self, job_id: str, artifact: Mapping[str, Any], directory: Path
    ) -> Path:
        """Download one same-origin artifact and verify its complete byte hash.

        Args:
            job_id: Published job identifier.
            artifact: Server manifest entry with a basename and SHA-256.
            directory: Private local output directory.
        Returns:
            The checksum-verified regular file.
        Raises:
            StreamError: Identity, transport, status, or checksum failed.
        """
        _identifier(job_id)
        name, expected = _artifact_identity(artifact)
        target = directory / name
        digest = hashlib.sha256()
        try:
            with self._client.stream(
                "GET", f"/v1/jobs/{job_id}/artifacts/{name}"
            ) as response:
                if response.status_code != 200:
                    raise StreamError(
                        f"Stream artifact returned HTTP {response.status_code}"
                    )
                fd = os.open(
                    target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
                )
                with os.fdopen(fd, "wb") as output:
                    for chunk in response.iter_bytes():
                        digest.update(chunk)
                        output.write(chunk)
        except (httpx.HTTPError, OSError):
            raise StreamError(
                "Stream artifact download failed; resume the retained job"
            ) from None
        if digest.hexdigest() != expected:
            target.unlink()
            raise StreamError("Stream artifact checksum mismatch")
        return target


def _identifier(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise StreamError("Stream returned an invalid identifier")


def _artifact_identity(artifact: Mapping[str, Any]) -> tuple[str, str]:
    name, digest = artifact.get("name"), artifact.get("sha256")
    if (
        not isinstance(name, str)
        or ".." in name
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
    ):
        raise StreamError("Stream returned an unsafe artifact name")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise StreamError("Stream returned an invalid artifact checksum")
    return name, digest
