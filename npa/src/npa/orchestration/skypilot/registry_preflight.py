"""Prove a workflow's images can actually be pulled before any GPU time is spent.

A present registry token is not evidence that a pull will succeed. Nebius
Container Registry speaks the standard Docker Registry v2 auth flow: an
unauthenticated request returns ``401`` with a ``WWW-Authenticate: Bearer`` realm,
the client exchanges its credentials there for a scoped token, and only the final
manifest request enforces the permission. The token endpoint hands out tokens
optimistically, and listing tags is a different scope from pulling, so an operator
can read ``/v2/<repo>/tags/list`` and still watch every worker pod fail with
``403 Forbidden``.

Kubernetes then retries image pulls forever, so the job sits in
``PENDING``/``ImagePullBackOff`` instead of failing. This module reproduces the
pull the worker will attempt, with the same credentials the run injects, and
turns the answer into a fail-fast result.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import re
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_TIMEOUT_SECONDS = 30
MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.oci.image.index.v1+json",
    )
)


class RegistryPreflightError(RuntimeError):
    """Raised when an image reference cannot be understood."""


@dataclass(frozen=True)
class ImageReference:
    """A parsed ``registry/repository:tag`` (or ``@digest``) reference."""

    registry: str
    repository: str
    reference: str
    raw: str

    @property
    def manifest_url(self) -> str:
        return f"https://{self.registry}/v2/{self.repository}/manifests/{self.reference}"

    @property
    def pull_scope(self) -> str:
        return f"repository:{self.repository}:pull"


@dataclass(frozen=True)
class ImagePullCheck:
    """The outcome of reproducing one image pull."""

    image: str
    status: str
    http_status: int | None = None
    detail: str = ""
    remedy: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def render(self) -> str:
        line = f"{self.image}: {self.status}"
        if self.http_status is not None:
            line = f"{line} (HTTP {self.http_status})"
        if self.detail:
            line = f"{line} - {self.detail}"
        if self.remedy:
            line = f"{line}\n  Suggested action: {self.remedy}"
        return line


def parse_image_reference(image: str) -> ImageReference:
    """Parse a fully qualified image reference.

    Only registry-qualified references are supported: an unqualified name would
    resolve to Docker Hub, which is not a pull NPA workflows make.
    """

    raw = str(image or "").strip()
    value = raw.removeprefix("docker:").strip()
    if not value:
        raise RegistryPreflightError("image reference is empty")
    if "/" not in value:
        raise RegistryPreflightError(
            f"image {raw!r} has no registry host; expected <registry>/<repository>:<tag>"
        )
    host, remainder = value.split("/", 1)
    if "." not in host and ":" not in host and host != "localhost":
        raise RegistryPreflightError(
            f"image {raw!r} has no registry host; expected <registry>/<repository>:<tag>"
        )
    if "@" in remainder:
        repository, reference = remainder.split("@", 1)
    elif ":" in remainder.rsplit("/", 1)[-1]:
        repository, reference = remainder.rsplit(":", 1)
    else:
        repository, reference = remainder, "latest"
    repository = repository.strip("/")
    if not repository or not reference:
        raise RegistryPreflightError(f"image {raw!r} is missing a repository or tag")
    return ImageReference(
        registry=host, repository=repository, reference=reference, raw=raw
    )


HttpResponse = tuple[int, dict[str, str], bytes]
Fetcher = Callable[[str, dict[str, str], int], HttpResponse]


def _fetch(url: str, headers: dict[str, str], timeout: int) -> HttpResponse:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (
                int(response.status),
                {key.lower(): value for key, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as exc:
        return (
            int(exc.code),
            {key.lower(): value for key, value in (exc.headers or {}).items()},
            exc.read() if hasattr(exc, "read") else b"",
        )


def _parse_www_authenticate(header: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in re.findall(r'(\w+)="([^"]*)"', header or ""):
        fields[key.lower()] = value
    return fields


def _basic_auth(username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


def _registry_error_detail(body: bytes) -> str:
    try:
        payload: Any = json.loads(body.decode("utf-8", errors="replace") or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    errors = payload.get("errors") if isinstance(payload, dict) else None
    if not isinstance(errors, list) or not errors:
        return ""
    first = errors[0]
    if not isinstance(first, dict):
        return ""
    code = str(first.get("code") or "").strip()
    message = str(first.get("message") or "").strip()
    return f"{code}: {message}".strip(": ")


def check_image_pull(
    image: str,
    *,
    username: str = "",
    password: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    fetcher: Fetcher | None = None,
) -> ImagePullCheck:
    """Reproduce the manifest fetch a worker performs when pulling ``image``."""

    fetch = fetcher or _fetch
    try:
        reference = parse_image_reference(image)
    except RegistryPreflightError as exc:
        return ImagePullCheck(
            image=str(image),
            status="invalid",
            detail=str(exc),
            remedy="fix the image reference in the workflow spec or --image override",
        )

    headers = {"Accept": MANIFEST_ACCEPT}
    try:
        status, response_headers, body = fetch(reference.manifest_url, headers, timeout)
    except OSError as exc:
        return ImagePullCheck(
            image=reference.raw,
            status="unreachable",
            detail=str(exc),
            remedy=f"check network access to https://{reference.registry}/v2/ from this host",
        )

    if status == 401:
        challenge = _parse_www_authenticate(response_headers.get("www-authenticate", ""))
        realm = challenge.get("realm", "")
        if not realm:
            return ImagePullCheck(
                image=reference.raw,
                status="unauthorized",
                http_status=status,
                detail="registry requires authentication but sent no Bearer realm",
                remedy="verify the registry host is a Docker Registry v2 endpoint",
            )
        if not password:
            return ImagePullCheck(
                image=reference.raw,
                status="no_credentials",
                http_status=status,
                detail="registry requires authentication and no credentials were supplied",
                remedy=(
                    "export SKYPILOT_DOCKER_PASSWORD (or make `nebius iam get-access-token` "
                    "work) so submit can mint a pull token"
                ),
            )
        query = {"service": challenge.get("service", reference.registry), "scope": reference.pull_scope}
        token_url = f"{realm}?{urllib.parse.urlencode(query)}"
        try:
            token_status, _, token_body = fetch(
                token_url,
                {"Authorization": _basic_auth(username or "iam", password)},
                timeout,
            )
        except OSError as exc:
            return ImagePullCheck(
                image=reference.raw,
                status="unreachable",
                detail=f"token endpoint unreachable: {exc}",
                remedy=f"check network access to {realm} from this host",
            )
        if token_status >= 400:
            return ImagePullCheck(
                image=reference.raw,
                status="unauthorized",
                http_status=token_status,
                detail=_registry_error_detail(token_body)
                or "registry rejected the supplied credentials",
                remedy=(
                    "the credentials this run injects are not valid for "
                    f"{reference.registry}; re-mint them (`nebius iam get-access-token`) "
                    "and confirm the active profile is the one that owns the registry"
                ),
            )
        try:
            payload = json.loads(token_body.decode("utf-8", errors="replace") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        bearer = str(payload.get("token") or payload.get("access_token") or "")
        if not bearer:
            return ImagePullCheck(
                image=reference.raw,
                status="unauthorized",
                http_status=token_status,
                detail="registry token endpoint returned no token",
                remedy="re-mint registry credentials and retry",
            )
        headers = {"Accept": MANIFEST_ACCEPT, "Authorization": f"Bearer {bearer}"}
        try:
            status, response_headers, body = fetch(
                reference.manifest_url, headers, timeout
            )
        except OSError as exc:
            return ImagePullCheck(
                image=reference.raw,
                status="unreachable",
                detail=str(exc),
                remedy=f"check network access to https://{reference.registry}/v2/ from this host",
            )

    detail = _registry_error_detail(body)
    if 200 <= status < 300:
        return ImagePullCheck(image=reference.raw, status="ok", http_status=status)
    if status == 403:
        return ImagePullCheck(
            image=reference.raw,
            status="forbidden",
            http_status=status,
            detail=detail or "registry authenticated the credentials but refused the pull",
            remedy=(
                f"grant this run's identity pull access to {reference.repository} in "
                f"{reference.registry}. Being able to list tags is a different permission "
                "from pulling, so a readable tag list does not rule this out. Every worker "
                "pod will sit in ImagePullBackOff until this is fixed."
            ),
        )
    if status == 401:
        return ImagePullCheck(
            image=reference.raw,
            status="unauthorized",
            http_status=status,
            detail=detail or "registry rejected the pull token",
            remedy="re-mint registry credentials and confirm the active Nebius profile",
        )
    if status == 404:
        return ImagePullCheck(
            image=reference.raw,
            status="not_found",
            http_status=status,
            detail=detail or "manifest not found",
            remedy=_missing_image_remedy(reference),
        )
    return ImagePullCheck(
        image=reference.raw,
        status="error",
        http_status=status,
        detail=detail or f"unexpected registry response {status}",
        remedy=f"inspect https://{reference.registry}/v2/{reference.repository}/manifests/{reference.reference}",
    )


def _missing_image_remedy(reference: ImageReference) -> str:
    """Explain a missing tag, with the build command when it is a workbench image.

    A fresh project's registry has none of these images, and the deploy guide's
    tags can drift from the ones the code pins -- so name the command that builds
    the exact tag this run asked for.
    """

    base = (
        f"the tag {reference.reference!r} does not exist in "
        f"{reference.registry}/{reference.repository}"
    )
    try:
        from npa.deploy.images import build_and_push_command

        command = build_and_push_command(reference.raw)
    except Exception:  # noqa: BLE001 - the remedy must never be the thing that fails
        command = ""
    copy_hint = _server_side_copy_hint(reference)
    if not command:
        return f"{base}; build and push it, or pin a tag that exists.{copy_hint}"
    return (
        f"{base}. This is an NPA workbench image. Authenticate with "
        f"`printf '%s' \"$(nebius iam get-access-token)\" | docker login "
        f"{reference.registry} -u iam --password-stdin`, then either copy it "
        f"server-side (preferred) or build it:{copy_hint}\n    {command}"
    )


def _server_side_copy_hint(reference: ImageReference) -> str:
    """Suggest a registry-to-registry copy before a local rebuild.

    These images run to tens of GB. Building or `docker pull`+`push` moves every
    layer through the local machine, where a long transfer gets killed; `crane
    copy` moves them registry-to-registry and never materializes them locally. If
    the tag exists anywhere already, copying is both faster and far more likely to
    finish.
    """

    try:
        from npa.deploy.images import backup_container_registry, primary_container_registry
    except Exception:  # noqa: BLE001 - the hint must never be what fails
        return ""
    target = f"{reference.registry}/{reference.repository}:{reference.reference}"
    sources = []
    for candidate in (primary_container_registry(), backup_container_registry()):
        source_registry = str(candidate or "").rstrip("/")
        if not source_registry:
            continue
        repository = reference.repository.rsplit("/", 1)[-1]
        source = f"{source_registry}/{repository}:{reference.reference}"
        if source != target and source not in sources:
            sources.append(source)
    if not sources:
        return ""
    lines = "".join(f"\n    crane copy {source} {target}" for source in sources)
    return (
        "\n  If the tag already exists in another registry, copy it server-side "
        "instead of moving tens of GB through this machine:" + lines
    )


def resolve_registry_credentials(*, mint: bool = True) -> tuple[str, str]:
    """Return the (username, password) a submit injects for Nebius registry pulls.

    Preflight is only meaningful if it uses the very credentials the run will use,
    so the render path and the preflight path both resolve them here.
    """

    import os

    username = (
        os.environ.get("SKYPILOT_DOCKER_USERNAME")
        or os.environ.get("NPA_REGISTRY_USERNAME")
        or "iam"
    )
    password = (
        os.environ.get("SKYPILOT_DOCKER_PASSWORD")
        or os.environ.get("NPA_REGISTRY_PASSWORD")
        or ""
    )
    if not password and mint:
        from npa.workflows.sim2real.registry_auth import mint_nebius_registry_token

        password = mint_nebius_registry_token()
    return username, password


def check_image_pulls(
    images: list[str],
    *,
    username: str = "",
    password: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    fetcher: Fetcher | None = None,
) -> list[ImagePullCheck]:
    """Check each distinct image once, preserving first-seen order."""

    seen: list[str] = []
    for image in images:
        value = str(image or "").strip()
        if value and value not in seen:
            seen.append(value)
    return [
        check_image_pull(
            image,
            username=username,
            password=password,
            timeout=timeout,
            fetcher=fetcher,
        )
        for image in seen
    ]
