"""Prove a workflow's images can actually be pulled before any GPU time is spent.

A present registry token is not evidence that a pull will succeed. OCI
registries use the standard Docker Registry v2 auth flow: an
unauthenticated request returns ``401`` with a ``WWW-Authenticate: Bearer`` realm,
the client exchanges its credentials there for a scoped token, and only the final
manifest request enforces the permission. The token endpoint hands out tokens
optimistically, and listing tags is a different scope from pulling, so an operator
can read ``/v2/<repo>/tags/list`` and still watch every worker pod fail with
``403 Forbidden``.

Kubernetes then retries image pulls forever, so the job sits in
``PENDING``/``ImagePullBackOff`` instead of failing. This module verifies each
execution path independently: host manifest evidence for VM paths and an owned,
exact-context, exact-namespace probe pod for every Kubernetes path.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import secrets
import subprocess
import time
from typing import Any, Callable, Collection, Mapping
import urllib.error
import urllib.parse
import urllib.request

import yaml

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
    def api_registry(self) -> str:
        """Return the Registry v2 API host for the declared image registry."""

        # ``docker.io`` is the canonical pull-name authority understood by
        # container runtimes, while Docker Hub serves Registry v2 traffic from
        # this distinct endpoint. Keep ``registry`` unchanged for credential
        # matching and Kubernetes pull-secret checks.
        return "registry-1.docker.io" if self.registry == "docker.io" else self.registry

    @property
    def manifest_url(self) -> str:
        return f"https://{self.api_registry}/v2/{self.repository}/manifests/{self.reference}"

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
    operator_status: str = ""
    target_status: str = "unverified"
    authority: str = "operator"
    digest: str = ""
    acceptable_digests: tuple[str, ...] = ()

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


@dataclass(frozen=True)
class KubernetesPullCheck:
    """Bounded evidence from an exact-target Kubernetes image-pull probe."""

    status: str
    digest: str = ""
    cleanup_status: str = "verified"

    @property
    def ok(self) -> bool:
        return self.status == "verified" and self.cleanup_status == "verified"


@dataclass(frozen=True)
class KubernetesPullTarget:
    """Effective SkyPilot namespace and config-level pull-secret set."""

    namespace: str
    pull_secret_names: tuple[str, ...] = ()
    pull_secret_names_configured: bool = False


def merge_skypilot_pull_secret_names(
    base_names: tuple[str, ...] | None,
    override_names: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Apply SkyPilot 0.12's legacy imagePullSecrets override semantics."""

    if override_names is None:
        return base_names or ()
    if base_names is None:
        return override_names
    if not base_names:
        raise RegistryPreflightError(
            "SkyPilot cannot merge imagePullSecrets into an existing empty list"
        )
    if len(override_names) != 1:
        raise RegistryPreflightError(
            "SkyPilot imagePullSecrets override must contain exactly one entry"
        )
    return (override_names[0], *base_names[1:])


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
        if ":" in repository.rsplit("/", 1)[-1]:
            repository = repository.rsplit(":", 1)[0]
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


def fetch_image_config_metadata(
    image: str,
    *,
    username: str = "",
    password: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    fetcher: Fetcher | None = None,
) -> tuple[str, dict[str, str]]:
    """Return immutable digest and OCI config labels for the selected amd64 image."""

    fetch = fetcher or _fetch
    reference = parse_image_reference(image)
    headers = {"Accept": MANIFEST_ACCEPT}

    def fetch_bounded(
        url: str, request_headers: dict[str, str], phase: str
    ) -> HttpResponse:
        try:
            return fetch(url, request_headers, timeout)
        except OSError as exc:
            raise RegistryPreflightError(
                f"{phase} unavailable ({type(exc).__name__})"
            ) from None

    status, response_headers, body = fetch_bounded(
        reference.manifest_url, headers, "registry manifest request"
    )
    if status == 401:
        challenge = _parse_www_authenticate(
            response_headers.get("www-authenticate", "")
        )
        realm = challenge.get("realm", "")
        if not realm:
            raise RegistryPreflightError(
                "registry authentication challenge has no realm"
            )
        parsed_realm = urllib.parse.urlsplit(realm)
        query = dict(urllib.parse.parse_qsl(parsed_realm.query, keep_blank_values=True))
        query.update(
            {
                "service": challenge.get("service", reference.registry),
                "scope": reference.pull_scope,
            }
        )
        token_url = urllib.parse.urlunsplit(
            parsed_realm._replace(query=urllib.parse.urlencode(query))
        )
        token_headers = (
            {"Authorization": _basic_auth(username or "iam", password)}
            if password
            else {}
        )
        token_status, _, token_body = fetch_bounded(
            token_url, token_headers, "registry token request"
        )
        if token_status >= 400:
            raise RegistryPreflightError(
                f"registry token request failed with HTTP {token_status}"
            )
        token_payload = json.loads(token_body.decode("utf-8", errors="replace") or "{}")
        bearer = str(
            token_payload.get("token") or token_payload.get("access_token") or ""
        )
        if not bearer:
            raise RegistryPreflightError("registry token response contains no token")
        headers = {"Accept": MANIFEST_ACCEPT, "Authorization": f"Bearer {bearer}"}
        status, response_headers, body = fetch_bounded(
            reference.manifest_url,
            headers,
            "authenticated registry manifest request",
        )
    if not 200 <= status < 300:
        raise RegistryPreflightError(f"manifest fetch failed with HTTP {status}")
    top_digest = str(response_headers.get("docker-content-digest") or "").strip()
    manifest = json.loads(body.decode("utf-8", errors="replace") or "{}")
    manifests = manifest.get("manifests") if isinstance(manifest, dict) else None
    if isinstance(manifests, list):
        selected = next(
            (
                item
                for item in manifests
                if isinstance(item, dict)
                and str((item.get("platform") or {}).get("os") or "") == "linux"
                and str((item.get("platform") or {}).get("architecture") or "")
                == "amd64"
            ),
            None,
        )
        if not selected:
            raise RegistryPreflightError("image index has no linux/amd64 manifest")
        selected_digest = str(selected.get("digest") or "")
        selected_url = f"https://{reference.api_registry}/v2/{reference.repository}/manifests/{selected_digest}"
        status, selected_headers, body = fetch_bounded(
            selected_url, headers, "platform manifest request"
        )
        if not 200 <= status < 300:
            raise RegistryPreflightError(
                f"platform manifest fetch failed with HTTP {status}"
            )
        manifest = json.loads(body.decode("utf-8", errors="replace") or "{}")
        # Pin the index digest when the original reference resolves to a
        # multi-platform index. Kubernetes then selects the platform manifest,
        # while the immutable reference still describes exactly what was
        # resolved during preflight.
        top_digest = top_digest or str(
            selected_headers.get("docker-content-digest") or selected_digest
        )
    config = manifest.get("config") if isinstance(manifest, dict) else None
    config_digest = str(config.get("digest") or "") if isinstance(config, dict) else ""
    if not config_digest:
        raise RegistryPreflightError("image manifest contains no config digest")
    config_url = f"https://{reference.api_registry}/v2/{reference.repository}/blobs/{config_digest}"
    status, _, config_body = fetch_bounded(config_url, headers, "image config request")
    if not 200 <= status < 300:
        raise RegistryPreflightError(f"image config fetch failed with HTTP {status}")
    config_payload = json.loads(config_body.decode("utf-8", errors="replace") or "{}")
    labels_raw = (config_payload.get("config") or {}).get("Labels") or {}
    labels = (
        {str(key): str(value) for key, value in labels_raw.items()}
        if isinstance(labels_raw, dict)
        else {}
    )
    if not top_digest:
        top_digest = (
            reference.reference if reference.reference.startswith("sha256:") else ""
        )
    return top_digest, labels


class _RegistryRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not leak registry bearer credentials to signed blob-storage URLs."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        source = urllib.parse.urlsplit(req.full_url)
        target = urllib.parse.urlsplit(newurl)
        if (source.scheme, source.netloc) != (target.scheme, target.netloc):
            redirected.remove_header("Authorization")
            redirected.remove_header("Proxy-Authorization")
        return redirected


def _fetch(url: str, headers: dict[str, str], timeout: int) -> HttpResponse:
    request = urllib.request.Request(url, headers=headers, method="GET")
    opener = urllib.request.build_opener(_RegistryRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
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
    return code if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) else ""


def _manifest_pull_digests(body: bytes, top_digest: str) -> tuple[str, ...]:
    """Return the index and platform-manifest digests a pull may report."""

    digests = []
    if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", top_digest):
        digests.append(top_digest.lower())
    try:
        payload: Any = json.loads(body.decode("utf-8", errors="replace") or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}
    manifests = payload.get("manifests") if isinstance(payload, Mapping) else None
    for item in manifests if isinstance(manifests, list) else []:
        digest = str(item.get("digest") or "") if isinstance(item, Mapping) else ""
        if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
            digests.append(digest.lower())
    return tuple(dict.fromkeys(digests))


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
    except RegistryPreflightError:
        return ImagePullCheck(
            image=str(image),
            status="invalid",
            detail="image reference is invalid",
            remedy="fix the image reference in the workflow spec or --image override",
        )

    headers = {"Accept": MANIFEST_ACCEPT}
    try:
        status, response_headers, body = fetch(reference.manifest_url, headers, timeout)
    except OSError as exc:
        return ImagePullCheck(
            image=reference.raw,
            status="unreachable",
            detail=f"registry manifest request unavailable ({type(exc).__name__})",
            remedy=f"check network access to https://{reference.registry}/v2/ from this host",
        )

    if status == 401:
        challenge = _parse_www_authenticate(
            response_headers.get("www-authenticate", "")
        )
        realm = challenge.get("realm", "")
        if not realm:
            return ImagePullCheck(
                image=reference.raw,
                status="unauthorized",
                http_status=status,
                detail="registry requires authentication but sent no Bearer realm",
                remedy="verify the registry host is a Docker Registry v2 endpoint",
            )
        parsed_realm = urllib.parse.urlsplit(realm)
        query = dict(urllib.parse.parse_qsl(parsed_realm.query, keep_blank_values=True))
        query.update(
            {
                "service": challenge.get("service", reference.registry),
                "scope": reference.pull_scope,
            }
        )
        token_url = urllib.parse.urlunsplit(
            parsed_realm._replace(query=urllib.parse.urlencode(query))
        )
        # A public registry (GHCR, Docker Hub) issues a pull token to an anonymous
        # caller, so "no credentials" is not the same as "cannot pull". Ask the
        # token endpoint before concluding anything.
        token_headers = (
            {"Authorization": _basic_auth(username or "iam", password)}
            if password
            else {}
        )
        try:
            token_status, _, token_body = fetch(token_url, token_headers, timeout)
        except OSError as exc:
            return ImagePullCheck(
                image=reference.raw,
                status="unreachable",
                detail=f"registry token request unavailable ({type(exc).__name__})",
                remedy=(
                    "check network access to the registry token endpoint for "
                    f"{reference.registry} from this host"
                ),
            )
        if token_status >= 400:
            if not password:
                return ImagePullCheck(
                    image=reference.raw,
                    status="no_credentials",
                    http_status=token_status,
                    detail="registry requires authentication and no credentials were supplied",
                    remedy=(
                        "export exact-host SKYPILOT_DOCKER_USERNAME and "
                        "SKYPILOT_DOCKER_PASSWORD credentials supplied by the "
                        "operator-controlled registry"
                    ),
                )
            return ImagePullCheck(
                image=reference.raw,
                status="unauthorized",
                http_status=token_status,
                detail=_registry_error_detail(token_body)
                or "registry rejected the supplied credentials",
                remedy=(
                    "the credentials this run injects are not valid for "
                    f"{reference.registry}; refresh them through that registry's "
                    "standard authentication flow and confirm the exact host scope"
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
                detail=(
                    "authenticated registry manifest request unavailable "
                    f"({type(exc).__name__})"
                ),
                remedy=f"check network access to https://{reference.registry}/v2/ from this host",
            )

    detail = _registry_error_detail(body)
    if 200 <= status < 300:
        digest = str(response_headers.get("docker-content-digest") or "").strip()
        digest = (
            reference.reference if reference.reference.startswith("sha256:") else digest
        )
        return ImagePullCheck(
            image=reference.raw,
            status="ok",
            http_status=status,
            digest=digest,
            acceptable_digests=_manifest_pull_digests(body, digest),
        )
    if status == 403:
        return ImagePullCheck(
            image=reference.raw,
            status="forbidden",
            http_status=status,
            detail=detail
            or "registry authenticated the credentials but refused the pull",
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
            remedy=(
                f"authenticate to {reference.registry} with that registry's normal "
                "credential and configure the exact-server SkyPilot/NPA Docker variables"
            ),
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
        f"{base}. This is an NPA workbench image. For the official public GHCR "
        "channel, select a published release tag. For an operator-controlled "
        f"registry, authenticate with `docker login {reference.registry}` (or that "
        f"registry's equivalent) and build it:{copy_hint}\n    {command}"
    )


def _server_side_copy_hint(reference: ImageReference) -> str:
    """Do not invent an official source for a missing image."""
    del reference
    return ""


def canonical_registry_host(value: str) -> str:
    """Return one credential authority for equivalent registry spellings."""

    cleaned = str(value or "").strip().removeprefix("docker:")
    cleaned = cleaned.removeprefix("https://").removeprefix("http://")
    host = cleaned.split("/", 1)[0].rstrip("/").casefold()
    if host in {"docker.io", "index.docker.io", "registry-1.docker.io"}:
        return "docker.io"
    return host


def resolve_registry_credentials(
    registry: str = "", *, image: str = "", mint: bool = True
) -> tuple[str, str]:
    """Return explicit credentials scoped to the selected registry host.

    NPA never mints cloud-provider registry tokens. Official public GHCR tags
    use anonymous pulls; operator-controlled registries must supply an exact-
    server username/password through the documented environment.
    """
    del mint

    if image:
        from npa.deploy.images import is_official_public_image

        # A stale credential can turn a valid anonymous GHCR pull into HTTP 403.
        # Official releases are deliberately public, so never attach operator or
        # legacy private-registry credentials to these exact package namespaces.
        if is_official_public_image(image):
            return "", ""

    import os

    target = canonical_registry_host(registry)
    configured_server = canonical_registry_host(
        os.environ.get("SKYPILOT_DOCKER_SERVER")
        or os.environ.get("NPA_REGISTRY_SERVER")
        or os.environ.get("NPA_REGISTRY")
        or ""
    )
    username = (
        os.environ.get("SKYPILOT_DOCKER_USERNAME")
        or os.environ.get("NPA_REGISTRY_USERNAME")
        or ""
    )
    password = (
        os.environ.get("SKYPILOT_DOCKER_PASSWORD")
        or os.environ.get("NPA_REGISTRY_PASSWORD")
        or ""
    )
    if target and configured_server != target:
        return "", ""
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


def check_image_pulls_with_credentials(
    images: list[str],
    *,
    mint: bool = True,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    fetcher: Fetcher | None = None,
    pull_secret_names: tuple[str, ...] = (),
    pull_secrets_by_image: Mapping[str, tuple[str, ...]] | None = None,
    pull_secret_sets_by_image: (
        Mapping[str, tuple[tuple[str, ...], ...]] | None
    ) = None,
    operator_images: Collection[str] | None = None,
    kubernetes_images: Collection[str] | None = None,
    namespace: str = "",
    context: str = "",
    secret_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    target_pull_timeout_seconds: int = 300,
    target_pull_verifier: Callable[..., KubernetesPullCheck] | None = None,
) -> list[ImagePullCheck]:
    """Check every classified VM and Kubernetes pull path independently."""

    seen: list[str] = []
    for image in images:
        value = str(image or "").strip()
        if value and value not in seen:
            seen.append(value)
    classified_paths = operator_images is not None or kubernetes_images is not None
    operator_image_set = {str(item) for item in operator_images or ()}
    kubernetes_image_set = {str(item) for item in kubernetes_images or ()}
    checks: list[ImagePullCheck] = []
    for image in seen:
        image_secret_names = (
            tuple((pull_secrets_by_image or {}).get(image, ()))
            if pull_secrets_by_image is not None
            else pull_secret_names
        )
        image_secret_sets = (
            tuple(
                tuple(dict.fromkeys(str(name).strip() for name in names))
                for names in pull_secret_sets_by_image.get(image, ())
            )
            if pull_secret_sets_by_image is not None
            else (image_secret_names,)
        )
        if classified_paths:
            requires_operator = image in operator_image_set
            requires_kubernetes = image in kubernetes_image_set
        elif pull_secrets_by_image is not None:
            # An empty tuple previously erased whether this was a VM path or a
            # Kubernetes path with no declared Secret. Refuse that ambiguity.
            requires_operator = False
            requires_kubernetes = bool(image_secret_names)
        else:
            requires_kubernetes = bool(image_secret_names)
            requires_operator = not requires_kubernetes
        if requires_kubernetes and not image_secret_sets:
            image_secret_sets = ((),)
        if pull_secret_sets_by_image is not None:
            image_secret_names = tuple(
                dict.fromkeys(name for names in image_secret_sets for name in names)
            )
        if not requires_operator and not requires_kubernetes:
            checks.append(
                ImagePullCheck(
                    image=image,
                    status="target_path_unclassified",
                    detail="image execution path is not classified as VM or Kubernetes",
                    remedy=(
                        "preserve the rendered execution target when planning image "
                        "preflight; do not infer it from imagePullSecret presence"
                    ),
                    operator_status="not_checked",
                    target_status="unverified",
                    authority="none",
                )
            )
            continue
        username = ""
        password = ""
        try:
            host = parse_image_reference(image).registry
            username, password = resolve_registry_credentials(
                host, image=image, mint=mint
            )
        except (RegistryPreflightError, RuntimeError) as exc:
            operator_check = ImagePullCheck(
                image=image,
                status="no_credentials",
                detail=f"registry credential resolution failed ({type(exc).__name__})",
                remedy="configure credentials for this registry and retry",
            )
            try:
                host = parse_image_reference(image).registry
            except RegistryPreflightError:
                checks.append(operator_check)
                continue
        else:
            operator_check = check_image_pull(
                image,
                username=username,
                password=password,
                timeout=timeout,
                fetcher=fetcher,
            )
        operator_verified = operator_check.ok
        target_verified = not requires_kubernetes
        target_status = "not_applicable"
        target_detail = "Kubernetes target pull is not required"
        target_digest = ""
        if requires_kubernetes:
            if not str(context or "").strip() or not _KUBERNETES_NAME_RE.fullmatch(
                namespace
            ):
                target_verified = False
                target_status = "exact_context_required"
                target_detail = (
                    "an exact Kubernetes context and effective namespace are required"
                )
            else:
                target_verified = True
                for path_secret_names in image_secret_sets:
                    if not path_secret_names and not (
                        operator_verified and not username and not password
                    ):
                        target_verified = False
                        target_status = "pull_secret_required"
                        target_detail = (
                            "private Kubernetes path has no declared imagePullSecret"
                        )
                        break
                    inventory_verified = True
                    inventory_detail = ""
                    if path_secret_names:
                        inventory_verified, inventory_detail = (
                            verify_kubernetes_pull_secret(
                                host,
                                path_secret_names,
                                namespace=namespace,
                                context=context,
                                timeout=timeout,
                                runner=secret_runner,
                            )
                        )
                    if not inventory_verified:
                        target_verified = False
                        target_status = "pull_secret_unverified"
                        target_detail = inventory_detail
                        break
                    verifier = target_pull_verifier or verify_kubernetes_image_pull
                    try:
                        target_check = verifier(
                            image=image,
                            secret_names=path_secret_names,
                            namespace=namespace,
                            context=context,
                            timeout_seconds=target_pull_timeout_seconds,
                        )
                    except (
                        OSError,
                        RuntimeError,
                        ValueError,
                        subprocess.SubprocessError,
                    ) as exc:
                        target_check = KubernetesPullCheck(
                            status=f"verifier_unavailable_{type(exc).__name__}",
                            cleanup_status="unverified",
                        )
                    if not target_check.ok:
                        target_verified = False
                        target_status = (
                            "cleanup_unverified"
                            if target_check.status == "verified"
                            else target_check.status
                        )
                        target_detail = _target_pull_status_detail(target_check.status)
                        if target_check.cleanup_status != "verified":
                            target_detail = (
                                f"{target_detail}; target pull probe cleanup was "
                                "not verified"
                            )
                        break
                    acceptable_digests = operator_check.acceptable_digests or tuple(
                        digest
                        for digest in (operator_check.digest or target_digest,)
                        if digest
                    )
                    if (
                        target_check.digest
                        and acceptable_digests
                        and target_check.digest not in acceptable_digests
                    ):
                        target_verified = False
                        target_status = "digest_mismatch"
                        target_detail = _target_pull_status_detail(target_status)
                        break
                    target_digest = target_digest or target_check.digest
                if target_verified:
                    has_secret = any(image_secret_sets)
                    target_status = (
                        "verified_pull_secret"
                        if len(image_secret_sets) == 1 and has_secret
                        else "verified_target_pull"
                        if len(image_secret_sets) == 1
                        else "verified_target_paths"
                    )
                    target_detail = (
                        f"{len(image_secret_sets)} exact target pull path(s) verified"
                    )
        operator_requirement_met = not requires_operator or operator_verified
        if operator_requirement_met and target_verified:
            if requires_operator and requires_kubernetes:
                authority = (
                    "operator_and_kubernetes_image_pull_secret"
                    if image_secret_names
                    else "operator_and_kubernetes_target"
                )
            elif requires_kubernetes:
                authority = (
                    "kubernetes_image_pull_secret"
                    if image_secret_names
                    else "kubernetes_target_pull"
                )
            else:
                authority = "operator"
            checks.append(
                ImagePullCheck(
                    image=operator_check.image,
                    status="ok",
                    http_status=operator_check.http_status,
                    detail=(
                        f"operator-side manifest check was {operator_check.status}; "
                        f"target pull authority: {target_detail}"
                    ),
                    operator_status=(
                        "verified" if operator_verified else operator_check.status
                    ),
                    target_status=target_status,
                    authority=authority,
                    digest=operator_check.digest or target_digest,
                )
            )
            continue
        target_remedy = ""
        if requires_kubernetes:
            target_namespace = namespace
            target_remedy = (
                f"prove the exact image pull in namespace {target_namespace!r} and context "
                f"{context or '<missing>'!r}"
            )
            if image_secret_names:
                target_remedy = (
                    f"declare a valid imagePullSecret for {host} in namespace "
                    f"{target_namespace!r}, then prove it in context "
                    f"{context or '<missing>'!r}"
                )
            elif target_status == "pull_secret_required":
                target_remedy = (
                    f"declare an operator-managed imagePullSecret for {host} in "
                    f"namespace {target_namespace!r}"
                )
        remedy = operator_check.remedy if requires_operator else ""
        checks.append(
            ImagePullCheck(
                image=operator_check.image,
                status=(
                    "target_pull_unverified"
                    if requires_kubernetes and not target_verified
                    else operator_check.status
                ),
                http_status=operator_check.http_status,
                detail=(
                    f"operator-side check: {operator_check.status}"
                    + f"; target pull authority: {target_detail}"
                ),
                remedy="; ".join(item for item in (remedy, target_remedy) if item),
                operator_status=(
                    "verified" if operator_verified else operator_check.status
                ),
                target_status=target_status,
                authority="none",
                digest=operator_check.digest or target_digest,
            )
        )
    return checks


def _target_pull_status_detail(status: str) -> str:
    """Render only bounded probe classifications, never subprocess text."""

    known = {
        "verified": "exact target pull verified",
        "exact_context_required": "an exact Kubernetes context is required",
        "create_rejected": "target rejected the pull probe",
        "image_pull_failed": "target image pull failed",
        "inventory_unavailable": "target pod inventory is unavailable",
        "timed_out": "target image pull probe timed out",
        "cleanup_unverified": "target pull succeeded but probe cleanup was not verified",
        "identity_mismatch": "target pull probe identity could not be verified",
        "invalid_image": "target pull probe image is invalid",
        "digest_mismatch": "host and target resolved different immutable image bytes",
        "target_probe_failed": "target probe could not start after pulling the image",
        "pull_secret_required": "private Kubernetes path requires an imagePullSecret",
    }
    if status in known:
        return known[status]
    if status.startswith("verifier_unavailable_"):
        return "target pull verifier is unavailable"
    return "target image pull is unverified"


_KUBERNETES_NAME_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$")
_IMAGE_PULL_FAILURE_REASONS = {
    "ErrImagePull",
    "ImagePullBackOff",
    "ImageInspectError",
    "ErrImageNeverPull",
    "InvalidImageName",
    "RegistryUnavailable",
}
_TARGET_PROBE_FAILURE_REASONS = {
    "CreateContainerConfigError",
    "CreateContainerError",
    "RunContainerError",
    "StartError",
}


def resolve_kubernetes_pull_target(
    *,
    context: str,
    global_config_path: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> KubernetesPullTarget:
    """Resolve SkyPilot's kubeconfig namespace and config-level Secrets."""

    selected_context = str(context or "").strip()
    if not selected_context:
        raise RegistryPreflightError("an exact Kubernetes context is required")
    secret_names: tuple[str, ...] = ()
    secret_names_configured = False
    if global_config_path is not None:
        try:
            document = (
                yaml.safe_load(global_config_path.read_text(encoding="utf-8")) or {}
            )
        except (OSError, yaml.YAMLError) as exc:
            raise RegistryPreflightError(
                f"selected SkyPilot config is unavailable ({type(exc).__name__})"
            ) from None
        if not isinstance(document, Mapping):
            raise RegistryPreflightError("selected SkyPilot config must be a mapping")
        kubernetes = document.get("kubernetes") or {}
        if not isinstance(kubernetes, Mapping):
            raise RegistryPreflightError(
                "selected SkyPilot kubernetes config must be a mapping"
            )
        context_configs = kubernetes.get("context_configs") or {}
        if not isinstance(context_configs, Mapping):
            raise RegistryPreflightError(
                "selected SkyPilot context_configs must be a mapping"
            )
        context_config = context_configs.get(selected_context) or {}
        if not isinstance(context_config, Mapping):
            raise RegistryPreflightError(
                "selected SkyPilot context config must be a mapping"
            )
        base_secret_names = _configured_pull_secret_names(kubernetes)
        override_secret_names = _configured_pull_secret_names(context_config)
        secret_names = merge_skypilot_pull_secret_names(
            base_secret_names,
            override_secret_names,
        )
        secret_names_configured = (
            base_secret_names is not None or override_secret_names is not None
        )
    execute = runner or subprocess.run
    try:
        result = execute(
            [
                "kubectl",
                "--context",
                selected_context,
                "config",
                "view",
                "--minify",
                "-o",
                "json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RegistryPreflightError(
            f"Kubernetes context namespace is unavailable ({type(exc).__name__})"
        ) from None
    if result.returncode != 0:
        raise RegistryPreflightError(
            f"Kubernetes context namespace lookup failed (exit {result.returncode})"
        )
    try:
        payload = json.loads(result.stdout or "{}")
        contexts = payload["contexts"]
        matches = [
            item
            for item in contexts
            if isinstance(item, Mapping)
            and str(item.get("name") or "") == selected_context
        ]
        if len(matches) != 1:
            raise ValueError
        context_payload = matches[0]["context"]
        if not isinstance(context_payload, Mapping):
            raise ValueError
        namespace = str(context_payload.get("namespace") or "default").strip()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise RegistryPreflightError(
            "Kubernetes context namespace response is invalid"
        ) from None
    if not _KUBERNETES_NAME_RE.fullmatch(namespace):
        raise RegistryPreflightError("effective Kubernetes namespace is invalid")
    invalid_secret = next(
        (name for name in secret_names if not _KUBERNETES_NAME_RE.fullmatch(name)),
        "",
    )
    if invalid_secret:
        raise RegistryPreflightError(
            "selected SkyPilot config contains an invalid imagePullSecret name"
        )
    return KubernetesPullTarget(
        namespace=namespace,
        pull_secret_names=secret_names,
        pull_secret_names_configured=secret_names_configured,
    )


def _configured_pull_secret_names(
    config: Mapping[str, Any],
) -> tuple[str, ...] | None:
    pod_config = config.get("pod_config") or {}
    if not isinstance(pod_config, Mapping):
        raise RegistryPreflightError("SkyPilot pod_config must be a mapping")
    pod_spec = pod_config.get("spec") or {}
    if not isinstance(pod_spec, Mapping):
        raise RegistryPreflightError("SkyPilot pod_config.spec must be a mapping")
    if "imagePullSecrets" not in pod_spec:
        return None
    raw_names = pod_spec["imagePullSecrets"]
    if not isinstance(raw_names, list):
        raise RegistryPreflightError(
            "SkyPilot imagePullSecrets must be a list of name mappings"
        )
    names: list[str] = []
    for item in raw_names:
        if not isinstance(item, Mapping):
            raise RegistryPreflightError(
                "SkyPilot imagePullSecrets must contain name mappings"
            )
        name = str(item.get("name") or "").strip()
        if not name:
            raise RegistryPreflightError(
                "SkyPilot imagePullSecrets entries require a name"
            )
        names.append(name)
    return tuple(names)


def verify_kubernetes_pull_secret(
    registry: str,
    secret_names: tuple[str, ...],
    *,
    namespace: str = "default",
    context: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[bool, str]:
    """Inspect a declared Secret before the separate target pull probe."""

    host = canonical_registry_host(registry)
    if not secret_names:
        return False, "no imagePullSecret is declared for this execution path"
    if not str(context or "").strip():
        return False, "an exact Kubernetes context is required"
    if not _KUBERNETES_NAME_RE.fullmatch(namespace):
        return False, f"invalid Kubernetes namespace reference {namespace!r}"
    execute = runner or subprocess.run
    failures: list[str] = []
    for name in secret_names:
        if not _KUBERNETES_NAME_RE.fullmatch(name):
            failures.append(f"invalid secret reference {name!r}")
            continue
        command = ["kubectl"]
        if context:
            command.extend(["--context", context])
        command.extend(["--namespace", namespace, "get", "secret", name, "-o", "json"])
        try:
            result = execute(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(
                f"{name}: Kubernetes inventory unavailable ({type(exc).__name__})"
            )
            continue
        if result.returncode != 0:
            failures.append(
                f"{name}: Kubernetes rejected the secret lookup "
                f"(exit {result.returncode})"
            )
            continue
        try:
            secret = json.loads(result.stdout or "{}")
            secret_type = str(secret.get("type") or "")
            encoded = str((secret.get("data") or {}).get(".dockerconfigjson") or "")
            docker_config = json.loads(base64.b64decode(encoded, validate=True))
            auths = docker_config.get("auths")
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(
                f"{name}: invalid docker-config secret ({type(exc).__name__})"
            )
            continue
        if secret_type != "kubernetes.io/dockerconfigjson" or not isinstance(
            auths, dict
        ):
            failures.append(f"{name}: not a kubernetes.io/dockerconfigjson secret")
            continue
        matching_entries = [
            entry
            for registry_name, entry in auths.items()
            if canonical_registry_host(str(registry_name)) == host
        ]
        if not matching_entries:
            failures.append(f"{name}: docker config does not cover registry {host}")
            continue
        if not any(
            _docker_auth_entry_has_credential(entry) for entry in matching_entries
        ):
            failures.append(
                f"{name}: docker config covers registry {host} but contains no "
                "usable credential fields"
            )
            continue
        return True, f"secret {namespace}/{name} covers registry {host}"
    return False, "; ".join(failures) or "no declared secret could be verified"


def _docker_auth_entry_has_credential(entry: Any) -> bool:
    """Whether one Docker config auth entry contains nonempty pull credentials."""

    if not isinstance(entry, dict):
        return False
    encoded_auth = str(entry.get("auth") or "").strip()
    if encoded_auth:
        try:
            decoded = base64.b64decode(encoded_auth, validate=True)
            username, password = decoded.split(b":", 1)
        except (ValueError, TypeError):
            pass
        else:
            if username and password:
                return True
    if any(
        str(entry.get(key) or "").strip() for key in ("identitytoken", "registrytoken")
    ):
        return True
    return bool(
        str(entry.get("username") or "").strip()
        and str(entry.get("password") or "").strip()
    )


def _cleanup_pull_probe_iteration(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
    name: str,
    namespace: str,
    nonce: str,
    image: str,
    expected_uid: str,
) -> tuple[str, str]:
    """Inspect and UID-delete one owned probe, returning a bounded outcome."""

    try:
        current = run(["get", "pod", name, "--ignore-not-found=true", "-o", "json"])
    except (OSError, subprocess.SubprocessError):
        return "retry", expected_uid
    if current.returncode != 0:
        return "retry", expected_uid
    if not current.stdout.strip():
        return "absent", expected_uid
    identity = _pull_probe_identity(
        current.stdout,
        name=name,
        nonce=nonce,
        image=image,
        expected_uid=expected_uid,
    )
    if identity is None:
        return "identity_mismatch", expected_uid
    observed_uid = identity[0]
    delete_options = {
        "apiVersion": "meta.k8s.io/v1",
        "kind": "DeleteOptions",
        "gracePeriodSeconds": 0,
        "preconditions": {"uid": observed_uid},
    }
    try:
        deleted = run(
            [
                "delete",
                "--raw",
                f"/api/v1/namespaces/{namespace}/pods/{name}",
                "-f",
                "-",
            ],
            input_text=json.dumps(delete_options, separators=(",", ":")),
        )
    except (OSError, subprocess.SubprocessError):
        return "retry", observed_uid
    return ("deleted" if deleted.returncode == 0 else "retry"), observed_uid


def _cleanup_kubernetes_pull_probe(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]],
    name: str,
    namespace: str,
    nonce: str,
    image: str,
    expected_uid: str,
    creation_confirmed: bool,
) -> tuple[str, KeyboardInterrupt | None]:
    """Finish bounded owned cleanup before propagating an operator interrupt."""

    interrupted: KeyboardInterrupt | None = None
    absent_observations = 0
    for _attempt in range(4):
        try:
            outcome, expected_uid = _cleanup_pull_probe_iteration(
                run=run,
                name=name,
                namespace=namespace,
                nonce=nonce,
                image=image,
                expected_uid=expected_uid,
            )
        except KeyboardInterrupt as exc:
            interrupted = interrupted or exc
            continue
        if outcome == "identity_mismatch":
            return "identity_mismatch", interrupted
        if outcome != "absent":
            absent_observations = 0
            continue
        absent_observations += 1
        if creation_confirmed or absent_observations >= 2:
            return "verified", interrupted
    return "unverified", interrupted


def verify_kubernetes_image_pull(
    *,
    image: str,
    secret_names: tuple[str, ...],
    namespace: str,
    context: str,
    timeout_seconds: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    poll_interval_seconds: float = 1.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(12),
) -> KubernetesPullCheck:
    """Prove an image reaches one exact target, then delete the owned probe pod."""

    try:
        parse_image_reference(image)
    except RegistryPreflightError:
        return KubernetesPullCheck(status="invalid_image")
    if (
        not str(context or "").strip()
        or not _KUBERNETES_NAME_RE.fullmatch(namespace)
        or any(not _KUBERNETES_NAME_RE.fullmatch(name) for name in secret_names)
        or timeout_seconds < 0
    ):
        return KubernetesPullCheck(status="exact_context_required")
    nonce = str(nonce_factory() or "").lower()
    if not re.fullmatch(r"[a-z0-9]{1,24}", nonce):
        return KubernetesPullCheck(status="identity_mismatch")
    image_key = hashlib.sha256(image.encode("utf-8")).hexdigest()[:12]
    name = f"npa-pull-{image_key}-{nonce}"
    labels = {
        "npa.nebius.com/owned": "true",
        "npa.nebius.com/purpose": "image-pull-preflight",
        "npa.nebius.com/probe-id": nonce,
    }
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 0,
            # Nebius GPU-only pools use this expected taint. The probe requests
            # no GPU; tolerating the taint lets kubelet test registry delivery
            # without making scheduler capacity part of the auth verdict.
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            ],
            "containers": [
                {
                    "name": "pull",
                    "image": image,
                    "imagePullPolicy": "Always",
                    "command": ["/bin/sh", "-c", "exit 0"],
                }
            ],
            "imagePullSecrets": [{"name": item} for item in secret_names],
        },
    }
    if timeout_seconds:
        manifest["spec"]["activeDeadlineSeconds"] = max(1, timeout_seconds)
    execute = runner or subprocess.run
    common = ["kubectl", "--context", context, "--namespace", namespace]

    def run(
        args: list[str], *, input_text: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return execute(
            [*common, *args],
            input=input_text or None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            check=False,
        )

    create_attempted = False
    creation_confirmed = False
    expected_uid = ""
    status = "create_rejected"
    digest = ""
    cleanup_status = "not_applicable"
    try:
        create_attempted = True
        try:
            create = run(
                ["create", "-f", "-", "-o", "json"],
                input_text=json.dumps(manifest, separators=(",", ":")),
            )
        except (OSError, subprocess.SubprocessError):
            status = "create_rejected"
        else:
            if create.returncode == 0:
                creation_confirmed = True
                created_identity = _pull_probe_identity(
                    create.stdout,
                    name=name,
                    nonce=nonce,
                    image=image,
                )
                if created_identity is not None:
                    expected_uid = created_identity[0]
                deadline = monotonic() + timeout_seconds if timeout_seconds else None
                while True:
                    try:
                        observed = run(["get", "pod", name, "-o", "json"])
                    except (OSError, subprocess.SubprocessError):
                        status = "inventory_unavailable"
                        break
                    if observed.returncode != 0:
                        status = "inventory_unavailable"
                        break
                    identity = _pull_probe_identity(
                        observed.stdout,
                        name=name,
                        nonce=nonce,
                        image=image,
                        expected_uid=expected_uid,
                    )
                    if identity is None:
                        status = "identity_mismatch"
                        break
                    expected_uid, image_id, waiting_reason = identity
                    digest_match = re.search(r"@(sha256:[0-9a-fA-F]{64})", image_id)
                    if image_id:
                        status = "verified"
                        digest = digest_match.group(1).lower() if digest_match else ""
                        break
                    if waiting_reason in _IMAGE_PULL_FAILURE_REASONS:
                        status = "image_pull_failed"
                        break
                    if waiting_reason in _TARGET_PROBE_FAILURE_REASONS:
                        status = "target_probe_failed"
                        break
                    if deadline is not None and monotonic() >= deadline:
                        status = "timed_out"
                        break
                    sleeper(max(0.0, poll_interval_seconds))
            else:
                status = "create_rejected"
    finally:
        if create_attempted:
            cleanup_status, cleanup_interrupt = _cleanup_kubernetes_pull_probe(
                run=run,
                name=name,
                namespace=namespace,
                nonce=nonce,
                image=image,
                expected_uid=expected_uid,
                creation_confirmed=creation_confirmed,
            )
            if cleanup_interrupt is not None:
                raise cleanup_interrupt
    return KubernetesPullCheck(
        status=status, digest=digest, cleanup_status=cleanup_status
    )


def _pull_probe_identity(
    raw: str,
    *,
    name: str,
    nonce: str,
    image: str,
    expected_uid: str = "",
) -> tuple[str, str, str] | None:
    """Read only immutable ownership and bounded pull-state fields."""

    try:
        payload = json.loads(raw or "{}")
        if not isinstance(payload, Mapping):
            return None
        metadata = payload["metadata"]
        if not isinstance(metadata, Mapping):
            return None
        labels = metadata["labels"]
        if not isinstance(labels, Mapping):
            return None
        uid = str(metadata["uid"])
        spec = payload["spec"]
        if not isinstance(spec, Mapping):
            return None
        containers = spec["containers"]
        actual_image = str(containers[0]["image"])
    except (AttributeError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None
    if (
        str(metadata.get("name") or "") != name
        or str(labels.get("npa.nebius.com/owned") or "") != "true"
        or str(labels.get("npa.nebius.com/purpose") or "") != "image-pull-preflight"
        or str(labels.get("npa.nebius.com/probe-id") or "") != nonce
        or not uid
        or actual_image != image
        or (expected_uid and uid != expected_uid)
    ):
        return None
    status = payload.get("status")
    status = status if isinstance(status, Mapping) else {}
    statuses = status.get("containerStatuses") or []
    image_id = ""
    waiting_reason = ""
    for row in statuses if isinstance(statuses, list) else []:
        if not isinstance(row, Mapping):
            continue
        image_id = str(row.get("imageID") or "").strip()
        state = row.get("state")
        waiting = state.get("waiting") if isinstance(state, Mapping) else {}
        if isinstance(waiting, Mapping):
            candidate = str(waiting.get("reason") or "").strip()
            if candidate in _IMAGE_PULL_FAILURE_REASONS | _TARGET_PROBE_FAILURE_REASONS:
                waiting_reason = candidate
        if image_id:
            break
    return uid, image_id, waiting_reason
