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
``PENDING``/``ImagePullBackOff`` instead of failing. This module reproduces the
pull the worker will attempt, with the same credentials the run injects, and
turns the answer into a fail-fast result.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import re
import subprocess
from typing import Any, Callable, Mapping
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
    def api_registry(self) -> str:
        """Return the Registry v2 API host for the declared image registry."""

        # ``docker.io`` is the canonical pull-name authority understood by
        # container runtimes, while Docker Hub serves Registry v2 traffic from
        # this distinct endpoint. Keep ``registry`` unchanged for credential
        # matching and Kubernetes pull-secret checks.
        return "registry-1.docker.io" if self.registry == "docker.io" else self.registry

    @property
    def manifest_url(self) -> str:
        return (
            f"https://{self.api_registry}/v2/{self.repository}/manifests/{self.reference}"
        )

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
    status, response_headers, body = fetch(reference.manifest_url, headers, timeout)
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
        token_status, _, token_body = fetch(token_url, token_headers, timeout)
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
        status, response_headers, body = fetch(reference.manifest_url, headers, timeout)
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
        status, selected_headers, body = fetch(selected_url, headers, timeout)
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
    config_url = (
        f"https://{reference.api_registry}/v2/{reference.repository}/blobs/{config_digest}"
    )
    status, _, config_body = fetch(config_url, headers, timeout)
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
                detail=f"token endpoint unreachable: {exc}",
                remedy=f"check network access to {realm} from this host",
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
                detail=str(exc),
                remedy=f"check network access to https://{reference.registry}/v2/ from this host",
            )

    detail = _registry_error_detail(body)
    if 200 <= status < 300:
        digest = str(response_headers.get("docker-content-digest") or "").strip()
        return ImagePullCheck(
            image=reference.raw,
            status="ok",
            http_status=status,
            digest=(
                reference.reference
                if reference.reference.startswith("sha256:")
                else digest
            ),
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


def _registry_host(value: str) -> str:
    cleaned = str(value or "").strip().removeprefix("docker:")
    cleaned = cleaned.removeprefix("https://").removeprefix("http://")
    return cleaned.split("/", 1)[0].rstrip("/")


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

    target = _registry_host(registry)
    configured_server = _registry_host(
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
    namespace: str = "default",
    context: str = "",
    secret_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> list[ImagePullCheck]:
    """Check operator or declared in-cluster pull authority for each registry."""

    seen: list[str] = []
    for image in images:
        value = str(image or "").strip()
        if value and value not in seen:
            seen.append(value)
    checks: list[ImagePullCheck] = []
    for image in seen:
        image_secret_names = (
            tuple((pull_secrets_by_image or {}).get(image, ()))
            if pull_secrets_by_image is not None
            else pull_secret_names
        )
        try:
            host = parse_image_reference(image).registry
            username, password = resolve_registry_credentials(
                host, image=image, mint=mint
            )
        except (RegistryPreflightError, RuntimeError) as exc:
            operator_check = ImagePullCheck(
                image=image,
                status="no_credentials",
                detail=str(exc),
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
        if operator_check.ok:
            checks.append(
                ImagePullCheck(
                    **{
                        **operator_check.__dict__,
                        "operator_status": "verified",
                        "target_status": "satisfied_by_operator_credential",
                        "authority": "operator",
                    }
                )
            )
            continue
        verified, detail = verify_kubernetes_pull_secret(
            host,
            image_secret_names,
            namespace=namespace,
            context=context,
            timeout=timeout,
            runner=secret_runner,
        )
        if verified:
            checks.append(
                ImagePullCheck(
                    image=image,
                    status="ok",
                    detail=(
                        f"operator-side manifest check was {operator_check.status}; "
                        f"target pull authority verified: {detail}"
                    ),
                    operator_status=operator_check.status,
                    target_status="verified_pull_secret",
                    authority="kubernetes_image_pull_secret",
                )
            )
            continue
        remedy = operator_check.remedy
        target_remedy = (
            f"declare a valid imagePullSecret for {host} in namespace "
            f"{namespace!r} and make it readable in context {context or '<current>'!r}"
        )
        checks.append(
            ImagePullCheck(
                image=operator_check.image,
                status=(
                    "target_pull_unverified"
                    if image_secret_names
                    else operator_check.status
                ),
                http_status=operator_check.http_status,
                detail=(
                    f"operator-side check: {operator_check.status}"
                    + (f" ({operator_check.detail})" if operator_check.detail else "")
                    + f"; target pull authority unverified: {detail}"
                ),
                remedy=f"{remedy}; {target_remedy}" if remedy else target_remedy,
                operator_status=operator_check.status,
                target_status="unverified",
                authority="none",
            )
        )
    return checks


_KUBERNETES_NAME_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$")


def verify_kubernetes_pull_secret(
    registry: str,
    secret_names: tuple[str, ...],
    *,
    namespace: str = "default",
    context: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[bool, str]:
    """Verify a declared docker-config secret covers ``registry`` without leaking it."""

    host = _registry_host(registry)
    if not secret_names:
        return False, "no imagePullSecret is declared for this execution path"
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
            failures.append(f"{name}: Kubernetes inventory unavailable ({exc})")
            continue
        if result.returncode != 0:
            detail = (
                result.stderr or result.stdout or f"exit {result.returncode}"
            ).strip()
            failures.append(f"{name}: Kubernetes rejected the secret lookup ({detail})")
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
            if _registry_host(str(registry_name)) == host
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
    if any(
        str(entry.get(key) or "").strip()
        for key in ("auth", "identitytoken", "registrytoken")
    ):
        return True
    return bool(
        str(entry.get("username") or "").strip()
        and str(entry.get("password") or "").strip()
    )
