"""Hugging Face model access checks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import quote, urlparse

import httpx


@dataclass(frozen=True)
class HFAccessResult:
    repo: str
    ok: bool
    status_code: int | None = None
    error: str = ""
    revision: str = ""
    filename: str = ""


def hf_model_url(repo: str) -> str:
    return f"https://huggingface.co/{repo}"


def validate_hf_identity(token: str, *, timeout: float = 10.0) -> HFAccessResult:
    """Authenticate a token without treating public-repository access as proof."""

    if not token:
        return HFAccessResult(repo="whoami-v2", ok=False, error="HF_TOKEN is absent")
    try:
        response = httpx.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        return HFAccessResult(repo="whoami-v2", ok=False, error=str(exc))
    if response.status_code == 200:
        return HFAccessResult(repo="whoami-v2", ok=True, status_code=200)
    if response.status_code in {401, 403}:
        return HFAccessResult(
            repo="whoami-v2",
            ok=False,
            status_code=response.status_code,
            error="HF_TOKEN was rejected by Hugging Face",
        )
    return HFAccessResult(
        repo="whoami-v2",
        ok=False,
        status_code=response.status_code,
        error=f"Hugging Face identity probe returned HTTP {response.status_code}",
    )


def validate_hf_access(
    token: str, repo: str, repo_type: str = "model", *, timeout: float = 10.0
) -> HFAccessResult:
    """Check authenticated or anonymous access to a Hugging Face repository."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    kind = "datasets" if repo_type == "dataset" else "models"
    url = f"https://huggingface.co/api/{kind}/{repo}"
    try:
        response = httpx.head(
            url, headers=headers, timeout=timeout, follow_redirects=True
        )
        if response.status_code == 405:
            response = httpx.get(
                url, headers=headers, timeout=timeout, follow_redirects=True
            )
    except httpx.HTTPError as exc:
        return HFAccessResult(repo=repo, ok=False, error=str(exc))

    if response.status_code in {401, 403}:
        return HFAccessResult(
            repo=repo,
            ok=False,
            status_code=response.status_code,
            error=(
                f"Error: HF_TOKEN does not have access to {repo}. "
                f"Request access at {hf_model_url(repo)} and retry."
            ),
        )
    if 200 <= response.status_code < 400:
        return HFAccessResult(repo=repo, ok=True, status_code=response.status_code)
    return HFAccessResult(
        repo=repo,
        ok=False,
        status_code=response.status_code,
        error=f"Unable to validate Hugging Face access to {repo}: HTTP {response.status_code}",
    )


def validate_hf_file_access(
    token: str,
    repo: str,
    revision: str,
    filename: str,
    *,
    timeout: float = 10.0,
) -> HFAccessResult:
    """Verify one pinned checkpoint path without downloading its bytes.

    Redirects are intentionally not followed so the bearer token never leaves
    ``huggingface.co``. Only Hugging Face's artifact-cache and signed-object
    redirect forms count as access; login, consent, and model-page redirects do
    not prove authorization for the exact pinned file.
    """

    normalized_repo = str(repo or "").strip("/")
    normalized_revision = str(revision or "").strip()
    normalized_filename = str(filename or "").strip("/")
    if not token:
        return HFAccessResult(
            repo=normalized_repo,
            revision=normalized_revision,
            filename=normalized_filename,
            ok=False,
            error="HF_TOKEN is required to verify the selected gated checkpoint",
        )
    url = (
        f"https://huggingface.co/{normalized_repo}/resolve/"
        f"{quote(normalized_revision, safe='')}/{quote(normalized_filename, safe='/')}"
    )
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = httpx.head(
            url, headers=headers, timeout=timeout, follow_redirects=False
        )
        if response.status_code == 405:
            response = httpx.get(
                url,
                headers={**headers, "Range": "bytes=0-0"},
                timeout=timeout,
                follow_redirects=False,
            )
    except httpx.HTTPError as exc:
        return HFAccessResult(
            repo=normalized_repo,
            revision=normalized_revision,
            filename=normalized_filename,
            ok=False,
            error=f"checkpoint access probe failed: {type(exc).__name__}",
        )
    status = response.status_code
    if 200 <= status < 300:
        return HFAccessResult(
            repo=normalized_repo,
            revision=normalized_revision,
            filename=normalized_filename,
            ok=True,
            status_code=status,
        )
    if status in {301, 302, 303, 307, 308}:
        location = str(response.headers.get("location") or "").strip()
        target = urlparse(location)
        host = str(target.hostname or "").casefold()
        artifact_cache = bool(
            not target.scheme
            and not target.netloc
            and target.path.startswith("/api/resolve-cache/")
        )
        signed_object = bool(
            target.scheme == "https"
            and not target.username
            and not target.password
            and target.path not in {"", "/"}
            and bool(target.query)
            and (
                re.fullmatch(r"cdn-lfs(?:-[a-z0-9-]+)?\.hf\.co", host)
                or host == "cas-bridge.xethub.hf.co"
                # Hugging Face's current Xet redirect is region/provider scoped,
                # for example ``us.aws.cdn.hf.co/xet-bridge-us/...``.  Trust only
                # the exact ``cdn.hf.co`` DNS boundary (plus subdomains); a mere
                # substring/suffix such as ``cdn.hf.co.attacker.invalid`` must
                # remain rejected.  The HTTPS, non-empty signed query, path, and
                # no-userinfo checks above still apply.
                or host == "cdn.hf.co"
                or host.endswith(".cdn.hf.co")
            )
        )
        if location and (artifact_cache or signed_object):
            return HFAccessResult(
                repo=normalized_repo,
                revision=normalized_revision,
                filename=normalized_filename,
                ok=True,
                status_code=status,
            )
        error = (
            "checkpoint access probe returned an untrusted or missing redirect "
            "target; exact artifact authorization remains unverified"
        )
        return HFAccessResult(
            repo=normalized_repo,
            revision=normalized_revision,
            filename=normalized_filename,
            ok=False,
            status_code=status,
            error=error,
        )
    if status in {401, 403}:
        error = (
            f"HF token cannot access gated repo {normalized_repo}; request access "
            f"at {hf_model_url(normalized_repo)}"
        )
    elif status == 404:
        error = (
            f"pinned checkpoint was not found in {normalized_repo}; verify revision "
            "and filename against the pinned Cosmos Transfer source"
        )
    else:
        error = (
            f"could not verify exact checkpoint access for {normalized_repo}: "
            f"HTTP {status}"
        )
    return HFAccessResult(
        repo=normalized_repo,
        revision=normalized_revision,
        filename=normalized_filename,
        ok=False,
        status_code=status,
        error=error,
    )
