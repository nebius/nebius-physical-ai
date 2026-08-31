"""Hugging Face model access checks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import quote, urljoin, urlparse

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
    token: str,
    repo: str,
    repo_type: str = "model",
    revision: str = "",
    filename: str = "",
    *,
    timeout: float = 10.0,
) -> HFAccessResult:
    """Check repository metadata, or exact payload bytes when *filename* is set."""
    if filename:
        return validate_hf_file_access(
            token,
            repo,
            revision,
            filename,
            repo_type=repo_type,
            timeout=timeout,
        )
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    kind = "datasets" if repo_type == "dataset" else "models"
    url = f"https://huggingface.co/api/{kind}/{repo}"
    if revision:
        url += f"/revision/{quote(revision, safe='')}"
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
            revision=revision,
            ok=False,
            status_code=response.status_code,
            error=(
                f"Error: HF_TOKEN does not have access to {repo}. "
                f"Request access at {hf_model_url(repo)} and retry."
            ),
        )
    if 200 <= response.status_code < 400:
        return HFAccessResult(
            repo=repo,
            revision=revision,
            ok=True,
            status_code=response.status_code,
        )
    return HFAccessResult(
        repo=repo,
        revision=revision,
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
    repo_type: str = "model",
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
    if not normalized_revision or not normalized_filename:
        return HFAccessResult(
            repo=normalized_repo,
            revision=normalized_revision,
            filename=normalized_filename,
            ok=False,
            error="exact revision and payload filename are required for byte access proof",
        )
    if not token:
        return HFAccessResult(
            repo=normalized_repo,
            revision=normalized_revision,
            filename=normalized_filename,
            ok=False,
            error="HF_TOKEN is required to verify the selected gated checkpoint",
        )
    repo_prefix = "datasets/" if repo_type == "dataset" else ""
    url = (
        f"https://huggingface.co/{repo_prefix}{normalized_repo}/resolve/"
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
    redirect_statuses = {301, 302, 303, 307, 308}

    def trusted_redirect(location: str) -> bool:
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
        return bool(location and (artifact_cache or signed_object))

    if 200 <= status < 300:
        return HFAccessResult(
            repo=normalized_repo,
            revision=normalized_revision,
            filename=normalized_filename,
            ok=True,
            status_code=status,
        )
    if status in redirect_statuses:
        current_url = url
        current_response = response
        # A redirect name alone is not byte authorization. Follow only the
        # narrowly trusted HF cache/signed-object chain, never with the bearer,
        # and require the target itself to authorize HEAD (or one byte Range).
        for _hop in range(3):
            location = str(current_response.headers.get("location") or "").strip()
            if not trusted_redirect(location):
                break
            current_url = urljoin(current_url, location)
            try:
                current_response = httpx.head(
                    current_url,
                    headers={},
                    timeout=timeout,
                    follow_redirects=False,
                )
                if current_response.status_code == 405:
                    current_response = httpx.get(
                        current_url,
                        headers={"Range": "bytes=0-0"},
                        timeout=timeout,
                        follow_redirects=False,
                    )
            except httpx.HTTPError as exc:
                return HFAccessResult(
                    repo=normalized_repo,
                    revision=normalized_revision,
                    filename=normalized_filename,
                    ok=False,
                    error=f"artifact target byte probe failed: {type(exc).__name__}",
                )
            if 200 <= current_response.status_code < 300:
                return HFAccessResult(
                    repo=normalized_repo,
                    revision=normalized_revision,
                    filename=normalized_filename,
                    ok=True,
                    status_code=current_response.status_code,
                )
            if current_response.status_code not in redirect_statuses:
                break
        else:
            return HFAccessResult(
                repo=normalized_repo,
                revision=normalized_revision,
                filename=normalized_filename,
                ok=False,
                status_code=current_response.status_code,
                error="trusted artifact redirect chain exceeded the bounded hop limit",
            )
        status = current_response.status_code
        if status in {401, 403}:
            error = (
                "trusted artifact target denied the token-free byte probe; exact "
                "artifact authorization remains unverified"
            )
            return HFAccessResult(
                repo=normalized_repo,
                revision=normalized_revision,
                filename=normalized_filename,
                ok=False,
                status_code=status,
                error=error,
            )
        if status not in redirect_statuses:
            return HFAccessResult(
                repo=normalized_repo,
                revision=normalized_revision,
                filename=normalized_filename,
                ok=False,
                status_code=status,
                error=(
                    "trusted artifact target did not authorize the bounded byte "
                    f"probe: HTTP {status}"
                ),
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
