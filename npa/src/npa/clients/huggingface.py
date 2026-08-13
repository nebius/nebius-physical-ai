"""Hugging Face model access checks."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

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


def validate_hf_access(token: str, repo: str, *, timeout: float = 10.0) -> HFAccessResult:
    """Check whether *token* can access a Hugging Face model repo."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/api/models/{repo}"
    try:
        response = httpx.head(url, headers=headers, timeout=timeout, follow_redirects=True)
        if response.status_code == 405:
            response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
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

    Redirects are intentionally not followed: an authenticated 3xx to object
    storage proves Hugging Face authorized the exact revision/path while keeping
    the bearer token on the huggingface.co origin.
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
    if 200 <= status < 400:
        return HFAccessResult(
            repo=normalized_repo,
            revision=normalized_revision,
            filename=normalized_filename,
            ok=True,
            status_code=status,
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
