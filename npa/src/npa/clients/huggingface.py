"""Hugging Face model access checks."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class HFAccessResult:
    repo: str
    ok: bool
    status_code: int | None = None
    error: str = ""


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

