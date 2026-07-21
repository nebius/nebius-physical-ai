"""Generic credential preflight shared across workbench tools and deploys.

Validates the credentials nearly every GPU job or deploy needs — Hugging Face,
NVIDIA NGC, Nebius object storage (S3), and Nebius Token Factory — as explicit
PASS/WARN/FAIL/SKIP checks, so a customer hits them as a clear preflight
instead of a mid-pipeline failure.

Every check is a pure function that takes the resolved credentials plus an
injectable probe. The CLI wires real probes (Hugging Face HEAD, S3 list, Token
Factory models); unit tests inject fakes. Nothing here imports GPU-heavy
packages or touches infrastructure at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from npa.workflows.sim2real_health import (
    FAIL,
    PASS,
    WARN,
    CheckResult,
    has_failure,
)

# A tiny, always-public HF repo used only to confirm a token is accepted (or
# that anonymous access works). It never requires access to a gated repo.
HF_PROBE_REPO = "hf-internal-testing/tiny-random-gpt2"

# Canonical order a customer should reason about credentials in.
CREDENTIAL_CHECKS: tuple[str, ...] = ("hf", "ngc", "s3", "token_factory")


@dataclass
class CredentialProbes:
    """Injectable side-effecting dependencies for credential checks.

    Defaults are ``None`` so the engine stays pure and import-safe. The CLI fills
    these with real implementations; tests pass fakes. When a probe is ``None``
    the corresponding live check is downgraded to a "present but unverified"
    PASS/WARN rather than reaching the network.
    """

    hf_validator: Callable[[str, str], Any] | None = None
    s3_client_factory: Callable[[], Any] | None = None
    token_factory_verifier: Callable[[], list[str]] | None = None


def _looks_like_auth_failure(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        marker in lowered
        for marker in ("401", "403", "unauthorized", "forbidden", "invalid api key", "authentication")
    )


def check_hf(credentials: Any, probes: CredentialProbes) -> CheckResult:
    """Check the Hugging Face token is present and (optionally) accepted."""

    token = getattr(credentials, "hf_token", "") or ""
    if not token:
        return CheckResult(
            name="hf",
            status=WARN,
            summary="HF_TOKEN is not set.",
            remedy=(
                "Create a read token at https://huggingface.co/settings/tokens and "
                "run `npa configure` (public models still download; gated repos fail)."
            ),
        )
    if probes.hf_validator is None:
        return CheckResult(
            name="hf",
            status=PASS,
            summary="HF_TOKEN is set (not verified against Hugging Face).",
        )
    result = probes.hf_validator(token, HF_PROBE_REPO)
    if getattr(result, "ok", False):
        # The probe hits a public repo, which confirms presence + connectivity
        # (and that the token is not outright rejected), but Hugging Face serves
        # public metadata even for an expired token, so this is not full
        # validation. Say "reachable", not "accepted".
        return CheckResult(
            name="hf",
            status=PASS,
            summary="HF_TOKEN is set and Hugging Face is reachable.",
        )
    status_code = getattr(result, "status_code", None)
    error = getattr(result, "error", "") or "unknown error"
    if status_code in {401, 403}:
        return CheckResult(
            name="hf",
            status=FAIL,
            summary="HF_TOKEN was rejected by Hugging Face.",
            remedy="Regenerate the token at https://huggingface.co/settings/tokens.",
            details=(error,),
        )
    # Non-auth failure (e.g. transient network / rate limit): don't hard-fail.
    return CheckResult(
        name="hf",
        status=WARN,
        summary="HF_TOKEN is set but could not be verified against Hugging Face.",
        remedy="Retry when the network is available; token presence looks fine.",
        details=(error,),
    )


def check_ngc(credentials: Any, probes: CredentialProbes) -> CheckResult:
    """Check the NVIDIA NGC API key is present and well-formed."""

    key = getattr(credentials, "ngc_api_key", "") or ""
    if not key:
        return CheckResult(
            name="ngc",
            status=WARN,
            summary="NGC_API_KEY is not set.",
            remedy=(
                "Needed for GR00T / Cosmos NVIDIA container + model pulls. Create one "
                "at https://org.ngc.nvidia.com/setup/api-key and run `npa configure`."
            ),
        )
    # Personal NGC API keys are prefixed 'nvapi-' (older docs sometimes show
    # 'nvapi_'); accept either separator.
    if not key.lower().startswith(("nvapi-", "nvapi_")):
        return CheckResult(
            name="ngc",
            status=WARN,
            summary="NGC_API_KEY is set but does not look like an NGC key.",
            remedy="NGC keys start with 'nvapi-'. Re-check the value in ~/.npa/credentials.yaml.",
        )
    return CheckResult(name="ngc", status=PASS, summary="NGC_API_KEY is set.")


def check_s3(credentials: Any, probes: CredentialProbes) -> CheckResult:
    """Check Nebius object storage credentials and (optionally) reachability."""

    access = getattr(credentials, "s3_access_key_id", "") or ""
    secret = getattr(credentials, "s3_secret_access_key", "") or ""
    endpoint = getattr(credentials, "s3_endpoint", "") or ""
    bucket = getattr(credentials, "s3_bucket", "") or ""

    if not (access and secret):
        return CheckResult(
            name="s3",
            status=WARN,
            summary="No S3 access key configured.",
            remedy=(
                "Run `npa configure` (auto-provisions a bucket + key) or set "
                "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
            ),
        )
    if not endpoint:
        return CheckResult(
            name="s3",
            status=WARN,
            summary="S3 credentials set but no endpoint configured.",
            remedy="Set AWS_ENDPOINT_URL (e.g. https://storage.eu-north1.nebius.cloud).",
        )
    if not bucket:
        return CheckResult(
            name="s3",
            status=WARN,
            summary="S3 credentials set but no bucket configured.",
            remedy="Set NEBIUS_S3_BUCKET / storage.bucket to an s3://... URI.",
        )
    if probes.s3_client_factory is None:
        return CheckResult(
            name="s3",
            status=PASS,
            summary="S3 credentials, endpoint, and bucket are set (not probed).",
        )
    try:
        client = probes.s3_client_factory()
        client.list_checkpoints(bucket)
    except Exception as exc:  # noqa: BLE001 - surface any reachability/auth error
        text = str(exc)
        remedy = (
            "Check the S3 access key/secret, endpoint region, and bucket name."
            if _looks_like_auth_failure(text)
            else "Confirm the endpoint is reachable and the bucket exists."
        )
        return CheckResult(
            name="s3",
            status=FAIL,
            summary=f"S3 endpoint/bucket not reachable with these credentials ({bucket}).",
            remedy=remedy,
            details=(text,),
        )
    return CheckResult(
        name="s3",
        status=PASS,
        summary=f"S3 endpoint reachable and bucket listable ({bucket}).",
    )


def check_token_factory(credentials: Any, probes: CredentialProbes) -> CheckResult:
    """Check the Nebius Token Factory key is present and (optionally) authenticates."""

    key = getattr(credentials, "token_factory_api_key", "") or ""
    if not key:
        return CheckResult(
            name="token_factory",
            status=WARN,
            summary="NEBIUS_TOKEN_FACTORY_KEY is not set.",
            remedy=(
                "Required for `npa agent` chat and zero-GPU token-factory tools. Get a "
                "key (starts with 'v1.') at https://tokenfactory.nebius.com/ and run "
                "`npa configure --token-factory-key <key>`."
            ),
        )
    if probes.token_factory_verifier is None:
        return CheckResult(
            name="token_factory",
            status=PASS,
            summary="NEBIUS_TOKEN_FACTORY_KEY is set (not verified).",
        )
    try:
        models = probes.token_factory_verifier()
    except Exception as exc:  # noqa: BLE001 - surface any auth/connectivity error
        return CheckResult(
            name="token_factory",
            status=FAIL,
            summary="Token Factory key did not authenticate.",
            remedy="Confirm the key at https://tokenfactory.nebius.com/ -> API keys.",
            details=(str(exc),),
        )
    return CheckResult(
        name="token_factory",
        status=PASS,
        summary=f"Token Factory authenticated ({len(models)} models available).",
    )


_CHECK_FUNCS: dict[str, Callable[[Any, CredentialProbes], CheckResult]] = {
    "hf": check_hf,
    "ngc": check_ngc,
    "s3": check_s3,
    "token_factory": check_token_factory,
}


def run_credential_preflight(
    credentials: Any,
    *,
    probes: CredentialProbes | None = None,
    checks: Iterable[str] | None = None,
) -> list[CheckResult]:
    """Run the selected credential checks and return their results in order."""

    active_probes = probes or CredentialProbes()
    selected = list(checks) if checks is not None else list(CREDENTIAL_CHECKS)
    unknown = [name for name in selected if name not in _CHECK_FUNCS]
    if unknown:
        raise ValueError(
            f"unknown credential check(s): {', '.join(unknown)}. "
            f"Choices: {', '.join(CREDENTIAL_CHECKS)}."
        )
    return [_CHECK_FUNCS[name](credentials, active_probes) for name in selected]


__all__ = [
    "CREDENTIAL_CHECKS",
    "CredentialProbes",
    "HF_PROBE_REPO",
    "check_hf",
    "check_ngc",
    "check_s3",
    "check_token_factory",
    "has_failure",
    "run_credential_preflight",
]
