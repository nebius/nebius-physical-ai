"""Generic credential preflight shared across workbench tools and deploys.

Validates the service credentials nearly every GPU job or deploy needs as
explicit PASS/WARN/FAIL/SKIP checks. An optional Nebius CLI check verifies the
control-plane authentication required for provisioning.

Every check is a pure function that takes the resolved credentials plus an
injectable probe. The CLI wires real probes (Hugging Face identity, NGC token
exchange, S3 list, Token Factory models); unit tests inject fakes. Nothing here imports GPU-heavy
packages or touches infrastructure at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from npa.workflows.sim2real_health import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    has_failure,
)

# Preserve the lightweight default for hosted-inference users. The explicit
# ``all`` selection also checks the Nebius CLI profile needed for cloud work.
DEFAULT_CREDENTIAL_CHECKS: tuple[str, ...] = ("hf", "ngc", "s3", "token_factory")
SUPPORTED_CREDENTIAL_CHECKS: tuple[str, ...] = (*DEFAULT_CREDENTIAL_CHECKS, "nebius")
# Backward-compatible name for callers that use the default check set.
CREDENTIAL_CHECKS = DEFAULT_CREDENTIAL_CHECKS


@dataclass
class CredentialProbes:
    """Injectable side-effecting dependencies for credential checks.

    Defaults are ``None`` so the engine stays pure and import-safe. The CLI fills
    these with real implementations; tests pass fakes. When a service probe is
    ``None``, the check reports presence without reaching the network. A missing
    Nebius profile probe produces SKIP because profile presence is not proof of
    usable authentication.
    """

    hf_validator: Callable[[str], Any] | None = None
    ngc_validator: Callable[[str], str] | None = None
    s3_client_factory: Callable[[], Any] | None = None
    token_factory_verifier: Callable[[], list[str]] | None = None
    nebius_profile_verifier: Callable[[], Any] | None = None


def _looks_like_auth_failure(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "authentication",
        )
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
    result = probes.hf_validator(token)
    if getattr(result, "ok", False):
        return CheckResult(
            name="hf",
            status=PASS,
            summary="HF_TOKEN is authenticated by Hugging Face.",
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
    """Check the NVIDIA NGC API key is present and optionally authenticated."""

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
    if probes.ngc_validator is None:
        return CheckResult(
            name="ngc",
            status=PASS,
            summary="NGC_API_KEY is set (not verified against NGC).",
        )
    outcome = probes.ngc_validator(key)
    if outcome == "reachable":
        return CheckResult(
            name="ngc", status=PASS, summary="NGC_API_KEY is authenticated by NGC."
        )
    if outcome in {"entitlement-required", "tags-401", "tags-403", "tags-404"}:
        return CheckResult(
            name="ngc",
            status=PASS,
            summary=(
                "NGC_API_KEY completed token exchange; access to the probe artifact "
                "is separate and not implied."
            ),
            details=(outcome,),
        )
    if outcome in {"auth-401", "auth-403", "auth-no-token"}:
        return CheckResult(
            name="ngc",
            status=FAIL,
            summary="NGC_API_KEY was rejected by NGC.",
            remedy="Regenerate the key at https://org.ngc.nvidia.com/setup/api-key.",
            details=(outcome,),
        )
    return CheckResult(
        name="ngc",
        status=WARN,
        summary="NGC_API_KEY is set but could not be authenticated against NGC.",
        remedy="Retry when NGC is reachable.",
        details=(outcome,),
    )


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
                "export NEBIUS_TOKEN_FACTORY_KEY, then run `npa configure "
                "--no-interactive --save-env-credentials`."
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


def check_nebius(_credentials: Any, probes: CredentialProbes) -> CheckResult:
    """Check that the selected Nebius CLI profile can call the control plane."""

    if probes.nebius_profile_verifier is None:
        return CheckResult(
            name="nebius",
            status=SKIP,
            summary="Nebius CLI authentication was not verified in offline mode.",
            remedy="Run the same check without `--offline` before provisioning.",
        )
    verification = probes.nebius_profile_verifier()
    profile_source = (
        "Configured Nebius CLI profile"
        if getattr(verification, "profile", "")
        else "Default Nebius CLI profile"
    )
    if verification.identity_verified and verification.iam_token_minted:
        return CheckResult(
            name="nebius",
            status=PASS,
            summary=f"{profile_source} is authenticated.",
        )
    failure_reason = getattr(verification, "failure_reason", "")
    if failure_reason == "cli_unavailable":
        return CheckResult(
            name="nebius",
            status=FAIL,
            summary="Nebius CLI is not available.",
            remedy="Install the Nebius CLI or put `nebius` on PATH, then retry.",
        )
    if failure_reason == "timeout":
        return CheckResult(
            name="nebius",
            status=FAIL,
            summary="Nebius CLI authentication check timed out.",
            remedy=(
                "Check connectivity to Nebius IAM and retry. Re-authenticate the "
                "selected profile if the timeout persists."
            ),
        )
    if failure_reason == "probe_error":
        return CheckResult(
            name="nebius",
            status=FAIL,
            summary="Nebius CLI authentication check could not run.",
            remedy=(
                "Run `nebius --profile <profile> --no-browser --no-check-update iam "
                "whoami` with the selected profile to diagnose, then retry."
            ),
        )
    if failure_reason == "token_mint_failed" or verification.identity_verified:
        return CheckResult(
            name="nebius",
            status=FAIL,
            summary=f"{profile_source} resolved identity but could not mint an IAM token.",
            remedy="Re-authenticate the selected Nebius CLI profile, then retry.",
        )
    return CheckResult(
        name="nebius",
        status=FAIL,
        summary=f"{profile_source} could not resolve an authenticated identity.",
        remedy="Authenticate the selected Nebius CLI profile, then retry.",
    )


_CHECK_FUNCS: dict[str, Callable[[Any, CredentialProbes], CheckResult]] = {
    "hf": check_hf,
    "ngc": check_ngc,
    "s3": check_s3,
    "token_factory": check_token_factory,
    "nebius": check_nebius,
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
            f"Choices: {', '.join(SUPPORTED_CREDENTIAL_CHECKS)}."
        )
    return [_CHECK_FUNCS[name](credentials, active_probes) for name in selected]


__all__ = [
    "CREDENTIAL_CHECKS",
    "DEFAULT_CREDENTIAL_CHECKS",
    "SUPPORTED_CREDENTIAL_CHECKS",
    "CredentialProbes",
    "check_hf",
    "check_ngc",
    "check_nebius",
    "check_s3",
    "check_token_factory",
    "has_failure",
    "run_credential_preflight",
]
