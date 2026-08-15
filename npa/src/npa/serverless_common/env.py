"""Environment construction for Workbench Serverless Jobs."""

from __future__ import annotations

import os

from collections.abc import Mapping


class MissingS3CredentialsError(ValueError):
    """Raised when a serverless job would launch without usable S3 credentials."""


class MissingIsaacEulaAcceptanceError(ValueError):
    """Raised before remote work when Isaac EULA acceptance is explicitly disabled."""


def require_s3_credentials(
    s3_credentials: Mapping[str, str] | None,
    *,
    context: str = "the serverless job",
) -> None:
    """Fail fast if any S3 credential a remote job needs is missing.

    Serverless jobs allocate paid GPUs *before* the container starts, so a
    silent fall-back to empty S3 credentials only surfaces minutes later inside
    the running job (after the GPU is already billing). Validate at submit time
    instead, and name every missing field so the fix is obvious.

    Note: ``endpoint_url`` is required because these jobs run against Nebius
    object storage, which always needs an explicit S3 endpoint. This check is
    Nebius-specific and would need relaxing to support the implicit AWS default
    endpoint.
    """

    creds = s3_credentials or {}
    missing = [
        label
        for label, key in (
            ("access key id (AWS_ACCESS_KEY_ID)", "aws_access_key_id"),
            ("secret access key (AWS_SECRET_ACCESS_KEY)", "aws_secret_access_key"),
            ("endpoint url (AWS_ENDPOINT_URL)", "endpoint_url"),
        )
        if not str(creds.get(key, "") or "").strip()
    ]
    if missing:
        raise MissingS3CredentialsError(
            f"Missing S3 {', '.join(missing)} for {context}. Run `npa configure` "
            "or export the AWS_* variables before submitting so the remote job "
            "can read inputs and write artifacts."
        )


#: NVIDIA licence acceptance for the Isaac images, forwarded from the CALLER's environment.
#:
#: The Isaac workbench images ship no Isaac Sim and default NVIDIA acceptance on at runtime.
#: An explicit empty/non-Y value opts out and makes them refuse before fetching (exit 78).
#:
#: This lives in the SHARED builder, not in one caller, because every CLI serverless path
#: (isaac_lab, groot, genesis, cosmos, fiftyone) and the golden-eval runner go through here.
#: Fixing it in one place fixes `npa workbench isaac-lab train --runtime serverless` too.
#:
#: Isaac CLI entrypoints default this value to Y. Explicit empty/non-Y values remain an
#: opt-out and fail before provisioning.
ISAAC_EULA_ENV = "ACCEPT_EULA"
ISAAC_EULA_VARS = (ISAAC_EULA_ENV,)


def require_isaac_eula_acceptance(*, context: str, resume_command: str) -> None:
    """Default acceptance on, while honoring an explicit empty/non-Y opt-out."""

    if ISAAC_EULA_ENV not in os.environ:
        os.environ[ISAAC_EULA_ENV] = "Y"
        return

    if str(os.environ.get(ISAAC_EULA_ENV) or "").strip() == "Y":
        return
    resume = f"ACCEPT_EULA=Y {resume_command.strip()}"
    raise MissingIsaacEulaAcceptanceError(
        f"Refusing to provision {context}: NVIDIA EULA acceptance was explicitly disabled "
        "for ACCEPT_EULA=Y. The required agreements are the NVIDIA "
        "Omniverse Licence Agreement, NVIDIA Isaac Sim Additional Software and "
        "Materials Licence, and NVIDIA Software Licence Agreement; official links "
        "are listed at https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses.html. "
        "No expensive action has begun. Accept only those named agreements, then "
        f"resume exactly with: {resume}"
    )


def isaac_eula_env() -> dict[str, str]:
    """Return the official run-scoped Isaac EULA value when explicitly set."""

    value = str(os.environ.get(ISAAC_EULA_ENV) or "").strip()
    return {ISAAC_EULA_ENV: value} if value else {}


def build_serverless_job_env(
    *,
    output_path: str,
    hf_token: str | None = None,
    s3_credentials: Mapping[str, str] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build standardized environment variables for a Serverless Job."""

    env = {
        "NPA_OUTPUT_PATH": output_path,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/tmp/hf_home",
        "LEROBOT_HF_HOME": "/tmp/hf_home",
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
    if s3_credentials:
        if access_key := s3_credentials.get("aws_access_key_id"):
            env["AWS_ACCESS_KEY_ID"] = access_key
        if secret_key := s3_credentials.get("aws_secret_access_key"):
            env["AWS_SECRET_ACCESS_KEY"] = secret_key
        if endpoint := s3_credentials.get("endpoint_url"):
            env["AWS_ENDPOINT_URL"] = endpoint
            env["S3_ENDPOINT_URL"] = endpoint
            env["NEBIUS_S3_ENDPOINT"] = endpoint
    # Before extra_env, so an explicit caller value still wins.
    env.update(isaac_eula_env())
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    return env


def split_serverless_env(
    env: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Split env vars into safe command-line vars and secret vars."""

    secret_marker_words = ("TOKEN", "KEY", "SECRET", "PASSWORD")
    safe: dict[str, str] = {}
    secret: dict[str, str] = {}
    for key, value in env.items():
        target = (
            secret
            if any(marker in key.upper() for marker in secret_marker_words)
            else safe
        )
        target[key] = value
    return safe, secret
