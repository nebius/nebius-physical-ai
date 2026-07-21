"""Environment construction for Workbench Serverless Jobs."""

from __future__ import annotations

from collections.abc import Mapping


class MissingS3CredentialsError(ValueError):
    """Raised when a serverless job would launch without usable S3 credentials."""


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
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    return env


def split_serverless_env(env: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Split env vars into safe command-line vars and secret vars."""

    secret_marker_words = ("TOKEN", "KEY", "SECRET", "PASSWORD")
    safe: dict[str, str] = {}
    secret: dict[str, str] = {}
    for key, value in env.items():
        target = secret if any(marker in key.upper() for marker in secret_marker_words) else safe
        target[key] = value
    return safe, secret
