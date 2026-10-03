"""Environment construction for Workbench Serverless Jobs."""

from __future__ import annotations

import os

from collections.abc import Mapping


class MissingS3CredentialsError(ValueError):
    """Raised when a serverless job would launch without usable S3 credentials."""


class MissingIsaacEulaAcceptanceError(ValueError):
    """Raised before remote work when Isaac EULA acceptance is explicitly disabled."""


class InvalidIsaacEulaValueError(MissingIsaacEulaAcceptanceError):
    """Raised when ACCEPT_EULA is neither a recognized acceptance nor opt-out."""


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


#: NVIDIA licence acceptance for Isaac-backed execution only.
#:
#: The parser is shared so Python command builders use the same spellings as the
#: shell bootstrap. Generic serverless jobs do not inherit this value implicitly:
#: each Isaac-dependent caller must preflight and propagate the canonical result.
ISAAC_EULA_ENV = "ACCEPT_EULA"
ISAAC_EULA_VARS = (ISAAC_EULA_ENV,)
ISAAC_EULA_AFFIRMATIVE_VALUES = frozenset({"Y", "YES", "1", "TRUE"})
ISAAC_EULA_NEGATIVE_VALUES = frozenset({"", "N", "NO", "0", "FALSE"})


def resolve_isaac_eula_acceptance(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return canonical ``Y`` or an explicit empty opt-out without side effects.

    An absent value follows the product default and resolves to ``Y``. Legacy
    affirmative spellings migrate case-insensitively; recognized negatives and
    an explicitly empty value resolve to the canonical empty opt-out. Everything
    else is rejected so a typo cannot silently change the operator's choice.
    """

    source = os.environ if environ is None else environ
    if ISAAC_EULA_ENV not in source:
        return "Y"
    raw = str(source[ISAAC_EULA_ENV]).strip()
    normalized = raw.upper()
    if normalized in ISAAC_EULA_AFFIRMATIVE_VALUES:
        return "Y"
    if normalized in ISAAC_EULA_NEGATIVE_VALUES:
        return ""
    allowed = "Y, YES, 1, TRUE, N, NO, 0, FALSE, or an empty string"
    raise InvalidIsaacEulaValueError(
        f"Invalid ACCEPT_EULA value {raw!r}; expected one of {allowed} "
        "(case-insensitive). No expensive action has begun."
    )


def require_isaac_eula_acceptance(*, context: str, resume_command: str) -> str:
    """Return canonical acceptance, honoring opt-out without mutating the process."""

    value = resolve_isaac_eula_acceptance()
    if value == "Y":
        return value
    resume = f"ACCEPT_EULA=Y {resume_command.strip()}"
    raise MissingIsaacEulaAcceptanceError(
        f"Refusing to provision {context}: NVIDIA EULA acceptance was explicitly disabled "
        "through ACCEPT_EULA. The applicable agreements are the NVIDIA "
        "Omniverse Licence Agreement, NVIDIA Isaac Sim Additional Software and "
        "Materials Licence, and NVIDIA Software Licence Agreement; official links "
        "are listed at https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses.html. "
        "No expensive action has begun. Remove the opt-out or resume exactly with: "
        f"{resume}"
    )


def isaac_eula_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Normalize an explicitly present run-scoped value, preserving opt-out."""

    source = os.environ if environ is None else environ
    if ISAAC_EULA_ENV not in source:
        return {}
    return {ISAAC_EULA_ENV: resolve_isaac_eula_acceptance(source)}


def resolved_isaac_eula_env(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the canonical environment for a route known to require Isaac.

    Unlike :func:`isaac_eula_env`, this includes the product default when the
    public variable is absent.  Every value passes through the shared parser,
    so callers never forward an unvalidated spelling to a remote runtime.
    """

    return {ISAAC_EULA_ENV: resolve_isaac_eula_acceptance(environ)}


def build_serverless_job_env(
    *,
    output_path: str,
    hf_token: str | None = None,
    s3_credentials: Mapping[str, str] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build standardized environment variables for a Serverless Job.

    ``/tmp/hf_home`` is the fallback, not the goal: it is writable in every
    workbench image (several bake a read-only ``HOME``), but it dies with the job,
    so a gated checkpoint the image is not allowed to bake is downloaded again on
    every submission. When the operator has configured durable weight storage,
    the whole cache family is redirected there instead — see
    :mod:`npa.workbench.model_cache`.
    """

    from npa.workbench.model_cache import (
        RUNTIME_SERVERLESS,
        model_cache_env,
        resolve_model_cache_root,
    )

    env = {
        "NPA_OUTPUT_PATH": output_path,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/tmp/hf_home",
        "LEROBOT_HF_HOME": "/tmp/hf_home",
    }
    env.update(model_cache_env(resolve_model_cache_root(runtime=RUNTIME_SERVERLESS)))
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
