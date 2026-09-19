"""Resolve configured workflow credentials without exposing their values."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Mapping, Sequence

from npa.clients.config import resolve_project_storage
from npa.clients.credentials import (
    load_credentials,
    shared_credential_env,
    storage_endpoint_url,
)

STORAGE_ENDPOINT_ENV_NAMES = (
    "AWS_ENDPOINT_URL_S3",
    "AWS_ENDPOINT_URL",
    "NEBIUS_S3_ENDPOINT",
    "NPA_STORAGE_ENDPOINT",
    "S3_ENDPOINT_URL",
)

PROCESS_ENVIRONMENT_CREDENTIAL_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    }
)
PROCESS_ENVIRONMENT_REQUEST_NAMES = (
    PROCESS_ENVIRONMENT_CREDENTIAL_NAMES | frozenset(STORAGE_ENDPOINT_ENV_NAMES)
)


def storage_endpoint_from_environment(environ: Mapping[str, str]) -> str:
    """Honor the service-specific boto endpoint before generic aliases."""
    return next(
        (
            str(environ[key]).strip()
            for key in STORAGE_ENDPOINT_ENV_NAMES
            if environ.get(key)
        ),
        "",
    )


@dataclass(frozen=True)
class SubmitCredentialContext:
    endpoint_url: str = ""
    bucket: str = ""
    access_key_id: str = field(default="", repr=False)
    secret_access_key: str = field(default="", repr=False)
    session_token: str = field(default="", repr=False)
    secret_values: Mapping[str, str] = field(default_factory=dict, repr=False)
    missing: tuple[str, ...] = ()
    provenance: Mapping[str, str] = field(default_factory=dict)


def resolve_submit_credentials(
    *,
    project: str = "",
    explicit_endpoint: str = "",
    requested: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    workflow_env: Mapping[str, str] | None = None,
    require_process_environment_triplet: bool = False,
) -> SubmitCredentialContext:
    """Resolve endpoint and explicitly requested secret envs.

    Explicit endpoint wins, then process environment, workflow environment,
    the selected project's storage stanza,
    then the host's configured credential file. Values are returned only for
    direct injection into the owner-only SkyPilot environment and are excluded
    from this object's repr.
    """

    process_env = environ if environ is not None else os.environ
    if require_process_environment_triplet:
        credential_names = (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        )
        values = tuple(str(process_env.get(name) or "") for name in credential_names)
        if not all(values):
            raise ValueError(
                "Control-plane-authorized storage credentials require one complete "
                "process-environment access/secret/session triplet"
            )
        environment_endpoint = storage_endpoint_from_environment(process_env)
        if not environment_endpoint:
            raise ValueError(
                "Control-plane-authorized storage credentials require a process-environment endpoint"
            )
        if explicit_endpoint and storage_endpoint_url(
            explicit_endpoint
        ) != storage_endpoint_url(environment_endpoint):
            raise ValueError(
                "Explicit storage endpoint differs from control-plane-authorized environment"
            )
        requested_names = tuple(
            dict.fromkeys(
                str(name or "").strip()
                for name in requested
                if str(name or "").strip()
            )
        )
        unauthorized = set(requested_names) - PROCESS_ENVIRONMENT_REQUEST_NAMES
        if unauthorized:
            raise ValueError(
                "Control-plane-authorized storage credentials may expose only the "
                "process-environment storage credential triplet"
            )
        requested_values = {
            name: str(process_env.get(name) or "")
            for name in requested_names
            if str(process_env.get(name) or "")
        }
        missing = tuple(
            name for name in requested_names if not str(process_env.get(name) or "")
        )
        if environment_endpoint:
            normalized_endpoint = storage_endpoint_url(environment_endpoint)
            requested_values.update(
                {
                    name: normalized_endpoint
                    for name in requested_names
                    if name in STORAGE_ENDPOINT_ENV_NAMES
                }
            )
            missing = tuple(
                name for name in missing if name not in STORAGE_ENDPOINT_ENV_NAMES
            )
        bucket = str(
            process_env.get("NPA_S3_BUCKET")
            or process_env.get("NPA_CHECKPOINT_BUCKET")
            or process_env.get("NEBIUS_S3_BUCKET")
            or ""
        ).strip()
        return SubmitCredentialContext(
            endpoint_url=storage_endpoint_url(environment_endpoint),
            bucket=bucket,
            access_key_id=values[0],
            secret_access_key=values[1],
            session_token=values[2],
            secret_values=requested_values,
            missing=missing,
            provenance={
                "credentials": "control-plane-authorized.environment",
                "endpoint": "control-plane-authorized.environment",
                "bucket": "environment" if bucket else "missing",
            },
        )
    env = dict(workflow_env or {})
    env.update({key: value for key, value in process_env.items() if value})
    project_storage = resolve_project_storage(project or None)
    configured = load_credentials(environ=env)
    available = shared_credential_env(configured)
    available.update({key: value for key, value in configured.tokens.items() if value})
    # Project-scoped saved credentials win over the shared credential file, but
    # never over an explicit process environment value.
    project_values = {
        "AWS_ACCESS_KEY_ID": project_storage.aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": project_storage.aws_secret_access_key,
        "AWS_ENDPOINT_URL": project_storage.endpoint_url,
        "NEBIUS_S3_ENDPOINT": project_storage.endpoint_url,
        "NEBIUS_S3_BUCKET": project_storage.checkpoint_bucket,
    }
    for key, value in project_values.items():
        if value:
            available[key] = value
    # Select an atomic credential pair. Combining an explicit access key with a
    # different saved principal's secret can pass unrelated presence checks but
    # can never authenticate the execution target.
    sources = (
        (
            "environment",
            process_env.get("AWS_ACCESS_KEY_ID", ""),
            process_env.get("AWS_SECRET_ACCESS_KEY", ""),
            process_env.get("AWS_SESSION_TOKEN", ""),
        ),
        (
            "workflow.env",
            (workflow_env or {}).get("AWS_ACCESS_KEY_ID", ""),
            (workflow_env or {}).get("AWS_SECRET_ACCESS_KEY", ""),
            (workflow_env or {}).get("AWS_SESSION_TOKEN", ""),
        ),
        (
            "project.storage",
            project_storage.aws_access_key_id,
            project_storage.aws_secret_access_key,
            getattr(project_storage, "aws_session_token", "")
            or getattr(project_storage, "session_token", ""),
        ),
        (
            "credentials",
            configured.s3_access_key_id,
            configured.s3_secret_access_key,
            getattr(configured, "s3_session_token", "")
            or getattr(configured, "session_token", ""),
        ),
    )
    access_key = secret_key = session_token = ""
    credential_source = "missing"
    for source, access, secret, token in sources:
        if access or secret:
            if not access or not secret:
                raise ValueError(
                    f"Incomplete S3 credential pair in {source}; set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in that source"
                )
            access_key = str(access)
            secret_key = str(secret)
            session_token = str(token or "")
            credential_source = source
            break
    available["AWS_ACCESS_KEY_ID"] = access_key
    available["AWS_SECRET_ACCESS_KEY"] = secret_key
    hf_token = available.get("HF_TOKEN", "") or available.get(
        "HUGGING_FACE_HUB_TOKEN", ""
    )
    if hf_token:
        available["HF_TOKEN"] = hf_token
        available["HUGGING_FACE_HUB_TOKEN"] = hf_token

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for raw_name in requested:
        name = str(raw_name or "").strip()
        if not name or name in resolved or name in missing:
            continue
        if name == "AWS_SESSION_TOKEN" and access_key and secret_key:
            # Keep the optional token bound to the same source as the selected
            # access/secret pair; never mix principals across environment,
            # workflow, project, or configured credentials.
            value = session_token
        else:
            value = str(
                available.get(name)
                if name in {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
                else env.get(name) or available.get(name) or ""
            )
        if value:
            resolved[name] = value
        else:
            missing.append(name)

    endpoint = (
        str(explicit_endpoint or "").strip()
        or storage_endpoint_from_environment(process_env)
        or storage_endpoint_from_environment(workflow_env or {})
        or str(project_storage.endpoint_url or configured.s3_endpoint or "").strip()
    )
    bucket = (
        str(
            env.get("NPA_S3_BUCKET")
            or env.get("NPA_CHECKPOINT_BUCKET")
            or env.get("NEBIUS_S3_BUCKET")
            or ""
        ).strip()
        or str(project_storage.checkpoint_bucket or configured.s3_bucket or "").strip()
    )
    endpoint = storage_endpoint_url(endpoint)
    for name in STORAGE_ENDPOINT_ENV_NAMES:
        if name in requested and endpoint:
            resolved[name] = endpoint
            if name in missing:
                missing.remove(name)
    return SubmitCredentialContext(
        endpoint_url=endpoint,
        bucket=bucket,
        access_key_id=access_key,
        secret_access_key=secret_key,
        session_token=session_token,
        secret_values=resolved,
        missing=tuple(missing),
        provenance={
            "credentials": credential_source,
            "endpoint": "cli"
            if explicit_endpoint
            else (
                "environment"
                if storage_endpoint_from_environment(process_env)
                else "workflow.env"
                if storage_endpoint_from_environment(workflow_env or {})
                else "project.storage"
                if project_storage.endpoint_url
                else "credentials"
            ),
            "bucket": "environment"
            if any(
                process_env.get(key)
                for key in (
                    "NPA_S3_BUCKET",
                    "NPA_CHECKPOINT_BUCKET",
                    "NEBIUS_S3_BUCKET",
                )
            )
            else (
                "workflow.env"
                if any(
                    env.get(key)
                    for key in (
                        "NPA_S3_BUCKET",
                        "NPA_CHECKPOINT_BUCKET",
                        "NEBIUS_S3_BUCKET",
                    )
                )
                else "project.storage"
                if project_storage.checkpoint_bucket
                else "credentials"
            ),
        },
    )
