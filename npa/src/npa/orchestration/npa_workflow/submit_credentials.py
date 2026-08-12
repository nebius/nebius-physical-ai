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


@dataclass(frozen=True)
class SubmitCredentialContext:
    endpoint_url: str = ""
    bucket: str = ""
    access_key_id: str = field(default="", repr=False)
    secret_access_key: str = field(default="", repr=False)
    secret_values: Mapping[str, str] = field(default_factory=dict, repr=False)
    missing: tuple[str, ...] = ()


def resolve_submit_credentials(
    *,
    project: str = "",
    explicit_endpoint: str = "",
    requested: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
) -> SubmitCredentialContext:
    """Resolve endpoint and explicitly requested secret envs.

    Process environment wins, followed by the selected project's storage stanza,
    then the host's configured credential file. Values are returned only for
    direct injection into the owner-only SkyPilot environment and are excluded
    from this object's repr.
    """

    env = environ if environ is not None else os.environ
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
        value = str(env.get(name) or available.get(name) or "")
        if value:
            resolved[name] = value
        else:
            missing.append(name)

    endpoint = (
        str(explicit_endpoint or "").strip()
        or str(
            env.get("AWS_ENDPOINT_URL") or env.get("NEBIUS_S3_ENDPOINT") or ""
        ).strip()
        or str(project_storage.endpoint_url or configured.s3_endpoint or "").strip()
    )
    bucket = (
        str(
            env.get("NPA_CHECKPOINT_BUCKET") or env.get("NEBIUS_S3_BUCKET") or ""
        ).strip()
        or str(project_storage.checkpoint_bucket or configured.s3_bucket or "").strip()
    )
    return SubmitCredentialContext(
        endpoint_url=storage_endpoint_url(endpoint),
        bucket=bucket,
        access_key_id=str(
            env.get("AWS_ACCESS_KEY_ID")
            or project_storage.aws_access_key_id
            or configured.s3_access_key_id
            or ""
        ),
        secret_access_key=str(
            env.get("AWS_SECRET_ACCESS_KEY")
            or project_storage.aws_secret_access_key
            or configured.s3_secret_access_key
            or ""
        ),
        secret_values=resolved,
        missing=tuple(missing),
    )
