"""Keep artifact-reading credentials separate from the Agent's deployment writes."""

from __future__ import annotations

import os

# NPA_EMBED_STANDALONE_START
if __name__ == "npa.cli.agent_storage_runtime":
    from fastapi import HTTPException

    from npa.workflows.artifacts import (
        ArtifactDiscoveryError,
        build_s3_client,
        encode_run_ref,
        parse_s3_uri,
    )
# NPA_EMBED_STANDALONE_END


def _agent_output_s3_settings() -> dict[str, str]:
    """Return only the deployment's writable storage configuration."""
    return {
        "bucket": os.environ.get("NPA_AGENT_S3_BUCKET", "").strip(),
        "prefix": os.environ.get("NPA_AGENT_S3_PREFIX", "").strip().strip("/"),
        "endpoint": os.environ.get("NPA_AGENT_S3_ENDPOINT", "").strip(),
        "access_key": os.environ.get("AWS_ACCESS_KEY_ID", "").strip(),
        "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip(),
        "region": os.environ.get("AWS_REGION", "eu-north1").strip() or "eu-north1",
    }


def _agent_s3_settings() -> dict[str, str]:
    """Return the explicit artifact-read identity, or the deployment default."""
    if os.environ.get("NPA_AGENT_ARTIFACT_S3_CONFIGURED") != "1":
        return _agent_output_s3_settings()
    names = {
        "bucket": "BUCKET",
        "prefix": "PREFIX",
        "endpoint": "ENDPOINT",
        "access_key": "ACCESS_KEY_ID",
        "secret_key": "SECRET_ACCESS_KEY",
        "region": "REGION",
    }
    return {
        key: os.environ.get("NPA_AGENT_ARTIFACT_S3_" + suffix, "").strip()
        for key, suffix in names.items()
    }


def _agent_state_s3_prefix(requested: str) -> str:
    """Keep a legacy session-state override inside the deployment output subtree."""
    output = _agent_output_s3_settings()["prefix"]
    override = requested.strip().strip("/")
    if not output:
        return override or "npa-agent/session-state"
    prefix = override or output + "/npa-agent/session-state"
    if (
        "://" in prefix
        or "\\" in prefix
        or "%" in prefix
        or any(ord(char) < 32 for char in prefix)
        or any(part in {"", ".", ".."} for part in prefix.split("/"))
        or not (prefix == output or prefix.startswith(output + "/"))
    ):
        raise HTTPException(
            status_code=400,
            detail="Agent session-state prefix must stay inside the deployment output subtree.",
        )
    return prefix


def _storage_client(settings: dict[str, str]):
    """Build a client only from the explicitly selected storage role."""
    if not all(settings.get(key) for key in ("bucket", "access_key", "secret_key")):
        raise HTTPException(
            status_code=400, detail="Agent storage role is not configured."
        )
    try:
        client_kwargs = {
            "endpoint_url": settings["endpoint"],
            "aws_access_key_id": settings["access_key"],
            "aws_secret_access_key": settings["secret_key"],
            "region_name": settings["region"],
        }
        return build_s3_client(**client_kwargs), settings
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Agent storage client initialization failed."
        ) from exc


def _agent_s3_client():
    """Create an artifact-read client without changing any write target."""
    return _storage_client(_agent_s3_settings())


def _agent_output_s3_client():
    """Create a client for the deployment's explicitly configured outputs."""
    return _storage_client(_agent_output_s3_settings())


def _agent_output_s3_client_optional():
    """Keep optional chat persistence unavailable when deployment storage is absent."""
    try:
        return _agent_output_s3_client()
    except HTTPException:
        return None, _agent_output_s3_settings()


def _recorded_output_source(*, state: dict, run_id: str, uri: str, run_ref: str):
    """Authorize only an exact URI retained by this Agent's own completed upload."""
    runs = state.get("sim2real_runs") or {}
    run = runs.get(run_id) if isinstance(runs, dict) else None
    artifacts = run.get("artifact_uris") if isinstance(run, dict) else None
    if not isinstance(artifacts, list) or uri not in artifacts:
        return None
    settings = _agent_output_s3_settings()
    try:
        bucket, key = parse_s3_uri(uri)
    except ArtifactDiscoveryError as exc:
        raise HTTPException(
            status_code=403, detail="Recorded output URI is invalid."
        ) from exc
    prefix = "/".join(part for part in (settings["prefix"], "sim2real-b") if part)
    project = os.environ.get("NEBIUS_PROJECT_ID", "").strip()
    if (
        not project
        or bucket != settings["bucket"]
        or not key.startswith(prefix + "/" + run_id + "/")
    ):
        raise HTTPException(
            status_code=403,
            detail="Recorded output differs from the deployment run scope.",
        )
    expected_ref = encode_run_ref(bucket, prefix, run_id)
    if run_ref and run_ref != expected_ref:
        raise HTTPException(
            status_code=400, detail="RRD URI is outside the selected run."
        )
    return {
        "bucket": bucket,
        "key": key,
        "project_id": project,
        "resolved_prefix": prefix,
        "run_ref": expected_ref,
    }
