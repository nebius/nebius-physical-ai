"""SDK helpers for durable Workbench workflow monitoring."""

from __future__ import annotations

from typing import Any

from npa.orchestration.skypilot.workflow_state import (
    WorkflowS3Config,
    list_artifacts,
    list_runs,
    read_manifest,
    read_stage_log,
    read_stage_status,
    resolve_workflow_s3_config,
)


def status(
    run_id: str,
    *,
    project: str | None = None,
    workflow_s3_uri: str = "",
    workflow_s3_prefix: str = "",
    s3_bucket: str = "",
    s3_endpoint: str = "",
) -> dict[str, Any]:
    """Read durable workflow manifest and per-stage status from S3."""

    state = _state(
        run_id,
        project=project,
        workflow_s3_uri=workflow_s3_uri,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
    )
    manifest = read_manifest(state)
    stages: dict[str, dict[str, Any]] = {}
    for stage, info in (manifest.get("stages", {}) or {}).items():
        merged = dict(info) if isinstance(info, dict) else {"name": str(stage)}
        stage_status = read_stage_status(state, str(stage))
        if stage_status:
            merged.update(stage_status)
        stages[str(stage)] = merged
    return {
        "run_id": manifest.get("run_id") or run_id,
        "workflow_name": manifest.get("workflow_name", ""),
        "sky_job_id": manifest.get("sky_job_id", ""),
        "run_prefix_uri": manifest.get("run_prefix_uri") or state.uri,
        "stages": stages,
    }


def logs(
    run_id: str,
    *,
    stage: str,
    project: str | None = None,
    workflow_s3_uri: str = "",
    workflow_s3_prefix: str = "",
    s3_bucket: str = "",
    s3_endpoint: str = "",
) -> str:
    """Read a stage's durable redacted run log from S3."""

    state = _state(
        run_id,
        project=project,
        workflow_s3_uri=workflow_s3_uri,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
    )
    return read_stage_log(state, stage)


def artifacts(
    run_id: str,
    *,
    stage: str | None = None,
    project: str | None = None,
    workflow_s3_uri: str = "",
    workflow_s3_prefix: str = "",
    s3_bucket: str = "",
    s3_endpoint: str = "",
) -> list[str]:
    """List durable artifact URIs for a run or stage."""

    state = _state(
        run_id,
        project=project,
        workflow_s3_uri=workflow_s3_uri,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
    )
    return list_artifacts(state, stage)


def runs(parent_state: WorkflowS3Config, *, limit: int = 50) -> list[dict[str, Any]]:
    """List workflow manifests below a parent S3 prefix."""

    return list_runs(state_parent=parent_state, limit=limit)


def _state(
    run_id: str,
    *,
    project: str | None,
    workflow_s3_uri: str,
    workflow_s3_prefix: str,
    s3_bucket: str,
    s3_endpoint: str,
) -> WorkflowS3Config:
    return resolve_workflow_s3_config(
        run_id=run_id,
        project=project,
        workflow_s3_uri=workflow_s3_uri,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
    )


__all__ = ["artifacts", "logs", "runs", "status"]
