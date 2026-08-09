"""Persist Sim2Real workflow state and emit active progress evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workflows.sim2real.models import Sim2RealLoopConfig, Sim2RealLoopError
from npa.workflows.sim2real.utils import _artifact_root_uri, _write_json_artifact
from npa.workflows.sim2real.viz_contract import visualization_run_metadata


def _workflow_state_path(local_dir: Path) -> Path:
    return local_dir / "state" / "workflow_state.json"


def _storage_client(config: Sim2RealLoopConfig) -> StorageClient:
    return StorageClient.from_environment(endpoint_url=config.s3_endpoint)


def sync_workflow_state_to_s3(
    config: Sim2RealLoopConfig, local_dir: Path
) -> dict[str, Any] | None:
    """Upload ``state/workflow_state.json`` for live status polling."""

    if not config.upload_artifacts or not config.s3_bucket:
        return None
    state_path = _workflow_state_path(local_dir)
    if not state_path.is_file():
        return None
    destination = f"{_artifact_root_uri(config)}/state/workflow_state.json"
    uri = _storage_client(config).upload_file(str(state_path), destination)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    from npa.workflows.sim2real.resume_state import DurableStateStore

    checkpoint_uri = DurableStateStore(config, local_dir).persist_workflow_checkpoint(
        payload
    )
    return {"status": "uploaded", "uri": uri, "checkpoint_uri": checkpoint_uri}


def emit_active_progress_rerun(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Refresh a viewable progress recording while the real run is active."""

    components = list(state.get("components") or [])
    if not config.rerun_enabled or len(components) < 3:
        return {"status": "not_ready", "stage_count": len(components)}
    progress_path = local_dir / "reports" / "sim2real-progress.rrd"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from npa.workflows.sim2real_viz import emit_sim2real_rerun

        result = emit_sim2real_rerun(
            local_dir=local_dir,
            inner_evidence=dict(state.get("final_inner") or {}),
            heldout_report=dict(state.get("final_eval") or {}) or None,
            stage_components=components,
            outer_history=list(state.get("outer_history") or []),
            run_metadata=visualization_run_metadata(
                config=config,
                artifact_root=_artifact_root_uri(config),
                policy_checkpoint=str(
                    (state.get("final_decision") or {}).get("checkpoint_uri") or ""
                ),
                progress=True,
            ),
            output_rrd=progress_path,
            write_mp4=False,
            allow_progress_only=True,
        )
        if not progress_path.is_file() or progress_path.stat().st_size <= 0:
            raise Sim2RealLoopError("active progress Rerun recording is empty")
        progress: dict[str, Any] = {
            "status": "written",
            "local_path": str(progress_path),
            "size_bytes": progress_path.stat().st_size,
            "stage_count": len(components),
            "recording": result.to_dict(),
            "viewer_command": (
                "npa workbench sim2real rerun serve "
                f"--run-id {config.run_id} --s3-bucket {config.s3_bucket} "
                f"--s3-prefix {config.s3_prefix}"
            ),
        }
        if config.upload_artifacts and config.s3_bucket:
            progress["s3_uri"] = _storage_client(config).upload_file(
                str(progress_path),
                f"{_artifact_root_uri(config)}/reports/sim2real-progress.rrd",
            )
        _write_json_artifact(local_dir / "reports" / "sim2real-progress.json", progress)
        return progress
    except Exception as exc:  # noqa: BLE001 - progress must not mask real stages
        progress = {
            "status": "blocked",
            "reason": str(exc),
            "stage_count": len(components),
        }
        _write_json_artifact(local_dir / "reports" / "sim2real-progress.json", progress)
        return progress


def _write_workflow_state(
    local_dir: Path,
    payload: dict[str, Any],
    *,
    config: Sim2RealLoopConfig | None = None,
) -> dict[str, Any]:
    record = _write_json_artifact(_workflow_state_path(local_dir), payload)
    if config is not None:
        sync_workflow_state_to_s3(config, local_dir)
    return record["payload"]


def _read_workflow_state(local_dir: Path) -> dict[str, Any]:
    path = _workflow_state_path(local_dir)
    if not path.exists():
        raise Sim2RealLoopError(f"workflow state file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Sim2RealLoopError("workflow state payload must be a JSON object")
    return payload
