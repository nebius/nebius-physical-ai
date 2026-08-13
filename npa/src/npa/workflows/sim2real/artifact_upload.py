"""Post-finalize artifact uploads for the Sim2Real workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.utils import _artifact_root_uri


def _upload_final_report(
    config: Sim2RealLoopConfig, report_path: Path
) -> dict[str, Any]:
    """Upload the final report after optional viewer metadata is written."""

    if not config.s3_bucket or not report_path.exists():
        return {"status": "skipped", "reason": "report or s3_bucket missing"}
    try:
        uri = f"{_artifact_root_uri(config)}/reports/sim2real-report.json"
        StorageClient.from_environment(endpoint_url=config.s3_endpoint).upload_file(
            str(report_path), uri
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"report re-upload failed: {exc}",
            "next_action": "CONTINUE",
        }
    return {"status": "uploaded", "uri": uri, "artifact": "sim2real-report.json"}


def upload_run_artifacts(config: Sim2RealLoopConfig, local_dir: Path) -> dict[str, Any]:
    """Upload the run artifact tree to S3-compatible storage."""

    if not config.s3_bucket:
        return {"status": "skipped", "reason": "s3_bucket is not configured"}
    try:
        client = StorageClient.from_environment(endpoint_url=config.s3_endpoint)
        destination = f"{_artifact_root_uri(config)}/"
        uploaded = client.upload_directory(str(local_dir), destination)
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"S3 upload failed: {exc}",
            "next_action": "CONTINUE",
        }
    return {"status": "uploaded", "uri": uploaded}
