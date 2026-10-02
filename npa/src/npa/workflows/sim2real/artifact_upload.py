"""Post-finalize artifact uploads for the Sim2Real workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.publication import replace_unjournaled_legacy_file
from npa.workflows.sim2real.utils import _artifact_root_uri


_RESERVED_PUBLICATION_PATHS = frozenset(
    {
        "reports/sim2real-report.json",
        "reports/sim2real.rrd",
        "reports/sim2real.mcap",
        "components/stage_14.json",
    }
)
_PUBLICATION_JOURNAL_PATH = "reports/.sim2real-publication.json"


def _upload_final_report(
    config: Sim2RealLoopConfig, report_path: Path
) -> dict[str, Any]:
    """Upload the final report after optional viewer metadata is written."""

    if not config.s3_bucket or not report_path.exists():
        return {"status": "skipped", "reason": "report or s3_bucket missing"}
    try:
        uri = f"{_artifact_root_uri(config)}/reports/sim2real-report.json"
        replace_unjournaled_legacy_file(
            StorageClient.from_environment(endpoint_url=config.s3_endpoint),
            report_path,
            uri,
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"report re-upload failed: {exc}",
            "next_action": "CONTINUE",
        }
    return {"status": "uploaded", "uri": uri, "artifact": "sim2real-report.json"}


def _upload_legacy_tree_without_reserved_aliases(
    client: StorageClient,
    local_dir: Path,
    destination: str,
) -> None:
    root = Path(local_dir)
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact upload tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == _PUBLICATION_JOURNAL_PATH:
            continue
        uri = f"{destination.rstrip('/')}/{relative}"
        if relative in _RESERVED_PUBLICATION_PATHS:
            replace_unjournaled_legacy_file(client, path, uri)
        else:
            client.upload_file(str(path), uri)


def upload_run_artifacts(config: Sim2RealLoopConfig, local_dir: Path) -> dict[str, Any]:
    """Upload the run artifact tree to S3-compatible storage."""

    if not config.s3_bucket:
        return {"status": "skipped", "reason": "s3_bucket is not configured"}
    try:
        client = StorageClient.from_environment(endpoint_url=config.s3_endpoint)
        destination = f"{_artifact_root_uri(config)}/"
        _upload_legacy_tree_without_reserved_aliases(client, local_dir, destination)
        uploaded = destination
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"S3 upload failed: {exc}",
            "next_action": "CONTINUE",
        }
    return {"status": "uploaded", "uri": uploaded}
