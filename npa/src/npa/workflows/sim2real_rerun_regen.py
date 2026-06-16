"""Regenerate Sim2Real Rerun recordings and optionally re-run held-out Isaac capture."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient, StorageError
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.utils import _artifact_root_uri
from npa.workflows.sim2real_viz import Sim2RealVizResult, emit_sim2real_rerun


class Sim2RealRerunRegenError(ValueError):
    """Raised when regen sync, held-out rerun, or .rrd emission fails."""


DEFAULT_REGEN_ROOT = Path("/tmp/sim2real-regen")


@dataclass(frozen=True)
class RegenResult:
    run_id: str
    local_dir: str
    local_rrd_path: str
    upload_uri: str
    heldout_frame_count: int
    rollout_count: int
    frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "local_dir": self.local_dir,
            "local_rrd_path": self.local_rrd_path,
            "upload_uri": self.upload_uri,
            "heldout_frame_count": self.heldout_frame_count,
            "rollout_count": self.rollout_count,
            "frame_count": self.frame_count,
        }


def resolve_local_rrd_path(
    run_id: str,
    *,
    override: str = "",
    local_dir: Path | None = None,
) -> Path:
    """Return the on-disk .rrd path (LOCAL_RRD_PATH env or run-scoped default)."""

    explicit = (override or os.environ.get("LOCAL_RRD_PATH", "")).strip()
    if explicit:
        return Path(explicit)
    if local_dir is not None:
        return local_dir / "reports" / "sim2real.rrd"
    return DEFAULT_REGEN_ROOT / run_id / "reports" / "sim2real.rrd"


def default_regen_local_dir(run_id: str, *, override: str = "") -> Path:
    explicit = (override or os.environ.get("NPA_SIM2REAL_REGEN_LOCAL_DIR", "")).strip()
    if explicit:
        return Path(explicit)
    return DEFAULT_REGEN_ROOT / run_id


def run_prefix_uri(config: Sim2RealLoopConfig) -> str:
    return f"{_artifact_root_uri(config).rstrip('/')}/"


def _sibling_uri(uri: str, filename: str) -> str:
    base = uri.rsplit("/", 1)[0] if "/" in uri else uri
    return f"{base.rstrip('/')}/{filename}"


def _storage_client_for_config(config: Sim2RealLoopConfig) -> StorageClient:
    from npa.workflows.sim2real.engine import _storage_client

    return _storage_client(config)


def _list_common_prefixes(client: StorageClient, prefix_uri: str) -> list[str]:
    bucket, prefix = _parse_s3(prefix_uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    paginator = client._s3.get_paginator("list_objects_v2")
    names: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for item in page.get("CommonPrefixes", []) or []:
            names.append(str(item.get("Prefix", "")))
    return [name for name in names if name]


def _parse_s3(uri: str) -> tuple[str, str]:
    from npa.clients.storage import _parse_bucket_uri

    return _parse_bucket_uri(uri)


def _download_if_exists(client: StorageClient, uri: str, local_path: Path) -> bool:
    try:
        client.download_path(uri, str(local_path))
    except (StorageError, OSError):
        return False
    return local_path.exists() and local_path.stat().st_size > 0


def sync_regen_inputs(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    client: StorageClient | None = None,
) -> None:
    """Download artifacts required for emit_sim2real_rerun from the run prefix."""

    storage = client or _storage_client_for_config(config)
    prefix = run_prefix_uri(config)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    singles = {
        "inner_loop/outer-01/evidence.json": local_dir / "inner_loop/outer-01/evidence.json",
        "eval/heldout/report.json": local_dir / "eval/heldout/report.json",
    }
    for rel, dest in singles.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _download_if_exists(storage, f"{prefix}{rel}", dest)

    for rel in ("actions", "vlm_eval", "training_signal"):
        try:
            storage.download_directory(f"{prefix}{rel}/", str(local_dir / rel))
        except (StorageError, OSError):
            pass

    heldout_report: dict[str, Any] = {}
    heldout_path = local_dir / "eval" / "heldout/report.json"
    if heldout_path.is_file():
        heldout_report = json.loads(heldout_path.read_text(encoding="utf-8"))
    sync_heldout_renders(config, local_dir, heldout_report=heldout_report, client=storage)


def sync_heldout_renders(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    heldout_report: dict[str, Any] | None = None,
    client: StorageClient | None = None,
) -> bool:
    """Sync held-out PNG tree into local_dir/eval/heldout/renders; return True if any PNGs."""

    storage = client or _storage_client_for_config(config)
    prefix = run_prefix_uri(config)
    renders_dir = Path(local_dir) / "eval" / "heldout" / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    canonical = f"{prefix}eval/heldout/renders/"
    try:
        storage.download_directory(canonical, str(renders_dir))
    except (StorageError, OSError):
        pass
    if _has_camera_pngs(renders_dir):
        return True

    component_root = f"{prefix}component-io/heldout-eval/"
    bucket, _ = _parse_s3(component_root)
    prefixes = sorted(_list_common_prefixes(storage, component_root))
    for component_prefix in reversed(prefixes):
        sibling_renders = f"s3://{bucket}/{component_prefix}output/renders/"
        try:
            storage.download_directory(sibling_renders, str(renders_dir))
        except (StorageError, OSError):
            continue
        if _has_camera_pngs(renders_dir):
            manifest_uri = f"s3://{bucket}/{component_prefix}output/render-manifest.json"
            manifest_path = Path(local_dir) / "eval" / "heldout" / "render-manifest.sibling.json"
            if _download_if_exists(storage, manifest_uri, manifest_path) and heldout_report is not None:
                report_path = Path(local_dir) / "eval" / "heldout/report.json"
                if report_path.is_file():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                else:
                    report = dict(heldout_report or {})
                report["render_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return True
    return _has_camera_pngs(renders_dir)


def _has_camera_pngs(renders_dir: Path) -> bool:
    return any(renders_dir.rglob("camera-*.png"))


def download_rrd_from_s3(
    config: Sim2RealLoopConfig,
    *,
    dest_path: Path,
    client: StorageClient | None = None,
) -> Path:
    """Download reports/sim2real.rrd for a run to dest_path."""

    storage = client or _storage_client_for_config(config)
    uri = f"{run_prefix_uri(config)}reports/sim2real.rrd"
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not _download_if_exists(storage, uri, dest_path):
        raise Sim2RealRerunRegenError(f"Rerun recording not found at {uri}")
    return dest_path


def publish_regen_outputs(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    client: StorageClient | None = None,
) -> str:
    """Upload regenerated held-out report/renders and .rrd back to the run prefix."""

    storage = client or _storage_client_for_config(config)
    prefix = run_prefix_uri(config)
    local_dir = Path(local_dir)

    report_path = local_dir / "eval" / "heldout/report.json"
    if report_path.is_file():
        storage.upload_file(str(report_path), f"{prefix}eval/heldout/report.json")

    renders_dir = local_dir / "eval" / "heldout/renders"
    if renders_dir.is_dir() and _has_camera_pngs(renders_dir):
        storage.upload_directory(str(renders_dir), f"{prefix}eval/heldout/renders")

    rrd_path = local_dir / "reports" / "sim2real.rrd"
    if not rrd_path.is_file():
        raise Sim2RealRerunRegenError(f"missing regenerated recording: {rrd_path}")
    upload_uri = storage.upload_file(str(rrd_path), f"{prefix}reports/sim2real.rrd")
    return upload_uri


def regen_sim2real_rrd(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path | None = None,
    local_rrd_path: Path | None = None,
    upload: bool = False,
    sync_inputs: bool = True,
    client: StorageClient | None = None,
) -> RegenResult:
    """Sync artifacts (optional), emit .rrd locally, optionally upload to S3."""

    work_dir = Path(local_dir) if local_dir is not None else default_regen_local_dir(config.run_id)
    output_rrd = (
        Path(local_rrd_path)
        if local_rrd_path is not None
        else resolve_local_rrd_path(config.run_id, local_dir=work_dir)
    )
    storage = client or _storage_client_for_config(config)

    if sync_inputs:
        sync_regen_inputs(config, work_dir, client=storage)

    inner_path = work_dir / "inner_loop/outer-01/evidence.json"
    heldout_path = work_dir / "eval/heldout/report.json"
    if not inner_path.is_file():
        raise Sim2RealRerunRegenError(f"missing inner evidence: {inner_path}")
    if not heldout_path.is_file():
        raise Sim2RealRerunRegenError(f"missing held-out report: {heldout_path}")

    inner_evidence = json.loads(inner_path.read_text(encoding="utf-8"))
    heldout_report = json.loads(heldout_path.read_text(encoding="utf-8"))
    result = emit_sim2real_rerun(
        local_dir=work_dir,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_rrd=output_rrd,
    )
    upload_uri = ""
    if upload:
        upload_uri = publish_regen_outputs(config, work_dir, client=storage)
    return _regen_result_from_viz(config.run_id, work_dir, result, upload_uri=upload_uri)


def rerun_heldout_eval_only(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path | None = None,
    outer_iteration: int = 1,
    publish: bool = True,
    client: StorageClient | None = None,
) -> dict[str, Any]:
    """Re-run stage 10 Isaac held-out eval on cluster for an existing run."""

    from npa.workflows.sim2real.engine import run_heldout_eval

    work_dir = Path(local_dir) if local_dir is not None else default_regen_local_dir(config.run_id)
    storage = client or _storage_client_for_config(config)
    prefix = run_prefix_uri(config)

    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        storage.download_directory(f"{prefix}envs/heldout/", str(work_dir / "envs" / "heldout"))
    except (StorageError, OSError) as exc:
        raise Sim2RealRerunRegenError(
            f"failed to sync envs/heldout for {config.run_id}: {exc}"
        ) from exc

    inner_path = work_dir / "inner_loop/outer-01/evidence.json"
    inner_path.parent.mkdir(parents=True, exist_ok=True)
    if not _download_if_exists(storage, f"{prefix}inner_loop/outer-01/evidence.json", inner_path):
        raise Sim2RealRerunRegenError(f"missing inner evidence at {prefix}inner_loop/outer-01/evidence.json")

    inner_evidence = json.loads(inner_path.read_text(encoding="utf-8"))
    report = run_heldout_eval(
        config,
        local_dir=work_dir,
        inner_evidence=inner_evidence,
        outer_iteration=outer_iteration,
    )

    invocation = report.get("component_invocation") or {}
    output_uri = str(invocation.get("output_uri") or "").strip()
    renders_dir = work_dir / "eval" / "heldout" / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    if output_uri:
        try:
            storage.download_directory(_sibling_uri(output_uri, "renders/"), str(renders_dir))
        except (StorageError, OSError):
            sync_heldout_renders(config, work_dir, heldout_report=report, client=storage)
    else:
        sync_heldout_renders(config, work_dir, heldout_report=report, client=storage)

    if not _has_camera_pngs(renders_dir):
        raise Sim2RealRerunRegenError(
            "held-out rerun completed but no camera-*.png renders were synced; "
            "check NPA_SIM2REAL_HELDOUT_RENDER_FRAMES=1 and Isaac sibling logs"
        )

    if publish:
        publish_regen_outputs(config, work_dir, client=storage)
    return report


def _regen_result_from_viz(
    run_id: str,
    local_dir: Path,
    result: Sim2RealVizResult,
    *,
    upload_uri: str = "",
) -> RegenResult:
    if result.heldout_frame_count <= 0:
        raise Sim2RealRerunRegenError(
            "regenerated .rrd has heldout_frame_count=0; sync eval/heldout/renders or rerun held-out eval"
        )
    return RegenResult(
        run_id=run_id,
        local_dir=str(local_dir),
        local_rrd_path=result.output_rrd_path,
        upload_uri=upload_uri,
        heldout_frame_count=result.heldout_frame_count,
        rollout_count=result.rollout_count,
        frame_count=result.frame_count,
    )
