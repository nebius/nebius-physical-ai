"""Regenerate Sim2Real viewer recordings and optionally re-run held-out Isaac capture."""

from __future__ import annotations

import json
import os
import hashlib
import shlex
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient, StorageError
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.reporting import build_progress_metrics
from npa.workflows.sim2real.utils import _artifact_root_uri
from npa.workflows.sim2real_viz import (
    Sim2RealVizResult,
    emit_sim2real_mcap_if_enabled,
    emit_sim2real_rerun,
)


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
    local_mcap_path: str = ""
    mcap_upload_uri: str = ""
    mcap_status: str = ""
    synthetic_frame_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "local_dir": self.local_dir,
            "local_rrd_path": self.local_rrd_path,
            "upload_uri": self.upload_uri,
            "heldout_frame_count": self.heldout_frame_count,
            "rollout_count": self.rollout_count,
            "frame_count": self.frame_count,
            "local_mcap_path": self.local_mcap_path,
            "mcap_upload_uri": self.mcap_upload_uri,
            "mcap_status": self.mcap_status,
            "synthetic_frame_count": self.synthetic_frame_count,
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


def _gold_eval_relative_dir(config: Sim2RealLoopConfig) -> Path:
    return Path("eval") / "gold-heldout" / f"outer-{config.outer_iterations:02d}"


def _gold_report_path(config: Sim2RealLoopConfig, local_dir: Path) -> Path:
    canonical = Path(local_dir) / _gold_eval_relative_dir(config) / "report.json"
    legacy = Path(local_dir) / "eval" / "heldout" / "report.json"
    return canonical if canonical.is_file() or not legacy.is_file() else legacy


def _renders_dir_for_report(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    heldout_report: dict[str, Any] | None,
) -> Path:
    report = heldout_report or {}
    recorded = Path(str(report.get("local_renders_dir") or ""))
    try:
        if recorded.is_absolute() and recorded.resolve().is_relative_to(
            Path(local_dir).resolve()
        ):
            return recorded
    except OSError:
        pass
    if report.get("evaluation_split") == "gold_heldout":
        outer = int(report.get("outer_iteration") or config.outer_iterations)
        return (
            Path(local_dir) / "eval" / "gold-heldout" / f"outer-{outer:02d}" / "renders"
        )
    return Path(local_dir) / "eval" / "heldout" / "renders"


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

    inner_evidence_rel = _latest_inner_evidence_rel(storage, prefix)
    gold_eval_rel = _gold_eval_relative_dir(config).as_posix()
    singles = {
        inner_evidence_rel: local_dir / inner_evidence_rel,
        f"{gold_eval_rel}/report.json": local_dir / gold_eval_rel / "report.json",
        "eval/heldout/report.json": local_dir / "eval/heldout/report.json",
        "reports/sim2real-report.json": local_dir / "reports/sim2real-report.json",
        "checkpoints/candidate/candidate.json": local_dir
        / "checkpoints/candidate/candidate.json",
        "outer_loop/decision.json": local_dir / "outer_loop/decision.json",
        "outer_loop/loopback.json": local_dir / "outer_loop/loopback.json",
        "tokens/manifest.json": local_dir / "tokens/manifest.json",
        "envs/train/envs.jsonl": local_dir / "envs/train/envs.jsonl",
        "envs/heldout/envs.jsonl": local_dir / "envs/heldout/envs.jsonl",
        "envs/manifest/split-manifest.json": local_dir
        / "envs/manifest/split-manifest.json",
        "envs/split-manifest.json": local_dir / "envs/split-manifest.json",
        "stage_01_trigger/trigger.json": local_dir / "stage_01_trigger/trigger.json",
        "stage_02_assets/assets_manifest.json": local_dir
        / "stage_02_assets/assets_manifest.json",
        "stage_02_assets/consumed_robot_spec.json": local_dir
        / "stage_02_assets/consumed_robot_spec.json",
        "stage_02_assets/consumed_scene_spec.json": local_dir
        / "stage_02_assets/consumed_scene_spec.json",
        "stage_12_external_validation/external_stub.json": local_dir
        / "stage_12_external_validation/external_stub.json",
        "stage_13_retrigger/retrigger.json": local_dir
        / "stage_13_retrigger/retrigger.json",
    }
    for rel, dest in singles.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _download_if_exists(storage, f"{prefix}{rel}", dest)

    # The viewer plots improvement across every outer/inner pass, not only the
    # latest evidence object selected for backward compatibility above.
    for outer_prefix in _list_common_prefixes(
        storage, f"{prefix.rstrip('/')}/inner_loop/"
    ):
        outer_name = Path(outer_prefix.rstrip("/")).name
        if not outer_name.startswith("outer-"):
            continue
        destination = local_dir / "inner_loop" / outer_name / "evidence.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        bucket, _run_key = _parse_s3(prefix)
        _download_if_exists(
            storage,
            f"s3://{bucket}/{outer_prefix.rstrip('/')}/evidence.json",
            destination,
        )

    for rel in ("actions", "vlm_eval", "training_signal", "augment", "envs/raw"):
        try:
            storage.download_directory(f"{prefix}{rel}/", str(local_dir / rel))
        except (StorageError, OSError):
            pass
    for evidence_path in sorted(
        (local_dir / "inner_loop").glob("outer-*/evidence.json")
    ):
        _rewrite_inner_evidence_paths(local_dir, evidence_path)

    heldout_report: dict[str, Any] = {}
    heldout_path = _gold_report_path(config, local_dir)
    if heldout_path.is_file():
        heldout_report = json.loads(heldout_path.read_text(encoding="utf-8"))
    sync_heldout_renders(
        config, local_dir, heldout_report=heldout_report, client=storage
    )


def sync_heldout_renders(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    heldout_report: dict[str, Any] | None = None,
    client: StorageClient | None = None,
) -> bool:
    """Sync the report's exact render tree; never guess for sealed gold."""

    storage = client or _storage_client_for_config(config)
    prefix = run_prefix_uri(config)
    renders_dir = _renders_dir_for_report(config, local_dir, heldout_report)
    renders_dir.mkdir(parents=True, exist_ok=True)

    if (heldout_report or {}).get("evaluation_split") == "gold_heldout":
        lineage = dict((heldout_report or {}).get("render_lineage") or {})
        canonical = str(lineage.get("renders_s3_uri") or "").strip()
        if not canonical:
            raise Sim2RealRerunRegenError(
                "sealed gold report has no exact render_lineage.renders_s3_uri"
            )
        if lineage.get("evaluation_split") != "gold_heldout":
            raise Sim2RealRerunRegenError(
                "sealed gold render lineage has the wrong evaluation split"
            )
    else:
        canonical = f"{prefix}eval/heldout/renders/"
    try:
        storage.download_directory(canonical, str(renders_dir))
    except (StorageError, OSError):
        pass
    if _has_camera_pngs(renders_dir):
        return True

    # Gold metrics may only be paired with the exact render prefix recorded by
    # that evaluation. Lexicographic component/BYO discovery is retained solely
    # for legacy validation reports where no sealed split is involved.
    if (heldout_report or {}).get("evaluation_split") == "gold_heldout":
        return False

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
            manifest_uri = (
                f"s3://{bucket}/{component_prefix}output/render-manifest.json"
            )
            manifest_path = renders_dir.parent / "render-manifest.sibling.json"
            if _download_if_exists(storage, manifest_uri, manifest_path):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                manifest = _render_manifest_from_png_tree(renders_dir)
            _write_report_render_manifest(config, local_dir, heldout_report, manifest)
            return True
    byo_root = f"{prefix}byo-eval/"
    bucket, _ = _parse_s3(byo_root)
    prefixes = sorted(_list_common_prefixes(storage, byo_root))
    for component_prefix in reversed(prefixes):
        sibling_renders = f"s3://{bucket}/{component_prefix}renders/"
        try:
            storage.download_directory(sibling_renders, str(renders_dir))
        except (StorageError, OSError):
            continue
        if _has_camera_pngs(renders_dir):
            manifest_uri = f"s3://{bucket}/{component_prefix}render-manifest.json"
            manifest_path = renders_dir.parent / "render-manifest.byo.json"
            if _download_if_exists(storage, manifest_uri, manifest_path):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                manifest = _render_manifest_from_png_tree(renders_dir)
            _write_report_render_manifest(config, local_dir, heldout_report, manifest)
            return True
    return _has_camera_pngs(renders_dir)


def _has_camera_pngs(renders_dir: Path) -> bool:
    return any(renders_dir.rglob("camera-*.png"))


def _latest_inner_evidence_rel(client: StorageClient, prefix_uri: str) -> str:
    """Return the latest inner_loop/outer-*/evidence.json relative to the run prefix."""

    default = "inner_loop/outer-01/evidence.json"
    _bucket, run_prefix = _parse_s3(prefix_uri)
    if run_prefix and not run_prefix.endswith("/"):
        run_prefix += "/"
    candidates: list[tuple[int, str]] = []
    for outer_prefix in _list_common_prefixes(
        client, f"{prefix_uri.rstrip('/')}/inner_loop/"
    ):
        outer_name = Path(outer_prefix.rstrip("/")).name
        if not outer_name.startswith("outer-"):
            continue
        try:
            outer_index = int(outer_name.removeprefix("outer-"))
        except ValueError:
            continue
        key = f"{outer_prefix.rstrip('/')}/evidence.json"
        candidates.append(
            (outer_index, key[len(run_prefix) :] if key.startswith(run_prefix) else key)
        )
    if not candidates:
        return default
    return sorted(candidates)[-1][1]


def _latest_local_inner_evidence(local_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for evidence_path in sorted(
        (Path(local_dir) / "inner_loop").glob("outer-*/evidence.json")
    ):
        try:
            outer_index = int(evidence_path.parent.name.removeprefix("outer-"))
        except ValueError:
            continue
        candidates.append((outer_index, evidence_path))
    if candidates:
        return sorted(candidates)[-1][1]
    return Path(local_dir) / "inner_loop/outer-01/evidence.json"


def _rewrite_inner_evidence_paths(local_dir: Path, evidence_path: Path) -> None:
    try:
        payload = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    changed = False
    markers = {
        "actions_dir": "actions",
        "vlm_eval_dir": "vlm_eval",
        "signal_dir": "training_signal",
    }
    for record in payload.get("iterations") or []:
        if not isinstance(record, dict):
            continue
        for key, marker in markers.items():
            rewritten = _path_under_marker(local_dir, record.get(key), marker)
            if rewritten is not None and str(record.get(key) or "") != str(rewritten):
                record[key] = str(rewritten)
                changed = True
    if changed:
        Path(evidence_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _path_under_marker(local_dir: Path, value: Any, marker: str) -> Path | None:
    if not value:
        return None
    parts = Path(str(value)).parts
    try:
        index = parts.index(marker)
    except ValueError:
        return None
    return Path(local_dir) / Path(*parts[index:])


def _render_manifest_from_png_tree(renders_dir: Path) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for env_dir in sorted(
        path for path in Path(renders_dir).iterdir() if path.is_dir()
    ):
        frames = [path.name for path in sorted(env_dir.glob("camera-*.png"))]
        if frames:
            episodes.append({"env_id": env_dir.name, "frames": frames})
    return {"schema": "npa.sim2real.heldout_renders.v1", "episodes": episodes}


def _write_report_render_manifest(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    heldout_report: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> None:
    report_path = _gold_report_path(config, Path(local_dir))
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = dict(heldout_report or {})
    report["render_manifest"] = manifest
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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

    report_path = _gold_report_path(config, local_dir)
    if report_path.is_file():
        rel = report_path.relative_to(local_dir).as_posix()
        storage.upload_file(str(report_path), f"{prefix}{rel}")

    renders_dir = _renders_dir_for_report(
        config,
        local_dir,
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {},
    )
    if renders_dir.is_dir() and _has_camera_pngs(renders_dir):
        rel = renders_dir.relative_to(local_dir).as_posix()
        storage.upload_directory(str(renders_dir), f"{prefix}{rel}")

    rrd_path = local_dir / "reports" / "sim2real.rrd"
    if not rrd_path.is_file():
        raise Sim2RealRerunRegenError(f"missing regenerated recording: {rrd_path}")
    visual_index_path = local_dir / "reports" / "sim2real-visual-index.json"
    if visual_index_path.is_file():
        storage.upload_file(
            str(visual_index_path), f"{prefix}reports/sim2real-visual-index.json"
        )
    final_report_path = local_dir / "reports" / "sim2real-report.json"
    if final_report_path.is_file():
        storage.upload_file(
            str(final_report_path), f"{prefix}reports/sim2real-report.json"
        )
    candidate_path = local_dir / "checkpoints" / "candidate" / "candidate.json"
    if candidate_path.is_file():
        storage.upload_file(
            str(candidate_path), f"{prefix}checkpoints/candidate/candidate.json"
        )
    upload_uri = storage.upload_file(str(rrd_path), f"{prefix}reports/sim2real.rrd")
    return upload_uri


def publish_regen_mcap(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    client: StorageClient | None = None,
) -> str:
    """Upload the regenerated ``reports/sim2real.mcap``, if one was emitted."""

    mcap_path = Path(local_dir) / "reports" / "sim2real.mcap"
    if not mcap_path.is_file() or mcap_path.stat().st_size == 0:
        return ""
    storage = client or _storage_client_for_config(config)
    prefix = run_prefix_uri(config)
    return storage.upload_file(str(mcap_path), f"{prefix}reports/sim2real.mcap")


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

    work_dir = (
        Path(local_dir)
        if local_dir is not None
        else default_regen_local_dir(config.run_id)
    )
    output_rrd = (
        Path(local_rrd_path)
        if local_rrd_path is not None
        else resolve_local_rrd_path(config.run_id, local_dir=work_dir)
    )
    storage = client or _storage_client_for_config(config)

    if sync_inputs:
        sync_regen_inputs(config, work_dir, client=storage)

    inner_path = _latest_local_inner_evidence(work_dir)
    _rewrite_inner_evidence_paths(work_dir, inner_path)
    heldout_path = _gold_report_path(config, work_dir)
    if not inner_path.is_file():
        raise Sim2RealRerunRegenError(f"missing inner evidence: {inner_path}")
    if not heldout_path.is_file():
        raise Sim2RealRerunRegenError(f"missing held-out report: {heldout_path}")

    inner_evidence = json.loads(inner_path.read_text(encoding="utf-8"))
    heldout_report = json.loads(heldout_path.read_text(encoding="utf-8"))
    report_path = work_dir / "reports" / "sim2real-report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {}
    )
    policy_access = _ensure_policy_access_metadata(
        config, work_dir, storage=storage, report=report
    )
    outer_history = list((report.get("outer_loop") or {}).get("history") or [])
    viewer_command = (
        "npa workbench sim2real rerun serve "
        f"--run-id {config.run_id} --s3-bucket {config.s3_bucket} "
        f"--s3-prefix {config.s3_prefix}"
    )
    rerun_started = time.monotonic()
    result = emit_sim2real_rerun(
        local_dir=work_dir,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        stage_components=list(report.get("components") or []),
        outer_history=outer_history,
        run_metadata={
            "run_id": config.run_id,
            "artifact_root": run_prefix_uri(config),
            "rrd_s3_uri": f"{run_prefix_uri(config)}reports/sim2real.rrd",
            "candidate_s3_uri": (
                f"{run_prefix_uri(config)}checkpoints/candidate/candidate.json"
            ),
            "policy_checkpoint": policy_access.get("checkpoint_uri", ""),
            "policy_checkpoint_identity": policy_access.get("identity", ""),
            "policy_checkpoint_sha256": policy_access.get("sha256", ""),
            "policy_checkpoint_size_bytes": policy_access.get("size_bytes", 0),
            "policy_download_command": policy_access.get(
                "authenticated_download_command", ""
            ),
            "policy_ui_action": policy_access.get("ui_action", ""),
            "policy_deployable": policy_access.get("deployable_policy", False),
            "orchestrator_job_name": config.run_id,
            "orchestrator_node_product": config.k8s_gpu_product,
            "viewer_command": viewer_command,
        },
        output_rrd=output_rrd,
    )
    rerun_duration_s = round(time.monotonic() - rerun_started, 3)
    if report:
        from npa.workflows.sim2real.engine import gpu_fallback_report_contract

        report["policy_access"] = policy_access
        report["progress_metrics"] = build_progress_metrics(work_dir, outer_history)
        report["gpu_fallback_contract"] = gpu_fallback_report_contract(
            config, list(report.get("components") or [])
        )
        report["visualization"] = {
            **result.to_dict(),
            "rrd_s3_uri": f"{run_prefix_uri(config)}reports/sim2real.rrd",
            "rrd_size_bytes": output_rrd.stat().st_size,
            "viewer_command": viewer_command,
        }
        for component in report.get("components") or []:
            if component.get("name") == "stage_14_rerun_viz":
                component["tier"] = "WORKS"
                component["evidence"] = (
                    f"Wrote the complete Rerun recording from every persisted pass: "
                    f"{result.rollout_count} real policy rollout(s), "
                    f"{result.frame_count} synchronized policy camera frame(s), and "
                    f"{result.heldout_frame_count} held-out frame(s)."
                )
                component.setdefault("artifacts", {}).update(
                    {
                        "rrd": f"{run_prefix_uri(config)}reports/sim2real.rrd",
                        "rrd_local": str(output_rrd),
                        "rrd_size_bytes": output_rrd.stat().st_size,
                        "duration_s": rerun_duration_s,
                    }
                )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    # The finalize stage emits both viewer recordings from the same inputs, so regen
    # must too: refreshing only the .rrd leaves the run's MCAP frozen at whatever
    # the emitter produced when the run first completed. Best-effort, exactly as in
    # finalize, so a missing mcap writer can never fail a Rerun regen.
    mcap_result = emit_sim2real_mcap_if_enabled(
        local_dir=work_dir,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        output_mcap=work_dir / "reports" / "sim2real.mcap",
    )
    upload_uri = ""
    mcap_upload_uri = ""
    if upload:
        upload_uri = publish_regen_outputs(config, work_dir, client=storage)
        mcap_upload_uri = publish_regen_mcap(config, work_dir, client=storage)
    return _regen_result_from_viz(
        config.run_id,
        work_dir,
        result,
        upload_uri=upload_uri,
        mcap_result=mcap_result,
        mcap_upload_uri=mcap_upload_uri,
    )


def _ensure_policy_access_metadata(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    storage: StorageClient,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Resolve candidate bytes and write secret-free access and promotion state."""

    candidate_path = Path(local_dir) / "checkpoints" / "candidate" / "candidate.json"
    candidate = (
        json.loads(candidate_path.read_text(encoding="utf-8"))
        if candidate_path.is_file()
        else {}
    )
    checkpoint_uri = str(
        candidate.get("policy_checkpoint_uri")
        or ((report.get("outer_loop") or {}).get("latest_decision") or {}).get(
            "checkpoint_uri"
        )
        or ""
    ).strip()
    deployable = bool(
        candidate.get("deployable_policy")
        and checkpoint_uri.startswith("s3://")
        and checkpoint_uri.endswith(".pt")
    )
    bytes_available = bool(
        (candidate.get("policy_bytes_available") or deployable)
        and checkpoint_uri.startswith("s3://")
        and checkpoint_uri.endswith(".pt")
    )
    if bytes_available and (
        not candidate.get("policy_checkpoint_sha256")
        or not candidate.get("policy_checkpoint_size_bytes")
    ):
        with tempfile.TemporaryDirectory(prefix="npa-policy-regen-") as temporary:
            local_checkpoint = Path(temporary) / Path(checkpoint_uri).name
            storage.download_file(checkpoint_uri, str(local_checkpoint))
            candidate.update(
                {
                    "policy_checkpoint_identity": Path(checkpoint_uri).name,
                    "policy_checkpoint_sha256": hashlib.sha256(
                        local_checkpoint.read_bytes()
                    ).hexdigest(),
                    "policy_checkpoint_size_bytes": local_checkpoint.stat().st_size,
                }
            )
    if bytes_available:
        candidate.update(
            {
                "policy_download_command": (
                    "aws s3 cp "
                    f"{shlex.quote(checkpoint_uri)} ./model.pt "
                    '--endpoint-url "$AWS_ENDPOINT_URL"'
                ),
                "policy_ui_action": (
                    "Open Artifacts for this run, select the candidate .pt checkpoint, "
                    "and choose Download. Check deployable_policy before deployment; "
                    "the Rerun viewer links the weights but does not execute them."
                ),
            }
        )
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return {
        "deployable_policy": deployable,
        "policy_bytes_available": bytes_available,
        "identity": candidate.get("policy_checkpoint_identity", ""),
        "sha256": candidate.get("policy_checkpoint_sha256", ""),
        "size_bytes": candidate.get("policy_checkpoint_size_bytes", 0),
        "checkpoint_uri": checkpoint_uri if bytes_available else "",
        "candidate_manifest_uri": (
            f"{run_prefix_uri(config)}checkpoints/candidate/candidate.json"
        ),
        "authenticated_download_command": candidate.get("policy_download_command", ""),
        "ui_action": candidate.get("policy_ui_action", ""),
        "viewer_executes_policy": False,
    }


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

    work_dir = (
        Path(local_dir)
        if local_dir is not None
        else default_regen_local_dir(config.run_id)
    )
    storage = client or _storage_client_for_config(config)
    prefix = run_prefix_uri(config)

    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        storage.download_directory(
            f"{prefix}envs/heldout/", str(work_dir / "envs" / "heldout")
        )
    except (StorageError, OSError) as exc:
        raise Sim2RealRerunRegenError(
            f"failed to sync envs/heldout for {config.run_id}: {exc}"
        ) from exc

    inner_path = work_dir / "inner_loop/outer-01/evidence.json"
    inner_path.parent.mkdir(parents=True, exist_ok=True)
    if not _download_if_exists(
        storage, f"{prefix}inner_loop/outer-01/evidence.json", inner_path
    ):
        raise Sim2RealRerunRegenError(
            f"missing inner evidence at {prefix}inner_loop/outer-01/evidence.json"
        )

    inner_evidence = json.loads(inner_path.read_text(encoding="utf-8"))
    report = run_heldout_eval(
        config,
        local_dir=work_dir,
        inner_evidence=inner_evidence,
        outer_iteration=outer_iteration,
    )

    invocation = report.get("component_invocation") or {}
    output_uri = str(invocation.get("output_uri") or "").strip()
    renders_dir = _renders_dir_for_report(config, work_dir, report)
    renders_dir.mkdir(parents=True, exist_ok=True)
    if output_uri:
        try:
            storage.download_directory(
                _sibling_uri(output_uri, "renders/"), str(renders_dir)
            )
        except (StorageError, OSError):
            sync_heldout_renders(
                config, work_dir, heldout_report=report, client=storage
            )
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
    mcap_result: dict[str, Any] | None = None,
    mcap_upload_uri: str = "",
) -> RegenResult:
    if result.heldout_frame_count <= 0:
        raise Sim2RealRerunRegenError(
            "regenerated .rrd has heldout_frame_count=0; sync eval/heldout/renders or rerun held-out eval"
        )
    mcap = mcap_result or {}
    return RegenResult(
        run_id=run_id,
        local_dir=str(local_dir),
        local_rrd_path=result.output_rrd_path,
        upload_uri=upload_uri,
        heldout_frame_count=result.heldout_frame_count,
        rollout_count=result.rollout_count,
        frame_count=result.frame_count,
        local_mcap_path=str(mcap.get("output_mcap_path") or ""),
        mcap_upload_uri=mcap_upload_uri,
        mcap_status=str(mcap.get("status") or ""),
        synthetic_frame_count=int(getattr(result, "synthetic_frame_count", 0)),
    )
