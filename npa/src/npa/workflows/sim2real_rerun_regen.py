"""Regenerate Sim2Real viewer recordings and optionally re-run held-out Isaac capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from npa.clients.storage import StorageClient, StorageError
from npa.workflows.sim2real.checkpoint_selection import (
    CHECKPOINT_DIGEST_ALIASES,
    CHECKPOINT_SIZE_ALIASES,
    CHECKPOINT_URI_ALIASES,
    GENERATOR_DIGEST_ALIASES,
    resolve_selected_checkpoint,
    resolve_run_scoped_checkpoint,
)
from npa.workflows.sim2real.component_authority import (
    stage14_report_authority_sha256,
    validate_component_records,
    validate_remote_stage4_authority,
    validate_stage14_component_record,
)
from npa.workflows.sim2real.constants import SCHEMA_E2E_REPORT
from npa.workflows.sim2real.decision_authority import validate_stage11_decision
from npa.workflows.sim2real.hashing import sha256_file
from npa.workflows.sim2real.stage10_execution import (
    validate_materialized_render_tree,
)
from npa.workflows.sim2real.stage10_authority import (
    expected_byo_render_prefix,
    validate_stage10_input_scope,
)
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.publication import (
    MutablePublicationTransaction,
    RemoteObjectSnapshot,
    remote_object_snapshot,
    upload_immutable_file,
    upload_immutable_tree,
)
from npa.workflows.sim2real.reporting import build_progress_metrics
from npa.workflows.sim2real.utils import _artifact_root_uri, _write_json_artifact
from npa.workflows.sim2real.viz_contract import (
    _checkpoint_uri,
    selected_checkpoint_policy_metadata,
)
from npa.workflows.sim2real.workflow_io import (
    component_record_history_uri,
    parse_json_object,
)
from npa.workflows.sim2real_viz import (
    Sim2RealVizResult,
    _heldout_policy_metadata,
    emit_sim2real_mcap_if_enabled,
    emit_sim2real_rerun,
)

_LOGGER = logging.getLogger(__name__)


class Sim2RealRerunRegenError(ValueError):
    """Raised when regen sync, held-out rerun, or .rrd emission fails."""


DEFAULT_REGEN_ROOT = Path("/tmp/sim2real-regen")


def _safe_run_id(run_id: str) -> str:
    if not isinstance(run_id, str):
        raise Sim2RealRerunRegenError("run_id must be a string path segment")
    value = run_id
    if (
        not value
        or value != value.strip()
        or not value.isascii()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise Sim2RealRerunRegenError("run_id must be one non-traversing path segment")
    return value


def _default_regen_dir(run_id: str) -> Path:
    safe_run_id = _safe_run_id(run_id)
    if DEFAULT_REGEN_ROOT.is_symlink():
        raise Sim2RealRerunRegenError(
            f"default regeneration root must not be a symlink: {DEFAULT_REGEN_ROOT}"
        )
    return DEFAULT_REGEN_ROOT / safe_run_id


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
    return DEFAULT_REGEN_ROOT / _safe_run_id(run_id) / "reports" / "sim2real.rrd"


def default_regen_local_dir(run_id: str, *, override: str = "") -> Path:
    explicit = (override or os.environ.get("NPA_SIM2REAL_REGEN_LOCAL_DIR", "")).strip()
    if explicit:
        return Path(explicit)
    return _default_regen_dir(run_id)


def run_prefix_uri(config: Sim2RealLoopConfig) -> str:
    return f"{_artifact_root_uri(config).rstrip('/')}/"


def _canonical_outer_index(name: str) -> int | None:
    if not name.startswith("outer-"):
        return None
    try:
        index = int(name.removeprefix("outer-"))
    except ValueError:
        return None
    if index <= 0 or name != f"outer-{index:02d}":
        return None
    return index


def _gold_eval_relative_dir(inner_evidence: str | Path) -> Path:
    outer_name = Path(inner_evidence).parent.name
    outer_iteration = _canonical_outer_index(outer_name)
    if outer_iteration is None:
        raise Sim2RealRerunRegenError(
            f"inner evidence has no canonical outer iteration: {inner_evidence}"
        )
    return Path("eval") / "gold-heldout" / outer_name


def _completed_local_outer_pair(local_dir: Path) -> tuple[Path, Path] | None:
    pairs: list[tuple[int, Path, Path]] = []
    for evidence_path in (Path(local_dir) / "inner_loop").glob("outer-*/evidence.json"):
        outer_name = evidence_path.parent.name
        outer_index = _canonical_outer_index(outer_name)
        if outer_index is None:
            continue
        report_path = (
            Path(local_dir) / "eval" / "gold-heldout" / outer_name / "report.json"
        )
        if report_path.is_file():
            pairs.append((outer_index, evidence_path, report_path))
    if not pairs:
        return None
    _outer, evidence, report = max(pairs, key=lambda item: item[0])
    return evidence, report


def _gold_report_path(_config: Sim2RealLoopConfig, local_dir: Path) -> Path:
    completed = _completed_local_outer_pair(local_dir)
    if completed is not None:
        return completed[1]
    inner_evidence = _latest_local_inner_evidence(local_dir)
    canonical = (
        Path(local_dir) / _gold_eval_relative_dir(inner_evidence) / "report.json"
    )
    legacy = Path(local_dir) / "eval" / "heldout" / "report.json"
    return canonical if canonical.is_file() or not legacy.is_file() else legacy


def _selected_outer_iteration(
    inner_path: Path,
    inner_evidence: dict[str, Any],
) -> int:
    outer_iteration = _canonical_outer_index(inner_path.parent.name)
    if outer_iteration is None:
        raise Sim2RealRerunRegenError(
            "selected inner evidence path has no canonical outer iteration"
        )
    recorded_inner = inner_evidence.get("outer_iteration")
    if (
        not isinstance(recorded_inner, int)
        or isinstance(recorded_inner, bool)
        or recorded_inner != outer_iteration
    ):
        raise Sim2RealRerunRegenError(
            "selected inner evidence payload disagrees with its outer path"
        )
    return outer_iteration


def _assert_heldout_outer_iteration(
    heldout_path: Path,
    heldout_report: dict[str, Any],
    outer_iteration: int,
) -> None:
    heldout_outer = _canonical_outer_index(heldout_path.parent.name)
    recorded_heldout = heldout_report.get("outer_iteration")
    if heldout_outer is None:
        if recorded_heldout is not None and (
            not isinstance(recorded_heldout, int)
            or isinstance(recorded_heldout, bool)
            or recorded_heldout != outer_iteration
        ):
            raise Sim2RealRerunRegenError(
                "legacy held-out report disagrees with selected outer iteration"
            )
        return
    if (
        heldout_outer != outer_iteration
        or not isinstance(recorded_heldout, int)
        or isinstance(recorded_heldout, bool)
        or recorded_heldout != outer_iteration
    ):
        raise Sim2RealRerunRegenError(
            "selected held-out report payload disagrees with its outer path"
        )


def _assert_selected_pair_outer_iteration(
    inner_path: Path,
    heldout_path: Path,
    inner_evidence: dict[str, Any],
    heldout_report: dict[str, Any],
) -> None:
    outer_iteration = _selected_outer_iteration(inner_path, inner_evidence)
    _assert_heldout_outer_iteration(heldout_path, heldout_report, outer_iteration)


def _contained_render_path(
    root: Path,
    value: object,
    *,
    source: str,
    require_relative: bool = False,
) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Sim2RealRerunRegenError(f"{source} render path is malformed")
    candidate = Path(value)
    if require_relative and candidate.is_absolute():
        raise Sim2RealRerunRegenError(
            f"{source} render path must be relative and contained"
        )
    if ".." in candidate.parts:
        raise Sim2RealRerunRegenError(
            f"{source} render path must be contained without parent traversal"
        )
    if not candidate.is_absolute():
        candidate = root / candidate
    normalized = Path(os.path.abspath(candidate))
    if normalized == root or not normalized.is_relative_to(root):
        raise Sim2RealRerunRegenError(
            f"{source} render path must be below regeneration root"
        )
    return normalized


def _sealed_render_default(
    root: Path,
    report: dict[str, Any],
    lineage: dict[str, Any],
) -> Path | None:
    report_split = report.get("evaluation_split")
    lineage_split = lineage.get("evaluation_split")
    for source, split in (
        ("held-out render report", report_split),
        ("held-out render lineage", lineage_split),
    ):
        if split not in (None, "", "gold_heldout"):
            raise Sim2RealRerunRegenError(f"{source} has the wrong evaluation split")
    sealed = report_split == "gold_heldout" or lineage_split == "gold_heldout"
    if not sealed:
        return None
    raw_outer = report.get("outer_iteration")
    if type(raw_outer) is not int or raw_outer <= 0:
        raise Sim2RealRerunRegenError(
            "sealed gold report has an invalid outer iteration"
        )
    return _contained_render_path(
        root,
        f"eval/gold-heldout/outer-{raw_outer:02d}/renders",
        source="default held-out",
        require_relative=True,
    )


def _reported_render_paths(
    root: Path,
    report: dict[str, Any],
    lineage: dict[str, Any],
) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    if report.get("local_renders_dir") not in (None, ""):
        candidates.append(
            (
                "local_renders_dir",
                _contained_render_path(
                    root,
                    report["local_renders_dir"],
                    source="local_renders_dir",
                ),
            )
        )
    if lineage.get("local_relative_dir") not in (None, ""):
        candidates.append(
            (
                "render_lineage.local_relative_dir",
                _contained_render_path(
                    root,
                    lineage["local_relative_dir"],
                    source="render_lineage.local_relative_dir",
                    require_relative=True,
                ),
            )
        )
    return candidates


def _renders_dir_for_report(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    heldout_report: dict[str, Any] | None,
) -> Path:
    report = heldout_report or {}
    raw_lineage = report.get("render_lineage")
    if raw_lineage is not None and not isinstance(raw_lineage, dict):
        raise Sim2RealRerunRegenError("render_lineage must be an object")
    lineage = raw_lineage or {}
    root = Path(os.path.abspath(local_dir))
    sealed_default = _sealed_render_default(root, report, lineage)
    candidates = _reported_render_paths(root, report, lineage)
    if len({path for _source, path in candidates}) > 1:
        raise Sim2RealRerunRegenError("render path sources disagree")
    if candidates:
        if sealed_default is not None and candidates[0][1] != sealed_default:
            raise Sim2RealRerunRegenError(
                "sealed gold render path disagrees with its outer iteration"
            )
        return candidates[0][1]
    if sealed_default is not None:
        return sealed_default
    return _contained_render_path(
        root,
        "eval/heldout/renders",
        source="default held-out",
        require_relative=True,
    )


def _assert_safe_render_destination(local_dir: Path, renders_dir: Path) -> None:
    root = Path(os.path.abspath(local_dir))
    target = Path(os.path.abspath(renders_dir))
    if target == root:
        raise Sim2RealRerunRegenError(
            "render destination must be below regeneration root"
        )
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise Sim2RealRerunRegenError(
            "render destination must be contained in regeneration root"
        ) from exc
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise Sim2RealRerunRegenError(
                f"render destination has a symlinked ancestor: {current}"
            )


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


def _s3_object_exists(client: StorageClient, bucket: str, key: str) -> bool:
    try:
        response = client._s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return int(response.get("ContentLength") or 0) > 0


def _parse_s3(uri: str) -> tuple[str, str]:
    from npa.clients.storage import _parse_bucket_uri

    return _parse_bucket_uri(uri)


def _assert_no_symlinked_ancestors(
    path: Path,
    *,
    containment_root: Path,
) -> None:
    root = Path(os.path.abspath(containment_root))
    absolute = Path(os.path.abspath(path))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise Sim2RealRerunRegenError(
            "download destination is outside its containment root"
        ) from exc
    if relative == Path("."):
        raise Sim2RealRerunRegenError(
            "download destination must be below its containment root"
        )
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise Sim2RealRerunRegenError(
                f"download destination has a symlinked ancestor: {current}"
            )


def _discard_download_destination(path: Path, containment_root: Path) -> bool:
    _assert_no_symlinked_ancestors(path, containment_root=containment_root)
    path.unlink(missing_ok=True)
    return False


def _download_if_exists(
    client: StorageClient,
    uri: str,
    local_path: Path,
    *,
    containment_root: Path,
) -> bool:
    _assert_no_symlinked_ancestors(
        local_path,
        containment_root=containment_root,
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlinked_ancestors(
        local_path,
        containment_root=containment_root,
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{local_path.name}.", dir=local_path.parent
    ) as directory:
        staged = Path(directory) / local_path.name
        try:
            client.download_path(uri, str(staged))
        except (StorageError, OSError):
            return _discard_download_destination(local_path, containment_root)
        if not staged.is_file() or staged.stat().st_size <= 0:
            return _discard_download_destination(local_path, containment_root)
        _assert_no_symlinked_ancestors(
            local_path,
            containment_root=containment_root,
        )
        os.replace(staged, local_path)
    return True


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _remove_tree_after_failure(path: Path, failure: BaseException) -> None:
    try:
        _remove_tree(path)
    except OSError as cleanup_error:
        raise failure from cleanup_error


def _download_render_tree(
    storage: StorageClient,
    uri: str,
    renders_dir: Path,
    *,
    containment_root: Path,
) -> bool:
    _assert_safe_render_destination(containment_root, renders_dir)
    renders_dir.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_render_destination(containment_root, renders_dir)
    with tempfile.TemporaryDirectory(
        prefix=f".{renders_dir.name}.", dir=renders_dir.parent
    ) as directory:
        staged = Path(directory) / "renders"
        try:
            storage.download_directory(uri, str(staged))
        except (StorageError, OSError, ClientError) as exc:
            _assert_safe_render_destination(containment_root, renders_dir)
            _remove_tree_after_failure(renders_dir, exc)
            return False
        except BaseException as exc:
            _assert_safe_render_destination(containment_root, renders_dir)
            _remove_tree_after_failure(renders_dir, exc)
            raise
        if not _has_camera_pngs(staged):
            _assert_safe_render_destination(containment_root, renders_dir)
            _remove_tree(renders_dir)
            return False
        _assert_safe_render_destination(containment_root, renders_dir)
        _remove_tree(renders_dir)
        os.replace(staged, renders_dir)
    return True


def _download_directory_fresh(
    storage: StorageClient,
    uri: str,
    destination: Path,
    *,
    containment_root: Path,
) -> bool:
    """Replace a cached directory only with one complete remote download."""

    _assert_no_symlinked_ancestors(
        destination,
        containment_root=containment_root,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as directory:
        staged = Path(directory) / destination.name
        try:
            storage.download_directory(uri, str(staged))
        except (StorageError, OSError, ClientError) as exc:
            _remove_tree_after_failure(destination, exc)
            return False
        except BaseException as exc:
            _remove_tree_after_failure(destination, exc)
            raise
        if not staged.is_dir() or staged.is_symlink():
            _remove_tree(destination)
            return False
        _assert_no_symlinked_ancestors(
            destination,
            containment_root=containment_root,
        )
        _remove_tree(destination)
        os.replace(staged, destination)
    return True


def _clear_regen_pair_scopes(local_dir: Path) -> None:
    for stale_scope in (
        local_dir / "inner_loop",
        local_dir / "eval" / "gold-heldout",
    ):
        _remove_tree(stale_scope)


def _regen_single_files(
    local_dir: Path,
    inner_evidence_rel: str,
    gold_eval_rel: str,
) -> dict[str, Path]:
    return {
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


def _download_regen_single_files(
    storage: StorageClient,
    prefix: str,
    local_dir: Path,
    singles: dict[str, Path],
) -> dict[str, bool]:
    downloaded: dict[str, bool] = {}
    for rel, dest in singles.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        downloaded[rel] = _download_if_exists(
            storage,
            f"{prefix}{rel}",
            dest,
            containment_root=local_dir,
        )
    return downloaded


def _require_regen_pair_downloads(
    downloaded: dict[str, bool],
    inner_evidence_rel: str,
    gold_eval_rel: str,
) -> None:
    for required in (inner_evidence_rel, f"{gold_eval_rel}/report.json"):
        if not downloaded[required]:
            raise Sim2RealRerunRegenError(
                f"required completed run artifact could not be downloaded: {required}"
            )


def _sync_inner_evidence_history(
    storage: StorageClient,
    prefix: str,
    local_dir: Path,
) -> None:
    for outer_prefix in _list_common_prefixes(
        storage, f"{prefix.rstrip('/')}/inner_loop/"
    ):
        outer_name = Path(outer_prefix.rstrip("/")).name
        if _canonical_outer_index(outer_name) is None:
            continue
        destination = local_dir / "inner_loop" / outer_name / "evidence.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        bucket, _run_key = _parse_s3(prefix)
        _download_if_exists(
            storage,
            f"s3://{bucket}/{outer_prefix.rstrip('/')}/evidence.json",
            destination,
            containment_root=local_dir,
        )


def _sync_regen_directories(
    storage: StorageClient,
    prefix: str,
    local_dir: Path,
) -> None:
    for rel in ("actions", "vlm_eval", "training_signal", "augment", "envs/raw"):
        destination = local_dir / rel
        had_cached_evidence = destination.exists() or destination.is_symlink()
        if (
            not _download_directory_fresh(
                storage,
                f"{prefix}{rel}/",
                destination,
                containment_root=local_dir,
            )
            and had_cached_evidence
        ):
            raise Sim2RealRerunRegenError(
                f"failed to replace cached regeneration input directory: {rel}"
            )


def _rewrite_synced_inner_evidence(local_dir: Path, prefix: str) -> None:
    for evidence_path in sorted(
        (local_dir / "inner_loop").glob("outer-*/evidence.json")
    ):
        _rewrite_inner_evidence_paths(
            local_dir,
            evidence_path,
            expected_artifact_root=prefix.rstrip("/"),
        )


def _sync_selected_heldout_renders(
    config: Sim2RealLoopConfig,
    storage: StorageClient,
    local_dir: Path,
    inner_evidence_rel: str,
    gold_eval_rel: str,
) -> tuple[Path, Path]:
    inner_path = local_dir / inner_evidence_rel
    heldout_path = local_dir / gold_eval_rel / "report.json"
    heldout_report = _read_retained_json(heldout_path, source="held-out report")
    sync_heldout_renders(
        config, local_dir, heldout_report=heldout_report, client=storage
    )
    return inner_path, heldout_path


def sync_regen_inputs(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    client: StorageClient | None = None,
) -> tuple[Path, Path]:
    """Download artifacts required for Rerun regeneration.

    Args:
        config: Run configuration identifying the durable artifact prefix.
        local_dir: Run-scoped local regeneration directory.
        client: Optional configured storage client.
    Returns:
        Selected inner-evidence and held-out-report paths.
    Raises:
        Sim2RealRerunRegenError: If required durable inputs cannot be synchronized.
    """
    storage = client or _storage_client_for_config(config)
    prefix = run_prefix_uri(config)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    inner_evidence_rel = _latest_completed_inner_evidence_rel(storage, prefix)
    gold_eval_rel = _gold_eval_relative_dir(inner_evidence_rel).as_posix()
    _clear_regen_pair_scopes(local_dir)
    downloaded = _download_regen_single_files(
        storage,
        prefix,
        local_dir,
        _regen_single_files(local_dir, inner_evidence_rel, gold_eval_rel),
    )
    _require_regen_pair_downloads(downloaded, inner_evidence_rel, gold_eval_rel)
    _sync_inner_evidence_history(storage, prefix, local_dir)
    _sync_regen_directories(storage, prefix, local_dir)
    _rewrite_synced_inner_evidence(local_dir, prefix)
    return _sync_selected_heldout_renders(
        config, storage, local_dir, inner_evidence_rel, gold_eval_rel
    )


def _sealed_render_uri(lineage: dict[str, Any]) -> str:
    evidence = [
        (key, lineage[key])
        for key in ("canonical_s3_uri", "renders_s3_uri")
        if key in lineage
    ]
    values: list[str] = []
    for key, value in evidence:
        if (
            not isinstance(value, str)
            or value != value.strip()
            or not value.startswith("s3://")
            or not value.endswith("/")
        ):
            raise Sim2RealRerunRegenError(
                f"sealed gold render lineage {key} is malformed"
            )
        values.append(value)
    if not values:
        raise Sim2RealRerunRegenError(
            "sealed gold report has no exact render_lineage URI"
        )
    if len(set(values)) != 1:
        raise Sim2RealRerunRegenError("sealed gold render lineage URI sources disagree")
    return values[0]


def _heldout_render_lineage(
    heldout_report: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    raw_lineage = heldout_report.get("render_lineage")
    if raw_lineage is not None and not isinstance(raw_lineage, dict):
        raise Sim2RealRerunRegenError("render_lineage must be an object")
    lineage = raw_lineage or {}
    report_split = heldout_report.get("evaluation_split")
    lineage_split = lineage.get("evaluation_split")
    if report_split not in (None, "", "gold_heldout"):
        raise Sim2RealRerunRegenError(
            "held-out render report has the wrong evaluation split"
        )
    if lineage_split not in (None, "", "gold_heldout"):
        raise Sim2RealRerunRegenError(
            "held-out render lineage has the wrong evaluation split"
        )
    if report_split not in (None, "", lineage_split) and lineage_split not in (
        None,
        "",
    ):
        raise Sim2RealRerunRegenError(
            "sealed gold render lineage has the wrong evaluation split: "
            "report and lineage disagree"
        )
    sealed = report_split == "gold_heldout" or lineage_split == "gold_heldout"
    return lineage, sealed


def _sealed_render_outer(heldout_report: dict[str, Any]) -> int:
    outer = heldout_report.get("outer_iteration")
    if not isinstance(outer, int) or isinstance(outer, bool) or outer <= 0:
        raise Sim2RealRerunRegenError(
            "sealed gold report has an invalid outer iteration"
        )
    return outer


def _sealed_render_attempt(lineage: dict[str, Any], outer: int) -> str:
    attempt_tag = lineage.get("evaluation_attempt_tag")
    allowed_attempt_prefixes = (
        f"gold-o{outer:02d}-attempt-",
        f"gold_heldout-outer-{outer:02d}-attempt-",
    )
    if attempt_tag not in (None, "") and (
        not isinstance(attempt_tag, str)
        or not any(
            attempt_tag.startswith(candidate)
            and len(attempt_tag) == len(candidate) + 32
            for candidate in allowed_attempt_prefixes
        )
        or any(char not in "0123456789abcdef" for char in attempt_tag[-32:])
    ):
        raise Sim2RealRerunRegenError(
            "sealed gold render lineage attempt identity is invalid"
        )
    return str(attempt_tag or "")


def _allowed_sealed_render_uris(
    prefix: str,
    lineage: dict[str, Any],
    outer: int,
    attempt_tag: str,
) -> set[str]:
    expected_uri = (
        f"{prefix}eval/gold-heldout/outer-{outer:02d}/attempts/{attempt_tag}/renders/"
        if attempt_tag
        else f"{prefix}eval/gold-heldout/outer-{outer:02d}/renders/"
    )
    expected_legacy_uri = (
        f"{prefix}component-io/heldout-eval/"
        f"gold_heldout-outer-{outer:02d}/output/renders/"
    )
    has_canonical = lineage.get("canonical_s3_uri") not in (None, "")
    return {expected_uri} if has_canonical else {expected_uri, expected_legacy_uri}


def _heldout_render_source(
    config: Sim2RealLoopConfig,
    heldout_report: dict[str, Any],
) -> tuple[str, bool]:
    prefix = run_prefix_uri(config)
    lineage, sealed = _heldout_render_lineage(heldout_report)
    if not sealed:
        return f"{prefix}eval/heldout/renders/", False
    render_uri = _sealed_render_uri(lineage)
    outer = _sealed_render_outer(heldout_report)
    attempt_tag = _sealed_render_attempt(lineage, outer)
    allowed_uris = _allowed_sealed_render_uris(prefix, lineage, outer, attempt_tag)
    if render_uri not in allowed_uris:
        raise Sim2RealRerunRegenError(
            "sealed gold render lineage disagrees with the configured run and iteration"
        )
    return render_uri, True


def _sync_legacy_render_source(
    storage: StorageClient,
    root: str,
    renders_dir: Path,
    *,
    render_suffix: str,
    manifest_suffix: str,
    manifest_name: str,
    config: Sim2RealLoopConfig,
    local_dir: Path,
    heldout_report: dict[str, Any],
) -> bool:
    bucket, _ = _parse_s3(root)
    for source_prefix in reversed(sorted(_list_common_prefixes(storage, root))):
        base_uri = f"s3://{bucket}/{source_prefix}"
        if not _download_render_tree(
            storage,
            f"{base_uri}{render_suffix}",
            renders_dir,
            containment_root=local_dir,
        ):
            continue
        manifest_path = renders_dir.parent / manifest_name
        if _download_if_exists(
            storage,
            f"{base_uri}{manifest_suffix}",
            manifest_path,
            containment_root=local_dir,
        ):
            manifest = parse_json_object(
                manifest_path.read_text(encoding="utf-8"),
                source=f"{base_uri}{manifest_suffix}",
            )
        else:
            manifest = _render_manifest_from_png_tree(renders_dir)
        _write_report_render_manifest(config, local_dir, heldout_report, manifest)
        return True
    return False


def _sync_legacy_heldout_renders(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    storage: StorageClient,
    renders_dir: Path,
    report: dict[str, Any],
) -> bool:
    prefix = run_prefix_uri(config)
    common = {
        "config": config,
        "local_dir": local_dir,
        "heldout_report": report,
    }
    return _sync_legacy_render_source(
        storage,
        f"{prefix}component-io/heldout-eval/",
        renders_dir,
        render_suffix="output/renders/",
        manifest_suffix="output/render-manifest.json",
        manifest_name="render-manifest.sibling.json",
        **common,
    ) or _sync_legacy_render_source(
        storage,
        f"{prefix}byo-eval/",
        renders_dir,
        render_suffix="renders/",
        manifest_suffix="render-manifest.json",
        manifest_name="render-manifest.byo.json",
        **common,
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
    report = heldout_report or {}
    canonical, sealed = _heldout_render_source(config, report)
    renders_dir = _renders_dir_for_report(config, local_dir, report)
    _assert_safe_render_destination(local_dir, renders_dir)
    if _download_render_tree(
        storage,
        canonical,
        renders_dir,
        containment_root=local_dir,
    ):
        return True
    if sealed:
        return False
    return _sync_legacy_heldout_renders(config, local_dir, storage, renders_dir, report)


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
        outer_index = _canonical_outer_index(outer_name)
        if outer_index is None:
            continue
        key = f"{outer_prefix.rstrip('/')}/evidence.json"
        candidates.append(
            (outer_index, key[len(run_prefix) :] if key.startswith(run_prefix) else key)
        )
    if not candidates:
        return default
    return sorted(candidates)[-1][1]


def _latest_completed_inner_evidence_rel(
    client: StorageClient,
    prefix_uri: str,
) -> str:
    bucket, run_prefix = _parse_s3(prefix_uri)
    if run_prefix and not run_prefix.endswith("/"):
        run_prefix += "/"

    def canonical_prefixes(category: str) -> dict[int, str]:
        result: dict[int, str] = {}
        for outer_prefix in _list_common_prefixes(
            client,
            f"{prefix_uri.rstrip('/')}/{category}/",
        ):
            outer_name = Path(outer_prefix.rstrip("/")).name
            outer_index = _canonical_outer_index(outer_name)
            if outer_index is not None:
                result[outer_index] = outer_prefix
        return result

    inner = canonical_prefixes("inner_loop")
    gold = canonical_prefixes("eval/gold-heldout")
    for outer_index in reversed(sorted(set(inner).intersection(gold))):
        evidence_key = f"{inner[outer_index].rstrip('/')}/evidence.json"
        report_key = f"{gold[outer_index].rstrip('/')}/report.json"
        if _s3_object_exists(client, bucket, evidence_key) and _s3_object_exists(
            client, bucket, report_key
        ):
            return (
                evidence_key[len(run_prefix) :]
                if evidence_key.startswith(run_prefix)
                else evidence_key
            )
    return _latest_inner_evidence_rel(client, prefix_uri)


def _latest_local_inner_evidence(local_dir: Path) -> Path:
    completed = _completed_local_outer_pair(local_dir)
    if completed is not None:
        return completed[0]
    candidates: list[tuple[int, Path]] = []
    for evidence_path in sorted(
        (Path(local_dir) / "inner_loop").glob("outer-*/evidence.json")
    ):
        outer_index = _canonical_outer_index(evidence_path.parent.name)
        if outer_index is None:
            continue
        candidates.append((outer_index, evidence_path))
    if candidates:
        return sorted(candidates)[-1][1]
    return Path(local_dir) / "inner_loop/outer-01/evidence.json"


def _read_rewrite_payload(
    local_dir: Path,
    evidence_path: Path,
) -> tuple[dict[str, Any], int]:
    _assert_no_symlinked_ancestors(evidence_path, containment_root=local_dir)
    if evidence_path.is_symlink():
        raise Sim2RealRerunRegenError(
            f"inner-loop evidence must not be a symlink: {evidence_path}"
        )
    try:
        payload = parse_json_object(
            evidence_path.read_text(encoding="utf-8"),
            source="inner-loop evidence",
        )
    except (OSError, ValueError) as exc:
        raise Sim2RealRerunRegenError(
            f"cannot safely rewrite inner-loop evidence: {evidence_path}"
        ) from exc
    outer_iteration = _canonical_outer_index(evidence_path.parent.name)
    if outer_iteration is None:
        raise Sim2RealRerunRegenError(
            "inner-loop evidence path has no canonical outer iteration"
        )
    return payload, outer_iteration


def _assert_rewrite_outer_iteration(
    payload: dict[str, Any],
    outer_iteration: int,
) -> None:
    recorded_outer = payload.get("outer_iteration")
    if recorded_outer is not None and (
        not isinstance(recorded_outer, int)
        or isinstance(recorded_outer, bool)
        or recorded_outer != outer_iteration
    ):
        raise Sim2RealRerunRegenError(
            "inner-loop evidence payload disagrees with its outer path"
        )


def _iteration_rewrite_specs(
    record: dict[str, Any],
    outer_iteration: int,
) -> tuple[tuple[str, str, str, str], ...]:
    inner_iteration = record.get("iteration")
    if type(inner_iteration) is not int or inner_iteration < 1:
        raise Sim2RealRerunRegenError("inner-loop iteration must be a positive integer")
    scope = f"outer-{outer_iteration:02d}/iter-{inner_iteration:02d}"
    return (
        ("actions_dir", "actions_uri", "actions", f"actions/train/{scope}/"),
        (
            "vlm_eval_dir",
            "vlm_eval_uri",
            "vlm_eval",
            f"vlm_eval/train/{scope}/evaluations/",
        ),
        (
            "signal_dir",
            "signal_uri",
            "vlm_eval",
            f"vlm_eval/train/{scope}/signals/",
        ),
    )


def _rewrite_iteration_path(
    record: dict[str, Any],
    local_dir: Path,
    expected_artifact_root: str | None,
    spec: tuple[str, str, str, str],
) -> bool:
    key, uri_key, marker, expected_suffix = spec
    source = record.get(uri_key) or record.get(key)
    if uri_key in record:
        expected_uri = (
            f"{expected_artifact_root.rstrip('/')}/{expected_suffix}"
            if expected_artifact_root
            else ""
        )
        if not expected_uri or not isinstance(source, str):
            raise Sim2RealRerunRegenError(
                f"{uri_key} cannot be localized without the artifact root"
            )
        if source.rstrip("/") + "/" != expected_uri:
            raise Sim2RealRerunRegenError(
                f"{uri_key} lacks exact run/iteration authority"
            )
    rewritten = _path_under_marker(local_dir, source, marker)
    if uri_key in record and rewritten is None:
        raise Sim2RealRerunRegenError(
            f"{uri_key} cannot be localized below regeneration root"
        )
    if rewritten is None or str(record.get(key) or "") == str(rewritten):
        return False
    record[key] = str(rewritten)
    return True


def _rewrite_iterations(
    payload: dict[str, Any],
    local_dir: Path,
    outer_iteration: int,
    expected_artifact_root: str | None,
) -> bool:
    changed = False
    for record in payload.get("iterations") or []:
        if not isinstance(record, dict):
            raise Sim2RealRerunRegenError(
                "inner-loop iteration evidence must be an object"
            )
        for spec in _iteration_rewrite_specs(record, outer_iteration):
            changed |= _rewrite_iteration_path(
                record, local_dir, expected_artifact_root, spec
            )
    return changed


def _replace_rewritten_evidence(
    local_dir: Path,
    evidence_path: Path,
    payload: dict[str, Any],
) -> None:
    _assert_no_symlinked_ancestors(evidence_path, containment_root=local_dir)
    if evidence_path.is_symlink():
        raise Sim2RealRerunRegenError(
            f"inner-loop evidence became a symlink: {evidence_path}"
        )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{evidence_path.name}.",
        dir=evidence_path.parent,
        delete=False,
    ) as staged:
        staged.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        staged_path = Path(staged.name)
    try:
        os.replace(staged_path, evidence_path)
    finally:
        staged_path.unlink(missing_ok=True)


def _rewrite_inner_evidence_paths(
    local_dir: Path,
    evidence_path: Path,
    *,
    expected_artifact_root: str | None = None,
) -> None:
    local_dir, evidence_path = Path(local_dir), Path(evidence_path)
    payload, outer_iteration = _read_rewrite_payload(local_dir, evidence_path)
    _assert_rewrite_outer_iteration(payload, outer_iteration)
    if _rewrite_iterations(payload, local_dir, outer_iteration, expected_artifact_root):
        _replace_rewritten_evidence(local_dir, evidence_path, payload)


def _path_under_marker(local_dir: Path, value: Any, marker: str) -> Path | None:
    if not value:
        return None
    if not isinstance(value, str) or value != value.strip() or "\\" in value:
        raise Sim2RealRerunRegenError(f"{marker} evidence path is malformed")
    parts = Path(value).parts
    if any(part in {".", ".."} for part in parts):
        raise Sim2RealRerunRegenError(f"{marker} evidence path contains traversal")
    try:
        index = parts.index(marker)
    except ValueError:
        return None
    root = Path(os.path.abspath(local_dir))
    localized = Path(os.path.abspath(root / Path(*parts[index:])))
    if localized == root or not localized.is_relative_to(root):
        raise Sim2RealRerunRegenError(
            f"{marker} evidence path escapes regeneration root"
        )
    return localized


def _render_manifest_from_png_tree(renders_dir: Path) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for env_dir in sorted(
        path for path in Path(renders_dir).iterdir() if path.is_dir()
    ):
        frames = [path.name for path in sorted(env_dir.glob("camera-*.png"))]
        if frames:
            episodes.append({"env_id": env_dir.name, "frames": frames})
    return {"schema": "npa.sim2real.heldout_renders.v1", "episodes": episodes}


def _render_manifest_report_path(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    heldout_report: dict[str, Any] | None,
) -> Path:
    report_outer = (heldout_report or {}).get("outer_iteration")
    if (
        (heldout_report or {}).get("evaluation_split") == "gold_heldout"
        and isinstance(report_outer, int)
        and not isinstance(report_outer, bool)
        and report_outer > 0
    ):
        report_path = (
            Path(local_dir)
            / "eval"
            / "gold-heldout"
            / f"outer-{report_outer:02d}"
            / "report.json"
        )
        return report_path
    return _gold_report_path(config, Path(local_dir))


def _write_report_render_manifest(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    heldout_report: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> None:
    report_path = _render_manifest_report_path(config, local_dir, heldout_report)
    _assert_no_symlinked_ancestors(
        report_path,
        containment_root=Path(local_dir),
    )
    if report_path.is_symlink():
        raise Sim2RealRerunRegenError(
            f"held-out report must not be a symlink: {report_path}"
        )
    if report_path.is_file():
        report = _read_retained_json(
            report_path,
            source="held-out report",
        )
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
    absolute_dest = Path(os.path.abspath(dest_path))
    default_root = Path(os.path.abspath(DEFAULT_REGEN_ROOT))
    containment_root = (
        default_root
        if absolute_dest.is_relative_to(default_root)
        else absolute_dest.parent
    )
    if containment_root == default_root and DEFAULT_REGEN_ROOT.is_symlink():
        raise Sim2RealRerunRegenError(
            f"default regeneration root must not be a symlink: {DEFAULT_REGEN_ROOT}"
        )
    if not _download_if_exists(
        storage,
        uri,
        absolute_dest,
        containment_root=containment_root,
    ):
        raise Sim2RealRerunRegenError(f"Rerun recording not found at {uri}")
    return dest_path


def _assert_regular_publication_file(path: Path) -> None:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise Sim2RealRerunRegenError(
            f"publication source must be a non-empty regular file: {path}"
        )
    for parent in path.parents:
        if parent.is_symlink():
            raise Sim2RealRerunRegenError(
                f"publication source has a symlinked ancestor: {parent}"
            )


def _assert_safe_publication_tree(local_dir: Path, tree: Path) -> None:
    root = Path(os.path.abspath(local_dir))
    tree = Path(os.path.abspath(tree))
    _assert_no_symlinked_ancestors(tree, containment_root=root)
    try:
        tree.relative_to(root)
    except ValueError as exc:
        raise Sim2RealRerunRegenError(
            f"publication tree escapes regeneration root: {tree}"
        ) from exc
    if root.is_symlink() or tree.is_symlink() or not tree.is_dir():
        raise Sim2RealRerunRegenError(
            f"publication tree must be a non-symlink directory: {tree}"
        )
    for entry in tree.rglob("*"):
        if entry.is_symlink():
            raise Sim2RealRerunRegenError(
                f"publication tree contains a symlink: {entry}"
            )


def _recording_publication_id(rrd_path: Path, mcap_path: Path | None = None) -> str:
    material = f"rrd:{sha256_file(rrd_path)}"
    if mcap_path is not None:
        material += f":mcap:{sha256_file(mcap_path)}"
    return hashlib.sha256(material.encode()).hexdigest()


def _seal_regen_publication_report(
    report_path: Path,
    *,
    publication_id: str,
    rrd_uri: str,
    mcap_uri: str,
    report_uri: str,
    journal_uri: str,
) -> None:
    report = _read_retained_json(report_path, source="Sim2Real final report")
    report["publication"] = {
        "generation": publication_id,
        "rrd_uri": rrd_uri,
        "mcap_uri": mcap_uri,
        "report_uri": report_uri,
        "journal_uri": journal_uri,
    }
    report["publication_journal_uri"] = journal_uri
    report["rrd_uri"] = rrd_uri
    report["report_uri"] = report_uri
    if mcap_uri:
        report["mcap_uri"] = mcap_uri
        report["canonical_mcap_uri"] = mcap_uri
    else:
        report.pop("mcap_uri", None)
        report.pop("canonical_mcap_uri", None)
        summaries = report.get("recording_summaries")
        if isinstance(summaries, dict):
            summaries.pop("mcap", None)
    visualization = report.get("visualization")
    if isinstance(visualization, dict):
        visualization["rrd_s3_uri"] = rrd_uri
        if mcap_uri:
            visualization["mcap_s3_uri"] = mcap_uri
        else:
            visualization.pop("mcap_s3_uri", None)
    for component in _stage_components(report):
        if component.get("name") != "stage_14_rerun_viz":
            continue
        component.setdefault("artifacts", {})["rrd"] = rrd_uri
        component["artifacts"]["report"] = report_uri
        if mcap_uri:
            component["artifacts"]["mcap"] = mcap_uri
        else:
            component["artifacts"].pop("mcap", None)
        component["artifacts"]["report_authority_sha256"] = (
            stage14_report_authority_sha256(report)
        )
        material = {
            key: value for key, value in component.items() if key != "content_sha256"
        }
        component["content_sha256"] = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    _write_json_artifact(report_path, report)


@dataclass(frozen=True)
class _RegenPublication:
    prefix: str
    local_dir: Path
    report_path: Path
    renders_dir: Path
    publish_renders: bool
    rrd_path: Path
    publication_id: str
    generation_prefix: str
    immutable_rrd_uri: str
    visual_index_path: Path
    final_report_path: Path
    candidate_path: Path


def _optional_publication_file(path: Path) -> Path:
    if path.is_file():
        _assert_regular_publication_file(path)
    return path


def _regen_report_render_paths(
    config: Sim2RealLoopConfig,
    local_dir: Path,
) -> tuple[Path, Path, bool]:
    report_path = _gold_report_path(config, local_dir)
    _assert_no_symlinked_ancestors(report_path, containment_root=local_dir)
    if report_path.is_file():
        _assert_regular_publication_file(report_path)
    report = (
        _read_retained_json(report_path, source="held-out report")
        if report_path.is_file()
        else {}
    )
    renders_dir = _renders_dir_for_report(config, local_dir, report)
    _assert_no_symlinked_ancestors(renders_dir, containment_root=local_dir)
    publish_renders = renders_dir.is_dir() and _has_camera_pngs(renders_dir)
    if publish_renders:
        _assert_safe_publication_tree(local_dir, renders_dir)
    return report_path, renders_dir, publish_renders


def _prepare_regen_publication(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    rrd_path: Path | None,
    publication_id: str,
) -> _RegenPublication:
    prefix = run_prefix_uri(config)
    local_dir = Path(os.path.abspath(local_dir))
    report_path, renders_dir, publish_renders = _regen_report_render_paths(
        config, local_dir
    )
    rrd_path = Path(rrd_path) if rrd_path else local_dir / "reports" / "sim2real.rrd"
    if not rrd_path.is_file():
        raise Sim2RealRerunRegenError(f"missing regenerated recording: {rrd_path}")
    _assert_regular_publication_file(rrd_path)
    publication_id = publication_id or _recording_publication_id(rrd_path)
    generation_prefix = f"{prefix}reports/generations/{publication_id}/"
    return _RegenPublication(
        prefix=prefix,
        local_dir=local_dir,
        report_path=report_path,
        renders_dir=renders_dir,
        publish_renders=publish_renders,
        rrd_path=rrd_path,
        publication_id=publication_id,
        generation_prefix=generation_prefix,
        immutable_rrd_uri=f"{generation_prefix}sim2real.rrd",
        visual_index_path=_optional_publication_file(
            local_dir / "reports" / "sim2real-visual-index.json"
        ),
        final_report_path=_optional_publication_file(
            local_dir / "reports" / "sim2real-report.json"
        ),
        candidate_path=_optional_publication_file(
            local_dir / "checkpoints" / "candidate" / "candidate.json"
        ),
    )


def _publish_regen_supporting_artifacts(
    storage: StorageClient,
    publication: _RegenPublication,
) -> None:
    if publication.publish_renders:
        _assert_safe_publication_tree(publication.local_dir, publication.renders_dir)
        upload_immutable_tree(
            storage,
            publication.renders_dir,
            f"{publication.prefix}"
            f"{publication.renders_dir.relative_to(publication.local_dir).as_posix()}",
        )
    if publication.report_path.is_file():
        upload_immutable_file(
            storage,
            publication.report_path,
            f"{publication.prefix}"
            f"{publication.report_path.relative_to(publication.local_dir).as_posix()}",
        )
    if publication.visual_index_path.is_file():
        upload_immutable_file(
            storage,
            publication.visual_index_path,
            f"{publication.prefix}reports/sim2real-visual-index.json",
        )
    if publication.candidate_path.is_file():
        upload_immutable_file(
            storage,
            publication.candidate_path,
            f"{publication.prefix}checkpoints/candidate/candidate.json",
        )


def _publish_regen_final_report(
    storage: StorageClient,
    publication: _RegenPublication,
    mcap_uri: str,
) -> Path | None:
    if publication.final_report_path.is_file():
        _seal_regen_publication_report(
            publication.final_report_path,
            publication_id=publication.publication_id,
            rrd_uri=publication.immutable_rrd_uri,
            mcap_uri=mcap_uri,
            report_uri=f"{publication.generation_prefix}sim2real-report.json",
            journal_uri=(f"{publication.prefix}reports/.sim2real-publication.json"),
        )
        upload_immutable_file(
            storage,
            publication.final_report_path,
            f"{publication.generation_prefix}sim2real-report.json",
        )
        report = _read_retained_json(
            publication.final_report_path,
            source="sealed Sim2Real final report",
        )
        components = _stage_components(report)
        if not components:
            return
        stage14_record = components[-1]
        try:
            validate_stage14_component_record(
                stage14_record,
                report,
                expected_source_sha=report.get("source_sha"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Sim2RealRerunRegenError(
                f"regenerated Stage 14 ComponentRecord is invalid: {exc}"
            ) from exc
        record_path = publication.local_dir / "reports" / "stage_14.component.json"
        _write_json_artifact(record_path, stage14_record)
        history_uri = component_record_history_uri(
            publication.prefix.rstrip("/"),
            14,
            stage14_record["content_sha256"],
        )
        upload_immutable_file(storage, record_path, history_uri)
        return record_path
    return None


def _capture_regen_publication_snapshots(
    config: Sim2RealLoopConfig,
    storage: StorageClient,
) -> dict[str, RemoteObjectSnapshot | None]:
    """Capture every mutable generation target before authority validation."""

    prefix = run_prefix_uri(config)
    uris = (
        f"{prefix}reports/sim2real-report.json",
        f"{prefix}reports/sim2real.rrd",
        f"{prefix}reports/sim2real.mcap",
        f"{prefix}components/stage_14.json",
        f"{prefix}reports/.sim2real-publication.json",
    )
    return {uri: remote_object_snapshot(storage, uri) for uri in uris}


def publish_regen_outputs(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    rrd_path: Path | None = None,
    publication_id: str = "",
    mcap_uri: str = "",
    client: StorageClient | None = None,
    snapshots: dict[str, RemoteObjectSnapshot | None] | None = None,
) -> str:
    """Publish immutable regeneration evidence before fenced canonical aliases.

    Args:
        config: Run configuration identifying the destination prefix.
        local_dir: Run-scoped regeneration directory.
        rrd_path: Optional explicit RRD source path.
        publication_id: Optional immutable recording generation.
        mcap_uri: Immutable MCAP URI for the same generation.
        client: Optional configured storage client.
        snapshots: Mutable targets captured before source-authority validation.
    Returns:
        Canonical uploaded RRD URI.
    Raises:
        Sim2RealRerunRegenError: If a publication source is unsafe or incomplete.
    """
    storage = client or _storage_client_for_config(config)
    publication = _prepare_regen_publication(
        config, local_dir, rrd_path, publication_id
    )
    canonical_report_uri = f"{publication.prefix}reports/sim2real-report.json"
    canonical_rrd_uri = f"{publication.prefix}reports/sim2real.rrd"
    canonical_mcap_uri = f"{publication.prefix}reports/sim2real.mcap"
    component_pointer_uri = f"{publication.prefix}components/stage_14.json"
    lock_uri = f"{publication.prefix}reports/.sim2real-publication.json"
    mutable_uris = (
        canonical_report_uri,
        canonical_rrd_uri,
        canonical_mcap_uri,
        component_pointer_uri,
        lock_uri,
    )
    if snapshots is None:
        raise Sim2RealRerunRegenError(
            "regeneration publication snapshots were not captured before validation"
        )
    if any(uri not in snapshots for uri in mutable_uris):
        raise Sim2RealRerunRegenError(
            "regeneration publication snapshot set is incomplete"
        )
    _publish_regen_supporting_artifacts(storage, publication)
    upload_immutable_file(
        storage,
        publication.rrd_path,
        publication.immutable_rrd_uri,
    )
    component_path = _publish_regen_final_report(storage, publication, mcap_uri)
    mcap_path = publication.local_dir / "reports" / "sim2real.mcap"
    with MutablePublicationTransaction(
        storage,
        lock_uri=lock_uri,
        lock_snapshot=snapshots[lock_uri],
        transaction_id=publication.publication_id,
    ) as transaction:
        transaction.replace_file(
            publication.rrd_path,
            canonical_rrd_uri,
            snapshots[canonical_rrd_uri],
            immutable_uri=publication.immutable_rrd_uri,
        )
        if mcap_uri:
            transaction.replace_file(
                mcap_path,
                canonical_mcap_uri,
                snapshots[canonical_mcap_uri],
                immutable_uri=mcap_uri,
            )
        else:
            transaction.delete_file(
                canonical_mcap_uri,
                snapshots[canonical_mcap_uri],
            )
        if publication.final_report_path.is_file():
            transaction.replace_file(
                publication.final_report_path,
                canonical_report_uri,
                snapshots[canonical_report_uri],
                immutable_uri=(f"{publication.generation_prefix}sim2real-report.json"),
            )
        if component_path is not None:
            component = _read_retained_json(
                component_path,
                source="regenerated Stage 14 ComponentRecord",
            )
            transaction.replace_file(
                component_path,
                component_pointer_uri,
                snapshots[component_pointer_uri],
                immutable_uri=component_record_history_uri(
                    publication.prefix.rstrip("/"),
                    14,
                    str(component.get("content_sha256") or ""),
                ),
            )
    return canonical_rrd_uri


@dataclass(frozen=True)
class _HeldoutPublication:
    local_dir: Path
    report_path: Path
    prefix: str
    canonical_render_uri: str
    immutable_report_uri: str
    current_report: dict[str, Any]
    renders_dir: Path


def _heldout_publication_attempt(
    report: dict[str, Any],
    render_manifest: dict[str, Any],
    outer_iteration: int,
) -> str:
    report_attempt = report.get("evaluation_attempt_tag")
    manifest_attempt = render_manifest.get("evaluation_attempt_tag")
    if (
        report_attempt not in (None, "")
        and manifest_attempt not in (None, "")
        and report_attempt != manifest_attempt
    ):
        raise Sim2RealRerunRegenError(
            "held-out publication attempt identity sources disagree"
        )
    attempt_tag = report_attempt or manifest_attempt or ""
    prefix = f"gold_heldout-outer-{outer_iteration:02d}-attempt-"
    if attempt_tag and (
        not isinstance(attempt_tag, str)
        or not attempt_tag.startswith(prefix)
        or len(attempt_tag) != len(prefix) + 32
        or any(char not in "0123456789abcdef" for char in attempt_tag[len(prefix) :])
    ):
        raise Sim2RealRerunRegenError(
            "held-out publication attempt identity is invalid"
        )
    return str(attempt_tag)


def _heldout_publication_report_path(
    local_dir: Path,
    outer_iteration: int,
) -> Path:
    path = (
        local_dir
        / "eval"
        / "gold-heldout"
        / f"outer-{outer_iteration:02d}"
        / "report.json"
    )
    _assert_no_symlinked_ancestors(path, containment_root=local_dir)
    if path.is_symlink():
        raise Sim2RealRerunRegenError(f"held-out report must not be a symlink: {path}")
    return path


def _heldout_publication_report(
    prefix: str,
    report: dict[str, Any],
    manifest: dict[str, Any],
    outer_iteration: int,
    attempt_tag: str,
) -> tuple[dict[str, Any], str]:
    relative = (
        Path("eval") / "gold-heldout" / f"outer-{outer_iteration:02d}" / "renders"
    )
    canonical_uri = (
        f"{prefix}eval/gold-heldout/outer-{outer_iteration:02d}/"
        f"attempts/{attempt_tag}/renders/"
        if attempt_tag
        else f"{prefix}{relative.as_posix()}/"
    )
    current_report = {
        **report,
        "evaluation_split": "gold_heldout",
        "outer_iteration": outer_iteration,
        "local_renders_dir": relative.as_posix(),
        "render_lineage": {
            "evaluation_split": "gold_heldout",
            "evaluation_attempt_tag": attempt_tag,
            "source_s3_uri": str(manifest.get("renders_s3_uri") or ""),
            "canonical_s3_uri": canonical_uri,
            "local_relative_dir": relative.as_posix(),
        },
    }
    return current_report, canonical_uri


def _prepare_heldout_publication(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    report: dict[str, Any],
    outer_iteration: int,
) -> _HeldoutPublication:
    local_dir = Path(os.path.abspath(local_dir))
    report_path = _heldout_publication_report_path(local_dir, outer_iteration)
    prefix = run_prefix_uri(config)
    render_manifest = report.get("render_manifest") or {}
    if not isinstance(render_manifest, dict):
        raise Sim2RealRerunRegenError("held-out render manifest must be an object")
    attempt_tag = _heldout_publication_attempt(report, render_manifest, outer_iteration)
    if not attempt_tag:
        raise Sim2RealRerunRegenError(
            "held-out publication requires an immutable evaluation attempt"
        )
    current_report, canonical_render_uri = _heldout_publication_report(
        prefix, report, render_manifest, outer_iteration, attempt_tag
    )
    immutable_report_uri = (
        f"{prefix}eval/gold-heldout/outer-{outer_iteration:02d}/"
        f"attempts/{attempt_tag}/report.json"
    )
    current_report["report_uri"] = immutable_report_uri
    renders_dir = _renders_dir_for_report(config, local_dir, current_report)
    if not renders_dir.is_dir() or not _has_camera_pngs(renders_dir):
        raise Sim2RealRerunRegenError(
            "held-out publication requires current camera render bytes"
        )
    _assert_safe_publication_tree(local_dir, renders_dir)
    return _HeldoutPublication(
        local_dir,
        report_path,
        prefix,
        canonical_render_uri,
        immutable_report_uri,
        current_report,
        renders_dir,
    )


def _publish_heldout_eval_outputs(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    report: dict[str, Any],
    *,
    outer_iteration: int,
    storage: StorageClient,
) -> None:
    """Publish only evidence produced by the current held-out rerun."""

    publication = _prepare_heldout_publication(
        config, local_dir, report, outer_iteration
    )
    upload_immutable_tree(
        storage,
        publication.renders_dir,
        publication.canonical_render_uri,
    )
    _write_json_artifact(publication.report_path, publication.current_report)
    report.clear()
    report.update(publication.current_report)
    _assert_regular_publication_file(publication.report_path)
    upload_immutable_file(
        storage,
        publication.report_path,
        publication.immutable_report_uri,
    )


def publish_regen_mcap(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    emission: dict[str, Any],
    publication_id: str,
    client: StorageClient | None = None,
) -> str:
    """Upload only the MCAP proven written by the current emission call."""

    mcap_path = Path(local_dir) / "reports" / "sim2real.mcap"
    reported_path = str(emission.get("output_mcap_path") or "")
    if emission.get("status") != "written":
        return ""
    if Path(os.path.abspath(reported_path)) != Path(os.path.abspath(mcap_path)):
        raise Sim2RealRerunRegenError("MCAP emitter reported an unexpected output path")
    if not mcap_path.is_file() or mcap_path.stat().st_size == 0:
        raise Sim2RealRerunRegenError("MCAP emitter reported missing or empty bytes")
    _assert_regular_publication_file(mcap_path)
    storage = client or _storage_client_for_config(config)
    prefix = run_prefix_uri(config)
    immutable_uri = f"{prefix}reports/generations/{publication_id}/sim2real.mcap"
    upload_immutable_file(
        storage,
        mcap_path,
        immutable_uri,
    )
    return immutable_uri


@dataclass(frozen=True)
class _RegenState:
    inner_evidence: dict[str, Any]
    heldout_report: dict[str, Any]
    report_path: Path
    report: dict[str, Any]
    policy_access: dict[str, Any]
    publication_snapshots: dict[str, RemoteObjectSnapshot | None] | None = None


def _stage_components(report: dict[str, Any]) -> list[dict[str, Any]]:
    canonical = report.get("component_records")
    legacy = report.get("components")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise Sim2RealRerunRegenError("component record sources disagree")
    components = canonical if canonical is not None else legacy
    if components is None:
        return []
    if not isinstance(components, list) or not all(
        isinstance(item, dict) for item in components
    ):
        raise Sim2RealRerunRegenError("component records must be a list of objects")
    if canonical is not None and legacy is not None:
        report["components"] = canonical
    return components


def _regen_paths(
    config: Sim2RealLoopConfig,
    local_dir: Path | None,
    local_rrd_path: Path | None,
) -> tuple[Path, Path]:
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
    return work_dir, output_rrd


@dataclass(frozen=True)
class _RegenInputs:
    inner_evidence: dict[str, Any]
    heldout_report: dict[str, Any]
    heldout_path: Path
    report_path: Path
    report: dict[str, Any]
    current_decision: dict[str, Any]


def _selected_regen_paths(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    sync_inputs: bool,
) -> tuple[Path, Path]:
    if sync_inputs:
        return sync_regen_inputs(config, work_dir, client=storage)
    return _latest_local_inner_evidence(work_dir), _gold_report_path(config, work_dir)


def _load_regen_inputs(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    inner_path: Path,
    heldout_path: Path,
) -> _RegenInputs:
    _rewrite_inner_evidence_paths(
        work_dir,
        inner_path,
        expected_artifact_root=run_prefix_uri(config).rstrip("/"),
    )
    if not inner_path.is_file():
        raise Sim2RealRerunRegenError(f"missing inner evidence: {inner_path}")
    if not heldout_path.is_file():
        raise Sim2RealRerunRegenError(f"missing held-out report: {heldout_path}")
    inner_evidence = _read_retained_json(inner_path, source="inner-loop evidence")
    heldout_report = _read_retained_json(heldout_path, source="held-out report")
    _assert_selected_pair_outer_iteration(
        inner_path, heldout_path, inner_evidence, heldout_report
    )
    report_path = work_dir / "reports" / "sim2real-report.json"
    report = (
        _read_retained_json(report_path, source="Sim2Real final report")
        if report_path.is_file()
        else {}
    )
    decision_path = work_dir / "outer_loop" / "decision.json"
    decision = (
        _read_retained_json(decision_path, source="current outer-loop decision")
        if decision_path.is_file()
        else {}
    )
    return _RegenInputs(
        inner_evidence,
        heldout_report,
        heldout_path,
        report_path,
        report,
        decision,
    )


def _validate_regen_decision(
    config: Sim2RealLoopConfig,
    inputs: _RegenInputs,
) -> None:
    if (
        inputs.inner_evidence.get("schema") != "npa.sim2real.inner_loop_evidence.v1"
        and not inputs.current_decision
    ):
        return
    try:
        outer_iteration = inputs.inner_evidence.get("outer_iteration")
        if type(outer_iteration) is not int:
            raise ValueError("selected outer iteration is invalid")
        validate_stage10_input_scope(
            argparse.Namespace(
                run_id=config.run_id,
                outer_iteration=outer_iteration,
            ),
            root=run_prefix_uri(config).rstrip("/"),
            evidence=inputs.inner_evidence,
        )
        selection, _candidate = resolve_run_scoped_checkpoint(
            inputs.inner_evidence,
            run_root=run_prefix_uri(config).rstrip("/"),
            run_id=config.run_id,
        )
        sealed_decision = (inputs.report.get("outer_loop") or {}).get("decision")
        policy_gate = inputs.report.get("policy_gate_config")
        if (
            not isinstance(sealed_decision, dict)
            or sealed_decision != inputs.current_decision
        ):
            raise ValueError(
                "canonical report does not seal the current Stage 11 decision"
            )
        if policy_gate is None:
            policy_gate = {
                "threshold": sealed_decision.get("threshold"),
                "early_exit": sealed_decision.get("early_exit_enabled"),
            }
        if not isinstance(policy_gate, dict):
            raise ValueError("canonical report policy-gate authority is invalid")
        threshold = policy_gate.get("threshold")
        early_exit = policy_gate.get("early_exit")
        if (
            threshold != sealed_decision.get("threshold")
            or type(early_exit) is not bool
            or early_exit != sealed_decision.get("early_exit_enabled")
        ):
            raise ValueError("canonical report policy-gate authority is inconsistent")
        validate_stage11_decision(
            inputs.current_decision,
            run_id=config.run_id,
            root=run_prefix_uri(config).rstrip("/"),
            outer_iteration=outer_iteration,
            gold_report=inputs.heldout_report,
            checkpoint_uri=selection["checkpoint_uri"],
            expected_threshold=threshold,
            expected_early_exit=early_exit,
            gold_report_bytes_sha256=sha256_file(inputs.heldout_path),
        )
    except (RuntimeError, ValueError) as exc:
        raise Sim2RealRerunRegenError(
            f"current Stage 11 decision authority is invalid: {exc}"
        ) from exc


def _validate_regen_component_authority(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    inputs: _RegenInputs,
    *,
    verify_remote_authority: bool,
    require_canonical_authority: bool = False,
) -> None:
    canonical_architecture = "npa.workflow/v0.0.1_compositional_standard_runtime"
    architecture = inputs.report.get("architecture")
    claims_canonical_authority = (
        inputs.report.get("schema") == SCHEMA_E2E_REPORT
        or "component_records" in inputs.report
        or architecture == canonical_architecture
        or (
            inputs.inner_evidence.get("schema") == "npa.sim2real.inner_loop_evidence.v1"
        )
    )
    if architecture != canonical_architecture:
        if require_canonical_authority:
            raise Sim2RealRerunRegenError(
                "remote canonical ComponentRecord authority is required for upload"
            )
        if claims_canonical_authority:
            raise Sim2RealRerunRegenError(
                "canonical report architecture identity is missing or invalid"
            )
        return
    if inputs.report.get("schema") != SCHEMA_E2E_REPORT:
        raise Sim2RealRerunRegenError(
            "canonical report schema identity is missing or invalid"
        )
    if inputs.report.get("run_id") != config.run_id:
        raise Sim2RealRerunRegenError(
            "canonical report run identity is missing or invalid"
        )
    if "component_records" not in inputs.report:
        raise Sim2RealRerunRegenError(
            "canonical report ComponentRecord source is missing"
        )
    components = _stage_components(inputs.report)
    if len(components) != 14:
        raise Sim2RealRerunRegenError(
            "canonical report must retain all 14 ComponentRecords"
        )
    root = run_prefix_uri(config).rstrip("/")
    expected_source = inputs.report.get("source_sha")
    try:
        validate_component_records(
            components[:13],
            root=root,
            evidence=inputs.inner_evidence,
            gold=inputs.heldout_report,
            expected_source_sha=expected_source,
        )
        validate_stage14_component_record(
            components[13],
            inputs.report,
            expected_source_sha=expected_source,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Sim2RealRerunRegenError(
            f"canonical report ComponentRecord authority is invalid: {exc}"
        ) from exc
    if not verify_remote_authority:
        return
    retained: list[dict[str, Any]] = []
    authority_dir = work_dir / "component-authority"
    for stage in range(1, 15):
        pointer_uri = f"{root}/components/stage_{stage:02d}.json"
        pointer_path = authority_dir / f"stage_{stage:02d}.json"
        if not _download_if_exists(
            storage,
            pointer_uri,
            pointer_path,
            containment_root=work_dir,
        ):
            raise Sim2RealRerunRegenError(
                f"missing canonical ComponentRecord pointer for Stage {stage}"
            )
        pointer = _read_retained_json(
            pointer_path, source=f"Stage {stage} ComponentRecord pointer"
        )
        history_uri = component_record_history_uri(
            root, stage, pointer.get("content_sha256", "")
        )
        history_path = authority_dir / f"stage_{stage:02d}-history.json"
        if not _download_if_exists(
            storage,
            history_uri,
            history_path,
            containment_root=work_dir,
        ):
            raise Sim2RealRerunRegenError(
                f"missing immutable ComponentRecord history for Stage {stage}"
            )
        history = _read_retained_json(
            history_path, source=f"Stage {stage} ComponentRecord history"
        )
        if pointer != history:
            raise Sim2RealRerunRegenError(
                f"Stage {stage} ComponentRecord pointer/history mismatch"
            )
        retained.append(pointer)
    if retained != components:
        raise Sim2RealRerunRegenError(
            "report ComponentRecords disagree with current immutable history"
        )
    try:

        def load_stage4_authority(uri: str) -> dict[str, Any]:
            target = (
                authority_dir
                / "stage-04-nested"
                / f"{hashlib.sha256(uri.encode()).hexdigest()}.json"
            )
            if not _download_if_exists(
                storage,
                uri,
                target,
                containment_root=work_dir,
            ):
                raise ValueError(f"missing remote Stage 4 authority: {uri}")
            return _read_retained_json(
                target,
                source=f"Stage 4 nested authority {uri}",
            )

        validate_remote_stage4_authority(
            root,
            components[3],
            load_stage4_authority,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Sim2RealRerunRegenError(
            f"remote Stage 4 authority is invalid: {exc}"
        ) from exc


def _validate_regen_materialized_frames(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    inputs: _RegenInputs,
) -> None:
    """Revalidate canonical local renders against sealed frame-byte authority."""

    if inputs.inner_evidence.get("schema") != "npa.sim2real.inner_loop_evidence.v1":
        return
    try:
        _selection, candidate = resolve_run_scoped_checkpoint(
            inputs.inner_evidence,
            run_root=run_prefix_uri(config).rstrip("/"),
            run_id=config.run_id,
        )
        validate_materialized_render_tree(
            _renders_dir_for_report(config, work_dir, inputs.heldout_report),
            inputs.heldout_report.get("render_manifest") or {},
            candidate,
            require_frame_identity=True,
        )
    except (RuntimeError, ValueError) as exc:
        raise Sim2RealRerunRegenError(
            f"canonical regeneration frame bytes are invalid: {exc}"
        ) from exc


def _load_regen_state(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    *,
    sync_inputs: bool,
    verify_remote_authority: bool | None = None,
    require_canonical_authority: bool = False,
    publication_snapshots: dict[str, RemoteObjectSnapshot | None] | None = None,
) -> _RegenState:
    try:
        paths = _selected_regen_paths(config, work_dir, storage, sync_inputs)
        inputs = _load_regen_inputs(config, work_dir, *paths)
        _validate_regen_decision(config, inputs)
        _validate_regen_component_authority(
            config,
            work_dir,
            storage,
            inputs,
            verify_remote_authority=(
                sync_inputs
                if verify_remote_authority is None
                else verify_remote_authority
            ),
            require_canonical_authority=require_canonical_authority,
        )
        _validate_regen_materialized_frames(config, work_dir, inputs)
        policy_access = _ensure_policy_access_metadata(
            config,
            work_dir,
            storage=storage,
            report=inputs.report,
            heldout_report=inputs.heldout_report,
            inner_evidence=inputs.inner_evidence,
            current_decision=inputs.current_decision,
        )
    except Exception:
        _fail_closed_local_candidate(work_dir)
        raise
    return _RegenState(
        inputs.inner_evidence,
        inputs.heldout_report,
        inputs.report_path,
        inputs.report,
        policy_access,
        publication_snapshots,
    )


def _fail_closed_local_candidate(work_dir: Path) -> None:
    try:
        candidate_path, candidate = _read_candidate_manifest(work_dir)
    except Exception:
        _LOGGER.debug(
            "Unable to parse candidate while persisting fail-closed state",
            exc_info=True,
        )
        return
    _persist_failed_candidate(candidate_path, candidate)


def _heldout_metadata(
    heldout_report: dict[str, Any],
    policy_access: dict[str, Any],
) -> dict[str, Any]:
    heldout_identity_kwargs: dict[str, object] = {}
    if policy_access.get("checkpoint_uri"):
        heldout_identity_kwargs["checkpoint_fallback"] = policy_access["checkpoint_uri"]
    if policy_access.get("sha256"):
        heldout_identity_kwargs["checkpoint_sha256_fallback"] = policy_access["sha256"]
    if policy_access.get("size_bytes"):
        heldout_identity_kwargs["checkpoint_size_fallback"] = policy_access[
            "size_bytes"
        ]
    return _heldout_policy_metadata(heldout_report, **heldout_identity_kwargs)


def _viewer_command(config: Sim2RealLoopConfig) -> str:
    return (
        "npa workbench sim2real rerun serve "
        f"--run-id {config.run_id} --s3-bucket {config.s3_bucket} "
        f"--s3-prefix {config.s3_prefix}"
    )


def _regen_run_metadata(
    config: Sim2RealLoopConfig,
    state: _RegenState,
    viewer_command: str,
) -> dict[str, Any]:
    prefix = run_prefix_uri(config)
    policy_access = state.policy_access
    return {
        "run_id": config.run_id,
        "artifact_root": prefix,
        "rrd_s3_uri": f"{prefix}reports/sim2real.rrd",
        "candidate_s3_uri": f"{prefix}checkpoints/candidate/candidate.json",
        "policy_checkpoint": policy_access.get("checkpoint_uri", ""),
        "policy_checkpoint_identity": policy_access.get("identity", ""),
        "policy_checkpoint_sha256": policy_access.get("sha256", ""),
        "policy_checkpoint_size_bytes": policy_access.get("size_bytes", 0),
        "policy_download_command": policy_access.get(
            "authenticated_download_command", ""
        ),
        "policy_ui_action": policy_access.get("ui_action", ""),
        "policy_deployable": policy_access.get("deployable_policy", False),
        **_heldout_metadata(state.heldout_report, policy_access),
        "orchestrator_job_name": config.run_id,
        "orchestrator_node_product": config.k8s_gpu_product,
        "viewer_command": viewer_command,
    }


def _emit_regen_rrd(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    output_rrd: Path,
    state: _RegenState,
    outer_history: list[dict[str, Any]],
    viewer_command: str,
) -> tuple[Sim2RealVizResult, float]:
    rerun_started = time.monotonic()
    result = emit_sim2real_rerun(
        local_dir=work_dir,
        inner_evidence=state.inner_evidence,
        heldout_report=state.heldout_report,
        stage_components=list(_stage_components(state.report)),
        outer_history=outer_history,
        run_metadata=_regen_run_metadata(config, state, viewer_command),
        output_rrd=output_rrd,
    )
    if Path(os.path.abspath(result.output_rrd_path)) != Path(
        os.path.abspath(output_rrd)
    ):
        raise Sim2RealRerunRegenError(
            "Rerun emitter reported a different output path than requested"
        )
    return result, round(time.monotonic() - rerun_started, 3)


def _update_stage14_component(
    component: dict[str, Any],
    result: Sim2RealVizResult,
    output_rrd: Path,
    prefix: str,
    duration_s: float,
) -> None:
    component["tier"] = "WORKS"
    component["evidence"] = (
        "Wrote the complete Rerun recording from every persisted pass: "
        f"{result.rollout_count} real policy rollout(s), "
        f"{result.frame_count} synchronized policy camera frame(s), and "
        f"{result.heldout_frame_count} held-out frame(s)."
    )
    component.setdefault("artifacts", {}).update(
        {
            "rrd": f"{prefix}reports/sim2real.rrd",
            "rrd_local": str(output_rrd),
            "rrd_size_bytes": output_rrd.stat().st_size,
            "duration_s": duration_s,
        }
    )
    digest_material = {
        key: value for key, value in component.items() if key != "content_sha256"
    }
    component["content_sha256"] = hashlib.sha256(
        json.dumps(
            digest_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _persist_regen_report(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    output_rrd: Path,
    state: _RegenState,
    result: Sim2RealVizResult,
    outer_history: list[dict[str, Any]],
    viewer_command: str,
    duration_s: float,
) -> None:
    if not state.report:
        return
    from npa.workflows.sim2real.engine import gpu_fallback_report_contract

    prefix = run_prefix_uri(config)
    state.report["policy_access"] = state.policy_access
    state.report["progress_metrics"] = build_progress_metrics(work_dir, outer_history)
    components = _stage_components(state.report)
    state.report["gpu_fallback_contract"] = gpu_fallback_report_contract(
        config, list(components)
    )
    state.report["visualization"] = {
        **result.to_dict(),
        "rrd_s3_uri": f"{prefix}reports/sim2real.rrd",
        "rrd_size_bytes": output_rrd.stat().st_size,
        "viewer_command": viewer_command,
    }
    for component in components:
        if component.get("name") == "stage_14_rerun_viz":
            _update_stage14_component(component, result, output_rrd, prefix, duration_s)
    state.report_path.write_text(
        json.dumps(state.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _current_mcap_path(
    work_dir: Path,
    mcap_result: dict[str, Any],
) -> Path | None:
    path = (
        work_dir / "reports" / "sim2real.mcap"
        if mcap_result.get("status") == "written"
        else None
    )
    if path is not None:
        if not path.is_file() or path.stat().st_size <= 0:
            raise Sim2RealRerunRegenError(
                "current MCAP emission has no non-empty publication bytes"
            )
        _assert_regular_publication_file(path)
    return path


def _publish_regen_recordings(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    *,
    upload: bool,
    mcap_result: dict[str, Any],
    output_rrd: Path | None = None,
    publication_snapshots: dict[str, RemoteObjectSnapshot | None] | None = None,
) -> tuple[str, str]:
    if not upload:
        return "", ""
    rrd_path = output_rrd or (work_dir / "reports" / "sim2real.rrd")
    _assert_regular_publication_file(rrd_path)
    written_mcap = _current_mcap_path(work_dir, mcap_result)
    publication_id = _recording_publication_id(rrd_path, written_mcap)
    immutable_mcap_uri = (
        f"{run_prefix_uri(config)}reports/generations/{publication_id}/sim2real.mcap"
        if written_mcap is not None
        else ""
    )
    published_mcap_uri = publish_regen_mcap(
        config,
        work_dir,
        emission=mcap_result,
        publication_id=publication_id,
        client=storage,
    )
    return (
        publish_regen_outputs(
            config,
            work_dir,
            rrd_path=output_rrd,
            publication_id=publication_id,
            mcap_uri=immutable_mcap_uri,
            client=storage,
            snapshots=publication_snapshots,
        ),
        (
            f"{run_prefix_uri(config)}reports/sim2real.mcap"
            if published_mcap_uri
            else ""
        ),
    )


def _emit_regen_mcap(
    work_dir: Path,
    state: _RegenState,
) -> dict[str, Any]:
    mcap_path = work_dir / "reports" / "sim2real.mcap"
    _assert_no_symlinked_ancestors(mcap_path, containment_root=work_dir)
    mcap_path.unlink(missing_ok=True)
    return emit_sim2real_mcap_if_enabled(
        local_dir=work_dir,
        inner_evidence=state.inner_evidence,
        heldout_report=state.heldout_report,
        output_mcap=mcap_path,
    )


def _assert_regen_heldout_frames(result: Sim2RealVizResult) -> None:
    if result.heldout_frame_count <= 0:
        raise Sim2RealRerunRegenError(
            "regenerated .rrd has heldout_frame_count=0; "
            "sync gold held-out renders or rerun held-out eval"
        )


def _finalize_regen_result(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    state: _RegenState,
    result: Sim2RealVizResult,
    *,
    upload: bool,
    output_rrd: Path | None = None,
) -> RegenResult:
    _assert_regen_heldout_frames(result)
    mcap_result = _emit_regen_mcap(work_dir, state)
    upload_uri, mcap_upload_uri = _publish_regen_recordings(
        config,
        work_dir,
        storage,
        upload=upload,
        mcap_result=mcap_result,
        output_rrd=output_rrd or Path(result.output_rrd_path),
        publication_snapshots=state.publication_snapshots,
    )
    return _regen_result_from_viz(
        config.run_id,
        work_dir,
        result,
        upload_uri=upload_uri,
        mcap_result=mcap_result,
        mcap_upload_uri=mcap_upload_uri,
    )


def _regenerate_sim2real_rrd(
    config: Sim2RealLoopConfig,
    local_dir: Path | None,
    local_rrd_path: Path | None,
    upload: bool,
    sync_inputs: bool,
    client: StorageClient | None,
) -> RegenResult:
    work_dir, output_rrd = _regen_paths(config, local_dir, local_rrd_path)
    storage = client or _storage_client_for_config(config)
    publication_snapshots = (
        _capture_regen_publication_snapshots(config, storage) if upload else None
    )
    state = _load_regen_state(
        config,
        work_dir,
        storage,
        sync_inputs=sync_inputs,
        verify_remote_authority=sync_inputs or upload,
        require_canonical_authority=upload and not sync_inputs,
        publication_snapshots=publication_snapshots,
    )
    if publication_snapshots is not None and state.publication_snapshots is None:
        state = replace(state, publication_snapshots=publication_snapshots)
    outer_history = list((state.report.get("outer_loop") or {}).get("history") or [])
    viewer_command = _viewer_command(config)
    result, duration_s = _emit_regen_rrd(
        config, work_dir, output_rrd, state, outer_history, viewer_command
    )
    _assert_regen_heldout_frames(result)
    _persist_regen_report(
        config,
        work_dir,
        output_rrd,
        state,
        result,
        outer_history,
        viewer_command,
        duration_s,
    )
    return _finalize_regen_result(
        config,
        work_dir,
        storage,
        state,
        result,
        upload=upload,
        output_rrd=output_rrd,
    )


def regen_sim2real_rrd(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path | None = None,
    local_rrd_path: Path | None = None,
    upload: bool = False,
    sync_inputs: bool = True,
    client: StorageClient | None = None,
) -> RegenResult:
    """Regenerate and optionally publish one run's RRD.

    Args:
        config: Existing run configuration.
        local_dir: Optional run-scoped working directory.
        local_rrd_path: Optional explicit RRD destination.
        upload: Whether to publish regenerated artifacts.
        sync_inputs: Whether to refresh durable inputs first.
        client: Optional configured storage client.
    Returns:
        Regeneration paths, counts, and publication results.
    Raises:
        Sim2RealRerunRegenError: If authority, emission, or publication fails.
    """
    return _regenerate_sim2real_rrd(
        config, local_dir, local_rrd_path, upload, sync_inputs, client
    )


def _read_retained_json(path: Path, *, source: str) -> dict[str, Any]:
    try:
        return parse_json_object(
            path.read_text(encoding="utf-8"),
            source=source,
        )
    except (OSError, ValueError) as exc:
        raise Sim2RealRerunRegenError(str(exc)) from exc


def _read_candidate_manifest(local_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidate_path = Path(local_dir) / "checkpoints" / "candidate" / "candidate.json"
    if not candidate_path.is_file():
        return candidate_path, {}
    try:
        candidate = _read_retained_json(
            candidate_path,
            source="candidate checkpoint manifest",
        )
    except Sim2RealRerunRegenError:
        _persist_failed_candidate(candidate_path, {})
        raise
    return candidate_path, candidate


def _one_checkpoint_uri(evidence: tuple[tuple[str, object], ...]) -> str:
    sources: list[str] = []
    for source, value in evidence:
        checkpoint_uri = _checkpoint_uri(value)
        if checkpoint_uri is None:
            raise Sim2RealRerunRegenError(
                f"candidate checkpoint URI from {source} is malformed"
            )
        sources.append(checkpoint_uri)
    if len(set(sources)) > 1:
        raise Sim2RealRerunRegenError(
            "candidate checkpoint URI sources disagree during Rerun regeneration"
        )
    return sources[0] if sources else ""


def _one_checkpoint_digest(evidence: tuple[tuple[str, object], ...]) -> str:
    sources: list[str] = []
    for source, value in evidence:
        digest = value.lower() if isinstance(value, str) else ""
        if (
            not isinstance(value, str)
            or value != value.strip()
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise Sim2RealRerunRegenError(
                f"candidate checkpoint SHA-256 from {source} is malformed"
            )
        sources.append(digest)
    if len(set(sources)) > 1:
        raise Sim2RealRerunRegenError(
            "candidate checkpoint SHA-256 sources disagree during Rerun regeneration"
        )
    return sources[0] if sources else ""


def _one_checkpoint_size(evidence: tuple[tuple[str, object], ...]) -> int:
    sources: list[int] = []
    for source, value in evidence:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise Sim2RealRerunRegenError(
                f"candidate checkpoint size from {source} is malformed"
            )
        sources.append(value)
    if len(set(sources)) > 1:
        raise Sim2RealRerunRegenError(
            "candidate checkpoint size sources disagree during Rerun regeneration"
        )
    return sources[0] if sources else 0


def _optional_object(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, Any]:
    if key not in payload:
        return {}
    value = payload[key]
    if not isinstance(value, dict):
        raise Sim2RealRerunRegenError(f"{source}.{key} must be a JSON object")
    return value


def _field_evidence(
    payload: dict[str, Any],
    key: str,
    source: str,
) -> tuple[tuple[str, object], ...]:
    return ((source, payload[key]),) if key in payload else ()


def _canonical_decision(outer_loop: dict[str, Any]) -> dict[str, Any]:
    decisions: dict[str, dict[str, Any]] = {}
    for key in ("latest_decision", "decision"):
        if key in outer_loop:
            value = outer_loop[key]
            if not isinstance(value, dict):
                raise Sim2RealRerunRegenError(f"outer_loop.{key} must be a JSON object")
            decisions[key] = value
    canonical = dict(
        decisions.get("decision") or decisions.get("latest_decision") or {}
    )
    if len(decisions) == 2:
        latest = decisions["latest_decision"]
        declared = decisions["decision"]
        for field in (
            "decision",
            "checkpoint_uri",
            "deployable_policy",
            "policy_bytes_available",
            "candidate",
        ):
            if (
                field in latest
                and field in declared
                and latest[field] != declared[field]
            ):
                raise Sim2RealRerunRegenError(
                    f"outer-loop decision aliases disagree on {field}"
                )
            if field in latest and field not in canonical:
                canonical[field] = latest[field]
    return canonical


@dataclass(frozen=True)
class _RetainedPolicySources:
    candidate: dict[str, Any]
    report: dict[str, Any]
    outer_loop: dict[str, Any]
    decision: dict[str, Any]
    decision_candidate: dict[str, Any]
    selection: dict[str, Any]
    selected_candidate: dict[str, Any]
    current_decision: dict[str, Any]
    current_decision_candidate: dict[str, Any]
    current_selection: dict[str, Any]
    current_selected_candidate: dict[str, Any]
    policy_access: dict[str, Any]
    heldout_report: dict[str, Any]
    producer: dict[str, Any]
    current_heldout_report: dict[str, Any]
    current_heldout_report_present: bool
    current_producer: dict[str, Any]


_RETAINED_SOURCE_LABELS = (
    ("candidate", "candidate"),
    ("selected_candidate", "selected_checkpoint_candidate"),
    ("decision_candidate", "outer_loop.decision.candidate"),
    ("decision", "outer_loop.decision"),
    ("selection", "checkpoint_selection"),
    ("policy_access", "policy_access"),
    ("heldout_report", "latest_heldout_report"),
    ("producer", "policy_inference_provenance"),
    ("current_decision", "current_outer_loop.decision"),
    ("current_decision_candidate", "current_outer_loop.decision.candidate"),
    ("current_selection", "current_checkpoint_selection"),
    ("current_selected_candidate", "current_selected_checkpoint_candidate"),
    ("current_heldout_report", "current_heldout_report"),
    ("current_producer", "current_policy_inference_provenance"),
)
_CHECKPOINT_URI_KEYS = {
    "candidate": CHECKPOINT_URI_ALIASES,
    "selected_candidate": CHECKPOINT_URI_ALIASES,
    "decision_candidate": CHECKPOINT_URI_ALIASES,
    "decision": ("checkpoint_uri",),
    "selection": CHECKPOINT_URI_ALIASES,
    "current_decision": ("checkpoint_uri",),
    "current_decision_candidate": CHECKPOINT_URI_ALIASES,
    "current_selection": CHECKPOINT_URI_ALIASES,
    "current_selected_candidate": CHECKPOINT_URI_ALIASES,
    "policy_access": ("checkpoint_uri",),
    "heldout_report": ("policy_checkpoint", "policy_checkpoint_uri"),
    "producer": ("checkpoint_uri",),
    "current_heldout_report": ("policy_checkpoint", "policy_checkpoint_uri"),
    "current_producer": ("checkpoint_uri",),
}
_CHECKPOINT_DIGEST_KEYS = {
    "candidate": CHECKPOINT_DIGEST_ALIASES + GENERATOR_DIGEST_ALIASES,
    "selected_candidate": CHECKPOINT_DIGEST_ALIASES + GENERATOR_DIGEST_ALIASES,
    "decision_candidate": CHECKPOINT_DIGEST_ALIASES + GENERATOR_DIGEST_ALIASES,
    "selection": CHECKPOINT_DIGEST_ALIASES + GENERATOR_DIGEST_ALIASES,
    "current_decision_candidate": (
        CHECKPOINT_DIGEST_ALIASES + GENERATOR_DIGEST_ALIASES
    ),
    "current_selection": CHECKPOINT_DIGEST_ALIASES + GENERATOR_DIGEST_ALIASES,
    "current_selected_candidate": (
        CHECKPOINT_DIGEST_ALIASES + GENERATOR_DIGEST_ALIASES
    ),
    "policy_access": ("sha256",),
    "heldout_report": (
        "policy_checkpoint_sha256",
        "generator_policy_sha256",
        "policy_generator_sha256",
    ),
    "producer": ("checkpoint_sha256", "generator_policy_sha256"),
    "current_heldout_report": (
        "policy_checkpoint_sha256",
        "generator_policy_sha256",
        "policy_generator_sha256",
    ),
    "current_producer": ("checkpoint_sha256", "generator_policy_sha256"),
}
_CHECKPOINT_SIZE_KEYS = {
    "candidate": CHECKPOINT_SIZE_ALIASES,
    "selected_candidate": CHECKPOINT_SIZE_ALIASES,
    "decision_candidate": CHECKPOINT_SIZE_ALIASES,
    "selection": CHECKPOINT_SIZE_ALIASES,
    "current_decision_candidate": CHECKPOINT_SIZE_ALIASES,
    "current_selection": CHECKPOINT_SIZE_ALIASES,
    "current_selected_candidate": CHECKPOINT_SIZE_ALIASES,
    "policy_access": ("size_bytes",),
    "heldout_report": ("policy_checkpoint_size_bytes",),
    "producer": ("checkpoint_size_bytes",),
    "current_heldout_report": ("policy_checkpoint_size_bytes",),
    "current_producer": ("checkpoint_size_bytes",),
}


def _report_policy_children(
    report: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "decision_candidate": _optional_object(
            decision,
            "candidate",
            source="regeneration report.outer_loop.decision",
        ),
        "selection": _optional_object(
            report,
            "checkpoint_selection",
            source="regeneration report",
        ),
        "selected_candidate": _optional_object(
            report,
            "selected_checkpoint_candidate",
            source="regeneration report",
        ),
        "policy_access": _optional_object(
            report,
            "policy_access",
            source="regeneration report",
        ),
    }


def _producer_sources(
    retained_heldout: dict[str, Any],
    current_heldout: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "producer": _optional_object(
            retained_heldout,
            "policy_inference_provenance",
            source="regeneration report.outer_loop.latest_heldout_report",
        ),
        "current_producer": _optional_object(
            current_heldout,
            "policy_inference_provenance",
            source="current held-out report",
        ),
    }


def _current_checkpoint_sources(
    inner_evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    modern_schema = (
        inner_evidence.get("schema") == "npa.sim2real.inner_loop_evidence.v1"
    )
    identity_keys = {
        "selected_checkpoint_uri",
        "final_checkpoint_uri",
        "checkpoint_selection",
        "checkpoint_candidates",
    }
    if not identity_keys.intersection(inner_evidence):
        if modern_schema:
            raise Sim2RealRerunRegenError(
                "current inner-loop checkpoint identity is missing"
            )
        return {}, {}
    if all(
        inner_evidence.get(key) in (None, "", {}, [])
        for key in identity_keys
        if key in inner_evidence
    ):
        if modern_schema:
            raise Sim2RealRerunRegenError(
                "current inner-loop checkpoint identity is empty"
            )
        return {}, {}
    try:
        return resolve_selected_checkpoint(inner_evidence)
    except ValueError as exc:
        if "schema" not in inner_evidence:
            # Archived pre-identity evidence remains viewable, but contributes no
            # checkpoint access or deployment authority.
            return {}, {}
        raise Sim2RealRerunRegenError(
            f"current inner-loop checkpoint identity is invalid: {exc}"
        ) from exc


def _retained_policy_sources(
    candidate: dict[str, Any],
    report: dict[str, Any],
    heldout_report: dict[str, Any] | None = None,
    inner_evidence: dict[str, Any] | None = None,
    current_decision: dict[str, Any] | None = None,
) -> _RetainedPolicySources:
    outer_loop = _optional_object(report, "outer_loop", source="regeneration report")
    retained_heldout = _optional_object(
        outer_loop,
        "latest_heldout_report",
        source="regeneration report.outer_loop",
    )
    decision = _canonical_decision(outer_loop)
    current_heldout = heldout_report or {}
    current_selection, current_selected_candidate = _current_checkpoint_sources(
        inner_evidence or {}
    )
    current_decision = current_decision or {}
    return _RetainedPolicySources(
        candidate=candidate,
        report=report,
        outer_loop=outer_loop,
        decision=decision,
        current_decision=current_decision,
        current_decision_candidate=_optional_object(
            current_decision,
            "candidate",
            source="current outer-loop decision",
        ),
        current_selection=current_selection,
        current_selected_candidate=current_selected_candidate,
        heldout_report=retained_heldout,
        current_heldout_report=current_heldout,
        current_heldout_report_present=heldout_report is not None,
        **_report_policy_children(report, decision),
        **_producer_sources(retained_heldout, current_heldout),
    )


def _aliased_evidence(
    *groups: tuple[dict[str, Any], str, tuple[str, ...]],
) -> tuple[tuple[str, object], ...]:
    evidence: tuple[tuple[str, object], ...] = ()
    for payload, source, keys in groups:
        for key in keys:
            evidence += _field_evidence(payload, key, f"{source}.{key}")
    return evidence


def _retained_evidence(
    sources: _RetainedPolicySources,
    keys_by_source: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, object], ...]:
    groups = (
        (getattr(sources, name), label, keys_by_source[name])
        for name, label in _RETAINED_SOURCE_LABELS
        if name in keys_by_source
    )
    return _aliased_evidence(*groups)


def _retained_checkpoint_uri(sources: _RetainedPolicySources) -> str:
    return _one_checkpoint_uri(_retained_evidence(sources, _CHECKPOINT_URI_KEYS))


def _retained_checkpoint_digest(sources: _RetainedPolicySources) -> str:
    return _one_checkpoint_digest(_retained_evidence(sources, _CHECKPOINT_DIGEST_KEYS))


def _retained_checkpoint_size(sources: _RetainedPolicySources) -> int:
    return _one_checkpoint_size(_retained_evidence(sources, _CHECKPOINT_SIZE_KEYS))


def _assert_retained_leaf_identity(
    sources: _RetainedPolicySources,
    *,
    checkpoint_uri: str,
) -> None:
    expected = Path(checkpoint_uri).name
    evidence = _aliased_evidence(
        (
            sources.candidate,
            "candidate",
            ("policy_checkpoint_identity", "identity"),
        ),
        (
            sources.selected_candidate,
            "selected_checkpoint_candidate",
            ("policy_checkpoint_identity", "identity"),
        ),
        (
            sources.decision_candidate,
            "outer_loop.decision.candidate",
            ("policy_checkpoint_identity", "identity"),
        ),
        (
            sources.current_decision_candidate,
            "current_outer_loop.decision.candidate",
            ("policy_checkpoint_identity", "identity"),
        ),
        (
            sources.current_selected_candidate,
            "current_selected_checkpoint_candidate",
            ("policy_checkpoint_identity", "identity"),
        ),
        (sources.policy_access, "policy_access", ("identity",)),
    )
    for source, value in evidence:
        if not isinstance(value, str) or value != expected:
            raise Sim2RealRerunRegenError(f"{source} disagrees with checkpoint URI")


def _consensus_claim(
    evidence: tuple[tuple[str, object], ...],
    *,
    default: bool,
) -> bool:
    values: list[bool] = []
    for _source, value in evidence:
        if value is not True and value is not False:
            return False
        values.append(value)
    return all(values) if values else default


def _retained_claim_evidence(
    sources: _RetainedPolicySources,
    key: str,
) -> tuple[tuple[str, object], ...]:
    return _aliased_evidence(
        (sources.candidate, "candidate", (key,)),
        (sources.selected_candidate, "selected_checkpoint_candidate", (key,)),
        (sources.decision_candidate, "outer_loop.decision.candidate", (key,)),
        (sources.report, "report", (key,)),
        (sources.outer_loop, "outer_loop", (key,)),
        (sources.selection, "checkpoint_selection", (key,)),
        (sources.policy_access, "policy_access", (key,)),
        (sources.heldout_report, "latest_heldout_report", (key,)),
        (sources.current_heldout_report, "current_heldout_report", (key,)),
        (sources.decision, "outer_loop.decision", (key,)),
        (sources.current_decision, "current_outer_loop.decision", (key,)),
        (
            sources.current_decision_candidate,
            "current_outer_loop.decision.candidate",
            (key,),
        ),
        (sources.current_selection, "current_checkpoint_selection", (key,)),
        (
            sources.current_selected_candidate,
            "current_selected_checkpoint_candidate",
            (key,),
        ),
    )


def _retained_deployment_claim(
    sources: _RetainedPolicySources,
    *,
    checkpoint_uri: str,
    checkpoint_sha256: str,
    checkpoint_size: int,
) -> bool:
    evidence = _retained_claim_evidence(sources, "deployable_policy")
    for label, decision in (
        ("outer_loop.decision.decision", sources.decision),
        ("current_outer_loop.decision.decision", sources.current_decision),
    ):
        if not decision:
            continue
        decision_name = decision.get("decision")
        evidence += (
            (
                label,
                decision_name == "promote_checkpoint"
                if isinstance(decision_name, str) and decision_name
                else None,
            ),
        )
    deployable = _consensus_claim(evidence, default=False)
    if sources.current_heldout_report_present:
        metadata = _heldout_policy_metadata(sources.current_heldout_report)
        deployable = bool(
            deployable
            and metadata["heldout_policy_loaded_for_inference"] is True
            and metadata["heldout_policy_identity_verified"] is True
            and metadata["heldout_policy_learned_actor_only"] is True
            and sources.current_heldout_report.get("deployable_policy_eval") is True
            and metadata["heldout_policy_checkpoint"] == checkpoint_uri
            and metadata["heldout_policy_checkpoint_sha256"] == checkpoint_sha256
            and metadata["heldout_policy_checkpoint_size_bytes"] == checkpoint_size
        )
    return deployable


def _assert_promoted_decision_identity(sources: _RetainedPolicySources) -> None:
    for label, decision in (
        ("retained promotion decision", sources.decision),
        ("current promotion decision", sources.current_decision),
    ):
        if decision.get("decision") != "promote_checkpoint":
            continue
        if _checkpoint_uri(decision.get("checkpoint_uri")) is None:
            raise Sim2RealRerunRegenError(f"{label} lacks a valid checkpoint URI")


def _retained_bytes_claim(
    sources: _RetainedPolicySources,
    *,
    deployment_claim: bool,
) -> bool:
    evidence = _retained_claim_evidence(sources, "policy_bytes_available")
    return _consensus_claim(evidence, default=deployment_claim)


def _retained_candidate_identity(
    candidate: dict[str, Any],
    report: dict[str, Any],
    heldout_report: dict[str, Any] | None = None,
    inner_evidence: dict[str, Any] | None = None,
    current_decision: dict[str, Any] | None = None,
) -> tuple[str, str, int, bool, bool]:
    sources = _retained_policy_sources(
        candidate,
        report,
        heldout_report,
        inner_evidence,
        current_decision,
    )
    _assert_promoted_decision_identity(sources)
    checkpoint_uri = _retained_checkpoint_uri(sources)
    checkpoint_sha256 = _retained_checkpoint_digest(sources)
    checkpoint_size = _retained_checkpoint_size(sources)
    _assert_retained_leaf_identity(sources, checkpoint_uri=checkpoint_uri)
    deployment_claim = _retained_deployment_claim(
        sources,
        checkpoint_uri=checkpoint_uri,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size=checkpoint_size,
    )
    bytes_claim = _retained_bytes_claim(
        sources,
        deployment_claim=deployment_claim,
    )
    return (
        checkpoint_uri,
        checkpoint_sha256,
        checkpoint_size,
        deployment_claim,
        bytes_claim,
    )


def _hydrate_candidate_identity(
    candidate: dict[str, Any],
    *,
    checkpoint_uri: str,
    storage: StorageClient,
) -> None:
    with tempfile.TemporaryDirectory(prefix="npa-policy-regen-") as temporary:
        local_checkpoint = Path(temporary) / Path(checkpoint_uri).name
        storage.download_file(checkpoint_uri, str(local_checkpoint))
        actual_digest = hashlib.sha256(local_checkpoint.read_bytes()).hexdigest()
        actual_size = local_checkpoint.stat().st_size
        retained_digest = candidate.get("policy_checkpoint_sha256")
        retained_size = candidate.get("policy_checkpoint_size_bytes")
        if retained_digest and retained_digest != actual_digest:
            raise Sim2RealRerunRegenError(
                "downloaded checkpoint SHA-256 disagrees with retained identity"
            )
        if retained_size and retained_size != actual_size:
            raise Sim2RealRerunRegenError(
                "downloaded checkpoint size disagrees with retained identity"
            )
        candidate.update(
            {
                "policy_checkpoint_identity": Path(checkpoint_uri).name,
                "policy_checkpoint_sha256": actual_digest,
                "generator_policy_sha256": actual_digest,
                "policy_checkpoint_size_bytes": actual_size,
            }
        )


def _candidate_identity_complete(
    candidate: dict[str, Any],
    *,
    checkpoint_uri: str,
) -> bool:
    digest = candidate.get("policy_checkpoint_sha256")
    size = candidate.get("policy_checkpoint_size_bytes")
    return bool(
        checkpoint_uri
        and candidate.get("policy_checkpoint_identity") == Path(checkpoint_uri).name
        and isinstance(digest, str)
        and digest == digest.strip().lower()
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size > 0
    )


def _persist_candidate_access(
    candidate_path: Path,
    candidate: dict[str, Any],
    *,
    checkpoint_uri: str,
) -> None:
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
    _write_candidate_manifest(candidate_path, candidate)


def _write_candidate_manifest(
    candidate_path: Path,
    candidate: dict[str, Any],
) -> None:
    _write_json_artifact(candidate_path, candidate)


def _clear_candidate_access(candidate: dict[str, Any]) -> None:
    for key in (
        "policy_download_command",
        "authenticated_download_command",
        "policy_ui_action",
        "ui_action",
    ):
        candidate.pop(key, None)


def _policy_access_record(
    config: Sim2RealLoopConfig,
    candidate: dict[str, Any],
    *,
    checkpoint_uri: str,
    bytes_available: bool,
    deployable: bool,
) -> dict[str, Any]:
    access_allowed = bool(bytes_available and deployable)
    download_command = (
        candidate.get("policy_download_command", "") if access_allowed else ""
    )
    ui_action = candidate.get("policy_ui_action", "") if access_allowed else ""
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
        "authenticated_download_command": download_command,
        "ui_action": ui_action,
        "viewer_executes_policy": False,
    }


def _prepare_candidate_identity(
    candidate: dict[str, Any],
    *,
    checkpoint_uri: str,
    digest: str,
    size: int,
    deployment_claim: bool,
    bytes_claim: bool,
    storage: StorageClient,
) -> tuple[bool, bool]:
    _clear_candidate_access(candidate)
    candidate.update(
        {
            "policy_checkpoint_identity": (
                candidate.get("policy_checkpoint_identity")
                or (Path(checkpoint_uri).name if checkpoint_uri else "")
            ),
            "policy_checkpoint_sha256": digest,
            "policy_checkpoint_size_bytes": size,
            "policy_checkpoint_uri": checkpoint_uri,
        }
    )
    bytes_available = bool(bytes_claim and checkpoint_uri)
    if bytes_available:
        _hydrate_candidate_identity(
            candidate, checkpoint_uri=checkpoint_uri, storage=storage
        )
    deployable = bool(
        deployment_claim
        and bytes_available
        and _candidate_identity_complete(
            candidate,
            checkpoint_uri=checkpoint_uri,
        )
    )
    candidate["deployable_policy"] = deployable
    candidate["policy_bytes_available"] = bytes_available
    return bytes_available, deployable


def _persist_failed_candidate(
    candidate_path: Path,
    candidate: dict[str, Any],
) -> None:
    _clear_candidate_access(candidate)
    candidate["deployable_policy"] = False
    candidate["policy_bytes_available"] = False
    if not candidate_path.is_file():
        return
    try:
        _write_candidate_manifest(candidate_path, candidate)
    except Exception:
        _LOGGER.debug(
            "Unable to persist fail-closed candidate access metadata",
            exc_info=True,
        )


def _assert_candidate_run_ids(
    config: Sim2RealLoopConfig,
    payloads: tuple[tuple[str, dict[str, Any]], ...],
) -> None:
    for source, payload in payloads:
        retained_run_id = payload.get("run_id")
        if retained_run_id is not None and retained_run_id != config.run_id:
            raise Sim2RealRerunRegenError(f"{source} belongs to a different run_id")


def _assert_candidate_uri_scope(
    config: Sim2RealLoopConfig,
    checkpoint_uri: str,
) -> None:
    if not checkpoint_uri:
        return
    expected_bucket, expected_key = _parse_s3(run_prefix_uri(config))
    checkpoint_bucket, checkpoint_key = _parse_s3(checkpoint_uri)
    if (
        checkpoint_bucket != expected_bucket
        or not checkpoint_key.startswith(expected_key)
        or any(part in {".", ".."} for part in checkpoint_key.split("/"))
    ):
        raise Sim2RealRerunRegenError(
            "candidate checkpoint URI is outside the current run prefix"
        )


def _retained_candidate_result(
    config: Sim2RealLoopConfig,
    candidate: dict[str, Any],
    report: dict[str, Any],
    heldout_report: dict[str, Any] | None,
    storage: StorageClient,
    inner_evidence: dict[str, Any] | None = None,
    current_decision: dict[str, Any] | None = None,
) -> tuple[str, bool, bool]:
    payloads = (
        ("candidate checkpoint manifest", candidate),
        ("final report", report),
        ("current held-out report", heldout_report or {}),
        ("inner-loop evidence", inner_evidence or {}),
        ("current outer-loop decision", current_decision or {}),
    )
    _assert_candidate_run_ids(config, payloads)
    uri, digest, size, deployment_claim, bytes_claim = _retained_candidate_identity(
        candidate, report, heldout_report, inner_evidence, current_decision
    )
    _assert_candidate_uri_scope(config, uri)
    bytes_available, deployable = _prepare_candidate_identity(
        candidate,
        checkpoint_uri=uri,
        digest=digest,
        size=size,
        deployment_claim=deployment_claim,
        bytes_claim=bytes_claim,
        storage=storage,
    )
    return uri, bytes_available, deployable


def _prepare_retained_candidate(
    config: Sim2RealLoopConfig,
    candidate_path: Path,
    candidate: dict[str, Any],
    report: dict[str, Any],
    heldout_report: dict[str, Any] | None,
    storage: StorageClient,
    inner_evidence: dict[str, Any] | None = None,
    current_decision: dict[str, Any] | None = None,
) -> tuple[str, bool, bool]:
    try:
        return _retained_candidate_result(
            config,
            candidate,
            report,
            heldout_report,
            storage,
            inner_evidence,
            current_decision,
        )
    except Exception:
        _persist_failed_candidate(candidate_path, candidate)
        raise


def _ensure_policy_access_metadata(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    storage: StorageClient,
    report: dict[str, Any],
    heldout_report: dict[str, Any] | None = None,
    inner_evidence: dict[str, Any] | None = None,
    current_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve candidate bytes and write secret-free access and promotion state."""

    candidate_path, candidate = _read_candidate_manifest(local_dir)
    checkpoint_uri, bytes_available, deployable = _prepare_retained_candidate(
        config,
        candidate_path,
        candidate,
        report,
        heldout_report,
        storage,
        inner_evidence,
        current_decision,
    )
    if bytes_available and deployable:
        _persist_candidate_access(
            candidate_path, candidate, checkpoint_uri=checkpoint_uri
        )
    elif candidate_path.is_file():
        _write_candidate_manifest(candidate_path, candidate)
    return _policy_access_record(
        config,
        candidate,
        checkpoint_uri=checkpoint_uri,
        bytes_available=bytes_available,
        deployable=deployable,
    )


def _sync_heldout_eval_inputs(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    prefix: str,
    *,
    outer_iteration: int,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    if (
        not isinstance(outer_iteration, int)
        or isinstance(outer_iteration, bool)
        or outer_iteration <= 0
    ):
        raise Sim2RealRerunRegenError("outer_iteration must be a positive integer")
    if not _download_directory_fresh(
        storage,
        f"{prefix}envs/gold-heldout/",
        work_dir / "envs" / "gold-heldout",
        containment_root=work_dir,
    ):
        raise Sim2RealRerunRegenError(
            f"failed to sync envs/gold-heldout for {config.run_id}"
        )

    outer_name = f"outer-{outer_iteration:02d}"
    inner_path = work_dir / "inner_loop" / outer_name / "evidence.json"
    inner_path.parent.mkdir(parents=True, exist_ok=True)
    if not _download_if_exists(
        storage,
        f"{prefix}inner_loop/{outer_name}/evidence.json",
        inner_path,
        containment_root=work_dir,
    ):
        raise Sim2RealRerunRegenError(
            f"missing inner evidence at {prefix}inner_loop/{outer_name}/evidence.json"
        )
    return _read_retained_json(inner_path, source="inner-loop evidence")


def _resolve_heldout_eval_checkpoint(
    config: Sim2RealLoopConfig,
    prefix: str,
    evidence: dict[str, Any],
    *,
    outer_iteration: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recorded_outer = evidence.get("outer_iteration")
    if (
        not isinstance(recorded_outer, int)
        or isinstance(recorded_outer, bool)
        or recorded_outer != outer_iteration
    ):
        raise Sim2RealRerunRegenError(
            "held-out rerun evidence outer iteration disagrees with the request"
        )
    try:
        validate_stage10_input_scope(
            argparse.Namespace(
                run_id=config.run_id,
                outer_iteration=outer_iteration,
            ),
            root=prefix.rstrip("/"),
            evidence=evidence,
        )
        return resolve_run_scoped_checkpoint(
            evidence,
            run_root=prefix.rstrip("/"),
            run_id=config.run_id,
        )
    except (RuntimeError, ValueError) as exc:
        raise Sim2RealRerunRegenError(
            "held-out rerun selected checkpoint is not current-run evidence"
        ) from exc


def _assert_heldout_report_publishable(
    report: dict[str, Any],
    selection: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    metadata = selected_checkpoint_policy_metadata(report, selection, candidate)
    if (
        report.get("deployable_policy_eval") is not True
        or metadata.get("heldout_policy_loaded_for_inference") is not True
        or metadata.get("heldout_policy_identity_verified") is not True
        or metadata.get("heldout_policy_learned_actor_only") is not True
    ):
        raise Sim2RealRerunRegenError(
            "held-out rerun refuses to publish nondeployable or unverified "
            "learned-policy evidence"
        )


def _heldout_render_producer(
    config: Sim2RealLoopConfig,
    report: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    invocation = report.get("component_invocation") or {}
    if not isinstance(invocation, dict):
        raise Sim2RealRerunRegenError(
            "held-out render component invocation must be an object"
        )
    output_uri = str(invocation.get("output_uri") or "").strip()
    prefix = run_prefix_uri(config)
    if output_uri and not output_uri.startswith(prefix):
        raise Sim2RealRerunRegenError(
            "held-out render producer is outside the current run"
        )
    manifest = report.get("render_manifest") or {}
    if not isinstance(manifest, dict):
        raise Sim2RealRerunRegenError("held-out render manifest must be an object")
    manifest_uri = str(manifest.get("renders_s3_uri") or "").strip()
    if manifest_uri and not manifest_uri.startswith(prefix):
        raise Sim2RealRerunRegenError(
            "held-out render manifest producer is outside the current run"
        )
    return output_uri, manifest, manifest_uri


def _assert_current_render_manifest(
    config: Sim2RealLoopConfig,
    report: dict[str, Any],
    manifest: dict[str, Any],
    manifest_uri: str,
) -> None:
    if not manifest_uri:
        raise Sim2RealRerunRegenError(
            "held-out render manifest lacks its immutable producer URI"
        )
    outer = report.get("outer_iteration")
    attempt_tag = manifest.get("evaluation_attempt_tag")
    base = f"gold_heldout-outer-{outer:02d}-attempt-" if type(outer) is int else ""
    valid_attempt = (
        type(outer) is int
        and outer > 0
        and isinstance(attempt_tag, str)
        and attempt_tag.startswith(base)
        and len(attempt_tag) == len(base) + 32
        and report.get("evaluation_attempt_tag") == attempt_tag
        and all(char in "0123456789abcdef" for char in attempt_tag[len(base) :])
    )
    expected_uri = (
        expected_byo_render_prefix(
            root=run_prefix_uri(config).rstrip("/"),
            run_id=config.run_id,
            outer_iteration=outer,
            evaluation_tag=attempt_tag,
        )
        if valid_attempt
        else ""
    )
    if not valid_attempt or manifest_uri.rstrip("/") + "/" != expected_uri:
        raise Sim2RealRerunRegenError(
            "held-out render manifest is not the current immutable producer"
        )


def _sync_heldout_eval_renders(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    report: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    _output_uri, manifest, manifest_uri = _heldout_render_producer(config, report)
    _assert_current_render_manifest(config, report, manifest, manifest_uri)
    renders_dir = _renders_dir_for_report(config, work_dir, report)
    synced = _download_render_tree(
        storage,
        manifest_uri.rstrip("/") + "/",
        renders_dir,
        containment_root=work_dir,
    )
    if not synced:
        raise Sim2RealRerunRegenError(
            "held-out rerun immutable producer renders could not be synchronized"
        )
    try:
        validate_materialized_render_tree(
            renders_dir,
            manifest,
            candidate,
            require_frame_identity=True,
        )
    except RuntimeError as exc:
        raise Sim2RealRerunRegenError(
            f"held-out rerun render authority is invalid: {exc}"
        ) from exc


def _execute_heldout_eval(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    outer_iteration: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from npa.workflows.sim2real.engine import run_heldout_eval

    prefix = run_prefix_uri(config)
    inner_evidence = _sync_heldout_eval_inputs(
        config,
        work_dir,
        storage,
        prefix,
        outer_iteration=outer_iteration,
    )
    selection, candidate = _resolve_heldout_eval_checkpoint(
        config,
        prefix,
        inner_evidence,
        outer_iteration=outer_iteration,
    )
    report = run_heldout_eval(
        config,
        local_dir=work_dir,
        inner_evidence=inner_evidence,
        outer_iteration=outer_iteration,
        evaluation_split="gold_heldout",
    )
    return report, selection, candidate


def _heldout_eval_work_dir(
    config: Sim2RealLoopConfig,
    local_dir: Path | None,
) -> Path:
    return (
        Path(local_dir)
        if local_dir is not None
        else default_regen_local_dir(config.run_id)
    )


def rerun_heldout_eval_only(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path | None = None,
    outer_iteration: int = 1,
    publish: bool = True,
    client: StorageClient | None = None,
) -> dict[str, Any]:
    """Re-run Stage 10 gold evaluation for one existing run.

    Args:
        config: Existing run configuration.
        local_dir: Optional run-scoped regeneration directory.
        outer_iteration: Exact outer-loop iteration to evaluate.
        publish: Whether to publish validated results.
        client: Optional configured storage client.
    Returns:
        The current held-out evaluation report.
    Raises:
        Sim2RealRerunRegenError: If run authority or publication evidence fails.
    """
    work_dir = _heldout_eval_work_dir(config, local_dir)
    storage = client or _storage_client_for_config(config)
    report, selection, candidate = _execute_heldout_eval(
        config, work_dir, storage, outer_iteration
    )
    if publish:
        _assert_heldout_report_publishable(report, selection, candidate)
    _sync_heldout_eval_renders(config, work_dir, storage, report, candidate)
    if publish:
        _publish_heldout_eval_outputs(
            config,
            work_dir,
            report,
            outer_iteration=outer_iteration,
            storage=storage,
        )
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
