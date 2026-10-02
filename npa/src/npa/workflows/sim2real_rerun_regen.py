"""Regenerate Sim2Real viewer recordings and optionally re-run held-out Isaac capture."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import tempfile
import time
from dataclasses import dataclass
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
)
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real.reporting import build_progress_metrics
from npa.workflows.sim2real.utils import _artifact_root_uri, _write_json_artifact
from npa.workflows.sim2real.viz_contract import _checkpoint_uri
from npa.workflows.sim2real.workflow_io import parse_json_object
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

    def contained(
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

    report_split = report.get("evaluation_split")
    lineage_split = lineage.get("evaluation_split")
    for source, split in (
        ("held-out render report", report_split),
        ("held-out render lineage", lineage_split),
    ):
        if split not in (None, "", "gold_heldout"):
            raise Sim2RealRerunRegenError(f"{source} has the wrong evaluation split")
    sealed = report_split == "gold_heldout" or lineage_split == "gold_heldout"
    sealed_default: Path | None = None
    if sealed:
        raw_outer = report.get("outer_iteration")
        if (
            not isinstance(raw_outer, int)
            or isinstance(raw_outer, bool)
            or raw_outer <= 0
        ):
            raise Sim2RealRerunRegenError(
                "sealed gold report has an invalid outer iteration"
            )
        sealed_default = contained(
            f"eval/gold-heldout/outer-{raw_outer:02d}/renders",
            source="default held-out",
            require_relative=True,
        )

    candidates: list[tuple[str, Path]] = []
    if report.get("local_renders_dir") not in (None, ""):
        candidates.append(
            (
                "local_renders_dir",
                contained(
                    report["local_renders_dir"],
                    source="local_renders_dir",
                ),
            )
        )
    if lineage.get("local_relative_dir") not in (None, ""):
        candidates.append(
            (
                "render_lineage.local_relative_dir",
                contained(
                    lineage["local_relative_dir"],
                    source="render_lineage.local_relative_dir",
                    require_relative=True,
                ),
            )
        )
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
    return contained(
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
            _assert_no_symlinked_ancestors(
                local_path,
                containment_root=containment_root,
            )
            local_path.unlink(missing_ok=True)
            return False
        if not staged.is_file() or staged.stat().st_size <= 0:
            _assert_no_symlinked_ancestors(
                local_path,
                containment_root=containment_root,
            )
            local_path.unlink(missing_ok=True)
            return False
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

    inner_evidence_rel = _latest_completed_inner_evidence_rel(storage, prefix)
    gold_eval_rel = _gold_eval_relative_dir(inner_evidence_rel).as_posix()
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
        _download_if_exists(
            storage,
            f"{prefix}{rel}",
            dest,
            containment_root=local_dir,
        )

    # The viewer plots improvement across every outer/inner pass, not only the
    # latest evidence object selected for backward compatibility above.
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
        heldout_report = _read_retained_json(
            heldout_path,
            source="held-out report",
        )
    sync_heldout_renders(
        config, local_dir, heldout_report=heldout_report, client=storage
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


def _heldout_render_source(
    config: Sim2RealLoopConfig,
    heldout_report: dict[str, Any],
) -> tuple[str, bool]:
    prefix = run_prefix_uri(config)
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
    if not sealed:
        return f"{prefix}eval/heldout/renders/", False
    render_uri = _sealed_render_uri(lineage)
    if lineage_split not in (None, "gold_heldout"):
        raise Sim2RealRerunRegenError(
            "sealed gold render lineage has the wrong evaluation split"
        )
    outer = heldout_report.get("outer_iteration")
    if not isinstance(outer, int) or isinstance(outer, bool) or outer <= 0:
        raise Sim2RealRerunRegenError(
            "sealed gold report has an invalid outer iteration"
        )
    expected_uri = f"{prefix}eval/gold-heldout/outer-{outer:02d}/renders/"
    expected_legacy_uri = (
        f"{prefix}component-io/heldout-eval/"
        f"gold_heldout-outer-{outer:02d}/output/renders/"
    )
    has_canonical = lineage.get("canonical_s3_uri") not in (None, "")
    allowed_uris = (
        {expected_uri} if has_canonical else {expected_uri, expected_legacy_uri}
    )
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
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def _rewrite_inner_evidence_paths(local_dir: Path, evidence_path: Path) -> None:
    try:
        payload = parse_json_object(
            Path(evidence_path).read_text(encoding="utf-8"),
            source="inner-loop evidence",
        )
    except (OSError, ValueError):
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
    if not _download_if_exists(
        storage,
        uri,
        absolute_dest,
        containment_root=containment_root,
    ):
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
    local_dir = Path(os.path.abspath(local_dir))

    report_path = _gold_report_path(config, local_dir)
    if report_path.is_file():
        rel = report_path.relative_to(local_dir).as_posix()
        storage.upload_file(str(report_path), f"{prefix}{rel}")

    renders_dir = _renders_dir_for_report(
        config,
        local_dir,
        _read_retained_json(report_path, source="held-out report")
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


@dataclass(frozen=True)
class _RegenState:
    inner_evidence: dict[str, Any]
    heldout_report: dict[str, Any]
    report_path: Path
    report: dict[str, Any]
    policy_access: dict[str, Any]


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


def _load_regen_state(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    *,
    sync_inputs: bool,
) -> _RegenState:
    try:
        if sync_inputs:
            sync_regen_inputs(config, work_dir, client=storage)
        inner_path = _latest_local_inner_evidence(work_dir)
        _rewrite_inner_evidence_paths(work_dir, inner_path)
        heldout_path = _gold_report_path(config, work_dir)
        if not inner_path.is_file():
            raise Sim2RealRerunRegenError(f"missing inner evidence: {inner_path}")
        if not heldout_path.is_file():
            raise Sim2RealRerunRegenError(f"missing held-out report: {heldout_path}")
        inner_evidence = _read_retained_json(inner_path, source="inner-loop evidence")
        heldout_report = _read_retained_json(heldout_path, source="held-out report")
        report_path = work_dir / "reports" / "sim2real-report.json"
        report = (
            _read_retained_json(report_path, source="Sim2Real final report")
            if report_path.is_file()
            else {}
        )
        decision_path = work_dir / "outer_loop" / "decision.json"
        current_decision = (
            _read_retained_json(decision_path, source="current outer-loop decision")
            if decision_path.is_file()
            else {}
        )
        policy_access = _ensure_policy_access_metadata(
            config,
            work_dir,
            storage=storage,
            report=report,
            heldout_report=heldout_report,
            inner_evidence=inner_evidence,
            current_decision=current_decision,
        )
    except Exception:
        _fail_closed_local_candidate(work_dir)
        raise
    return _RegenState(
        inner_evidence,
        heldout_report,
        report_path,
        report,
        policy_access,
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


def _publish_regen_recordings(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    *,
    upload: bool,
) -> tuple[str, str]:
    if not upload:
        return "", ""
    return (
        publish_regen_outputs(config, work_dir, client=storage),
        publish_regen_mcap(config, work_dir, client=storage),
    )


def _finalize_regen_result(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    state: _RegenState,
    result: Sim2RealVizResult,
    *,
    upload: bool,
) -> RegenResult:
    mcap_result = emit_sim2real_mcap_if_enabled(
        local_dir=work_dir,
        inner_evidence=state.inner_evidence,
        heldout_report=state.heldout_report,
        output_mcap=work_dir / "reports" / "sim2real.mcap",
    )
    upload_uri, mcap_upload_uri = _publish_regen_recordings(
        config, work_dir, storage, upload=upload
    )
    return _regen_result_from_viz(
        config.run_id,
        work_dir,
        result,
        upload_uri=upload_uri,
        mcap_result=mcap_result,
        mcap_upload_uri=mcap_upload_uri,
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
    """Sync artifacts (optional), emit .rrd locally, optionally upload to S3."""

    work_dir, output_rrd = _regen_paths(config, local_dir, local_rrd_path)
    storage = client or _storage_client_for_config(config)
    state = _load_regen_state(config, work_dir, storage, sync_inputs=sync_inputs)
    outer_history = list((state.report.get("outer_loop") or {}).get("history") or [])
    viewer_command = _viewer_command(config)
    result, duration_s = _emit_regen_rrd(
        config, work_dir, output_rrd, state, outer_history, viewer_command
    )
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
        config, work_dir, storage, state, result, upload=upload
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
    identity_keys = {
        "selected_checkpoint_uri",
        "final_checkpoint_uri",
        "checkpoint_selection",
        "checkpoint_candidates",
    }
    if not identity_keys.intersection(inner_evidence):
        return {}, {}
    if all(
        inner_evidence.get(key) in (None, "", {}, [])
        for key in identity_keys
        if key in inner_evidence
    ):
        return {}, {}
    try:
        return resolve_selected_checkpoint(inner_evidence)
    except ValueError as exc:
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


def _prepare_retained_candidate(
    candidate_path: Path,
    candidate: dict[str, Any],
    report: dict[str, Any],
    heldout_report: dict[str, Any] | None,
    storage: StorageClient,
    inner_evidence: dict[str, Any] | None = None,
    current_decision: dict[str, Any] | None = None,
) -> tuple[str, bool, bool]:
    try:
        uri, digest, size, deployment_claim, bytes_claim = _retained_candidate_identity(
            candidate,
            report,
            heldout_report,
            inner_evidence,
            current_decision,
        )
        bytes_available, deployable = _prepare_candidate_identity(
            candidate,
            checkpoint_uri=uri,
            digest=digest,
            size=size,
            deployment_claim=deployment_claim,
            bytes_claim=bytes_claim,
            storage=storage,
        )
    except Exception:
        _persist_failed_candidate(candidate_path, candidate)
        raise
    return uri, bytes_available, deployable


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
) -> dict[str, Any]:
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
        storage,
        f"{prefix}inner_loop/outer-01/evidence.json",
        inner_path,
        containment_root=work_dir,
    ):
        raise Sim2RealRerunRegenError(
            f"missing inner evidence at {prefix}inner_loop/outer-01/evidence.json"
        )
    return _read_retained_json(inner_path, source="inner-loop evidence")


def _sync_heldout_eval_renders(
    config: Sim2RealLoopConfig,
    work_dir: Path,
    storage: StorageClient,
    report: dict[str, Any],
) -> None:
    invocation = report.get("component_invocation") or {}
    output_uri = str(invocation.get("output_uri") or "").strip()
    renders_dir = _renders_dir_for_report(config, work_dir, report)
    synced = bool(
        output_uri
        and _download_render_tree(
            storage,
            _sibling_uri(output_uri, "renders/"),
            renders_dir,
            containment_root=work_dir,
        )
    )
    if not synced:
        synced = sync_heldout_renders(
            config, work_dir, heldout_report=report, client=storage
        )
    if not synced or not _has_camera_pngs(renders_dir):
        raise Sim2RealRerunRegenError(
            "held-out rerun completed but no camera-*.png renders were synced; "
            "check NPA_SIM2REAL_HELDOUT_RENDER_FRAMES=1 and Isaac sibling logs"
        )


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
    inner_evidence = _sync_heldout_eval_inputs(config, work_dir, storage, prefix)
    report = run_heldout_eval(
        config,
        local_dir=work_dir,
        inner_evidence=inner_evidence,
        outer_iteration=outer_iteration,
    )
    _sync_heldout_eval_renders(config, work_dir, storage, report)
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
