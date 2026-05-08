"""LeRobotDataset import helpers for FiftyOne.

This module intentionally parses LeRobot parquet metadata directly instead of
depending on the ``lerobot`` package. It is copied to FiftyOne workbench VMs by
the CLI and executed inside the FiftyOne virtualenv.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
SUCCESS_TERMS = (
    "task_success",
    "is_success",
    "episode_success",
    "success",
    "succeeded",
    "failure",
    "failed",
    "outcome",
    "result",
    "reward",
    "done",
    "terminal",
)


@dataclass
class LeRobotSampleSpec:
    filepath: str
    fields: dict[str, Any]
    tags: list[str]


@dataclass
class LeRobotImportPlan:
    samples: list[LeRobotSampleSpec]
    warnings: list[str]
    media_keys: list[str]
    metadata_fields: list[str]


def materialize_lerobot_source(source: str, name: str, datasets_dir: Path) -> tuple[Path, str]:
    """Return a local LeRobotDataset directory for a local, S3, or HF source."""
    source = source.strip()
    if not source:
        raise ValueError("source must not be empty")

    if source.startswith("s3://"):
        return _download_s3_source(source, name, datasets_dir), "s3"

    local = Path(source).expanduser()
    if local.exists():
        return local, "local"

    return _download_huggingface_source(source, name, datasets_dir), "huggingface"


def build_lerobot_import_plan(root: Path, output_dir: Path) -> LeRobotImportPlan:
    """Parse a local LeRobotDataset and build FiftyOne sample specs.

    The parser is intentionally schema-tolerant. It maps fields when columns are
    present and records warnings instead of failing when optional fields are not
    available.
    """
    root = root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    warnings: list[str] = []
    info = _read_info_json(root, warnings)
    data_rows = _read_data_rows(root)
    if not data_rows:
        raise RuntimeError(f"No LeRobot data rows found under {root}")

    episode_rows = _read_episode_rows(root, warnings)
    episode_by_index = {
        int(row["episode_index"]): row
        for row in episode_rows
        if _coerce_int(row.get("episode_index")) is not None
    }
    task_by_index = _read_tasks(root, warnings)

    columns = set()
    for row in data_rows:
        columns.update(row)

    episode_col = _find_column(columns, ("episode_index", "episode", "episode_idx"))
    frame_col = _find_column(columns, ("frame_index", "step_index", "step", "frame"))
    timestamp_col = _find_column(columns, ("timestamp", "time", "time_sec"))
    task_index_col = _find_column(columns, ("task_index", "task_id"))
    frame_success_col = _find_success_column(columns)
    episode_success_col = _find_success_column(_episode_columns(episode_rows))

    metadata_fields: list[str] = []
    if episode_col:
        metadata_fields.append("episode_index")
    else:
        warnings.append("LeRobot data parquet has no episode_index column; skipping episode_index field")
    if frame_col:
        metadata_fields.append("frame_index")
    else:
        warnings.append("LeRobot data parquet has no frame_index column; skipping frame_index field")
    if timestamp_col:
        metadata_fields.append("timestamp")
    else:
        warnings.append("LeRobot data parquet has no timestamp column; skipping timestamp field")
    if frame_success_col or episode_success_col:
        metadata_fields.append("task_success")
    else:
        warnings.append("LeRobot metadata has no success/failure column; skipping task_success field")

    video_keys, image_keys = _discover_media_keys(root, info, columns)
    if not video_keys and not image_keys:
        raise RuntimeError(
            "Could not find observation image or video features in LeRobot dataset"
        )

    rows = _normalize_rows(
        data_rows,
        episode_col=episode_col,
        frame_col=frame_col,
        timestamp_col=timestamp_col,
        task_index_col=task_index_col,
        frame_success_col=frame_success_col,
        episode_success_col=episode_success_col,
        episode_by_index=episode_by_index,
        task_by_index=task_by_index,
    )

    samples: list[LeRobotSampleSpec] = []
    for image_key in image_keys:
        samples.extend(_build_image_samples(root, rows, image_key, metadata_fields, warnings))
    for video_key in video_keys:
        samples.extend(
            _build_video_samples(root, output_dir, rows, episode_by_index, info, video_key, metadata_fields, warnings)
        )

    if not samples:
        raise RuntimeError("LeRobot metadata parsed successfully, but no media samples were found")

    media_keys = [*image_keys, *video_keys]
    return LeRobotImportPlan(
        samples=samples,
        warnings=_dedupe(warnings),
        media_keys=media_keys,
        metadata_fields=metadata_fields,
    )


def import_lerobot_dataset(name: str, source: str, datasets_dir: Path) -> dict[str, Any]:
    """Materialize and import a LeRobotDataset into FiftyOne."""
    import fiftyone as fo

    datasets_dir = datasets_dir.expanduser().resolve()
    source_root, source_type = materialize_lerobot_source(source, name, datasets_dir)
    output_dir = datasets_dir / name / "lerobot_frames"
    plan = build_lerobot_import_plan(source_root, output_dir)

    if name in fo.list_datasets():
        fo.delete_dataset(name)

    dataset = fo.Dataset(name=name)
    dataset.persistent = True

    fo_samples = []
    for spec in plan.samples:
        sample = fo.Sample(filepath=spec.filepath, tags=spec.tags)
        for field, value in spec.fields.items():
            if value is not None:
                sample[field] = value
        fo_samples.append(sample)

    dataset.add_samples(fo_samples)
    dataset.save()

    return {
        "status": "loaded",
        "name": dataset.name,
        "source": source,
        "source_type": source_type,
        "format": "lerobot",
        "samples": len(dataset),
        "media_keys": plan.media_keys,
        "metadata_fields": plan.metadata_fields,
        "warnings": plan.warnings,
    }


def _download_s3_source(uri: str, name: str, datasets_dir: Path) -> Path:
    import boto3

    parsed = urlparse(uri)
    if not parsed.netloc:
        raise ValueError(f"S3 URI must include a bucket: {uri}")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    target_root = datasets_dir / name / "lerobot_source" / "s3"
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    endpoint = os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or None
    s3 = boto3.client("s3", endpoint_url=endpoint)
    paginator = s3.get_paginator("list_objects_v2")

    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):].lstrip("/") if prefix else key
            if not rel:
                continue
            dest = target_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            count += 1

    if count == 0:
        raise RuntimeError(f"No S3 objects found at {uri}")
    return target_root


def _download_huggingface_source(repo_id: str, name: str, datasets_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    target_root = datasets_dir / name / "lerobot_source" / "hf"
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_root),
    )
    return target_root


def _read_info_json(root: Path, warnings: list[str]) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.exists():
        warnings.append("LeRobot meta/info.json not found; media features will be inferred from files")
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        warnings.append(f"Could not parse meta/info.json: {exc}")
        return {}


def _read_table(path: Path):
    import pyarrow.parquet as pq

    return pq.read_table(path)


def _read_data_rows(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        paths = [
            path
            for path in sorted(root.glob("**/*.parquet"))
            if "/meta/" not in path.as_posix()
        ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_table(path).to_pylist())
    return rows


def _read_episode_rows(root: Path, warnings: list[str]) -> list[dict[str, Any]]:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            rows.extend(_read_table(path).to_pylist())
        except Exception as exc:
            warnings.append(f"Could not read episode metadata {path}: {exc}")
    return rows


def _read_tasks(root: Path, warnings: list[str]) -> dict[int, str]:
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        return {}
    try:
        rows = _read_table(path).to_pylist()
    except Exception as exc:
        warnings.append(f"Could not read task metadata {path}: {exc}")
        return {}
    tasks: dict[int, str] = {}
    for row in rows:
        idx = _coerce_int(row.get("task_index"))
        task = row.get("task")
        if idx is not None and task is not None:
            tasks[idx] = str(task)
    return tasks


def _episode_columns(rows: list[dict[str, Any]]) -> set[str]:
    columns: set[str] = set()
    for row in rows:
        columns.update(row)
    return columns


def _find_column(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def _find_success_column(columns: set[str]) -> str | None:
    for col in sorted(columns):
        normalized = col.lower()
        if normalized.startswith("stats/"):
            continue
        if any(term in normalized for term in SUCCESS_TERMS):
            return col
    return None


def _discover_media_keys(
    root: Path,
    info: dict[str, Any],
    data_columns: set[str],
) -> tuple[list[str], list[str]]:
    video_keys: list[str] = []
    image_keys: list[str] = []

    for key, spec in (info.get("features") or {}).items():
        dtype = str((spec or {}).get("dtype", "")).lower()
        if dtype == "video":
            video_keys.append(key)
        elif dtype in {"image", "pil", "path"}:
            image_keys.append(key)

    for key in data_columns:
        if not key.startswith(("observation.image", "observation.images")):
            continue
        if _has_files(root / "videos" / key, VIDEO_EXTENSIONS):
            video_keys.append(key)
        elif _has_files(root / "images" / key, IMAGE_EXTENSIONS):
            image_keys.append(key)
        elif key not in video_keys:
            image_keys.append(key)

    if not video_keys:
        videos_root = root / "videos"
        if videos_root.exists():
            for path in sorted(videos_root.iterdir()):
                if path.is_dir() and _has_files(path, VIDEO_EXTENSIONS):
                    video_keys.append(path.name)

    if not image_keys:
        images_root = root / "images"
        if images_root.exists():
            for path in sorted(images_root.iterdir()):
                if path.is_dir() and _has_files(path, IMAGE_EXTENSIONS):
                    image_keys.append(path.name)

    return _dedupe(video_keys), _dedupe(image_keys)


def _normalize_rows(
    data_rows: list[dict[str, Any]],
    *,
    episode_col: str | None,
    frame_col: str | None,
    timestamp_col: str | None,
    task_index_col: str | None,
    frame_success_col: str | None,
    episode_success_col: str | None,
    episode_by_index: dict[int, dict[str, Any]],
    task_by_index: dict[int, str],
) -> list[dict[str, Any]]:
    per_episode_frame: dict[int, int] = {}
    normalized: list[dict[str, Any]] = []
    for row_index, row in enumerate(data_rows):
        episode = _coerce_int(row.get(episode_col)) if episode_col else 0
        if episode is None:
            episode = 0
        inferred_frame = per_episode_frame.get(episode, 0)
        frame = _coerce_int(row.get(frame_col)) if frame_col else inferred_frame
        if frame is None:
            frame = inferred_frame
        per_episode_frame[episode] = max(per_episode_frame.get(episode, 0), frame + 1)

        timestamp = _coerce_float(row.get(timestamp_col)) if timestamp_col else None
        task_index = _coerce_int(row.get(task_index_col)) if task_index_col else None
        episode_row = episode_by_index.get(episode, {})
        task = task_by_index.get(task_index) if task_index is not None else None
        if task is None:
            tasks = episode_row.get("tasks")
            if isinstance(tasks, list) and tasks:
                task = str(tasks[0])

        success = None
        if frame_success_col:
            success = _coerce_success(row.get(frame_success_col), frame_success_col)
        if success is None and episode_success_col:
            success = _coerce_success(episode_row.get(episode_success_col), episode_success_col)

        normalized.append(
            {
                "raw": row,
                "_row_index": row_index,
                "_episode": episode,
                "_frame": frame,
                "episode_index": episode if episode_col else None,
                "frame_index": frame if frame_col else None,
                "timestamp": timestamp,
                "task_success": success,
                "task_index": task_index,
                "task": task,
            }
        )
    return normalized


def _build_image_samples(
    root: Path,
    rows: list[dict[str, Any]],
    image_key: str,
    metadata_fields: list[str],
    warnings: list[str],
) -> list[LeRobotSampleSpec]:
    samples: list[LeRobotSampleSpec] = []
    fallback_images = _sorted_media(root / "images" / image_key, IMAGE_EXTENSIONS)
    for idx, row in enumerate(rows):
        raw_value = row["raw"].get(image_key)
        image_path = _resolve_image_path(root, image_key, raw_value)
        if image_path is None and idx < len(fallback_images):
            image_path = fallback_images[idx]
        if image_path is None or not image_path.exists():
            if idx == 0:
                warnings.append(f"No image files found for LeRobot feature {image_key}; skipping feature")
            continue
        samples.append(
            LeRobotSampleSpec(
                filepath=str(image_path),
                fields=_sample_fields(row, image_key, metadata_fields),
                tags=["lerobot", _safe_name(image_key)],
            )
        )
    return samples


def _build_video_samples(
    root: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    episode_by_index: dict[int, dict[str, Any]],
    info: dict[str, Any],
    video_key: str,
    metadata_fields: list[str],
    warnings: list[str],
) -> list[LeRobotSampleSpec]:
    samples: list[LeRobotSampleSpec] = []
    rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_episode.setdefault(int(row["_episode"]), []).append(row)

    for episode, episode_rows in sorted(rows_by_episode.items()):
        video_path = _resolve_video_path(root, info, video_key, episode_by_index.get(episode, {}), episode)
        if video_path is None or not video_path.exists():
            warnings.append(f"Video for feature {video_key}, episode {episode} was not found; skipping episode")
            continue

        frame_dir = output_dir / _safe_name(video_key) / f"episode_{episode:06d}"
        extracted = _extract_video_frames(video_path, frame_dir)
        if not extracted:
            warnings.append(f"No frames extracted from {video_path}; skipping episode")
            continue
        by_frame = _index_extracted_frames(extracted)

        for row in sorted(episode_rows, key=lambda item: int(item["_frame"])):
            frame = int(row["_frame"])
            image_path = by_frame.get(frame)
            if image_path is None and frame < len(extracted):
                image_path = extracted[frame]
            if image_path is None:
                warnings.append(f"Missing extracted frame {frame} for {video_key}, episode {episode}")
                continue
            samples.append(
                LeRobotSampleSpec(
                    filepath=str(image_path),
                    fields=_sample_fields(row, video_key, metadata_fields),
                    tags=["lerobot", _safe_name(video_key)],
                )
            )
    return samples


def _sample_fields(row: dict[str, Any], media_key: str, metadata_fields: list[str]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "source_video_key": media_key,
        "camera": _media_label(media_key),
    }
    for field in metadata_fields:
        value = row.get(field)
        if value is not None:
            fields[field] = value
    if row.get("task_index") is not None:
        fields["task_index"] = row["task_index"]
    if row.get("task") is not None:
        fields["task"] = row["task"]
    return fields


def _resolve_image_path(root: Path, image_key: str, raw_value: Any) -> Path | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        for key in ("path", "filepath", "file_name", "filename", "image_path"):
            if key in raw_value:
                return _resolve_image_path(root, image_key, raw_value[key])
        return None
    if isinstance(raw_value, bytes):
        return None
    value = str(raw_value)
    if not value:
        return None
    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                root / path,
                root / "images" / image_key / path,
                root / "videos" / image_key / path,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _resolve_video_path(
    root: Path,
    info: dict[str, Any],
    video_key: str,
    episode_row: dict[str, Any],
    episode: int,
) -> Path | None:
    chunk_index = _coerce_int(episode_row.get(f"videos/{video_key}/chunk_index")) or 0
    file_index = _coerce_int(episode_row.get(f"videos/{video_key}/file_index"))
    if file_index is None:
        file_index = episode

    template = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    try:
        rel = template.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
            episode_index=episode,
        )
        candidate = root / rel
        if candidate.exists():
            return candidate
    except Exception:
        pass

    feature_dir = root / "videos" / video_key
    media = _sorted_media(feature_dir, VIDEO_EXTENSIONS)
    for path in media:
        if path.stem.endswith(f"{file_index:03d}") or path.stem.endswith(str(file_index)):
            return path
    if 0 <= file_index < len(media):
        return media[file_index]
    return None


def _extract_video_frames(video_path: Path, frame_dir: Path) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = _sorted_media(frame_dir, IMAGE_EXTENSIONS)
    if existing:
        return existing

    output_pattern = frame_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-start_number",
        "0",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    subprocess.run(cmd, check=True)
    return _sorted_media(frame_dir, IMAGE_EXTENSIONS)


def _index_extracted_frames(frames: list[Path]) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in frames:
        match = re.search(r"(\d+)$", path.stem)
        if match:
            indexed[int(match.group(1))] = path
    return indexed


def _has_files(root: Path, extensions: set[str]) -> bool:
    return bool(_sorted_media(root, extensions, limit=1))


def _sorted_media(root: Path, extensions: set[str], limit: int | None = None) -> list[Path]:
    if not root.exists():
        return []
    paths = sorted(
        path
        for path in root.glob("**/*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    return paths[:limit] if limit else paths


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return cleaned or "media"


def _media_label(media_key: str) -> str:
    return media_key.rsplit(".", 1)[-1].replace("/", "_")


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_success(value: Any, column: str) -> bool | None:
    if value is None:
        return None
    invert = "fail" in column.lower() or "failure" in column.lower()
    result: bool | None
    if isinstance(value, bool):
        result = value
    elif isinstance(value, (int, float)):
        if value not in (0, 1, 0.0, 1.0):
            return None
        result = bool(value)
    else:
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "success", "succeeded", "pass", "passed", "done"}:
            result = True
        elif normalized in {"false", "0", "no", "failure", "failed", "fail"}:
            result = False
        else:
            return None
    return not result if invert else result


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
