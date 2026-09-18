"""Round-trip FiftyOne temporal tags into a derived LeRobot dataset."""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


SUBTASK_TAG_PREFIX = "subtask:"
NANOSECONDS_PER_SECOND = 1_000_000_000
FIFTYONE_DURATION_INDEX = 2


class SubtaskLabelError(ValueError):
    """Raised when subtask labels cannot produce a valid LeRobot dataset.

    Args:
        message: Human-readable validation failure.

    Returns:
        None.

    Raises:
        None.
    """


@dataclass(frozen=True)
class SubtaskSegment:
    """A half-open subtask interval on one logical episode.

    Args:
        episode_index: Zero-based episode index in the exported dataset.
        label: Human-readable atomic action label.
        start_ns: Inclusive interval start in elapsed nanoseconds.
        end_ns: Exclusive interval end in elapsed nanoseconds.

    Returns:
        None.

    Raises:
        None.
    """

    episode_index: int
    label: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class _FileAnnotationPlan:
    path: Path
    table: pa.Table
    subtask_indices: list[int]


def segments_from_temporal_tags(
    episode_by_sample_id: Mapping[str, int],
    temporal_tags: Iterable[Any],
) -> list[SubtaskSegment]:
    """Parse ``subtask:`` FiftyOne temporal tags into validated segments.

    Args:
        episode_by_sample_id: Output episode index keyed by FiftyOne sample ID.
        temporal_tags: FiftyOne ``TemporalTag`` objects or equivalent mappings.

    Returns:
        Sorted, non-overlapping subtask segments.

    Raises:
        SubtaskLabelError: If a subtask tag is malformed or overlaps another.
    """
    segments = []
    for temporal_tag in temporal_tags:
        segment = _segment_from_temporal_tag(episode_by_sample_id, temporal_tag)
        if segment is not None:
            segments.append(segment)

    segments.sort(key=lambda item: (item.episode_index, item.start_ns, item.end_ns, item.label))
    _validate_non_overlapping_segments(segments)
    return segments


def apply_subtask_segments(
    dataset_root: Path,
    segments: Iterable[SubtaskSegment],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Write temporal subtasks into a copied LeRobot v3 dataset.

    Args:
        dataset_root: Root of the self-contained LeRobot dataset to update.
        segments: Validated subtask intervals keyed by episode index.
        require_complete: Whether every frame must receive exactly one subtask.

    Returns:
        A JSON-serializable export report.

    Raises:
        SubtaskLabelError: If metadata is missing or frame coverage is invalid.
    """
    root = dataset_root.expanduser().resolve()
    normalized = sorted(
        segments,
        key=lambda item: (item.episode_index, item.start_ns, item.end_ns, item.label),
    )
    _validate_non_overlapping_segments(normalized)
    labels = sorted({segment.label for segment in normalized})
    if not labels:
        raise SubtaskLabelError(f"No {SUBTASK_TAG_PREFIX!r} temporal tags were found")

    plans, episode_indexes, gap_count = _plan_frame_annotations(
        root,
        normalized,
        labels,
        require_complete=require_complete,
    )
    for plan in plans:
        _write_annotated_table(plan)
    _write_subtask_metadata(root, normalized, labels)
    return _build_export_report(normalized, labels, episode_indexes, gap_count)


def export_fiftyone_subtasks(
    dataset_name: str,
    output_dir: Path,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Export one FiftyOne LeRobot dataset and materialize its subtask tags.

    Args:
        dataset_name: Persistent FiftyOne dataset containing LeRobot episodes.
        output_dir: New local directory for the derived LeRobot dataset.
        require_complete: Whether every exported frame must have a subtask.

    Returns:
        A JSON-serializable export report.

    Raises:
        SubtaskLabelError: If the dataset or labels violate the export contract.
    """
    import fiftyone as fo

    dataset_type, dataset, samples = _load_review_dataset(fo, dataset_name)
    export_view, episode_by_sample_id, episode_mapping = _ordered_export(dataset, samples)
    segments = segments_from_temporal_tags(
        episode_by_sample_id,
        dataset.temporal_tags.values(),
    )
    destination = output_dir.expanduser().resolve()
    _require_empty_destination(destination)
    export_view.export(export_dir=str(destination), dataset_type=dataset_type)
    report = apply_subtask_segments(destination, segments, require_complete=require_complete)
    report.update({"dataset_name": dataset_name, "output_dir": str(destination)})
    report["episode_index_mapping"] = episode_mapping
    return report


def _ordered_export(dataset: Any, samples: list[Any]) -> tuple[Any, dict[str, int], list[dict[str, int]]]:
    indexed_samples = [(_source_episode_index(sample), sample) for sample in samples]
    indexed_samples.sort(key=lambda item: item[0])
    source_indexes = [index for index, _sample in indexed_samples]
    if len(set(source_indexes)) != len(source_indexes):
        raise SubtaskLabelError("LeRobot source episode indexes must be unique")
    sample_ids = [str(sample.id) for _index, sample in indexed_samples]
    # FiftyOne 1.22 rewrites episode_index in collection order, not source order.
    # Freeze that order for both label mapping and the native export's second read.
    view = dataset.select(sample_ids, ordered=True)
    mapping = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    provenance = [
        {"source_episode_index": source_index, "export_episode_index": output_index}
        for output_index, source_index in enumerate(source_indexes)
    ]
    return view, mapping, provenance


def _source_episode_index(sample: Any) -> int:
    try:
        index = sample["episode_index"]
    except (KeyError, AttributeError) as exc:
        raise SubtaskLabelError("LeRobot sample is missing episode_index") from exc
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise SubtaskLabelError("LeRobot sample episode_index must be a nonnegative integer")
    return index


def _load_review_dataset(fo: Any, dataset_name: str) -> tuple[Any, Any, list[Any]]:
    dataset_type = getattr(fo.types, "LeRobotDataset", None)
    if dataset_type is None:
        raise SubtaskLabelError("FiftyOne 1.22 or newer is required for LeRobot subtask export")
    if dataset_name not in fo.list_datasets():
        raise SubtaskLabelError(f"FiftyOne dataset not found: {dataset_name}")
    dataset = fo.load_dataset(dataset_name)
    if dataset.media_type != "multimodal":
        raise SubtaskLabelError(f"FiftyOne dataset is not a native LeRobot multimodal dataset: {dataset_name}")
    samples = list(dataset.iter_samples())
    if not samples:
        raise SubtaskLabelError(f"FiftyOne dataset has no episodes: {dataset_name}")
    return dataset_type, dataset, samples


def export_fiftyone_subtasks_to_s3(
    dataset_name: str,
    output_uri: str,
    *,
    require_complete: bool = True,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Export reviewed FiftyOne subtasks to a new S3 LeRobot dataset prefix.

    Args:
        dataset_name: Persistent FiftyOne dataset containing reviewed episodes.
        output_uri: Empty S3 prefix that will receive the derived dataset.
        require_complete: Whether every exported frame must have a subtask.
        storage_client: Shared StorageClient; otherwise built from the environment.

    Returns:
        A JSON-serializable export and upload report.

    Raises:
        SubtaskLabelError: If the dataset or annotations are invalid.
        StorageError: If the S3 URI is invalid or its prefix is nonempty.
        Exception: If the storage service rejects or cannot complete the upload.
    """
    with tempfile.TemporaryDirectory(prefix="npa-lerobot-subtasks-") as temporary_directory:
        output_dir = Path(temporary_directory) / "dataset"
        report = export_fiftyone_subtasks(
            dataset_name,
            output_dir,
            require_complete=require_complete,
        )
        uploaded_files = _upload_directory(output_dir, output_uri, storage_client)
    report.pop("output_dir", None)
    report.update({"output_path": output_uri, "uploaded_files": uploaded_files})
    return report


def existing_subtask_segments(dataset_root: Path) -> list[SubtaskSegment]:
    """Read existing LeRobot ``subtask_index`` values as temporal segments.

    Args:
        dataset_root: Root of a LeRobot dataset.

    Returns:
        Existing contiguous subtask segments, or an empty list when absent.

    Raises:
        SubtaskLabelError: If subtask metadata cannot resolve an index.
    """
    root = dataset_root.expanduser().resolve()
    labels = _read_subtask_labels(root)
    if not labels:
        return []
    fps = _read_dataset_fps(root)
    rows = _read_frame_subtask_rows(root)
    return _collapse_subtask_rows(rows, labels, fps)


def _segment_from_temporal_tag(
    episode_by_sample_id: Mapping[str, int],
    temporal_tag: Any,
) -> SubtaskSegment | None:
    tag = str(_tag_value(temporal_tag, "tag") or "")
    if not tag.startswith(SUBTASK_TAG_PREFIX):
        return None
    label = tag.removeprefix(SUBTASK_TAG_PREFIX).strip()
    if not label:
        raise SubtaskLabelError(f"Subtask tag must include a label after {SUBTASK_TAG_PREFIX!r}")
    sample_id = str(_tag_value(temporal_tag, "sample_id") or "")
    if sample_id not in episode_by_sample_id:
        raise SubtaskLabelError(f"Subtask tag references unknown sample {sample_id!r}")
    index_type = _tag_value(temporal_tag, "index_type")
    if index_type not in (None, FIFTYONE_DURATION_INDEX):
        raise SubtaskLabelError(f"Subtask tag {tag!r} must use elapsed nanoseconds")
    start = _tag_value(temporal_tag, "start")
    end = _tag_value(temporal_tag, "end")
    if start is None or end is None:
        raise SubtaskLabelError(f"Subtask tag {tag!r} is missing start or end")
    start_ns = int(start)
    end_ns = int(end)
    if start_ns < 0 or end_ns <= start_ns:
        raise SubtaskLabelError(f"Subtask tag {tag!r} has invalid interval [{start_ns}, {end_ns})")
    return SubtaskSegment(episode_by_sample_id[sample_id], label, start_ns, end_ns)


def _tag_value(temporal_tag: Any, name: str) -> Any:
    if isinstance(temporal_tag, Mapping):
        return temporal_tag.get(name)
    return getattr(temporal_tag, name, None)


def _validate_non_overlapping_segments(segments: list[SubtaskSegment]) -> None:
    previous_by_episode: dict[int, SubtaskSegment] = {}
    for segment in segments:
        previous = previous_by_episode.get(segment.episode_index)
        if previous is not None and segment.start_ns < previous.end_ns:
            raise SubtaskLabelError(
                f"Episode {segment.episode_index} has overlapping subtasks "
                f"{previous.label!r} and {segment.label!r}"
            )
        previous_by_episode[segment.episode_index] = segment


def _plan_frame_annotations(
    root: Path,
    segments: list[SubtaskSegment],
    labels: list[str],
    *,
    require_complete: bool,
) -> tuple[list[_FileAnnotationPlan], set[int], int]:
    _read_dataset_info(root)
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise SubtaskLabelError(f"No LeRobot data parquet files found under {root}")
    by_episode = _segments_by_episode(segments)
    label_indexes = {label: index for index, label in enumerate(labels)}
    plans = []
    episode_indexes: set[int] = set()
    gap_count = 0
    for path in data_files:
        table = pq.read_table(path)
        indices, file_episodes, file_gaps = _frame_subtask_indices(
            table,
            by_episode,
            label_indexes,
        )
        plans.append(_FileAnnotationPlan(path, table, indices))
        episode_indexes.update(file_episodes)
        gap_count += file_gaps
    if require_complete and gap_count:
        raise SubtaskLabelError(f"Subtask labels leave {gap_count} LeRobot frame(s) unlabeled")
    return plans, episode_indexes, gap_count


def _segments_by_episode(
    segments: list[SubtaskSegment],
) -> dict[int, list[SubtaskSegment]]:
    grouped: dict[int, list[SubtaskSegment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.episode_index].append(segment)
    return dict(grouped)


def _frame_subtask_indices(
    table: pa.Table,
    segments_by_episode: Mapping[int, list[SubtaskSegment]],
    label_indexes: Mapping[str, int],
) -> tuple[list[int], set[int], int]:
    required = {"episode_index", "timestamp"}
    missing = sorted(required.difference(table.column_names))
    if missing:
        raise SubtaskLabelError(f"LeRobot data parquet is missing columns: {', '.join(missing)}")
    episodes = table.column("episode_index").to_pylist()
    timestamps = table.column("timestamp").to_pylist()
    indices = []
    gaps = 0
    for episode, timestamp in zip(episodes, timestamps, strict=True):
        index = _subtask_index_for_timestamp(
            int(episode),
            float(timestamp),
            segments_by_episode,
            label_indexes,
        )
        indices.append(index)
        gaps += index < 0
    return indices, {int(episode) for episode in episodes}, gaps


def _subtask_index_for_timestamp(
    episode_index: int,
    timestamp: float,
    segments_by_episode: Mapping[int, list[SubtaskSegment]],
    label_indexes: Mapping[str, int],
) -> int:
    timestamp_ns = round(timestamp * NANOSECONDS_PER_SECOND)
    for segment in segments_by_episode.get(episode_index, []):
        if segment.start_ns <= timestamp_ns < segment.end_ns:
            return label_indexes[segment.label]
    return -1


def _write_annotated_table(plan: _FileAnnotationPlan) -> None:
    array = pa.array(plan.subtask_indices, type=pa.int64())
    if "subtask_index" in plan.table.column_names:
        position = plan.table.column_names.index("subtask_index")
        table = plan.table.set_column(position, "subtask_index", array)
    else:
        table = plan.table.append_column("subtask_index", array)
    temporary_path = plan.path.with_suffix(plan.path.suffix + ".tmp")
    pq.write_table(table, temporary_path, compression="snappy")
    temporary_path.replace(plan.path)


def _write_subtask_metadata(
    root: Path,
    segments: list[SubtaskSegment],
    labels: list[str],
) -> None:
    metadata_dir = root / "meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "subtask": labels,
            "subtask_index": list(range(len(labels))),
        }
    )
    pq.write_table(table, metadata_dir / "subtasks.parquet", compression="snappy")
    _update_info_json(metadata_dir / "info.json")
    _write_annotation_state(metadata_dir / "lerobot_annotations.json", segments)


def _update_info_json(info_path: Path) -> None:
    info = _read_dataset_info(info_path.parent.parent)
    info.setdefault("features", {})["subtask_index"] = {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def _read_dataset_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise SubtaskLabelError(f"LeRobot dataset is missing {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubtaskLabelError(f"LeRobot dataset metadata is invalid: {info_path}") from exc
    if not isinstance(info, dict):
        raise SubtaskLabelError(f"LeRobot dataset metadata must be an object: {info_path}")
    return info


def _write_annotation_state(path: Path, segments: list[SubtaskSegment]) -> None:
    episodes: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for episode_index, episode_segments in _segments_by_episode(segments).items():
        episodes[str(episode_index)] = {
            "subtasks": [
                {
                    "start": segment.start_ns / NANOSECONDS_PER_SECOND,
                    "end": segment.end_ns / NANOSECONDS_PER_SECOND,
                    "label": segment.label,
                }
                for segment in episode_segments
            ],
            "high_levels": [],
        }
    path.write_text(json.dumps({"version": 1, "episodes": episodes}, indent=2) + "\n", encoding="utf-8")


def _build_export_report(
    segments: list[SubtaskSegment],
    labels: list[str],
    episode_indexes: set[int],
    gap_count: int,
) -> dict[str, Any]:
    labeled_episodes = {segment.episode_index for segment in segments}
    return {
        "status": "exported",
        "format": "lerobot-v3-subtasks",
        "episode_count": len(episode_indexes),
        "labeled_episode_count": len(labeled_episodes),
        "segment_count": len(segments),
        "subtask_count": len(labels),
        "unlabeled_frame_count": gap_count,
        "tag_prefix": SUBTASK_TAG_PREFIX,
    }


def _require_empty_destination(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise SubtaskLabelError(f"Output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)


def _upload_directory(root: Path, output_uri: str, storage_client: Any = None) -> int:
    paths = sorted(root.rglob("*"))
    for path in paths:
        if path.is_symlink():
            raise SubtaskLabelError(f"Refusing to upload symlinked dataset asset: {path}")
    if storage_client is None:
        from npa.clients.storage import StorageClient

        storage_client = StorageClient.from_environment()
    storage_client.upload_directory(str(root), output_uri, require_empty=True)
    return sum(path.is_file() for path in paths)


def _read_subtask_labels(root: Path) -> dict[int, str]:
    path = root / "meta" / "subtasks.parquet"
    if not path.exists():
        return {}
    rows = pq.read_table(path).to_pylist()
    labels = {}
    for row in rows:
        index = row.get("subtask_index")
        label = row.get("subtask")
        if index is not None and label:
            labels[int(index)] = str(label)
    return labels


def _read_dataset_fps(root: Path) -> float:
    fps = float(_read_dataset_info(root).get("fps") or 0)
    if fps <= 0:
        raise SubtaskLabelError("LeRobot dataset must declare a positive fps")
    return fps


def _read_frame_subtask_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        required = {"episode_index", "timestamp", "subtask_index"}
        if not required.issubset(table.column_names):
            continue
        rows.extend(table.select(sorted(required)).to_pylist())
    rows.sort(key=lambda row: (int(row["episode_index"]), float(row["timestamp"])))
    return rows


def _collapse_subtask_rows(
    rows: list[dict[str, Any]],
    labels: Mapping[int, str],
    fps: float,
) -> list[SubtaskSegment]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_index"])].append(row)
    segments = []
    frame_ns = round(NANOSECONDS_PER_SECOND / fps)
    for episode_index, episode_rows in grouped.items():
        segments.extend(_collapse_episode_rows(episode_index, episode_rows, labels, frame_ns))
    return segments


def _collapse_episode_rows(
    episode_index: int,
    rows: list[dict[str, Any]],
    labels: Mapping[int, str],
    frame_ns: int,
) -> list[SubtaskSegment]:
    segments = []
    start_ns = 0
    active_index: int | None = None
    for row in rows:
        timestamp_ns = round(float(row["timestamp"]) * NANOSECONDS_PER_SECOND)
        subtask_index = int(row["subtask_index"])
        if subtask_index == active_index:
            continue
        if active_index is not None and active_index >= 0:
            segments.append(SubtaskSegment(episode_index, labels[active_index], start_ns, timestamp_ns))
        active_index = subtask_index
        start_ns = timestamp_ns
        if subtask_index >= 0 and subtask_index not in labels:
            raise SubtaskLabelError(f"Unknown subtask_index {subtask_index} in episode {episode_index}")
    if rows and active_index is not None and active_index >= 0:
        last_ns = round(float(rows[-1]["timestamp"]) * NANOSECONDS_PER_SECOND)
        segments.append(SubtaskSegment(episode_index, labels[active_index], start_ns, last_ns + frame_ns))
    return segments
