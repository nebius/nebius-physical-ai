"""Tests for FiftyOne temporal-tag to LeRobot subtask round trips."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.fiftyone_lerobot_subtasks import (
    SubtaskLabelError,
    SubtaskSegment,
    apply_subtask_segments,
    existing_subtask_segments,
    segments_from_temporal_tags,
)


def _write_dataset(root: Path) -> Path:
    metadata = root / "meta"
    metadata.mkdir(parents=True)
    (metadata / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 10, "features": {}}),
        encoding="utf-8",
    )
    table = pa.table(
        {
            "episode_index": [0, 0, 0, 1, 1],
            "frame_index": [0, 1, 2, 0, 1],
            "timestamp": [0.0, 0.1, 0.2, 0.0, 0.1],
            "action": [[0.0], [0.1], [0.2], [1.0], [1.1]],
        }
    )
    path = root / "data" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(table, path)
    return root


def test_segments_from_temporal_tags_uses_subtask_namespace() -> None:
    segments = segments_from_temporal_tags(
        {"sample-a": 0},
        [
            {"sample_id": "sample-a", "start": 0, "end": 100, "tag": "reviewed"},
            {
                "sample_id": "sample-a",
                "start": 0,
                "end": 100,
                "tag": "subtask:approach",
                "index_type": 2,
            },
        ],
    )

    assert segments == [SubtaskSegment(0, "approach", 0, 100)]


def test_segments_from_temporal_tags_rejects_overlap() -> None:
    with pytest.raises(SubtaskLabelError, match="overlapping subtasks"):
        segments_from_temporal_tags(
            {"sample-a": 0},
            [
                {"sample_id": "sample-a", "start": 0, "end": 100, "tag": "subtask:a"},
                {"sample_id": "sample-a", "start": 50, "end": 150, "tag": "subtask:b"},
            ],
        )


def test_apply_subtask_segments_writes_training_columns_and_metadata(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "dataset")
    segments = [
        SubtaskSegment(0, "approach", 0, 200_000_000),
        SubtaskSegment(0, "grasp", 200_000_000, 300_000_000),
        SubtaskSegment(1, "approach", 0, 200_000_000),
    ]

    report = apply_subtask_segments(root, segments)

    table = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
    assert table.column("subtask_index").to_pylist() == [0, 0, 1, 0, 0]
    assert table.column("action").to_pylist() == [[0.0], [0.1], [0.2], [1.0], [1.1]]
    subtasks = pq.read_table(root / "meta" / "subtasks.parquet").to_pylist()
    assert subtasks == [
        {"subtask": "approach", "subtask_index": 0},
        {"subtask": "grasp", "subtask_index": 1},
    ]
    annotations = json.loads((root / "meta" / "lerobot_annotations.json").read_text())
    assert annotations["episodes"]["0"]["subtasks"][1]["label"] == "grasp"
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["features"]["subtask_index"] == {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    assert report["segment_count"] == 3
    assert report["unlabeled_frame_count"] == 0


def test_apply_subtask_segments_fails_before_writing_when_frames_have_gaps(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "dataset")
    data_path = root / "data" / "chunk-000" / "file-000.parquet"

    with pytest.raises(SubtaskLabelError, match=r"3 LeRobot frame\(s\) unlabeled"):
        apply_subtask_segments(
            root,
            [SubtaskSegment(0, "approach", 0, 200_000_000)],
        )

    assert "subtask_index" not in pq.read_table(data_path).column_names
    assert not (root / "meta" / "subtasks.parquet").exists()


def test_existing_subtask_segments_reconstructs_contiguous_intervals(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "dataset")
    apply_subtask_segments(
        root,
        [
            SubtaskSegment(0, "approach", 0, 200_000_000),
            SubtaskSegment(0, "grasp", 200_000_000, 300_000_000),
            SubtaskSegment(1, "approach", 0, 200_000_000),
        ],
    )

    assert existing_subtask_segments(root) == [
        SubtaskSegment(0, "approach", 0, 200_000_000),
        SubtaskSegment(0, "grasp", 200_000_000, 300_000_000),
        SubtaskSegment(1, "approach", 0, 200_000_000),
    ]
