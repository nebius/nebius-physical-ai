"""Opt-in native FiftyOne/MongoDB round trip using real SO100 episode data."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from npa.fiftyone_lerobot import (
    _prepare_native_lerobot_source,
    _seed_native_subtask_tags,
)
from npa.fiftyone_lerobot_subtasks import export_fiftyone_subtasks

pytestmark = pytest.mark.e2e


@pytest.fixture
def review_dataset(tmp_path, monkeypatch):
    source = os.environ.get("NPA_E2E_FIFTYONE_LEROBOT_SOURCE")
    if not source:
        pytest.skip("Set NPA_E2E_FIFTYONE_LEROBOT_SOURCE to the pinned SO100 dataset")
    monkeypatch.setenv("FIFTYONE_DATABASE_DIR", str(tmp_path / "database"))
    monkeypatch.setenv("FIFTYONE_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("FIFTYONE_DO_NOT_TRACK", "1")
    import fiftyone as fo

    assert fo.__version__ == "1.22.0"
    normalized = _prepare_native_lerobot_source(Path(source), tmp_path / "staging")
    dataset = fo.Dataset.from_dir(
        dataset_dir=str(normalized),
        dataset_type=fo.types.LeRobotDataset,
        episodes=[7, 2],
        name=f"npa-subtask-native-{uuid4().hex}",
        persistent=True,
    )
    try:
        yield dataset, Path(source), fo
    finally:
        dataset.delete()


def _source_rows(source, episode):
    table = pq.read_table(source / "data/chunk-000/file-000.parquet")
    return table.filter(pc.equal(table["episode_index"], episode))


def _tag_real_episodes(dataset, source):
    from fiftyone.core.tags import TemporalTag

    for sample in dataset.iter_samples():
        rows = _source_rows(source, sample.episode_index)
        boundary = round(rows["timestamp"][len(rows) // 2].as_py() * 1e9)
        end = round((rows["timestamp"][-1].as_py() + 1 / sample.fps) * 1e9)
        dataset.temporal_tags.add(
            [
                TemporalTag(
                    sample.id,
                    start=0,
                    end=boundary,
                    tag=f"subtask:episode-{sample.episode_index}-first",
                ),
                TemporalTag(
                    sample.id,
                    start=boundary,
                    end=end,
                    tag=f"subtask:episode-{sample.episode_index}-second",
                ),
            ]
        )


def _assert_episode_identity(source, exported, output_episode, source_episode, labels):
    original = _source_rows(source, source_episode)
    actual = exported.filter(pc.equal(exported["episode_index"], output_episode))
    # Native export legitimately rebuilds episode/global/task indexes.
    preserved = [
        name
        for name in original.column_names
        if name not in {"episode_index", "index", "task_index"}
    ]
    assert original.select(preserved).equals(actual.select(preserved))
    resolved = [labels[index] for index in actual["subtask_index"].to_pylist()]
    middle = len(original) // 2
    assert resolved == (
        [f"episode-{source_episode}-first"] * middle
        + [f"episode-{source_episode}-second"] * (len(original) - middle)
    )


def test_real_fiftyone_122_export_and_reimport(
    review_dataset, tmp_path, record_property
):
    dataset, source, fo = review_dataset
    assert dataset.values("episode_index") == [7, 2]
    _tag_real_episodes(dataset, source)
    output = tmp_path / "reviewed"
    report = export_fiftyone_subtasks(dataset.name, output)
    exported = pq.read_table(output / "data/chunk-000/file-000.parquet")
    catalog = pq.read_table(output / "meta/subtasks.parquet").to_pylist()
    labels = {row["subtask_index"]: row["subtask"] for row in catalog}
    for output_episode, source_episode in enumerate([2, 7]):
        _assert_episode_identity(
            source, exported, output_episode, source_episode, labels
        )
    assert report["unlabeled_frame_count"] == 0
    assert report["subtask_count"] == report["segment_count"] == 4
    assert report["episode_index_mapping"] == [
        {"source_episode_index": 2, "export_episode_index": 0},
        {"source_episode_index": 7, "export_episode_index": 1},
    ]
    reimported = fo.Dataset.from_dir(
        dataset_dir=str(output), dataset_type=fo.types.LeRobotDataset
    )
    try:
        assert _seed_native_subtask_tags(reimported, output) == 4
        tags = list(reimported.temporal_tags.values())
        assert {tag.tag for tag in tags} == {
            f"subtask:{label}" for label in labels.values()
        }
    finally:
        reimported.delete()
    record_property("fiftyone_version", fo.__version__)
    record_property("source_episode_order", "7,2")
    record_property("export_episode_order", "2->0,7->1")
    record_property("verified_frame_count", len(exported))
    record_property("verified_temporal_tags", 4)
