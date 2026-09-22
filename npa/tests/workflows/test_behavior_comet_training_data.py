"""Focused contracts for the task-1 Comet training dataset boundary."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.workflows.behavior_challenge import comet_training_data as data


def _split(path: Path) -> str:
    path.write_text(
        json.dumps(
            {
                "schema": "npa.behavior.panel-episode-split.v1",
                "revision": data.DATASET_REVISION,
                "tasks": {
                    "1": {
                        "name": data.TASK_NAME,
                        "training": list(range(180)),
                        "holdout": list(range(180, 200)),
                    }
                },
            }
        )
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(episode: int = 0, frame: int = 0) -> dict:
    sample = {
        "observation.state": np.arange(61, dtype=np.float32),
        "action": np.arange(32 * 23, dtype=np.float32).reshape(32, 23),
        "episode_index": np.int64(episode),
        "frame_index": np.int64(frame),
    }
    for role, camera in data.CAMERAS.items():
        size = data.CAMERA_SIZES[role]
        shape = (3, size, size) if role == "head" else (size, size, 3)
        sample[f"observation.rgb.{camera}_camera_0"] = np.zeros(shape, dtype=np.uint8)
    return sample


class _Dataset:
    def __init__(self, sample: dict, episodes: list[int], length: int = 40):
        self.sample = sample
        self.meta = SimpleNamespace(
            episodes={episode: {"length": length} for episode in episodes}
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.sample


def _build(tmp_path: Path, sample: dict, *, partition: str = "training"):
    split = tmp_path / "split.json"
    split_sha256 = _split(split)
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return _Dataset(sample, kwargs["episodes"])

    dataset = data.CometTask1Dataset(
        tmp_path / "dataset",
        split,
        expected_split_sha256=split_sha256,
        partition=partition,
        dataset_factory=factory,
    )
    return dataset, calls


def _v3_row(episode: int, frame_start: int, video_start: float) -> dict:
    length = 40
    row = {
        "episode_index": episode,
        "tasks": [data.TASK_NAME],
        "length": length,
        "data/chunk_index": 1,
        "data/file_index": 0,
        "dataset_from_index": frame_start,
        "dataset_to_index": frame_start + length,
        "task_index": data.TASK_ID,
    }
    for camera in data.CAMERAS.values():
        prefix = f"videos/observation.rgb.{camera}_camera_0"
        row[f"{prefix}/chunk_index"] = 1
        row[f"{prefix}/file_index"] = 0
        row[f"{prefix}/from_timestamp"] = video_start
        row[f"{prefix}/to_timestamp"] = video_start + length / data.FPS
    return row


def _v3_info() -> dict:
    return {
        "codebase_version": "v3.0",
        "fps": data.FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [61]},
            "action": {"dtype": "float32", "shape": [23]},
        },
    }


def _v3_metadata(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-001").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(_v3_info()))
    (root / "meta/stats.json").write_text("{}")
    (root / "meta/tasks.jsonl").write_text(
        json.dumps(
            {
                "task_index": data.TASK_ID,
                "task_name": data.TASK_NAME,
                "task": "Put cans into the trash can.",
            }
        )
        + "\n"
    )
    rows = [_v3_row(200, 1000, 4.0), _v3_row(201, 1040, 20.0)]
    rows.extend(
        _v3_row(episode, 1040 + 40 * episode, 40.0 + episode)
        for episode in range(202, 400)
    )
    pq.write_table(
        pa.Table.from_pylist(rows), root / "meta/episodes/chunk-001/file-000.parquet"
    )
    (root / "data/chunk-001").mkdir(parents=True)
    (root / "data/chunk-001/file-000.parquet").write_bytes(b"packed-data")
    for camera in data.CAMERAS.values():
        path = root / f"videos/observation.rgb.{camera}_camera_0/chunk-001"
        path.mkdir(parents=True)
        (path / "file-000.mp4").write_bytes(b"packed-video")
    return root


def test_factory_receives_exact_partition_alignment_and_horizon(tmp_path):
    dataset, calls = _build(tmp_path, _sample(), partition="holdout")

    args, kwargs = calls[0]
    assert args == (data.DATASET_REPOSITORY,)
    assert kwargs["revision"] == data.DATASET_REVISION
    assert kwargs["episodes"] == list(range(180, 200))
    assert kwargs["tolerance_s"] == 5e-4
    assert kwargs["delta_timestamps"] == {"action": [step / 30 for step in range(32)]}
    assert kwargs["return_uint8"] is True
    assert len(dataset) == 1


def test_v3_metadata_maps_shared_data_and_video_without_relabeling(tmp_path):
    root = _v3_metadata(tmp_path)

    locations = data._load_locations(root, [200, 201])

    assert locations[200].data_path == locations[201].data_path
    for key in locations[200].video_paths:
        assert locations[200].video_paths[key] == locations[201].video_paths[key]
    assert locations[200].video_offsets != locations[201].video_offsets
    assert locations[200].dataset_from == 1000
    assert locations[201].dataset_from == 1040


def test_v3_metadata_rejects_task_and_timestamp_drift(tmp_path):
    root = _v3_metadata(tmp_path)
    path = root / "meta/episodes/chunk-001/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["tasks"] = ["wrong_task"]
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="task identity"):
        data._load_locations(root, [200])

    rows[0]["tasks"] = [data.TASK_NAME]
    key = f"videos/observation.rgb.{data.CAMERAS['head']}_camera_0/to_timestamp"
    rows[0][key] += 0.1
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="video interval"):
        data._load_locations(root, [200])


def test_v3_video_tolerance_has_no_relative_component(tmp_path):
    root = _v3_metadata(tmp_path)
    path = root / "meta/episodes/chunk-001/file-000.parquet"
    row = pq.read_table(path).to_pylist()[0]
    row["length"] = 7500
    row["dataset_to_index"] = row["dataset_from_index"] + row["length"]
    for camera in data.CAMERAS.values():
        prefix = f"videos/observation.rgb.{camera}_camera_0"
        row[f"{prefix}/to_timestamp"] = (
            row[f"{prefix}/from_timestamp"] + row["length"] / data.FPS + 0.001
        )

    with pytest.raises(ValueError, match="video interval"):
        data._episode_location(root, data._validate_v3_info(root), row)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("data/chunk_index", True, "packed file index"),
        ("data/file_index", -1, "packed file index"),
        ("dataset_from_index", -1, "frame interval"),
        ("length", 0, "frame interval"),
        (
            "videos/observation.rgb.zed_link_camera_0/from_timestamp",
            np.nan,
            "video interval",
        ),
    ],
)
def test_v3_metadata_rejects_invalid_indices_and_intervals(
    tmp_path, field, value, message
):
    root = _v3_metadata(tmp_path)
    path = root / "meta/episodes/chunk-001/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0][field] = value

    with pytest.raises(ValueError, match=message):
        data._episode_location(root, data._validate_v3_info(root), rows[0])


class _PackedTable:
    def __init__(self):
        self.features = {
            "index": SimpleNamespace(dtype="int64"),
            "episode_index": SimpleNamespace(dtype="int64"),
            "frame_index": SimpleNamespace(dtype="int64"),
            "timestamp": SimpleNamespace(dtype="float32"),
            "observation.state": SimpleNamespace(
                feature=SimpleNamespace(dtype="float32"),
                length=data.STATE_DIMENSION,
            ),
            "action": SimpleNamespace(
                feature=SimpleNamespace(dtype="float32"),
                length=data.ACTION_DIMENSION,
            ),
        }
        self.formatted = False
        self.rows = []
        for episode in (200, 201):
            for frame in range(40):
                self.rows.append(
                    {
                        "index": 1000 + len(self.rows),
                        "episode_index": episode,
                        "frame_index": frame,
                        "timestamp": np.float32(frame / data.FPS),
                        "observation.state": np.full(
                            data.STATE_DIMENSION, frame, dtype=np.float32
                        ),
                        "action": np.full(23, episode, dtype=np.float32),
                    }
                )

    def set_format(self, name):
        assert name == "numpy"
        self.formatted = True

    def __getitem__(self, index):
        if isinstance(index, list):
            return {
                key: np.asarray([self.rows[item][key] for item in index])
                for key in ("action", "episode_index", "frame_index")
            }
        return self.rows[index]


def test_v3_action_horizon_clamps_inside_shared_packed_file(tmp_path):
    shared = tmp_path / "packed.parquet"
    first = data._EpisodeLocation(200, 40, 1000, shared, {}, {})
    second = data._EpisodeLocation(201, 40, 1040, shared, {}, {})
    dataset = data._PackedV3Dataset.__new__(data._PackedV3Dataset)
    dataset._tables = {shared: _PackedTable()}

    sample = dataset._low_dimensional_sample(first, 39)

    assert sample["action"].shape == (32, 23)
    assert np.unique(sample["action"]).tolist() == [200.0]
    assert sample["episode_index"] == 200
    assert second.dataset_from == first.dataset_from + first.length

    dataset._tables[shared].rows[10]["episode_index"] = 201
    with pytest.raises(ValueError, match="action horizon crosses"):
        dataset._low_dimensional_sample(first, 0)


def test_v3_frame_tolerance_has_no_relative_component(tmp_path):
    class SparseTable:
        @staticmethod
        def row(frame):
            timestamp = frame / data.FPS
            if frame == 8000:
                timestamp += 0.001
            return {
                "index": 1000 + frame,
                "episode_index": 200,
                "frame_index": frame,
                "timestamp": np.float32(timestamp),
                "observation.state": np.zeros(data.STATE_DIMENSION, dtype=np.float32),
                "action": np.zeros(data.ACTION_DIMENSION, dtype=np.float32),
            }

        def __getitem__(self, index):
            if isinstance(index, list):
                rows = [self.row(item) for item in index]
                return {
                    key: np.asarray([row[key] for row in rows])
                    for key in ("action", "episode_index", "frame_index")
                }
            return self.row(index)

    packed = tmp_path / "packed.parquet"
    location = data._EpisodeLocation(200, 9000, 1000, packed, {}, {})
    dataset = data._PackedV3Dataset.__new__(data._PackedV3Dataset)
    dataset._tables = {packed: SparseTable()}

    with pytest.raises(ValueError, match="frame timestamp"):
        dataset._low_dimensional_sample(location, 8000)


def test_v3_full_getitem_applies_numpy_format_and_preserves_float32(
    tmp_path, monkeypatch
):
    root = _v3_metadata(tmp_path)
    table = _PackedTable()
    loaded_paths = []

    def from_parquet(path):
        loaded_paths.append(path)
        return table

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset=SimpleNamespace(from_parquet=from_parquet)),
    )
    monkeypatch.setattr(
        data,
        "_decode_rgb",
        lambda *args: np.zeros((3, 4, 4), dtype=np.float32),
    )

    dataset = data._PackedV3Dataset(
        data.DATASET_REPOSITORY,
        root=root,
        episodes=[200],
        revision=data.DATASET_REVISION,
        video_backend="pyav",
        tolerance_s=data.ALIGNMENT_TOLERANCE_SECONDS,
        delta_timestamps={
            "action": [step / data.FPS for step in range(data.ACTION_HORIZON)]
        },
    )
    sample = dataset[0]

    assert table.formatted
    assert loaded_paths == [str(root / "data/chunk-001/file-000.parquet")]
    assert sample["observation.state"].dtype == np.float32
    assert sample["action"].dtype == np.float32
    assert sample["action"].shape == (data.ACTION_HORIZON, data.ACTION_DIMENSION)
    assert int(sample["episode_index"]) == 200
    assert int(sample["frame_index"]) == 0


def test_v3_video_query_uses_episode_offset_in_shared_file(tmp_path, monkeypatch):
    shared = tmp_path / "packed.mp4"
    key = "observation.rgb.zed_link_camera_0"
    location = data._EpisodeLocation(
        201, 40, 1040, tmp_path / "data", {key: shared}, {key: 20.0}
    )
    queries = []

    def decode(path, query):
        queries.append((path, query))
        return np.zeros((3, 720, 720), dtype=np.uint8)

    monkeypatch.setattr(data, "_decode_rgb", decode)
    dataset = data._PackedV3Dataset.__new__(data._PackedV3Dataset)

    result = dataset._rgb_sample(location, 3, 0.1)

    assert queries == [(shared, [20.1])]
    assert result[key].shape == (3, 720, 720)


def test_full_action_chunk_has_no_padding_and_maps_cameras(tmp_path):
    dataset, _ = _build(tmp_path, _sample(frame=0))

    result = dataset[0]

    assert result["observation.state"].shape == (61,)
    assert result["action"].shape == (32, 23)
    assert result["prompt"] == data.TASK_NAME
    assert result["action_valid_mask"].tolist() == [True] * 32
    assert not result["action_is_pad"].any()
    assert set(key for key in result if key.startswith("observation.images")) == {
        "observation.images.rgb.head",
        "observation.images.rgb.left_wrist",
        "observation.images.rgb.right_wrist",
    }


@pytest.mark.parametrize(
    ("frame", "valid_steps"),
    [(20, 20), (39, 1)],
)
def test_terminal_chunk_is_masked_and_repeat_padded(tmp_path, frame, valid_steps):
    sample = _sample(frame=frame)
    original = sample["action"].copy()
    dataset, _ = _build(tmp_path, sample)

    result = dataset[0]

    assert result["action_valid_mask"].sum() == valid_steps
    assert result["action_is_pad"].sum() == 32 - valid_steps
    np.testing.assert_array_equal(
        result["action"][:valid_steps], original[:valid_steps]
    )
    np.testing.assert_array_equal(
        result["action"][valid_steps:],
        np.repeat(original[valid_steps - 1][None], 32 - valid_steps, axis=0),
    )


def test_sample_cannot_cross_requested_episode_partition(tmp_path):
    dataset, _ = _build(tmp_path, _sample(episode=199))

    with pytest.raises(ValueError, match="escaped the requested episode partition"):
        dataset[0]


def test_alignment_and_shape_fail_closed(tmp_path):
    sample = _sample()
    sample["action"] = sample["action"][:-1]
    dataset, _ = _build(tmp_path, sample)
    with pytest.raises(ValueError, match="state/action shape differs"):
        dataset[0]

    sample = _sample()
    sample["frame_index"] = np.float32(0)
    dataset, _ = _build(tmp_path, sample)
    with pytest.raises(ValueError, match="frame_index must be an integer scalar"):
        dataset[0]


def test_split_rejects_overlap_and_wrong_partition(tmp_path):
    split = tmp_path / "split.json"
    split_sha256 = _split(split)
    value = json.loads(split.read_text())
    value["tasks"]["1"]["holdout"][0] = 0
    split.write_text(json.dumps(value))

    changed_sha256 = hashlib.sha256(split.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="bytes differ"):
        data.load_task1_episodes(split, "training", expected_split_sha256=split_sha256)
    with pytest.raises(ValueError, match="overlaps"):
        data.load_task1_episodes(
            split, "training", expected_split_sha256=changed_sha256
        )
    with pytest.raises(ValueError, match="partition must"):
        data.load_task1_episodes(
            split, "validation", expected_split_sha256=changed_sha256
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("observation.state", np.zeros(61, dtype=np.float64), "state/action dtype"),
        ("action", np.zeros((32, 23), dtype=bool), "state/action dtype"),
        ("action", np.full((32, 23), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_state_and_action_semantics_fail_closed(tmp_path, key, value, message):
    sample = _sample()
    sample[key] = value
    dataset, _ = _build(tmp_path, sample)
    with pytest.raises(ValueError, match=message):
        dataset[0]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.full((720, 720, 3), np.nan, dtype=np.float32), "RGB dtype"),
        (np.zeros((719, 720, 3), dtype=np.uint8), "RGB shape"),
        (np.zeros((720, 720), dtype=np.uint8), "RGB shape"),
    ],
)
def test_rgb_semantics_fail_closed(tmp_path, value, message):
    sample = _sample()
    sample["observation.rgb.zed_link_camera_0"] = value
    dataset, _ = _build(tmp_path, sample)
    with pytest.raises(ValueError, match=message):
        dataset[0]


def test_pinned_float_decoder_is_converted_to_exact_uint8():
    sample = _sample()
    key = "observation.rgb.zed_link_camera_0"
    sample[key] = np.full((3, 720, 720), 0.5, dtype=np.float32)
    wrapped = data._Uint8LeRobotDataset(_Dataset(sample, list(range(180))))

    result = wrapped[0][key]

    assert result.dtype == np.uint8
    assert np.unique(result).tolist() == [127]


def test_pinned_float_decoder_rejects_nonfinite_rgb():
    sample = _sample()
    key = "observation.rgb.zed_link_camera_0"
    sample[key] = np.full((3, 720, 720), np.nan, dtype=np.float32)
    wrapped = data._Uint8LeRobotDataset(_Dataset(sample, list(range(180))))

    with pytest.raises(ValueError, match="floating decode differs"):
        wrapped[0]


def test_mask_rejects_invalid_episode_positions():
    with pytest.raises(ValueError, match="outside its episode"):
        data.action_valid_mask(40, 40)
    with pytest.raises(ValueError, match="32-step"):
        data.action_valid_mask(0, 40, horizon=30)
