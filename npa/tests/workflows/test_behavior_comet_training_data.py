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


def _three_task_split(path: Path) -> str:
    rows = {}
    for task_id, name in data.TASK_NAMES.items():
        first = task_id * 200
        rows[str(task_id)] = {
            "name": name,
            "training": list(range(first, first + 180)),
            "holdout": list(range(first + 180, first + 200)),
        }
    path.write_text(
        json.dumps(
            {
                "schema": "npa.behavior.panel-episode-split.v1",
                "revision": data.DATASET_REVISION,
                "tasks": rows,
            }
        )
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("task_id", [0, 1, 22])
def test_task_reader_keeps_holdout_membership_prompt_and_terminal_actions(
    tmp_path, task_id
):
    split = tmp_path / "split.json"
    digest = _three_task_split(split)
    episode = task_id * 200 + 180
    sample = _sample(episode, frame=39)
    calls = []

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return _Dataset(sample, kwargs["episodes"])

    dataset = data.CometTaskDataset(
        tmp_path,
        split,
        task_id=task_id,
        partition="holdout",
        expected_split_sha256=digest,
        dataset_factory=factory,
    )
    row = dataset[0]
    assert dataset.episodes == tuple(range(episode, episode + 20))
    assert row["prompt"] == data.TASK_NAMES[task_id]
    assert int(row["episode_index"]) == episode
    np.testing.assert_array_equal(row["action_valid_mask"], np.arange(32) == 0)
    np.testing.assert_array_equal(
        row["action"], np.repeat(sample["action"][:1], 32, axis=0)
    )
    assert calls[0].get("task_id", 1) == task_id


def test_task1_compatibility_reader_matches_parameterized_reader(tmp_path):
    split = tmp_path / "split.json"
    digest = _three_task_split(split)

    def factory(*args, **kwargs):
        return _Dataset(_sample(200, frame=39), kwargs["episodes"])

    arguments = dict(expected_split_sha256=digest, dataset_factory=factory)
    legacy = data.CometTask1Dataset(tmp_path, split, **arguments)
    general = data.CometTaskDataset(tmp_path, split, task_id=1, **arguments)
    assert legacy.episodes == general.episodes
    assert legacy[0].keys() == general[0].keys()
    for key, value in legacy[0].items():
        np.testing.assert_array_equal(value, general[0][key])


@pytest.mark.parametrize("task_id", [True, False, "1", -1, 2, 100])
def test_task_reader_rejects_unsupported_or_untyped_task_before_loading(
    tmp_path, task_id
):
    with pytest.raises(ValueError, match="task IDs"):
        data.CometTaskDataset(
            tmp_path,
            tmp_path / "absent.json",
            task_id=task_id,
            expected_split_sha256="a" * 64,
        )


@pytest.mark.parametrize("task_id", [0, 22])
def test_anchor_split_rejects_wrong_task_name_and_partition_overlap(tmp_path, task_id):
    split = tmp_path / "split.json"
    _three_task_split(split)
    value = json.loads(split.read_text())
    task = value["tasks"][str(task_id)]
    task["name"] = data.TASK_NAME
    split.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="split differs"):
        data.load_task_episodes(
            split,
            "holdout",
            task_id=task_id,
            expected_split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
        )
    task["name"] = data.TASK_NAMES[task_id]
    task["holdout"][0] = task["training"][0]
    split.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="overlaps"):
        data.load_task_episodes(
            split,
            "holdout",
            task_id=task_id,
            expected_split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize("task_id", [0, 22])
def test_packed_anchor_metadata_uses_selected_task_chunk_and_identity(
    tmp_path, task_id
):
    root = _v3_metadata(tmp_path)
    old = root / "meta/episodes/chunk-001/file-000.parquet"
    rows = pq.read_table(old).to_pylist()
    for row in rows:
        row["task_index"] = task_id
        row["tasks"] = [data.TASK_NAMES[task_id]]
    target = root / f"meta/episodes/chunk-{task_id:03d}/file-000.parquet"
    target.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), target)
    (root / "meta/tasks.jsonl").write_text(
        json.dumps(
            {
                "task_index": task_id,
                "task_name": data.TASK_NAMES[task_id],
            }
        )
        + "\n"
    )
    locations = data._load_locations(root, [200, 201], task_id=task_id)
    assert locations[201].dataset_from == 1040
    assert locations[201].video_offsets != locations[200].video_offsets
    rows[0]["task_index"] = 1
    pq.write_table(pa.Table.from_pylist(rows), target)
    with pytest.raises(ValueError, match="task identity"):
        data._load_locations(root, [200], task_id=task_id)


@pytest.mark.parametrize(("task_id", "malformed"), [(0, False), (1, True)])
def test_task_metadata_rejects_boolean_aliases(tmp_path, task_id, malformed):
    root = _v3_metadata(tmp_path)
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": malformed, "task_name": data.TASK_NAMES[task_id]})
        + "\n"
    )
    with pytest.raises(ValueError, match="identity differs"):
        data._validate_task_metadata(root, task_id)


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


def _reconstruction_contract(task_id=1):
    names = {0: "turning_on_radio", 1: "picking_up_trash", 22: "putting_shoes_on_rack"}
    return {
        "schema": data.DATA_RECONSTRUCTION_SCHEMA,
        "dataset_repository": data.DATASET_REPOSITORY,
        "dataset_revision": data.DATASET_REVISION,
        "task_id": task_id,
        "task_name": names[task_id],
        "modalities": ["rgb"],
        "tolerance_s": data.ALIGNMENT_TOLERANCE_SECONDS,
        "prompt_from_task": True,
        "fine_grained_level": 0,
    }


@pytest.mark.parametrize("task_id", [0, 1, 22])
def test_data_reconstruction_accepts_supported_tasks(task_id):
    contract = _reconstruction_contract(task_id)
    assert data.validate_data_reconstruction(contract) is contract


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("dataset_revision", "0" * 40),
        ("modalities", ["depth"]),
        ("tolerance_s", 1e-4),
    ],
)
def test_data_reconstruction_rejects_fixed_contract_drift(field, wrong):
    contract = _reconstruction_contract()
    contract[field] = wrong
    with pytest.raises(ValueError, match="data reconstruction contract differs"):
        data.validate_data_reconstruction(contract)


def test_data_reconstruction_rejects_task_name_mismatch():
    contract = _reconstruction_contract()
    contract["task_name"] = "turning_on_radio"
    with pytest.raises(ValueError, match="task identity differs"):
        data.validate_data_reconstruction(contract)


@pytest.mark.parametrize("task_id", [999, None, True, "1"])
def test_data_reconstruction_rejects_unknown_task_without_name(task_id):
    contract = _reconstruction_contract()
    contract["task_id"] = task_id
    contract.pop("task_name")
    with pytest.raises(ValueError, match="task identity differs"):
        data.validate_data_reconstruction(contract)
