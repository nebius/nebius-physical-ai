"""Focused contracts for the task-1 Comet training dataset boundary."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
