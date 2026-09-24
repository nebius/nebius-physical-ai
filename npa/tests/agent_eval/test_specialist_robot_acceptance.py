"""Reject unreplayed trace suffixes and falsely labeled native training features."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_EXAMPLE = Path(__file__).parents[2] / "examples/specialists/robot_workflow"


@pytest.fixture
def acceptance(monkeypatch):
    pytest.importorskip("av")
    monkeypatch.syspath_prepend(str(_EXAMPLE))
    return SimpleNamespace(
        replay=importlib.import_module("replay"),
        native=importlib.import_module("native_dataset"),
    )


def _trace(replay):
    result = {name: np.zeros(shape) for name, shape in replay._TRACE_SHAPES.items()}
    result["phase"] = np.full(185, "approach")
    result["environment_success"] = np.zeros(185, dtype=bool)
    return result


@pytest.mark.parametrize(
    "field",
    [
        "object_position",
        "next_object_position",
        "gripper_position",
        "next_gripper_position",
        "finger_contacts",
        "reward",
        "environment_success",
        "phase",
        "state",
        "actions",
        "next_state",
    ],
)
@pytest.mark.parametrize("count", [184, 186])
def test_every_trace_timestep_must_belong_to_the_native_replay(
    acceptance, field, count
):
    trace = _trace(acceptance.replay)
    acceptance.replay._trace_shapes(trace)
    trace[field] = np.resize(trace[field], (count, *trace[field].shape[1:]))
    with pytest.raises(ValueError, match="shape differs"):
        acceptance.replay._trace_shapes(trace)


@pytest.mark.parametrize(
    "field", ["reward", "goal", "gripper_position", "finger_contacts"]
)
def test_all_numeric_evidence_must_be_finite(acceptance, field):
    trace = _trace(acceptance.replay)
    trace[field].flat[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        acceptance.replay._trace_shapes(trace)


def test_report_contact_count_is_derived_from_trace_not_claim(acceptance):
    trace = _trace(acceptance.replay)
    trace["finger_contacts"][:15] = 2
    trace["finger_contacts"][15:25] = 1
    observed = acceptance.replay._physics(trace)
    assert observed["bilateral_contact_frames"] == 15
    assert observed["judge"] == "mujoco_state_and_contacts"
    assert observed["checks"]["bilateral_grasp_contact"] is True
    assert observed["accepted"] is False


def _info(native):
    features = {
        "action": {"dtype": "float32", "shape": [4], "names": native._ACTIONS},
        "observation.state": {
            "dtype": "float32",
            "shape": [9],
            "names": native._JOINTS,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    for name in ("episode_index", "frame_index", "index", "task_index"):
        features[name] = {"dtype": "int64", "shape": [1], "names": None}
    for camera in ("workspace", "wrist"):
        features["observation.images." + camera] = {
            "dtype": "video",
            "shape": [360, 480, 3],
            "names": ["height", "width", "channel"],
            "video_info": {"video.fps": 25, "video.is_depth_map": False},
        }
    return {
        "features": features,
        "codebase_version": "v3.0",
        "robot_type": "fetch",
        "fps": 25,
        "total_episodes": 2,
        "total_frames": 370,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("robot_type", "different_robot"),
        ("fps", 30),
        ("total_frames", 371),
        ("total_episodes", 3),
        ("codebase_version", "v2.1"),
    ],
)
def test_native_dataset_declared_contract_cannot_differ(
    acceptance, tmp_path, field, value
):
    info = _info(acceptance.native)
    path = tmp_path / "dataset/meta/info.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(info))
    acceptance.native._metadata(tmp_path, [{}, {}])
    info[field] = value
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="differs"):
        acceptance.native._metadata(tmp_path, [{}, {}])


@pytest.mark.parametrize(
    "field,property,value",
    [
        ("observation.state", "names", ["wrong_joint"] * 9),
        ("action", "names", ["wrong_action"] * 4),
        ("action", "dtype", "float64"),
        ("observation.images.wrist", "shape", [480, 360, 3]),
        ("observation.images.workspace", "names", ["width", "height", "channel"]),
        (
            "observation.images.workspace",
            "video_info",
            {"video.fps": 30, "video.is_depth_map": False},
        ),
    ],
)
def test_native_features_preserve_physical_meaning(
    acceptance, tmp_path, field, property, value
):
    info = deepcopy(_info(acceptance.native))
    info["features"][field][property] = value
    path = tmp_path / "dataset/meta/info.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="differ"):
        acceptance.native._metadata(tmp_path, [{}, {}])


def test_native_global_indices_cannot_reset_between_episodes(acceptance):
    row = {"simulation": {"task": "Pick and place"}, "dataset_episode_index": 1}
    sample = {
        "task": "Pick and place",
        "episode_index": 1,
        "frame_index": 0,
        "index": 185,
        "timestamp": 0.0,
    }
    acceptance.native._sample_metadata(sample, row, 0, 185)
    sample["index"] = 0
    with pytest.raises(ValueError, match="global index"):
        acceptance.native._sample_metadata(sample, row, 0, 185)


def test_native_reader_cannot_substitute_nearly_identical_camera_pixels(acceptance):
    original = np.full((4, 4, 3), 128, dtype=np.uint8)
    observed = original.copy()

    def sample():
        tensor = SimpleNamespace(
            numpy=lambda: observed.transpose(2, 0, 1).astype(np.float32) / 255
        )
        return {"observation.images.workspace": tensor}

    acceptance.native._sample_images(sample(), {"workspace": iter([original])})
    observed[1, 1, 0] += 1
    with pytest.raises(AssertionError, match="timeline differs"):
        acceptance.native._sample_images(sample(), {"workspace": iter([original])})
