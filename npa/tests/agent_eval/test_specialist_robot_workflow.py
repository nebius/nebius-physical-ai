"""Prepared robot workflow boundaries without paid inference or native simulation."""

from copy import deepcopy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLE = Path(__file__).parents[2] / "examples/specialists/robot_workflow"


@pytest.fixture
def modules(monkeypatch):
    pytest.importorskip("av")
    monkeypatch.syspath_prepend(str(_EXAMPLE))
    for name in ("matrix", "workload", "verify", "replay", "media", "native_dataset"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return SimpleNamespace(
        **{
            name: importlib.import_module(name)
            for name in ("matrix", "workload", "verify", "replay")
        }
    )


def test_matrix_preserves_real_negative_control(modules):
    matrix, digest = modules.matrix.load_matrix(_EXAMPLE / "scenes.json")
    assert [case["expected_accepted"] for case in matrix["cases"]] == [
        True,
        False,
        True,
    ]
    assert digest == modules.matrix.file_digest(_EXAMPLE / "scenes.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "seed_bool",
        "unknown_controller",
        "unexpected_field",
        "invalid_scene",
    ],
)
def test_matrix_rejects_ambiguous_cases(modules, tmp_path, mutation):
    matrix = json.loads((_EXAMPLE / "scenes.json").read_text())
    case = matrix["cases"][1]
    if mutation == "duplicate":
        case["id"] = matrix["cases"][0]["id"]
    elif mutation == "seed_bool":
        case["seed"] = True
    elif mutation == "unknown_controller":
        case["controller"] = "pretend_success"
    elif mutation == "unexpected_field":
        case["skip"] = True
    else:
        case["scene"]["goal_x"] = 10
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError):
        modules.matrix.load_matrix(path)


def test_open_gripper_restores_controller_after_failure(modules):
    def original(position, goal):
        return [("grasp", position, -1, 15)]

    simulation = SimpleNamespace(_phases=original)
    with (
        pytest.raises(RuntimeError),
        modules.workload._controller(simulation, "open_gripper"),
    ):
        assert simulation._phases([0], [1])[0][2] == 1
        raise RuntimeError("native failure")
    assert simulation._phases is original


def test_inventory_rejects_substituted_bytes_and_hidden_artifacts(modules, tmp_path):
    artifact = tmp_path / "data.bin"
    artifact.write_bytes(b"original")
    (tmp_path / "report.json").write_text("{}")
    manifest = {
        "data.bin": {"bytes": 8, "sha256": modules.matrix.file_digest(artifact)}
    }
    modules.verify._inventory(tmp_path, manifest)
    artifact.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash differs"):
        modules.verify._inventory(tmp_path, manifest)
    artifact.write_bytes(b"original")
    (tmp_path / "unreported").write_text("extra")
    with pytest.raises(ValueError, match="inventory differs"):
        modules.verify._inventory(tmp_path, manifest)


def test_inventory_rejects_directory_symlink(modules, tmp_path):
    root, outside = tmp_path / "run", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "report.json").write_text("{}")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="linked artifact"):
        modules.verify._inventory(root, {})


def test_accepted_indices_exclude_failed_episode(modules):
    observed = {"accepted": True, "checks": {"grasp": True}, **_metrics()}
    row = {
        "status": "accepted",
        "simulation": deepcopy(observed),
        "dataset_episode_index": 0,
    }
    accepted = []
    modules.verify._acceptance(row, observed, accepted)
    second = deepcopy(row)
    second["dataset_episode_index"] = 2
    with pytest.raises(ValueError, match="contiguous"):
        modules.verify._acceptance(second, observed, accepted)
    rejected = {
        "status": "rejected",
        "simulation": {"accepted": False, "checks": {"grasp": False}, **_metrics()},
        "dataset_episode_index": 1,
    }
    with pytest.raises(ValueError, match="rejected episode"):
        modules.verify._acceptance(rejected, rejected["simulation"], accepted)


def test_task_label_is_derived_from_frozen_scene(modules):
    case = json.loads((_EXAMPLE / "scenes.json").read_text())["cases"][0]
    row = {
        "id": case["id"],
        "scene": case["scene"],
        "simulation_seed": case["seed"],
        "controller": case["controller"],
        "episode_path": "episodes/episode_0000",
        "simulation": {
            "task": "Forged task",
            "frames": 185,
            "fps": 25,
            "environment": "FetchPickAndPlace-v4",
        },
    }
    with pytest.raises(ValueError, match="task label"):
        modules.verify._record_identity(row, case, 0)


def test_failed_verification_is_retained_without_claiming_success(modules, tmp_path):
    evidence = tmp_path / "verification"
    with pytest.raises(FileNotFoundError):
        modules.verify.verify_workflow(
            tmp_path / "missing",
            _EXAMPLE / "scenes.json",
            {},
            Path("missing-python"),
            evidence,
        )
    receipt = json.loads((evidence / "result.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["error_type"] == "FileNotFoundError"
    assert receipt["seconds"] >= 0


def test_all_rejected_run_requires_absent_dataset(modules, tmp_path):
    assert (
        modules.verify._native_reader(tmp_path, [], Path("unused"), tmp_path)[
            "native_frames"
        ]
        == 0
    )
    (tmp_path / "dataset").mkdir()
    with pytest.raises(ValueError, match="rejected cases"):
        modules.verify._native_reader(tmp_path, [], Path("unused"), tmp_path)


@pytest.fixture
def media(modules):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for real compressed timeline checks")
    return importlib.import_module("media")


def _moving_frames():
    import numpy as np

    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    for index in range(5):
        frames[index, 8:24, 4 * index : 4 * index + 8, 0] = 255
    return frames


def test_compressed_video_matches_independent_reference(media, tmp_path):
    frames = _moving_frames()
    video = tmp_path / "output.mp4"
    media._encode([frames], video)
    assert media.verify_video(video, [frames]) == 5


@pytest.mark.parametrize("change", ["shift", "duplicate", "truncate"])
def test_compressed_video_rejects_temporal_corruption(media, tmp_path, change):
    import numpy as np

    frames = _moving_frames()
    if change == "shift":
        corrupted = np.concatenate([frames[1:], frames[-1:]])
    elif change == "duplicate":
        corrupted = frames.copy()
        corrupted[2] = corrupted[1]
    else:
        corrupted = frames[:-1]
    video = tmp_path / "output.mp4"
    media._encode([corrupted], video)
    with pytest.raises((AssertionError, ValueError)):
        media.verify_video(video, [frames])


def _metrics():
    return {
        "final_distance_m": 0.005,
        "lift_height_m": 0.15,
        "settled_max_speed_m_s": 0.0,
        "bilateral_contact_frames": 90,
        "judge": "mujoco_state_and_contacts",
    }


@pytest.mark.parametrize("value", [0.9, float("nan"), float("inf"), True])
def test_reported_metrics_must_match_independent_physics(modules, value):
    expected = _metrics()
    claimed = dict(expected, final_distance_m=value)
    with pytest.raises(ValueError, match="metric differs"):
        modules.verify._metrics(claimed, expected)
