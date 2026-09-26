"""Keep failed conversions unpublished and bind provenance only to complete datasets."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from npa.adapter import sim_to_lerobot as adapter
from npa.workbench.token_factory import robot_artifacts


def _episodes(root):
    records = []
    for index in range(2):
        relative = f"episodes/episode_{index:04d}"
        episode = root / relative
        episode.mkdir(parents=True)
        for name in ("obs_workspace", "obs_wrist"):
            np.save(episode / name, np.zeros((3, 8, 8, 3), dtype=np.uint8))
        np.save(episode / "state", np.zeros((3, 9), dtype=np.float32))
        np.save(episode / "actions", np.zeros((3, 4), dtype=np.float32))
        records.append(
            {
                "id": str(index),
                "status": "accepted",
                "episode_path": relative,
                "simulation": {"task": f"move cube {index}"},
            }
        )
    return records


def _converted(_source, output, **_kwargs):
    (output / "meta").mkdir(parents=True)
    (output / "meta/info.json").write_text(
        json.dumps({"features": {"observation.state": {}, "action": {}}})
    )


def test_failed_conversion_leaves_no_dataset_or_phantom_indices(tmp_path, monkeypatch):
    records = _episodes(tmp_path)
    original = copy.deepcopy(records)

    def fail(_source, output, **_kwargs):
        output.mkdir(parents=True)
        (output / "partial-video.mp4").write_bytes(b"incomplete")
        raise adapter.AdapterError("injected encoder failure after partial output")

    monkeypatch.setattr(robot_artifacts, "convert", fail)
    with pytest.raises(adapter.AdapterError):
        robot_artifacts.export_robot_dataset(tmp_path, records)
    assert records == original
    assert not (tmp_path / "dataset").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["episodes"]
    monkeypatch.setattr(robot_artifacts, "convert", _converted)
    assert robot_artifacts.export_robot_dataset(tmp_path, records) == 2
    assert [record["dataset_episode_index"] for record in records] == [0, 1]


def test_metadata_failure_also_rolls_back(tmp_path, monkeypatch):
    records = _episodes(tmp_path)
    original = copy.deepcopy(records)

    def incomplete(_source, output, **_kwargs):
        (output / "meta").mkdir(parents=True)
        (output / "meta/info.json").write_text("{}")

    monkeypatch.setattr(robot_artifacts, "convert", incomplete)
    with pytest.raises((KeyError, ValueError, adapter.AdapterError)):
        robot_artifacts.export_robot_dataset(tmp_path, records)
    assert records == original and not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling_symlink"])
def test_existing_destination_is_never_touched(tmp_path, monkeypatch, kind):
    records = _episodes(tmp_path)
    destination = tmp_path / "dataset"
    target = tmp_path / "operator-data"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("operator-owned")
    if kind == "directory":
        destination.mkdir()
    elif kind == "file":
        destination.write_text("operator-owned")
    else:
        destination.symlink_to(target if kind == "symlink" else tmp_path / "missing")
    monkeypatch.setattr(
        robot_artifacts, "convert", lambda *_a, **_k: pytest.fail("conversion started")
    )
    with pytest.raises((FileExistsError, adapter.AdapterError)):
        robot_artifacts.export_robot_dataset(tmp_path, records)
    assert sentinel.read_text() == "operator-owned"
    assert all("dataset_episode_index" not in record for record in records)


def test_rejected_episodes_are_not_validated_or_assigned_indices(tmp_path, monkeypatch):
    records = _episodes(tmp_path)
    records.insert(1, {"status": "rejected", "episode_path": "absent-negative-control"})
    monkeypatch.setattr(robot_artifacts, "convert", _converted)
    assert robot_artifacts.export_robot_dataset(tmp_path, records) == 2
    assert [record.get("dataset_episode_index") for record in records] == [0, None, 1]


@pytest.mark.parametrize(
    "defect",
    [
        "length",
        "rgb_dtype",
        "state_nan",
        "action_inf",
        "empty",
        "width",
        "camera_shape",
    ],
)
@pytest.mark.parametrize("episode_index", [0, 1])
def test_every_episode_is_validated_before_output_or_encoding(
    tmp_path, monkeypatch, defect, episode_index
):
    records = _episodes(tmp_path)
    source = tmp_path / "episodes"
    episode = tmp_path / records[episode_index]["episode_path"]
    replacements = {
        "length": ("actions", np.zeros((2, 4), dtype=np.float32)),
        "rgb_dtype": ("obs_wrist", np.zeros((3, 8, 8, 3), dtype=np.float32)),
        "state_nan": ("state", np.full((3, 9), np.nan, dtype=np.float32)),
        "action_inf": ("actions", np.full((3, 4), np.inf, dtype=np.float32)),
        "empty": ("state", np.empty((0, 9), dtype=np.float32)),
        "width": ("state", np.zeros((3, 8), dtype=np.float32)),
        "camera_shape": ("obs_wrist", np.zeros((3, 10, 8, 3), dtype=np.uint8)),
    }
    name, data = replacements[defect]
    np.save(episode / name, data)
    output = tmp_path / "converted"
    monkeypatch.setattr(
        adapter,
        "encode_video",
        lambda *_a, **_k: pytest.fail("encoding started before validation"),
    )
    with pytest.raises(adapter.AdapterError):
        adapter.convert(source, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["state", "actions", "obs_workspace", "obs_wrist"])
def test_scalar_arrays_raise_adapter_error_before_encoding(tmp_path, monkeypatch, name):
    records = _episodes(tmp_path)
    np.save(tmp_path / records[0]["episode_path"] / name, np.array(1.0))
    monkeypatch.setattr(
        adapter, "encode_video", lambda *_a, **_k: pytest.fail("encoding started")
    )
    with pytest.raises(adapter.AdapterError):
        adapter.convert(tmp_path / "episodes", tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("dtype", [str, complex, bool])
def test_numeric_dataset_features_require_real_values(tmp_path, monkeypatch, dtype):
    records = _episodes(tmp_path)
    np.save(
        tmp_path / records[0]["episode_path"] / "actions",
        np.zeros((3, 4)).astype(dtype),
    )
    monkeypatch.setattr(
        adapter, "encode_video", lambda *_a, **_k: pytest.fail("encoding started")
    )
    with pytest.raises(adapter.AdapterError):
        adapter.convert(tmp_path / "episodes", tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
