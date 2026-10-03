"""Prevent training provenance from aliasing or escaping its recorded episode set."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from npa.adapter import sim_to_lerobot as adapter
from npa.workbench.token_factory import robot_artifacts


def _record(relative):
    return {
        "status": "accepted",
        "episode_path": relative,
        "simulation": {"task": "Recorded demonstration"},
    }


@pytest.mark.parametrize("kind", ["parent", "absolute", "linked"])
def test_accepted_episode_cannot_escape_run_root(tmp_path, monkeypatch, kind):
    root, outside = tmp_path / "run", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    relative = "../outside" if kind == "parent" else str(outside)
    if kind == "linked":
        (root / "linked").symlink_to(outside, target_is_directory=True)
        relative = "linked"
    records = [_record(relative)]
    original = copy.deepcopy(records)
    monkeypatch.setattr(
        robot_artifacts,
        "convert",
        lambda *_a, **_k: pytest.fail("escaped source reached adapter"),
    )
    with pytest.raises((ValueError, adapter.AdapterError)):
        robot_artifacts.export_robot_dataset(root, records)
    assert records == original and not (root / "dataset").exists()


@pytest.mark.parametrize("alias", [False, True])
def test_same_physical_episode_cannot_be_exported_twice(tmp_path, monkeypatch, alias):
    episode = tmp_path / "episode"
    episode.mkdir()
    relative = "episode"
    if alias:
        (tmp_path / "alias").symlink_to(episode, target_is_directory=True)
        relative = "alias"
    records = [_record("episode"), _record(relative)]
    monkeypatch.setattr(
        robot_artifacts,
        "convert",
        lambda *_a, **_k: pytest.fail("duplicate source reached adapter"),
    )
    with pytest.raises((ValueError, adapter.AdapterError)):
        robot_artifacts.export_robot_dataset(tmp_path, records)
    assert all("dataset_episode_index" not in record for record in records)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True])
def test_invalid_frame_rate_is_rejected_before_encoding(tmp_path, monkeypatch, fps):
    source = tmp_path / "source"
    episode = source / "episode_0000"
    episode.mkdir(parents=True)
    for key in ("obs_workspace", "obs_wrist"):
        np.save(episode / key, np.zeros((3, 8, 8, 3), dtype=np.uint8))
    for key, width in (("state", 9), ("actions", 4)):
        np.save(episode / key, np.zeros((3, width), dtype=np.float32))
    monkeypatch.setattr(
        adapter,
        "encode_video",
        lambda *_a, **_k: pytest.fail("invalid rate reached encoder"),
    )
    with pytest.raises(adapter.AdapterError):
        adapter.convert(source, tmp_path / "dataset", fps=fps)
    assert not (tmp_path / "dataset").exists()
