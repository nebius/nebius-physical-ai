from __future__ import annotations

import importlib
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.adapter.isaac_lab_lerobot import (
    G1_BONE_PAIRS,
    G1_STATE_DIM,
    LeRobotFeatureSpec,
    WORKSPACE_VIEW_KEY,
    convert,
)
from npa.viz.adapters.lerobot_to_rerun import (
    REPRESENTATIVE_JOINTS,
    _build_logical_blueprint,
    lerobot_dataset_logical_to_rerun,
    lerobot_to_rerun,
    verify_rerun_entities,
)
from npa.viz.lerobot import VizDataError


def _write_g1_raw_dataset(root: Path, *, frames: int = 10) -> Path:
    raw = root / "raw"
    episode = raw / "episode_000000"
    episode.mkdir(parents=True)
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    state = np.zeros((frames, G1_STATE_DIM), dtype=np.float32)
    state[:, 0] = np.sin(t * np.pi) * 0.10
    state[:, 6] = np.sin(t * np.pi * 2.0) * 0.25
    state[:, 15] = np.cos(t * np.pi * 2.0) * 0.20
    state[:, 29] = np.cos(t * np.pi * 2.0) * -0.20
    np.save(episode / "state.npy", state)
    np.save(episode / "actions.npy", state + 0.05)
    return raw


def _write_lerobot_dataset(root: Path, *, frames: int = 10, fps: int = 10) -> Path:
    return convert(
        _write_g1_raw_dataset(root, frames=frames),
        root / "lerobot",
        fps=fps,
        task="Isaac-Velocity-Flat-G1-v0",
    )


def _write_empty_lerobot_dataset(root: Path, *, fps: int = 10) -> Path:
    dataset = root / "empty-lerobot"
    data_path = dataset / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array([], type=pa.list_(pa.float32(), G1_STATE_DIM)),
                "index": pa.array([], type=pa.int64()),
            }
        ),
        data_path,
    )
    meta_path = dataset / "meta" / "info.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(f'{{"fps": {fps}, "robot_type": "unitree_g1"}}')
    return dataset


def _recording_chunks(path: Path):
    from rerun.recording import load_recording

    return list(load_recording(path).chunks())


def _entity_paths(chunks) -> set[str]:
    return {str(chunk.entity_path) for chunk in chunks}


def _dynamic_row_count(chunks, entity_path: str) -> int:
    return sum(
        int(chunk.num_rows)
        for chunk in chunks
        if str(chunk.entity_path) == entity_path and not chunk.is_static
    )


def test_lerobot_to_rerun_writes_expected_entities_and_frame_count(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=10, fps=10)
    output = tmp_path / "isaac-lab-trajectory.rrd"

    lerobot_to_rerun(dataset, output)

    assert output.exists()
    assert output.stat().st_size > 0
    chunks = _recording_chunks(output)
    entity_paths = _entity_paths(chunks)
    assert "/world/skeleton/joints" in entity_paths
    assert "/world/skeleton/bones" in entity_paths
    for joint_name in REPRESENTATIVE_JOINTS:
        assert f"/world/skeleton/angles/{joint_name}" in entity_paths

    assert _dynamic_row_count(chunks, "/world/skeleton/joints") == 10
    assert _dynamic_row_count(chunks, "/world/skeleton/bones") == 10
    for joint_name in REPRESENTATIVE_JOINTS:
        assert _dynamic_row_count(chunks, f"/world/skeleton/angles/{joint_name}") == 10


def test_lerobot_to_rerun_duration_cap_subsamples_to_five_seconds(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=100, fps=10)
    output = tmp_path / "capped.rrd"

    lerobot_to_rerun(dataset, output)

    chunks = _recording_chunks(output)
    assert _dynamic_row_count(chunks, "/world/skeleton/joints") == 50
    assert _dynamic_row_count(chunks, "/world/skeleton/bones") == 50


def test_lerobot_to_rerun_caps_trajectory_longer_than_ten_seconds(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=120, fps=10)
    output = tmp_path / "capped-long.rrd"

    lerobot_to_rerun(dataset, output)

    chunks = _recording_chunks(output)
    assert _dynamic_row_count(chunks, "/world/skeleton/joints") == 50


def test_lerobot_to_rerun_single_frame_dataset(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=1, fps=10)
    output = tmp_path / "single-frame.rrd"

    lerobot_to_rerun(dataset, output)

    chunks = _recording_chunks(output)
    assert _dynamic_row_count(chunks, "/world/skeleton/joints") == 1
    assert _dynamic_row_count(chunks, "/world/skeleton/bones") == 1


def test_lerobot_to_rerun_empty_dataset_is_clean_error(tmp_path: Path) -> None:
    dataset = _write_empty_lerobot_dataset(tmp_path)

    with pytest.raises(VizDataError, match="No observation.state rows found"):
        lerobot_to_rerun(dataset, tmp_path / "empty.rrd")


def test_lerobot_to_rerun_records_bone_segments(tmp_path: Path) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=10, fps=10)
    output = tmp_path / "bones.rrd"

    lerobot_to_rerun(dataset, output)

    chunks = _recording_chunks(output)
    bone_chunk = next(
        chunk for chunk in chunks if str(chunk.entity_path) == "/world/skeleton/bones" and not chunk.is_static
    )
    batch = bone_chunk.to_record_batch()
    strips = batch.column("LineStrips3D:strips").to_pylist()[0]
    assert len(strips) == len(G1_BONE_PAIRS)
    assert len(strips[0]) == 2
    assert len(strips[0][0]) == 3


def test_lerobot_to_rerun_uploads_s3_output_after_local_save(tmp_path: Path, mocker) -> None:
    dataset = _write_lerobot_dataset(tmp_path, frames=10, fps=10)
    storage = mocker.Mock()

    def upload_file(local_file: str, destination: str) -> str:
        local_path = Path(local_file)
        assert local_path.exists()
        assert local_path.suffix == ".rrd"
        assert local_path.stat().st_size > 0
        assert destination == "s3://bucket/visuals/out.rrd"
        return destination

    storage.upload_file.side_effect = upload_file
    adapter_module = importlib.import_module("npa.viz.adapters.lerobot_to_rerun")
    mocker.patch.object(adapter_module, "_storage_client", return_value=storage)

    lerobot_to_rerun(dataset, "s3://bucket/visuals/out.rrd")

    storage.upload_file.assert_called_once()


def test_verify_rerun_entities_uses_fallback_counts_without_recording_loader(tmp_path: Path, mocker) -> None:
    output = tmp_path / "logical.rrd"
    output.write_bytes(b"rrd")
    counts = {"/input_dataset/episodes/episode_000000/state/dim_00": 3}
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rerun.recording":
            raise ImportError("recording loader unavailable")
        return real_import(name, globals, locals, fromlist, level)

    mocker.patch("builtins.__import__", side_effect=fake_import)

    assert verify_rerun_entities(
        output,
        ["input_dataset/episodes/episode_000000/state/dim_00"],
        fallback_counts=counts,
    ) == counts


class _FakePanelState:
    Hidden = "hidden"
    Expanded = "expanded"


class _FakeBlueprintApi:
    PanelState = _FakePanelState

    def __getattr__(self, name: str):
        def construct(*args, **kwargs):
            return {"kind": name, "children": list(args), **kwargs}

        return construct


def _view_nodes(node: object) -> list[dict]:
    if not isinstance(node, dict):
        return []
    found = [node] if str(node.get("kind", "")).endswith("View") else []
    for child in node.get("children", []):
        found.extend(_view_nodes(child))
    return found


def test_logical_blueprint_without_cameras_opens_state_and_actions() -> None:
    blueprint = _build_logical_blueprint(
        _FakeBlueprintApi(),
        [],
        input_episode_indices=[],
        rollout_episode_indices=[0, 1],
    )
    views = _view_nodes(blueprint)

    assert not any(view["kind"] == "Spatial2DView" for view in views)
    assert {view.get("name") for view in views} == {"State", "Policy actions", "VLM/VLA eval"}
    assert next(view for view in views if view.get("name") == "State")["contents"] == [
        "policy_rollout/episodes/episode_000000/state/**",
        "policy_rollout/episodes/episode_000001/state/**",
    ]
    assert next(view for view in views if view.get("name") == "Policy actions")["contents"] == (
        [
            "policy_rollout/episodes/episode_000000/actions/**",
            "policy_rollout/episodes/episode_000001/actions/**",
        ]
    )


def test_logical_blueprint_with_cameras_keeps_images_and_signals_visible() -> None:
    blueprint = _build_logical_blueprint(
        _FakeBlueprintApi(),
        ["observation.images.workspace"],
        input_episode_indices=[0],
        rollout_episode_indices=[1],
    )
    views = _view_nodes(blueprint)

    assert [view.get("name") for view in views if view["kind"] == "Spatial2DView"] == [
        "Input demos",
        "Isaac environment — trained policy",
    ]
    assert next(view for view in views if view.get("name") == "Input demos")["contents"] == [
        "input_dataset/episodes/episode_000000/camera/**"
    ]
    assert next(
        view
        for view in views
        if view.get("name") == "Isaac environment — trained policy"
    )["contents"] == ["policy_rollout/episodes/episode_000001/camera/**"]
    assert {view.get("name") for view in views if view["kind"] == "TimeSeriesView"} == {
        "State",
        "Policy actions",
        "VLM/VLA eval",
    }


def test_logical_blueprint_rollout_rgb_is_primary_and_omits_empty_eval() -> None:
    blueprint = _build_logical_blueprint(
        _FakeBlueprintApi(),
        [WORKSPACE_VIEW_KEY],
        input_episode_indices=[],
        rollout_episode_indices=[0],
        has_feedback=False,
    )
    views = _view_nodes(blueprint)

    assert [view.get("name") for view in views if view["kind"] == "Spatial2DView"] == [
        "Isaac environment — trained policy"
    ]
    assert {view.get("name") for view in views if view["kind"] == "TimeSeriesView"} == {
        "State",
        "Policy actions",
    }
    horizontal = blueprint["children"][0]
    assert horizontal["column_shares"] == [4.0, 1.6]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_logical_rerun_maps_each_real_rgb_video_to_its_episode_timeline(
    tmp_path: Path,
) -> None:
    spec = LeRobotFeatureSpec(
        state_names=["cart", "pole"],
        action_names=["force"],
        robot_type="cartpole",
    )
    raw = tmp_path / "raw"
    for episode_index in range(2):
        episode = raw / f"episode_{episode_index:06d}"
        episode.mkdir(parents=True)
        frame_count = 4
        np.save(episode / "state.npy", np.zeros((frame_count, 2), dtype=np.float32))
        np.save(episode / "actions.npy", np.zeros((frame_count, 1), dtype=np.float32))
        pixels = np.zeros((frame_count, 32, 48, 3), dtype=np.uint8)
        pixels[..., episode_index] = 180
        pixels[:, 8:24, 18:30, 2] = 255
        np.save(episode / "rgb.npy", pixels)
    dataset = convert(raw, tmp_path / "lerobot", fps=10, spec=spec)
    output = tmp_path / "isaac-rgb.rrd"

    result = lerobot_dataset_logical_to_rerun(
        dataset,
        output,
        input_episode_indices=[],
        rollout_episode_indices=[0, 1],
        feedback_by_episode={},
        max_frames_per_episode=4,
    )

    assert result.entity_counts[
        "/policy_rollout/episodes/episode_000000/camera/observation_images_workspace"
    ] == 4
    chunks = _recording_chunks(output)
    assert _dynamic_row_count(
        chunks,
        "/policy_rollout/episodes/episode_000001/camera/observation_images_workspace",
    ) == 4
    paths = _entity_paths(chunks)
    assert "/videos/episode_000000/observation_images_workspace" in paths
    assert "/videos/episode_000001/observation_images_workspace" in paths
    assert not any(path.startswith("/eval/") for path in paths)
