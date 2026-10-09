"""Prevent independent photographs from sharing an incorrect rigid-rig pose."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workbench.nurec import photographic_timeline as timeline
from npa.workbench.nurec.nurec import NurecConfig, NurecError


def _photographs():
    cameras = {}
    trajectories = {}
    for name, offset in (("camera1", 0), ("camera2", 10)):
        times = np.array([0, 1000000], dtype=np.uint64)
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[:, 0, 3] = [offset, offset + 1]
        cameras[name] = SimpleNamespace(
            frames_timestamps_us=np.stack([times, times], axis=1), frames_count=2
        )
        trajectories[name] = poses, times
    reader = SimpleNamespace(
        sequence_timestamp_interval_us=SimpleNamespace(start=0, stop=2000001)
    )
    return reader, cameras, trajectories


def test_independent_photographs_have_unique_times_and_exact_source_poses():
    reader, cameras, trajectories = _photographs()
    frames = timeline._plan_frames(reader, cameras, trajectories)

    assert len(frames) == 4
    assert len({frame["training_timestamp_us"] for frame in frames}) == 4
    assert [frame["source_timestamp_us"] for frame in frames] == [0, 1000000] * 2
    for frame in frames:
        source = trajectories[frame["camera_id"]][0][frame["frame_index"]]
        np.testing.assert_array_equal(frame["T_camera_world"], source)
    assert frames == timeline._plan_frames(
        reader, dict(reversed(cameras.items())), trajectories
    )


def test_retiming_rejects_rolling_shutter_and_missing_camera_poses():
    reader, cameras, trajectories = _photographs()
    cameras["camera1"].frames_timestamps_us[1, 0] -= 1
    with pytest.raises(NurecError, match="instantaneous"):
        timeline._plan_frames(reader, cameras, trajectories)
    reader, cameras, trajectories = _photographs()
    del trajectories["camera2"]
    with pytest.raises(NurecError, match="no poses"):
        timeline._plan_frames(reader, cameras, trajectories)


def test_retiming_rejects_a_timeline_that_cannot_hold_every_photo():
    reader, cameras, trajectories = _photographs()
    reader.sequence_timestamp_interval_us.stop = 3
    with pytest.raises(NurecError, match="unique photographic timestamps"):
        timeline._plan_frames(reader, cameras, trajectories)


def test_dry_run_and_native_rig_routing_do_not_write(tmp_path):
    source = tmp_path / "capture.json"
    source.write_text("{}")
    config = NurecConfig(out_dir=tmp_path / "output")
    assert timeline.prepare_photographic_training(config, str(source)) == (
        config,
        str(source),
        "",
    )
    (tmp_path / "npa-rig.json").write_text(
        json.dumps(
            {
                "reference_camera": "camera2",
                "cameras": ["camera1", "camera2"],
                "poses_component_group": "npa_rig",
            }
        )
    )
    planned, metadata, evidence = timeline.prepare_photographic_training(
        config, str(source), dry_run=True
    )
    assert planned.poses_component_group == "npa_photographic"
    assert metadata.endswith("photographic-input/capture.json")
    assert evidence.endswith("photographic-timeline.json")
    assert not config.resolved_out_dir.exists()
    custom = replace(config, config_name="custom.yaml")
    assert timeline.prepare_photographic_training(custom, str(source)) == (
        custom,
        str(source),
        "",
    )


def test_preparation_round_trips_images_and_poses_with_public_ncore(tmp_path):
    pytest.importorskip("ncore.data.v4")
    from npa.workbench.nurec.ncore_rig import (
        camera_world_trajectories,
        derive_rig_poses,
    )

    source = _write_capture(tmp_path / "source")
    derived = derive_rig_poses(source, output_dir=tmp_path / "derived")
    assert derived.ok
    original = camera_world_trajectories(derived.output_meta)
    config, prepared, evidence = timeline.prepare_photographic_training(
        NurecConfig(out_dir=tmp_path / "output"), derived.output_meta
    )
    payload = json.loads(open(evidence).read())
    assert payload["frame_count"] == 4
    assert payload["camera_ids"] == ["camera1", "camera2"]
    assert config.poses_component_group == "npa_photographic"
    exported = camera_world_trajectories(prepared)
    for frame in payload["frames"]:
        poses, times = exported[frame["camera_id"]]
        actual = poses[np.searchsorted(times, frame["training_timestamp_us"])]
        expected = original[frame["camera_id"]][0][frame["frame_index"]]
        np.testing.assert_array_equal(actual, expected)


def _write_capture(root):
    from ncore.data.v4 import (
        CameraSensorComponent,
        PosesComponent,
        SequenceComponentGroupsReader,
        SequenceComponentGroupsWriter,
    )
    from ncore.impl.common.transformations import HalfClosedInterval
    from upath import UPath

    root.mkdir()
    writer = SequenceComponentGroupsWriter(
        UPath(root), "scene", "scene", HalfClosedInterval(0, 2000001), {}
    )
    poses = writer.register_component_writer(PosesComponent.Writer, "default")
    _, cameras, trajectories = _photographs()
    for name, camera in cameras.items():
        output = writer.register_component_writer(
            CameraSensorComponent.Writer, name, group_name=name
        )
        for index, times in enumerate(camera.frames_timestamps_us):
            output.store_frame(
                image_binary_data=f"encoded-{name}-{index}".encode(),
                image_format="JPEG",
                frame_timestamps_us=times,
                generic_data={"source": np.array([index])},
                generic_meta_data={"original": name},
            )
        matrices, times = trajectories[name]
        poses.store_dynamic_pose(
            name, "world", matrices, times, require_sequence_time_coverage=False
        )
    reader = SequenceComponentGroupsReader(writer.finalize())
    metadata = root / "scene.json"
    metadata.write_text(json.dumps(reader.get_sequence_meta().to_dict()))
    return metadata
