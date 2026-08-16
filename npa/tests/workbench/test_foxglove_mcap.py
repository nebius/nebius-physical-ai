"""MCAP conversion tests — real files in, real MCAP out, read back to prove it.

The `mcap` dependency is optional (`npa[foxglove]`), so the round-trip tests skip
when it is absent; the graceful-degradation path is asserted unconditionally.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import jsonschema
import pytest

from npa.workbench.foxglove import (
    FOXGLOVE_EMBED_SDK_VERSION,
    MCAP_MAGIC,
    sdk_assets_present,
    sdk_tarball_url,
)
from npa.workbench.foxglove.inspect import (
    McapInspectError,
    format_mcap_info,
    has_mcap_magic,
    summarize_mcap,
)
from npa.workbench.foxglove.mcap_writer import (
    COMPRESSED_IMAGE_SCHEMA,
    LOG_SCHEMA,
    SCENE_UPDATE_SCHEMA,
    SCENE_UPDATE_SCHEMA_SOURCE,
    FrameInput,
    LogInput,
    McapWriteError,
    MetricsInput,
    collect_run_inputs,
    convert_run_directory,
    safe_topic,
    write_run_mcap,
)

pytest.importorskip("PIL", reason="Pillow is required to build image fixtures")


def _run_fixture(tmp_path: Path, *, frames: int = 3) -> Path:
    from PIL import Image

    root = tmp_path / "run"
    (root / "camera" / "front").mkdir(parents=True)
    (root / "camera" / "wrist").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    for index in range(frames):
        Image.new("RGB", (16, 12), (10 * index, 90, 200)).save(
            root / "camera" / "front" / f"{index:04d}.png"
        )
        # .ppm frames are common in NPA sim rollouts and must be transcoded.
        Image.new("RGB", (8, 6), (200, 10 * index, 20)).save(
            root / "camera" / "wrist" / f"{index:04d}.ppm"
        )
    (root / "reports" / "metrics.json").write_text(
        json.dumps(
            {"success_rate": 0.82, "episodes": 12, "gate": "promote", "extra": {"a": 1}}
        ),
        encoding="utf-8",
    )
    (root / "reports" / "run.log").write_text(
        "stage 1 ok\nstage 2 ok\n\nstage 3 warn\n", encoding="utf-8"
    )
    (root / "reports" / "opaque.bin").write_bytes(b"\x00\x01\x02")
    return root


def test_safe_topic_is_stable_and_sanitized() -> None:
    assert safe_topic("front", prefix="/camera") == "/camera/front"
    assert safe_topic("weird name!", prefix="/camera") == "/camera/weird_name"
    assert safe_topic("", prefix="/camera") == "/camera/default"
    assert safe_topic("front", prefix="") == "/front"


def test_collect_run_inputs_classifies_artifacts(tmp_path: Path) -> None:
    root = _run_fixture(tmp_path)
    frames, metrics, logs, skipped = collect_run_inputs(root)

    assert {frame.camera for frame in frames} == {"front", "wrist"}
    assert len(frames) == 6
    assert [metric.name for metric in metrics] == ["metrics"]
    assert [metric.source for metric in metrics] == ["reports/metrics.json"]
    assert [entry.name for entry in logs] == ["run"]
    assert skipped == ["reports/opaque.bin"]

    limited, _metrics, _logs, _skipped = collect_run_inputs(root, max_frames=2)
    assert len(limited) == 2

    with pytest.raises(McapWriteError):
        collect_run_inputs(tmp_path / "missing")


def test_schemas_are_the_foxglove_well_known_shapes() -> None:
    assert COMPRESSED_IMAGE_SCHEMA["title"] == "foxglove.CompressedImage"
    assert set(COMPRESSED_IMAGE_SCHEMA["properties"]) == {
        "timestamp",
        "frame_id",
        "data",
        "format",
    }
    assert COMPRESSED_IMAGE_SCHEMA["properties"]["data"]["contentEncoding"] == "base64"
    assert LOG_SCHEMA["title"] == "foxglove.Log"
    assert {"timestamp", "level", "message", "name", "file", "line"} <= set(
        LOG_SCHEMA["properties"]
    )


def test_write_run_mcap_requires_input(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    with pytest.raises(McapWriteError, match="nothing to convert"):
        write_run_mcap(output=tmp_path / "empty.mcap")


def test_convert_run_directory_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    root = _run_fixture(tmp_path)
    output = tmp_path / "out" / "session.mcap"

    summary = convert_run_directory(
        input_path=root, output=output, fps=5.0, run_id="run-42"
    )

    assert output.is_file()
    assert has_mcap_magic(output)
    assert output.read_bytes()[: len(MCAP_MAGIC)] == MCAP_MAGIC
    assert summary.frames == 6
    assert summary.metrics == 1
    assert summary.logs == 3  # blank lines dropped
    assert summary.message_count == 10
    assert summary.timestamps == "synthetic-fps"
    assert summary.fps == 5.0
    assert "reports/opaque.bin" in summary.skipped
    assert summary.channels["/camera"] == 3

    # Read it back the way Foxglove would.
    info = summarize_mcap(output)
    assert info.message_count == 10
    assert info.schemas["/camera"] == "foxglove.CompressedImage"
    assert info.schemas["/camera/wrist"] == "foxglove.CompressedImage"
    assert info.schemas["/log"] == "foxglove.Log"
    assert info.schemas["/metrics/metrics"].startswith("npa.RunMetrics")
    assert info.metadata["npa"]["run_id"] == "run-42"
    # Timestamp provenance must be recorded, not implied.
    assert info.metadata["npa"]["timestamps"] == "synthetic-fps"
    assert info.duration_s > 0
    assert (
        info.channel_time_ranges["/camera"] == info.channel_time_ranges["/camera/wrist"]
    )
    assert "foxglove.CompressedImage" in format_mcap_info(info)

    # Message payloads must be real, decodable images / values.
    with output.open("rb") as handle:
        reader = make_reader(handle)
        seen: dict[str, list[dict]] = {}
        timestamps: list[int] = []
        for _schema, channel, message in reader.iter_messages():
            seen.setdefault(channel.topic, []).append(json.loads(message.data))
            timestamps.append(message.log_time)

    front = seen["/camera"][0]
    assert front["format"] == "png"
    assert front["frame_id"] == "front"
    assert base64.b64decode(front["data"])[:8] == b"\x89PNG\r\n\x1a\n"
    # .ppm frames are transcoded to PNG so Foxglove can render them.
    wrist = seen["/camera/wrist"][0]
    assert wrist["format"] == "png"
    assert base64.b64decode(wrist["data"])[:8] == b"\x89PNG\r\n\x1a\n"

    metrics_msg = seen["/metrics/metrics"][0]
    assert metrics_msg["success_rate"] == 0.82
    assert metrics_msg["episodes"] == 12
    # Nested structures are flattened to strings so the Plot panel stays usable.
    assert json.loads(metrics_msg["extra"]) == {"a": 1}

    log_msg = seen["/log"][0]
    assert log_msg["message"] == "stage 1 ok"
    assert log_msg["level"] == 2  # info

    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) > 1


def test_write_run_mcap_preserves_explicit_timestamp_provenance(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    log = tmp_path / "training.log"
    log.write_text("optimizer step complete\n", encoding="utf-8")
    output = tmp_path / "training.mcap"

    summary = write_run_mcap(
        output=output,
        logs=[LogInput(path=log, name="groot")],
        run_id="groot-run",
        metadata={
            "timestamps": "dataset/synthetic-fps",
            "dataset_source_uri": "s3://fixture/data/episode.mp4",
            "is_robot_capture_time": "false",
        },
    )
    info = summarize_mcap(output)

    assert summary.timestamps == "dataset/synthetic-fps"
    assert info.metadata["npa"]["run_id"] == "groot-run"
    assert info.metadata["npa"]["timestamps"] == "dataset/synthetic-fps"
    assert info.metadata["npa"]["is_robot_capture_time"] == "false"
    assert info.metadata["npa"]["dataset_source_uri"].endswith("episode.mp4")


def test_mcap_accepts_long_relative_timelines_and_large_optimizer_steps(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mcap")
    metrics = tmp_path / "loss.json"
    records = [
        {"optimizer_step": step, "loss": 1.0, "_timestamp_ns": 1 + step * 1_000_000_000}
        for step in (1000, 2500, 5000, 10000)
    ]
    metrics.write_text(json.dumps(records), encoding="utf-8")
    output = tmp_path / "long-training.mcap"

    write_run_mcap(
        output=output,
        metrics=[
            MetricsInput(path=metrics, name="train_loss", topic="/metrics/train_loss")
        ],
        start_time_ns=1,
        run_id="long-run",
        metadata={
            "timestamps": "relative dataset-index and optimizer-step clocks",
            "timeline_origin": "relative-zero-plus-1ns",
            "training_loss_clock": "optimizer_step-as-seconds",
            "is_robot_capture_time": "false",
        },
    )
    info = summarize_mcap(output)

    assert info.timestamps_in_int64_domain is True
    assert info.channels_monotonic is True
    assert info.duration_s > 1000
    assert info.end_time_ns == 10_000_000_000_001
    assert info.channel_time_ranges["/metrics/train_loss"] == {
        "start_time_ns": 1_000_000_000_001,
        "end_time_ns": 10_000_000_000_001,
    }


@pytest.mark.parametrize("timestamp", [-1, 2**63, float("inf")])
def test_mcap_rejects_timestamp_values_outside_nonnegative_int64(
    tmp_path: Path, timestamp: int | float
) -> None:
    pytest.importorskip("mcap")
    metrics = tmp_path / "bad-time.json"
    metrics.write_text(
        json.dumps([{"loss": 1.0, "_timestamp_ns": timestamp}]), encoding="utf-8"
    )
    with pytest.raises(McapWriteError, match="timestamp"):
        write_run_mcap(
            output=tmp_path / "bad.mcap",
            metrics=[MetricsInput(path=metrics, name="loss")],
        )


@pytest.mark.parametrize("timestamp", [-1, 2**63, float("inf")])
def test_mcap_rejects_invalid_start_time_with_stable_error(
    tmp_path: Path, timestamp: int | float
) -> None:
    pytest.importorskip("mcap")
    log = tmp_path / "input.log"
    log.write_text("event\n", encoding="utf-8")
    with pytest.raises(McapWriteError, match="timestamp"):
        write_run_mcap(
            output=tmp_path / "bad-start.mcap",
            logs=[LogInput(log)],
            start_time_ns=timestamp,
        )


def test_write_run_mcap_preserves_explicit_producer(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    log = tmp_path / "rollout.log"
    log.write_text("closed-loop rollout complete\n", encoding="utf-8")
    output = tmp_path / "task-performance.mcap"

    write_run_mcap(
        output=output,
        logs=[LogInput(path=log, name="rollout")],
        metadata={"producer": "npa.groot.task-performance"},
    )

    assert summarize_mcap(output).metadata["npa"]["producer"] == (
        "npa.groot.task-performance"
    )


def test_camera_topics_share_synthetic_epoch_and_primary_camera_topic(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mcap")
    from PIL import Image

    root = tmp_path / "images"
    root.mkdir()
    inputs: list[FrameInput] = []
    for camera, count in (("camera", 1), ("front", 3), ("wrist", 2)):
        for index in range(count):
            path = root / f"{camera}-{index}.png"
            Image.new("RGB", (4, 4), (index * 20, 40, 80)).save(path)
            inputs.append(FrameInput(path=path, camera=camera))

    output = tmp_path / "overlap.mcap"
    summary = write_run_mcap(output=output, frames=inputs, fps=2, start_time_ns=10_000)
    info = summarize_mcap(output)

    assert summary.channels == {"/camera": 1, "/camera/front": 3, "/camera/wrist": 2}
    assert info.channel_time_ranges["/camera"]["start_time_ns"] == 10_000
    assert info.channel_time_ranges["/camera/front"]["start_time_ns"] == 10_000
    assert info.channel_time_ranges["/camera/wrist"]["start_time_ns"] == 10_000
    assert info.channel_time_ranges["/camera/front"]["end_time_ns"] == 1_000_010_000
    assert info.channel_time_ranges["/camera/wrist"]["end_time_ns"] == 500_010_000
    assert info.metadata["npa"]["synthetic_timeline"] == (
        "shared-epoch-per-topic-frame-index"
    )
    assert info.metadata["npa"]["primary_image_topic"] == "/camera"


def test_source_frame_timestamps_are_preserved(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    from PIL import Image

    image = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(image)
    output = tmp_path / "source-time.mcap"
    summary = write_run_mcap(
        output=output,
        frames=[FrameInput(path=image, camera="camera", timestamp_ns=123_456_789)],
        fps=30,
    )
    info = summarize_mcap(output)

    assert summary.timestamps == "source"
    assert info.channel_time_ranges["/camera"] == {
        "start_time_ns": 123_456_789,
        "end_time_ns": 123_456_789,
    }
    assert info.metadata["npa"]["timestamps"] == "source"


def test_write_run_mcap_reports_unreadable_inputs(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    root = _run_fixture(tmp_path, frames=1)
    bad_metric = root / "reports" / "broken.json"
    bad_metric.write_text("{not json", encoding="utf-8")
    bad_image = root / "camera" / "front" / "broken.png"
    bad_image.write_bytes(b"not really a png")

    summary = write_run_mcap(
        output=tmp_path / "partial.mcap",
        frames=[FrameInput(path=bad_image, camera="front")]
        + [FrameInput(path=root / "camera" / "front" / "0000.png", camera="front")],
        metrics=[MetricsInput(path=bad_metric, name="broken")],
        logs=[LogInput(path=root / "reports" / "run.log", name="run")],
    )
    # The good frame is written; the bad inputs are reported, never faked.
    assert summary.frames == 1
    assert any("broken.json" in item for item in summary.skipped)
    assert any(
        "broken.png" in item and "unsupported" in item for item in summary.skipped
    )


def test_inspect_rejects_non_mcap(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mcap"
    fake.write_bytes(b"RIFF not an mcap file")
    assert not has_mcap_magic(fake)
    with pytest.raises(McapInspectError, match="magic"):
        summarize_mcap(fake)
    with pytest.raises(McapInspectError, match="not found"):
        summarize_mcap(tmp_path / "missing.mcap")


def test_missing_mcap_dependency_degrades_with_guidance(
    tmp_path: Path, monkeypatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("mcap"):
            raise ModuleNotFoundError("No module named 'mcap'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(McapWriteError, match=r"npa\[foxglove\]"):
        write_run_mcap(
            output=tmp_path / "x.mcap",
            logs=[LogInput(path=tmp_path / "nope.log", name="x")],
        )


def test_image_encoding_is_shared_with_the_lichtblick_writer(tmp_path: Path) -> None:
    """One encoder feeds both viewers — no second image implementation."""
    from npa.workbench.lichtblick import encode_frame_to_compressed_bytes

    from npa.workbench.foxglove import mcap_writer

    source = Path(inspect_module_source := mcap_writer.__file__).read_text(
        encoding="utf-8"
    )
    assert "encode_frame_to_compressed_bytes" in source, inspect_module_source
    assert "compressed_image_message" in source

    from PIL import Image

    frame = tmp_path / "frame.ppm"
    Image.new("RGB", (8, 6), (12, 34, 56)).save(frame)
    payload, fmt = mcap_writer._read_image(frame)
    shared_payload, shared_fmt = encode_frame_to_compressed_bytes(str(frame))
    assert (payload, fmt) == (shared_payload, shared_fmt)


def test_metric_field_named_timestamp_is_preserved(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    metrics = tmp_path / "m.json"
    metrics.write_text('{"timestamp": 12.5, "success_rate": 0.9}', encoding="utf-8")
    output = tmp_path / "metrics.mcap"

    write_run_mcap(output=output, metrics=[MetricsInput(path=metrics, name="m")])

    with output.open("rb") as handle:
        message = next(
            json.loads(msg.data) for _s, _c, msg in make_reader(handle).iter_messages()
        )
    # The time struct owns "timestamp"; the payload field keeps its value.
    assert message["timestamp"] == {
        "sec": message["timestamp"]["sec"],
        "nsec": message["timestamp"]["nsec"],
    }
    assert message["timestamp_value"] == 12.5
    assert message["success_rate"] == 0.9


def test_sdk_metadata_helpers() -> None:
    assert sdk_tarball_url().endswith(f"embed-{FOXGLOVE_EMBED_SDK_VERSION}.tgz")
    ready, reason = sdk_assets_present("/nonexistent/foxglove/sdk")
    assert not ready and reason


def test_metric_series_becomes_a_plottable_curve(tmp_path: Path) -> None:
    """A JSON array of records is a time series, not one opaque blob."""
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    series = tmp_path / "reward_curve.json"
    series.write_text(
        json.dumps([{"episode": i, "reward": i * 0.5} for i in range(5)]),
        encoding="utf-8",
    )
    output = tmp_path / "series.mcap"

    summary = write_run_mcap(
        output=output, metrics=[MetricsInput(path=series, name="reward_curve")], fps=5.0
    )

    assert summary.metrics == 5
    assert summary.channels["/metrics/reward_curve"] == 5
    with output.open("rb") as handle:
        messages = [
            json.loads(m.data) for _s, _c, m in make_reader(handle).iter_messages()
        ]
    assert [m["episode"] for m in messages] == [0, 1, 2, 3, 4]
    assert [m["reward"] for m in messages] == [0.0, 0.5, 1.0, 1.5, 2.0]
    # Each sample carries its own timestamp so the Plot panel can draw a curve.
    stamps = [
        m["timestamp"]["sec"] * 1_000_000_000 + m["timestamp"]["nsec"] for m in messages
    ]
    assert stamps == sorted(stamps) and len(set(stamps)) == 5


def test_real_state_series_emits_synchronized_pointcloud_and_transform(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "schema": "npa.foxglove.pointcloud-series.v1",
                "frame_id": "isaac_observation_state",
                "coordinate_semantics": "state vector grouped into XYZ triples",
                "samples": [
                    {
                        # Must not split this topic from the converter's common
                        # clock unless the producer declares absolute_ns.
                        "timestamp_ns": 123,
                        "points": [[0, 0, 0], [1, 2, 3]],
                        "colors": [[36, 184, 255], [255, 120, 36]],
                    },
                    {
                        "timestamp_ns": 456,
                        "points": [[0.1, 0.2, 0.3], [1.1, 2.1, 3.1]],
                        "colors": [[36, 184, 255], [255, 120, 36]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "trajectory.mcap"
    base_ns = 1_786_363_200_000_000_000

    summary = write_run_mcap(
        output=output,
        metrics=[MetricsInput(path=trajectory, name="trajectory")],
        fps=4,
        start_time_ns=base_ns,
    )

    assert summary.pointclouds == 2
    assert summary.transforms == 1
    assert summary.channels == {"/tf": 1, "/trajectory": 2}
    info = summarize_mcap(output)
    assert info.schemas["/trajectory"] == "foxglove.PointCloud"
    assert info.schemas["/tf"] == "foxglove.FrameTransform"
    assert info.channel_time_ranges["/trajectory"] == {
        "start_time_ns": base_ns,
        "end_time_ns": base_ns + 250_000_000,
    }
    with output.open("rb") as handle:
        messages = [
            (channel.topic, json.loads(message.data))
            for _schema, channel, message in make_reader(handle).iter_messages()
        ]
    cloud = next(payload for topic, payload in messages if topic == "/trajectory")
    assert cloud["frame_id"] == "isaac_observation_state"
    assert cloud["pose"]["orientation"]["w"] == 1
    transform = next(payload for topic, payload in messages if topic == "/tf")
    assert transform["parent_frame_id"] == "world"
    assert transform["child_frame_id"] == "isaac_observation_state"


def test_metric_document_that_is_neither_object_nor_records_is_reported(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mcap")
    bad = tmp_path / "weird.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    good = tmp_path / "ok.json"
    good.write_text('{"score": 1}', encoding="utf-8")

    summary = write_run_mcap(
        output=tmp_path / "x.mcap",
        metrics=[
            MetricsInput(path=bad, name="weird"),
            MetricsInput(path=good, name="ok"),
        ],
    )

    assert summary.metrics == 1
    assert any("weird.json" in item for item in summary.skipped)


def test_action_rollout_emits_meaningful_robot_motion_contract(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    from mcap.reader import make_reader

    root = _run_fixture(tmp_path, frames=3)
    actions = [
        {
            "step": index,
            "sim_step": index * 9,
            "action": [index + joint * 0.5 for joint in range(8)],
            "simulator_ground_truth": {
                "contact": index >= 1,
                "stable_grasp": index >= 2,
                "gripper_closed": index >= 1,
                "placement_stable": False,
                "termination_reason": "running",
            },
        }
        for index in range(3)
    ]
    (root / "reports" / "source-actions-manifest.json").write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.action_rollout.v1",
                "actions": actions,
                "camera_metadata": [
                    {
                        "name": "primary",
                        "pose_frame": "isaac_world",
                        "position": [-2.0, 0.0, 1.0],
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                        "quaternion_order": "wxyz",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "rich.mcap"

    summary = convert_run_directory(
        input_path=root,
        output=output,
        fps=4.0,
        run_id="real-source-diagnostic-visualization",
    )
    info = summarize_mcap(output)

    expected = {
        "/robot/diagnostic_scene": "foxglove.SceneUpdate",
        "/robot/diagnostic_pose": "foxglove.PoseInFrame",
        "/robot/diagnostic_trajectory": "foxglove.PosesInFrame",
        "/robot/diagnostic_joint_states": "foxglove.JointStates",
        "/actuators/commands": "npa.ActuatorCommands",
        "/run/state": "npa.RunState",
    }
    assert {topic: info.schemas[topic] for topic in expected} == expected
    assert summary.scenes == 3
    assert summary.poses == 6
    assert summary.joint_states == summary.actuator_states == summary.run_states == 3
    assert (
        info.metadata["npa"]["visualization_contract"] == "npa.foxglove.robot-motion.v3"
    )
    assert info.metadata["npa"]["scene_update_schema_source"] == (
        SCENE_UPDATE_SCHEMA_SOURCE
    )
    assert info.metadata["npa"]["visualization_fixed_frame"] == "npa_action_space"
    assert "not calibrated" in info.metadata["npa"]["visualization_fidelity"]
    assert info.channels_monotonic is True
    assert all(
        value["start_time_ns"] <= value["end_time_ns"]
        for value in info.channel_time_ranges.values()
    )

    messages: dict[str, list[dict]] = {}
    message_times: dict[str, list[int]] = {}
    wire_schemas: dict[str, dict] = {}
    with output.open("rb") as handle:
        for schema, channel, message in make_reader(handle).iter_messages():
            messages.setdefault(channel.topic, []).append(json.loads(message.data))
            message_times.setdefault(channel.topic, []).append(message.log_time)
            wire_schemas.setdefault(channel.topic, json.loads(schema.data))
    scene = messages["/robot/diagnostic_scene"][0]
    label = scene["entities"][0]["texts"][0]["text"]
    assert "DIAGNOSTIC action-space schematic" in label
    assert "not calibrated robot/world kinematics" in label
    assert len(scene["entities"][0]["lines"][0]["points"]) == 8
    assert [
        len(item["poses"]) for item in messages["/robot/diagnostic_trajectory"]
    ] == [1, 2, 3]
    assert messages["/run/state"][-1]["phase"] == "lift"
    assert messages["/run/state"][-1]["progress"] == 1.0
    assert messages["/actuators/commands"][1]["command_7"] == actions[1]["action"][7]
    joint_message = messages["/robot/diagnostic_joint_states"][1]
    assert "states" not in joint_message
    assert "frame_id" not in joint_message
    assert [joint["name"] for joint in joint_message["joints"]] == [
        f"diagnostic_joint_{index}" for index in range(1, 8)
    ]
    joint_schema = wire_schemas["/robot/diagnostic_joint_states"]
    assert joint_schema["required"] == ["timestamp", "joints"]
    assert set(joint_schema["properties"]) == {"timestamp", "joints"}
    assert joint_schema["properties"]["joints"]["items"]["required"] == ["name"]
    assert wire_schemas["/robot/diagnostic_scene"]["required"] == [
        "deletions",
        "entities",
    ]
    # Validate every emitted SceneUpdate through jsonschema, independently of
    # the MCAP writer/reader.  The explicit item-title table mirrors the current
    # official @foxglove/schemas 2.1.0 contract and catches the former channel
    # schema whose empty primitive arrays had no `items` shape.
    official_array_items = {
        "metadata": "foxglove.KeyValuePair",
        "arrows": "foxglove.ArrowPrimitive",
        "cubes": "foxglove.CubePrimitive",
        "spheres": "foxglove.SpherePrimitive",
        "cylinders": "foxglove.CylinderPrimitive",
        "lines": "foxglove.LinePrimitive",
        "triangles": "foxglove.TriangleListPrimitive",
        "texts": "foxglove.TextPrimitive",
        "models": "foxglove.ModelPrimitive",
    }
    scene_schema = wire_schemas["/robot/diagnostic_scene"]
    jsonschema.Draft7Validator.check_schema(scene_schema)
    validator = jsonschema.Draft7Validator(scene_schema)
    scene_messages = messages["/robot/diagnostic_scene"]
    assert len(scene_messages) == 3
    assert [list(validator.iter_errors(message)) for message in scene_messages] == [
        [],
        [],
        [],
    ]
    assert scene_schema == SCENE_UPDATE_SCHEMA
    assert scene_schema["properties"]["deletions"]["items"]["title"] == (
        "foxglove.SceneEntityDeletion"
    )
    entity_properties = scene_schema["properties"]["entities"]["items"]["properties"]
    assert {
        name: entity_properties[name]["items"]["title"] for name in official_array_items
    } == official_array_items

    def arrays_without_items(node, path="$"):
        if not isinstance(node, dict):
            return []
        missing = [path] if node.get("type") == "array" and "items" not in node else []
        for key, value in node.items():
            missing.extend(arrays_without_items(value, f"{path}.{key}"))
        return missing

    assert arrays_without_items(scene_schema) == []
    malformed = copy.deepcopy(scene_schema)
    del malformed["properties"]["entities"]["items"]["properties"]["models"]["items"]
    assert arrays_without_items(malformed) == [
        "$.properties.entities.items.properties.models"
    ]
    for topic in (
        "/robot/diagnostic_scene",
        "/robot/diagnostic_pose",
        "/robot/diagnostic_trajectory",
        "/robot/diagnostic_joint_states",
        "/actuators/commands",
        "/run/state",
    ):
        assert message_times[topic] == message_times["/camera"]
    assert messages["/tf"][0]["parent_frame_id"] == "isaac_world"


def test_multiple_action_rollouts_share_channels_and_one_monotonic_schedule(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mcap")
    from collections import Counter

    from mcap.reader import make_reader

    rollout_paths: list[Path] = []
    expected_steps: list[int] = []
    for rollout_index, action_count in enumerate((2, 3), start=1):
        path = tmp_path / f"rollout-{rollout_index}" / "action-rollout.json"
        path.parent.mkdir()
        actions = []
        for local_index in range(action_count):
            step = (rollout_index - 1) * 100 + local_index
            expected_steps.append(step)
            actions.append(
                {
                    "step": step,
                    "sim_step": step * 4,
                    "action": [
                        rollout_index * 10 + local_index + joint / 10
                        for joint in range(8)
                    ],
                    "simulator_ground_truth": {
                        "contact": local_index > 0,
                        "stable_grasp": rollout_index == 2 and local_index > 0,
                        "termination_reason": "running",
                    },
                }
            )
        path.write_text(
            json.dumps(
                {
                    "schema": "npa.sim2real.action_rollout.v1",
                    "actions": actions,
                    "camera_metadata": [
                        {
                            "name": f"rollout_{rollout_index}_camera",
                            "pose_frame": "isaac_world",
                            "position": [float(rollout_index), 0.0, 1.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0],
                            "quaternion_order": "wxyz",
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        rollout_paths.append(path)

    metrics = [
        MetricsInput(
            path=path,
            name=path.stem,
            source=path.relative_to(tmp_path).as_posix(),
        )
        for path in rollout_paths
    ]
    outputs = [tmp_path / "multi-a.mcap", tmp_path / "multi-b.mcap"]
    summaries = [
        write_run_mcap(
            output=output,
            metrics=metrics,
            fps=5.0,
            start_time_ns=1_000_000_000,
            run_id="multi-rollout",
        )
        for output in outputs
    ]

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert summaries[0].message_count == summaries[1].message_count == 32
    assert summaries[0].scenes == 5
    assert summaries[0].poses == 10
    assert summaries[0].joint_states == 5
    assert summaries[0].actuator_states == summaries[0].run_states == 5
    assert summaries[0].transforms == 2

    messages: dict[str, list[tuple[int, int, dict]]] = {}
    with outputs[0].open("rb") as handle:
        reader = make_reader(handle)
        mcap_summary = reader.get_summary()
        assert mcap_summary is not None
        channels_per_topic = Counter(
            channel.topic for channel in mcap_summary.channels.values()
        )
        for topic in (
            "/robot/diagnostic_scene",
            "/robot/diagnostic_pose",
            "/robot/diagnostic_trajectory",
            "/robot/diagnostic_joint_states",
            "/actuators/commands",
            "/run/state",
            "/tf",
        ):
            assert channels_per_topic[topic] == 1
        schema_names = Counter(schema.name for schema in mcap_summary.schemas.values())
        assert schema_names["foxglove.SceneUpdate"] == 1
        assert schema_names["foxglove.FrameTransform"] == 1

        handle.seek(0)
        for schema, channel, message in make_reader(handle).iter_messages():
            messages.setdefault(channel.topic, []).append(
                (message.log_time, message.sequence, json.loads(message.data))
            )
            if channel.topic == "/robot/diagnostic_scene":
                assert schema is not None
                wire_schema = json.loads(schema.data)
                assert wire_schema == SCENE_UPDATE_SCHEMA
                jsonschema.validate(json.loads(message.data), wire_schema)

        handle.seek(0)
        metadata = {
            record.name: dict(record.metadata)
            for record in make_reader(handle).iter_metadata()
        }["npa"]

    rich_topics = (
        "/robot/diagnostic_scene",
        "/robot/diagnostic_pose",
        "/robot/diagnostic_trajectory",
        "/robot/diagnostic_joint_states",
        "/actuators/commands",
        "/run/state",
    )
    expected_times = [1_000_000_000 + index * 200_000_000 for index in range(5)]
    for topic in rich_topics:
        times = [timestamp for timestamp, _sequence, _payload in messages[topic]]
        sequences = [sequence for _timestamp, sequence, _payload in messages[topic]]
        assert times == expected_times
        assert all(later > earlier for earlier, later in zip(times, times[1:]))
        assert sequences == list(range(5))
    tf_times = [timestamp for timestamp, _sequence, _payload in messages["/tf"]]
    assert tf_times == [1_000_000_000, 1_400_000_000]
    assert all(later >= earlier for earlier, later in zip(tf_times, tf_times[1:]))
    assert [payload["step"] for _time, _seq, payload in messages["/run/state"]] == (
        expected_steps
    )
    assert metadata["action_rollout_count"] == "2"
    assert json.loads(metadata["action_rollout_sources"]) == [
        path.relative_to(tmp_path).as_posix() for path in rollout_paths
    ]
    assert metadata["action_rollout_schedule"] == (
        "metric-input-order-global-synthetic-fps"
    )
    independently_inspected = summarize_mcap(outputs[0])
    assert independently_inspected.channels_monotonic is True
    assert independently_inspected.message_count == 32
