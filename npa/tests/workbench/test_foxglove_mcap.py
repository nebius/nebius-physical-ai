"""MCAP conversion tests — real files in, real MCAP out, read back to prove it.

The `mcap` dependency is optional (`npa[foxglove]`), so the round-trip tests skip
when it is absent; the graceful-degradation path is asserted unconditionally.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

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
        json.dumps({"success_rate": 0.82, "episodes": 12, "gate": "promote", "extra": {"a": 1}}),
        encoding="utf-8",
    )
    (root / "reports" / "run.log").write_text("stage 1 ok\nstage 2 ok\n\nstage 3 warn\n", encoding="utf-8")
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
    assert summary.channels["/camera/front"] == 3

    # Read it back the way Foxglove would.
    info = summarize_mcap(output)
    assert info.message_count == 10
    assert info.schemas["/camera/front"] == "foxglove.CompressedImage"
    assert info.schemas["/camera/wrist"] == "foxglove.CompressedImage"
    assert info.schemas["/log"] == "foxglove.Log"
    assert info.schemas["/metrics/metrics"].startswith("npa.RunMetrics")
    assert info.metadata["npa"]["run_id"] == "run-42"
    # Timestamp provenance must be recorded, not implied.
    assert info.metadata["npa"]["timestamps"] == "synthetic-fps"
    assert info.duration_s > 0
    assert "foxglove.CompressedImage" in format_mcap_info(info)

    # Message payloads must be real, decodable images / values.
    with output.open("rb") as handle:
        reader = make_reader(handle)
        seen: dict[str, list[dict]] = {}
        timestamps: list[int] = []
        for _schema, channel, message in reader.iter_messages():
            seen.setdefault(channel.topic, []).append(json.loads(message.data))
            timestamps.append(message.log_time)

    front = seen["/camera/front"][0]
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
    assert any("broken.png" in item and "unsupported" in item for item in summary.skipped)


def test_inspect_rejects_non_mcap(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mcap"
    fake.write_bytes(b"RIFF not an mcap file")
    assert not has_mcap_magic(fake)
    with pytest.raises(McapInspectError, match="magic"):
        summarize_mcap(fake)
    with pytest.raises(McapInspectError, match="not found"):
        summarize_mcap(tmp_path / "missing.mcap")


def test_missing_mcap_dependency_degrades_with_guidance(tmp_path: Path, monkeypatch) -> None:
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


def test_sdk_metadata_helpers() -> None:
    assert sdk_tarball_url().endswith(f"embed-{FOXGLOVE_EMBED_SDK_VERSION}.tgz")
    ready, reason = sdk_assets_present("/nonexistent/foxglove/sdk")
    assert not ready and reason
