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


def test_image_encoding_is_shared_with_the_lichtblick_writer(tmp_path: Path) -> None:
    """One encoder feeds both viewers — no second image implementation."""
    from npa.workbench.lichtblick import encode_frame_to_compressed_bytes

    from npa.workbench.foxglove import mcap_writer

    source = Path(inspect_module_source := mcap_writer.__file__).read_text(encoding="utf-8")
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
    assert message["timestamp"] == {"sec": message["timestamp"]["sec"], "nsec": message["timestamp"]["nsec"]}
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
        json.dumps([{"episode": i, "reward": i * 0.5} for i in range(5)]), encoding="utf-8"
    )
    output = tmp_path / "series.mcap"

    summary = write_run_mcap(
        output=output, metrics=[MetricsInput(path=series, name="reward_curve")], fps=5.0
    )

    assert summary.metrics == 5
    assert summary.channels["/metrics/reward_curve"] == 5
    with output.open("rb") as handle:
        messages = [json.loads(m.data) for _s, _c, m in make_reader(handle).iter_messages()]
    assert [m["episode"] for m in messages] == [0, 1, 2, 3, 4]
    assert [m["reward"] for m in messages] == [0.0, 0.5, 1.0, 1.5, 2.0]
    # Each sample carries its own timestamp so the Plot panel can draw a curve.
    stamps = [m["timestamp"]["sec"] * 1_000_000_000 + m["timestamp"]["nsec"] for m in messages]
    assert stamps == sorted(stamps) and len(set(stamps)) == 5


def test_metric_document_that_is_neither_object_nor_records_is_reported(tmp_path: Path) -> None:
    pytest.importorskip("mcap")
    bad = tmp_path / "weird.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    good = tmp_path / "ok.json"
    good.write_text('{"score": 1}', encoding="utf-8")

    summary = write_run_mcap(
        output=tmp_path / "x.mcap",
        metrics=[MetricsInput(path=bad, name="weird"), MetricsInput(path=good, name="ok")],
    )

    assert summary.metrics == 1
    assert any("weird.json" in item for item in summary.skipped)
