"""Independently decode real LeRobot video, frame labels, and their Rerun proof."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rerun.recording import load_recording

REPOSITORY = Path(__file__).parents[3]
EVIDENCE = REPOSITORY / "docs/workbench/evidence/lerobot-video-subtasks"
FRAME_FILE = "data/chunk-000/file-000.parquet"
PHASES = [("ready", 0, 80), ("approach", 80, 160), ("grasp", 160, 190),
          ("transfer", 190, 245), ("place", 245, 279), ("release", 279, 300),
          ("retract", 300, 360), ("complete", 360, 454)]


@pytest.fixture(scope="module")
def manifest():
    return json.loads((EVIDENCE / "manifest.json").read_text())


@pytest.fixture(scope="module")
def recorder():
    script = REPOSITORY / "npa/scripts/record_lerobot_video_subtasks.py"
    spec = importlib.util.spec_from_file_location("record_lerobot_video_subtasks", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_artifacts_and_recipe_bytes_match_manifest(manifest) -> None:
    for relative, digest in manifest["artifacts"].items():
        assert hashlib.sha256((EVIDENCE / relative).read_bytes()).hexdigest() == digest
    for relative, digest in manifest["provenance"]["recipe_sha256"].items():
        recipe = REPOSITORY / relative
        if relative == "npa/src/npa/fiftyone_lerobot_subtasks.py":
            recipe = EVIDENCE.parent / "lerobot-subtask-recipes" / f"{digest}.py"
        assert hashlib.sha256(recipe.read_bytes()).hexdigest() == digest
    assert manifest["input_unchanged"] is True
    assert "not upstream ground truth" in manifest["provenance"]["limitations"]
    assert str(REPOSITORY) not in json.dumps(manifest)


def test_real_episode_preserves_all_input_columns_and_labels_every_frame() -> None:
    original = pq.read_table(EVIDENCE / "input" / FRAME_FILE)
    reviewed = pq.read_table(EVIDENCE / "reviewed" / FRAME_FILE)
    assert original.equals(reviewed.select(original.column_names), check_metadata=True)
    assert original.num_rows == 454
    assert "subtask_index" not in original.column_names
    assert reviewed.column("subtask_index").type == pa.int64()
    labels = {row["subtask_index"]: row["subtask"] for row in
              pq.read_table(EVIDENCE / "reviewed/meta/subtasks.parquet").to_pylist()}
    observed = [labels[index] for index in reviewed.column("subtask_index").to_pylist()]
    assert observed == [label for label, start, end in PHASES for _ in range(start, end)]
    proof = json.loads((EVIDENCE / "subtask-proof.json").read_text())
    assert proof["summary"]["labeled_frame_count"] == 454
    assert proof["proof"]["subtask"] == "grasp"
    assert proof["proof"]["frame_index"] == 160
    assert proof["proof"]["source_parquet_sha256"] == hashlib.sha256((EVIDENCE / "reviewed" / FRAME_FILE).read_bytes()).hexdigest()
    assert json.loads((EVIDENCE / "workflow-report.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("camera", ["top", "wrist"])
def test_source_camera_clips_decode_exactly_one_episode(camera) -> None:
    path = EVIDENCE / f"input/videos/observation.images.{camera}/chunk-000/file-000.mp4"
    digest = hashlib.sha256()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.codec.id == av.Codec("av1", "r").id
        assert (stream.width, stream.height) == (640, 480)
        times = []
        for frame in container.decode(video=0):
            times.append(frame.time)
            assert frame.format.name == "yuv420p"
            # RGB conversion uses platform-dependent SIMD rounding. Hash decoded
            # native planes, excluding allocator-dependent row padding, instead.
            for plane in frame.planes:
                data = bytes(plane)
                for row in range(plane.height):
                    digest.update(data[row * plane.line_size:row * plane.line_size + plane.width])
    assert times == pytest.approx([index / 30 for index in range(454)], abs=1e-6)
    receipt = json.loads((EVIDENCE / "source-camera-planes.json").read_text())
    provenance = json.loads((EVIDENCE / "manifest.json").read_text())["provenance"]
    assert receipt["source_revision"] == provenance["source_revision"]
    assert len(times) == receipt["cameras"][camera]["frame_count"]
    assert digest.hexdigest() == receipt["cameras"][camera]["decoded_yuv420p_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["cameras"][camera]["clip_sha256"]
    reviewed = EVIDENCE / f"reviewed/videos/observation.images.{camera}/chunk-000/file-000.mp4"
    assert path.read_bytes() == reviewed.read_bytes()


def test_labeled_mp4_decodes_all_frames_and_shows_eight_distinct_label_colors() -> None:
    centers = {(start + end) // 2 for _label, start, end in PHASES}
    colors, camera_images = [], {"top": set(), "wrist": set()}
    with av.open(str(EVIDENCE / "lerobot-labeled.mp4")) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (1280, 720)
        times = []
        for index, frame in enumerate(container.decode(video=0)):
            times.append(frame.time)
            if index not in centers:
                continue
            pixels = frame.to_ndarray(format="rgb24")
            colors.append(tuple(int(value) for value in pixels[25, 1230]))
            for camera, left in (("top", 0), ("wrist", 640)):
                camera_images[camera].add(hashlib.sha256(pixels[130:560, left:left + 640].tobytes()).hexdigest())
    assert times == pytest.approx([index / 30 for index in range(454)], abs=1e-6)
    assert len(set(colors)) == 8
    assert all(len(images) == 8 for images in camera_images.values())
    expected = [(127, 155, 185), (80, 188, 232), (255, 204, 102), (186, 161, 255),
                (255, 159, 118), (247, 141, 197), (103, 211, 194), (141, 214, 138)]
    assert all(max(abs(left - right) for left, right in zip(color, target)) <= 5
               for color, target in zip(colors, expected, strict=True))


def test_rrd_embeds_exact_mp4_and_every_labeled_source_row(manifest) -> None:
    recording = load_recording(EVIDENCE / "lerobot-video-subtasks.rrd")
    assert recording.application_id() == "npa-lerobot-video-subtasks"
    assert recording.recording_id() == manifest["provenance"]["run_id"]
    rows, video_times, scalars, blobs = [], [], [], []
    for chunk in recording.chunks():
        batch = chunk.to_record_batch()
        if "AssetVideo:blob" in batch.schema.names:
            blobs.extend(bytes(value[0]) for value in batch.column("AssetVideo:blob").to_pylist())
        if "episode_time" not in batch.schema.names:
            continue
        times = batch.column("episode_time").cast(pa.int64()).to_pylist()
        if chunk.entity_path == "/labeled/frame":
            rows.extend(zip(times, [json.loads(value[0]) for value in batch.column("TextDocument:text").to_pylist()]))
        if chunk.entity_path == "/video":
            video_times.extend(zip(times, [value[0] for value in batch.column("VideoFrameReference:timestamp").to_pylist()]))
        if chunk.entity_path == "/subtasks/index":
            scalars.extend(zip(times, [value[0] for value in batch.column("Scalars:scalars").to_pylist()]))
    expected = json.loads((EVIDENCE / "labeled-frames.json").read_text())
    expected_times = [round(row["timestamp"] * 1e9) for row in expected]
    assert sorted(rows) == list(zip(expected_times, expected))
    assert sorted(scalars) == [(time, row["subtask_index"]) for time, row in zip(expected_times, expected)]
    assert [time for time, _reference in sorted(video_times)] == expected_times
    assert [reference for _time, reference in sorted(video_times)] == pytest.approx(
        [index / 30 * 1e9 for index in range(454)], abs=1)
    assert blobs == [(EVIDENCE / "lerobot-labeled.mp4").read_bytes()]


def test_producer_rejects_existing_output_without_touching_it(tmp_path, recorder) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        recorder.record_video_proof(tmp_path / "missing-source", output, "test")
    assert sentinel.read_text() == "keep"


def test_producer_rejects_wrong_source_hash(tmp_path, recorder, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "wrong.txt").write_text("wrong data")
    monkeypatch.setattr(recorder, "SOURCE_HASHES", {"wrong.txt": "0" * 64})
    with pytest.raises(ValueError, match="Pinned source hash mismatch"):
        recorder.record_video_proof(source, tmp_path / "new-proof", "test")
    assert not (tmp_path / "new-proof").exists()


def test_agent_ui_recording_decodes_and_displays_all_eight_phases(recorder) -> None:
    receipt = json.loads((EVIDENCE / "agent-ui-capture.json").read_text())
    video = receipt["video"]
    path = EVIDENCE / video["path"]
    assert path.stat().st_size == video["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == video["sha256"]
    source = receipt["source"]
    assert hashlib.sha256((EVIDENCE / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    phases = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == video["codec"]
        assert (stream.width, stream.height) == (1680, 1120)
        assert float(stream.average_rate) == video["fps"]
        assert float(stream.duration * stream.time_base) == pytest.approx(video["duration_seconds"])
        times = []
        for index, frame in enumerate(container.decode(video=0)):
            times.append(frame.time)
            label = _captured_phase(frame, recorder.PHASES)
            if not phases or phases[-1]["label"] != label:
                phases.append({"label": label, "capture_frame": index})
    assert len(times) == video["decoded_frame_count"]
    assert all(left < right for left, right in zip(times, times[1:]))
    assert phases == video["observed_phases"]
    assert [phase["label"] for phase in phases] == [label for label, _start, _end in PHASES]


def _captured_phase(frame, palette) -> str:
    from PIL import ImageColor

    pixel = frame.to_ndarray(format="rgb24")[140, 1610].astype(int)
    candidates = []
    for label, _start, _end, color in palette:
        rgb = ImageColor.getrgb(color)
        difference = max(abs(int(pixel[channel]) - rgb[channel]) for channel in range(3))
        candidates.append((difference, label))
    difference, label = min(candidates)
    # The browser colorspace conversion and two lossy encodes shift palette values.
    assert difference <= 25
    return label
