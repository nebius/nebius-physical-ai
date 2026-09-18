"""Record pinned real LeRobot footage with frame-aligned demonstration subtask labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import av
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image, ImageDraw, ImageFont

from npa.fiftyone_lerobot_subtasks import apply_subtask_segments, segments_from_temporal_tags

REPOSITORY = Path(__file__).resolve().parents[2]
DATASET = "lerobot/svla_so100_pickplace"
REVISION = "728583b5eaf9e739a7f119e2def466fa1d552402"
FRAME_FILE = "data/chunk-000/file-000.parquet"
EPISODE_FILE = "meta/episodes/chunk-000/file-000.parquet"
CAMERAS = ("top", "wrist")
FRAME_COUNT = 454
FPS = 30
TIMELINE = "episode_time"
APPLICATION = "npa-lerobot-video-subtasks"
PHASES = (
    ("ready", 0, 80, "#7F9BB9"),
    ("approach", 80, 160, "#50BCE8"),
    ("grasp", 160, 190, "#FFCC66"),
    ("transfer", 190, 245, "#BAA1FF"),
    ("place", 245, 279, "#FF9F76"),
    ("release", 279, 300, "#F78DC5"),
    ("retract", 300, 360, "#67D3C2"),
    ("complete", 360, 454, "#8DD68A"),
)
SOURCE_HASHES = {
    FRAME_FILE: "e74b2786f24bc5bb8d9e6b82f5f16cc4ae921fcaf42cabdcb9bf939ca78767e1",
    EPISODE_FILE: "c5714dd5018ec013fac6610bd6bc7e97fc9ec79623b898ab8c4adddc3da61260",
    "meta/info.json": "aec8d0c7bd3d2dc693c160bd58f9fb6aabcbb24d4b28e2f09950a3aa6b246eef",
    "meta/tasks.parquet": "8703df66cb1d03b2e808ee069cbf816447a91ef59acad8f49e2834a2f4587f89",
    "videos/observation.images.top/chunk-000/file-000.mp4":
        "7c7f2674c7af4ac9fde4c0301ad25325ef8c655f0bd455b4b3b8a9ece1965d73",
    "videos/observation.images.wrist/chunk-000/file-000.mp4":
        "f605a80ca6d299231aebec49c945768a18a1207f30bd9dce4add0ab535478e7c",
}
LIMITATIONS = (
    "Real SO-100 camera footage; labels are assistant-authored demonstration annotations "
    "visually checked against both cameras, not upstream ground truth or expert-certified labels. "
    "Programmatic temporal tags exercise the production FiftyOne tag parser and LeRobot writer; "
    "no interactive FiftyOne labeling session, policy training, or NVIDIA inference is claimed."
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inventory(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _digest(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _verify_source(source: Path) -> None:
    for relative, expected in SOURCE_HASHES.items():
        if _digest(source / relative) != expected:
            raise ValueError(f"Pinned source hash mismatch: {relative}")


def _video_path(root: Path, camera: str) -> Path:
    return root / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"


def _clip_camera(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-i", str(source),
        "-map", "0:v:0", "-frames:v", str(FRAME_COUNT), "-an", "-c:v", "copy",
        "-movflags", "+faststart", str(destination),
    ], check=True)
    with av.open(str(destination)) as video:
        timestamps = [frame.time for frame in video.decode(video=0)]
        if len(timestamps) != FRAME_COUNT or any(abs(timestamp - index / FPS) > 1e-5
                                                for index, timestamp in enumerate(timestamps)):
            raise ValueError("Camera clip is not exactly episode 0 at 30 fps")


def _materialize_episode(source: Path, root: Path) -> None:
    frames = pq.read_table(source / FRAME_FILE).slice(0, FRAME_COUNT)
    if set(frames.column("episode_index").to_pylist()) != {0}:
        raise ValueError("Source rows are not episode 0")
    for relative, table in ((FRAME_FILE, frames), (EPISODE_FILE, pq.read_table(source / EPISODE_FILE).slice(0, 1))):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, root / relative)
    info = json.loads((source / "meta/info.json").read_text())
    info.update(total_episodes=1, total_frames=FRAME_COUNT, splits={"train": "0:1"})
    _write_json(root / "meta/info.json", info)
    shutil.copyfile(source / "meta/tasks.parquet", root / "meta/tasks.parquet")
    episode = pq.read_table(root / EPISODE_FILE).to_pylist()[0]
    statistics: dict[str, dict] = {}
    for key, value in episode.items():
        if key.startswith("stats/"):
            _, feature, statistic = key.split("/")
            statistics.setdefault(feature, {})[statistic] = value
    _write_json(root / "meta/stats.json", statistics)
    for camera in CAMERAS:
        _clip_camera(_video_path(source, camera), _video_path(root, camera))


def _label_episode(root: Path) -> list[dict]:
    original = pq.read_table(root / "input" / FRAME_FILE)
    timestamps = original.column("timestamp").to_pylist()
    boundaries = [round(float(value) * 1e9) for value in timestamps] + [round(FRAME_COUNT / FPS * 1e9)]
    tags = [{"sample_id": "episode-0", "tag": f"subtask:{label}", "index_type": 2,
             "start": boundaries[start], "end": boundaries[end]} for label, start, end, _color in PHASES]
    _write_json(root / "temporal-tags.json", tags)
    shutil.copytree(root / "input", root / "reviewed")
    report = apply_subtask_segments(root / "reviewed", segments_from_temporal_tags({"episode-0": 0}, tags))
    _write_json(root / "export-report.json", report)
    reviewed = pq.read_table(root / "reviewed" / FRAME_FILE)
    if not original.equals(reviewed.select(original.column_names), check_metadata=True):
        raise ValueError("Labeling changed an original frame column")
    catalog = pq.read_table(root / "reviewed/meta/subtasks.parquet").to_pylist()
    labels = {row["subtask_index"]: row["subtask"] for row in catalog}
    rows = [{**row, "resolved_subtask": labels[row["subtask_index"]]} for row in reviewed.to_pylist()]
    for label, start, end, _color in PHASES:
        if any(row["resolved_subtask"] != label for row in rows[start:end]):
            raise ValueError("Temporal tag boundaries do not match the intended frames")
    _write_json(root / "labeled-frames.json", rows)
    return rows


def _run_yaml(root: Path, run_id: str) -> None:
    workflow = REPOSITORY / "workflows/testing/lerobot-subtask-proof.yaml"
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY / "npa/src"))
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    result = subprocess.run([
        sys.executable, "-m", "npa", "workbench", "workflow", "run-spec", str(workflow),
        "--run-id", run_id, "--execute", "--json", "--var", "reviewed_dataset_uri=reviewed",
        "--var", "proof_uri=subtask-proof.json",
    ], cwd=root, env=environment, text=True, capture_output=True, check=True)
    report = json.loads(result.stdout.replace(str(REPOSITORY), "<checkout>"))
    if report["status"] != "completed" or any(step["returncode"] != 0 for step in report["steps"]):
        raise ValueError("The subtask verification YAML failed")
    _write_json(root / "workflow-report.json", report)
    shutil.copyfile(workflow, root / "workflow.yaml")


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    candidates = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                  Path("/System/Library/Fonts/Supplemental/Arial.ttf")]
    font = next((path for path in candidates if path.exists()), None)
    if font is None:
        raise FileNotFoundError("Install DejaVu Sans or Arial to render the proof video")
    return {name: ImageFont.truetype(str(font), size) for name, size in
            (("title", 30), ("label", 28), ("body", 19), ("small", 16))}


def _draw_header(draw: ImageDraw.ImageDraw, row: dict, fonts: dict, phase: tuple) -> None:
    label, _start, _end, color = phase
    draw.text((24, 12), "LeRobot  /  Subtask labeling", font=fonts["title"], fill="#EFF6FF")
    draw.text((24, 51), "SO-100  |  Episode 0  |  Pick up the cube and place it in the box",
              font=fonts["body"], fill="#AEC2D9")
    draw.rounded_rectangle((899, 10, 1256, 73), radius=12, fill=color)
    draw.text((918, 16), f"{label.upper()}", font=fonts["label"], fill="#101B2B")
    draw.text((919, 49), f"Frame {row['frame_index']:03d}/453   |   {row['timestamp']:.2f} s",
              font=fonts["small"], fill="#101B2B")
    for offset, title in ((0, "TOP CAMERA"), (640, "WRIST CAMERA")):
        draw.rectangle((offset + 12, 94, offset + 172, 122), fill="#101B2B")
        draw.text((offset + 21, 99), title, font=fonts["small"], fill="white")


def _draw_timeline(draw: ImageDraw.ImageDraw, row: dict, fonts: dict) -> None:
    for index, (label, start, end, color) in enumerate(PHASES):
        left = 24 + index * 154
        active = label == row["resolved_subtask"]
        draw.rounded_rectangle((left, 580, left + 149, 625), radius=6,
                               fill=color if active else "#243248", outline=color, width=2)
        draw.text((left + 8, 593), f"{index + 1} {label.upper()}", font=fonts["small"],
                  fill="#101B2B" if active else "#E2EAF5")
        draw.rectangle((24 + start / FRAME_COUNT * 1232, 641, 24 + end / FRAME_COUNT * 1232, 651), fill=color)
    cursor = 24 + row["frame_index"] / (FRAME_COUNT - 1) * 1232
    draw.polygon([(cursor, 637), (cursor - 6, 630), (cursor + 6, 630)], fill="white")
    draw.line((cursor, 637, cursor, 655), fill="white", width=2)
    draw.text((24, 666), "REAL ROBOT VIDEO  |  454/454 frames labeled  |  Original state + action preserved",
              font=fonts["body"], fill="#E2EAF5")
    draw.text((24, 696), "Source: lerobot/svla_so100_pickplace (Apache-2.0)  |  Demo annotations, not upstream ground truth",
              font=fonts["small"], fill="#AEC2D9")


def _compose_frame(images: list[Image.Image], row: dict, fonts: dict) -> Image.Image:
    canvas = Image.new("RGB", (1280, 720), "#101B2B")
    for index, image in enumerate(images):
        canvas.paste(image, (index * 640, 88))
    draw = ImageDraw.Draw(canvas)
    phase = next(phase for phase in PHASES if phase[0] == row["resolved_subtask"])
    _draw_header(draw, row, fonts, phase)
    _draw_timeline(draw, row, fonts)
    return canvas


def _render_video(root: Path, rows: list[dict]) -> None:
    fonts = _fonts()
    with ExitStack() as stack:
        cameras = [stack.enter_context(av.open(str(_video_path(root / "input", camera)))) for camera in CAMERAS]
        decoders = [camera.decode(video=0) for camera in cameras]
        output = stack.enter_context(av.open(str(root / "lerobot-labeled.mp4"), "w", options={"movflags": "+faststart"}))
        stream = output.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for row in rows:
            frames = [next(decoder) for decoder in decoders]
            if any(abs(frame.time - row["timestamp"]) > 1e-5 for frame in frames):
                raise ValueError("Camera timestamp does not match the LeRobot frame")
            canvas = _compose_frame([frame.to_image() for frame in frames], row, fonts)
            frame = av.VideoFrame.from_image(canvas)
            frame.pts = row["frame_index"]
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Spatial2DView(origin="video", name="Real LeRobot episode | synchronized subtask labels"),
        rrb.TimePanel(timeline=TIMELINE, state="collapsed", play_state="paused", playback_speed=1.0),
        rrb.BlueprintPanel(state="collapsed"), rrb.SelectionPanel(state="collapsed"), auto_layout=False,
    )


def _record_rrd(root: Path, rows: list[dict], provenance: dict) -> None:
    recording = rr.RecordingStream(APPLICATION, recording_id=provenance["run_id"])
    try:
        recording.save(root / "lerobot-video-subtasks.rrd", default_blueprint=_blueprint())
        recording.log("provenance", rr.TextDocument(json.dumps(provenance, indent=2)), static=True)
        asset = rr.AssetVideo(path=root / "lerobot-labeled.mp4")
        recording.log("video", asset, static=True)
        for row, timestamp in zip(rows, asset.read_frame_timestamps_nanos(), strict=True):
            recording.set_time(TIMELINE, duration=float(row["timestamp"]))
            recording.log("video", rr.VideoFrameReference(nanoseconds=int(timestamp)))
            recording.log("labeled/frame", rr.TextDocument(json.dumps(row)))
            recording.log("subtasks/index", rr.Scalars(row["subtask_index"]))
        recording.flush()
    finally:
        recording.disconnect()
    subprocess.run([str(Path(sys.executable).with_name("rerun")), "rrd", "verify",
                    str(root / "lerobot-video-subtasks.rrd")], check=True)


def _provenance(run_id: str) -> dict:
    recipes = [Path(__file__), REPOSITORY / "workflows/testing/lerobot-subtask-proof.yaml",
               REPOSITORY / "npa/src/npa/fiftyone_lerobot_subtasks.py",
               REPOSITORY / "npa/src/npa/workflows/lerobot_subtask_proof.py"]
    return {
        "run_id": run_id, "source_dataset": DATASET, "source_revision": REVISION,
        "source_url": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
        "source_sha256": SOURCE_HASHES, "source_license": "Apache-2.0", "episode_index": 0,
        "frame_count": FRAME_COUNT, "fps": FPS, "duration_seconds": FRAME_COUNT / FPS,
        "phases": [{"label": label, "start_frame": start, "end_frame_exclusive": end}
                   for label, start, end, _color in PHASES],
        "modifications": "Episode 0 subset; unchanged AV1 camera packets remuxed into clips; "
                         "new subtask labels; H.264 two-camera composition with annotation overlays.",
        "limitations": LIMITATIONS, "recipe_sha256": {
            path.relative_to(REPOSITORY).as_posix(): _digest(path) for path in recipes},
        "versions": {"rerun": rr.__version__, "av": av.__version__},
    }


def record_video_proof(source: Path, output: Path, run_id: str) -> dict:
    """Build a self-contained, labeled episode and video proof from pinned public files.

    Args:
        source: Local snapshot of the pinned Hugging Face dataset.
        output: New directory for the episode, MP4, RRD, and verification reports.
        run_id: Recording and local workflow identifier.

    Returns:
        Hash-bound manifest describing actual source files and produced artifacts.

    Raises:
        FileExistsError: If the output directory exists.
        ValueError: If source hashes, frame alignment, or label coverage is invalid.
        subprocess.CalledProcessError: If clipping, the YAML, or RRD verification fails.
    """
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    _verify_source(source)
    output.mkdir(parents=True)
    _materialize_episode(source, output / "input")
    original = _inventory(output / "input")
    rows = _label_episode(output)
    _run_yaml(output, run_id)
    _render_video(output, rows)
    provenance = _provenance(run_id)
    _record_rrd(output, rows, provenance)
    if original != _inventory(output / "input"):
        raise ValueError("Original episode files were modified")
    manifest = {"schema": "npa.lerobot.video_subtask_proof.v1", "provenance": provenance,
                "input_unchanged": True, "original_frame_columns_unchanged": True,
                "artifacts": _inventory(output)}
    _write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    result = record_video_proof(arguments.source_dir, arguments.output_dir, arguments.run_id)
    print(json.dumps({"run_id": arguments.run_id, "frames": FRAME_COUNT, "labels": len(PHASES),
                      "input_unchanged": result["input_unchanged"]}))
