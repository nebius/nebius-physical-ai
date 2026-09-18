"""Assemble cached picture edits with centered dissolves and exact frame timing."""

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

if __package__:
    from .film_cache import _build_cached, _hash, _render_lock
else:
    from film_cache import _build_cached, _hash, _render_lock


@dataclass(frozen=True)
class PictureShot:
    """Describe an encoded shot, including extra footage for adjacent dissolves.

    Args:
        path: Local encoded video, with head, body, then tail frames.
        frames: Nominal duration in frames, excluding extra handles.
        head_frames: Extra frames before the nominal in point; zero for a cut.
        tail_frames: Extra frames after the nominal out point; zero for a cut.

    Returns:
        An immutable shot description; no media is changed.

    Raises:
        None. Assembly validates the complete timeline before encoding.
    """

    path: Path
    frames: int
    head_frames: int = 0
    tail_frames: int = 0


def _probe(path):
    result = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-of", "json", str(path),
    ], text=True)
    streams = json.loads(result)["streams"]
    if not streams:
        raise ValueError(f"Picture has no video stream: {path}")
    return streams[0]


def _validate_shots(shots, fps, output):
    if type(fps) is not int or fps <= 0 or not shots:
        raise ValueError("Picture needs shots and a positive integer fps")
    if shots[0].head_frames or shots[-1].tail_frames:
        raise ValueError("The first head and last tail must be zero")
    for index, shot in enumerate(shots):
        counts = (shot.frames, shot.head_frames, shot.tail_frames)
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("Shot frame counts must be nonnegative integers")
        if shot.frames <= shot.head_frames + shot.tail_frames:
            raise ValueError("Dissolves must leave at least one clear body frame")
        if index and shots[index - 1].tail_frames != shot.head_frames:
            raise ValueError("Adjacent tail and head handles must match")
        if Path(shot.path).resolve() == output.resolve():
            raise ValueError("Picture output must not overwrite an input shot")


def _verify_video(path, frames, fps, size=None):
    video = _probe(path)
    actual_size = (video["width"], video["height"])
    if int(video.get("nb_frames", -1)) != frames:
        raise ValueError(f"Picture frame count does not match the edit: {path}")
    if Fraction(video["r_frame_rate"]) != fps or Fraction(video["avg_frame_rate"]) != fps:
        raise ValueError(f"Picture must use constant {fps} fps: {path}")
    if abs(float(video.get("duration", 0)) - frames / fps) > 0.001:
        raise ValueError(f"Picture duration does not match the edit: {path}")
    if size is not None and actual_size != size:
        raise ValueError("All picture shots must have the same dimensions")
    if any(dimension % 2 for dimension in actual_size):
        raise ValueError("Picture dimensions must be even for yuv420p")
    return actual_size


def _source_records(shots, fps):
    records, size = [], None
    for shot in shots:
        path = Path(shot.path)
        size = _verify_video(path, shot.frames + shot.head_frames + shot.tail_frames, fps, size)
        records.append({"sha256": _hash(path), "frames": shot.frames,
                        "head_frames": shot.head_frames, "tail_frames": shot.tail_frames})
    return records, size


def _segments(shots):
    for index, shot in enumerate(shots):
        yield [(index, 2 * shot.head_frames)], shot.frames - shot.head_frames - shot.tail_frames
        if shot.tail_frames:
            start = shot.head_frames + shot.frames - shot.tail_frames
            yield [(index, start), (index + 1, 0)], 2 * shot.tail_frames


def _graph(inputs, count, fps):
    graph = []
    for index, (_, start) in enumerate(inputs):
        graph.append(
            f"[{index}:v]trim=start_frame={start}:end_frame={start + count},"
            f"settb=1/{fps},setpts=N,setsar=1,"
            "scale=iw:ih:out_color_matrix=bt709:out_range=tv,format=yuv420p,"
            "setparams=range=limited:color_primaries=bt709:"
            f"color_trc=bt709:colorspace=bt709[v{index}]"
        )
    if len(inputs) == 2:
        progress = f"((N-1)/{count - 1})"
        weight = f"({progress}*{progress}*(3-2*{progress}))"
        graph.append(f"[v0][v1]blend=all_expr='A*(1-{weight})+B*{weight}':shortest=1[out]")
    else:
        graph.append("[v0]null[out]")
    return ";".join(graph)


def _encode_segment(shots, inputs, count, fps, directory):
    command = ["ffmpeg", "-v", "error", "-xerror", "-y", "-filter_complex_threads", "2"]
    for index, _ in inputs:
        command += ["-i", str(shots[index].path)]
    target = directory / "segment.mp4"
    command += ["-filter_complex", _graph(inputs, count, fps), "-map", "[out]", "-an",
                "-frames:v", str(count), "-c:v", "libx264", "-preset", "fast", "-crf", "17",
                "-threads", "4", "-pix_fmt", "yuv420p", "-r", str(fps),
                "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-video_track_timescale", str(fps * 512),
                "-movflags", "+faststart", str(target)]
    subprocess.run(command, check=True)
    _verify_video(target, count, fps)


def _cached_segments(shots, records, fps, cache_dir):
    environment = {"code": _hash(Path(__file__)), "fps": fps,
                   "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}
    parts, reused = [], 0
    for inputs, count in _segments(shots):
        identity = {"environment": environment, "frames": count,
                    "inputs": [{"sha256": records[index]["sha256"], "start_frame": start}
                               for index, start in inputs]}
        directory, hit = _build_cached(
            cache_dir, "picture", identity,
            lambda staging: _encode_segment(shots, inputs, count, fps, staging),
        )
        parts.append(directory / "segment.mp4")
        reused += hit
    return parts, reused


def _concatenate(parts, target, staging):
    names = []
    for index, part in enumerate(parts):
        name = f"segment-{index:04d}.mp4"
        shutil.copyfile(part, staging / name)
        names.append(name)
    listing = staging / "segments.txt"
    listing.write_text("".join(f"file '{name}'\n" for name in names))
    subprocess.run([
        "ffmpeg", "-v", "error", "-xerror", "-y", "-f", "concat", "-safe", "1",
        "-i", str(listing), "-map", "0:v:0", "-an", "-c:v", "copy",
        "-movflags", "+faststart", str(target),
    ], check=True)


def _assemble(shots, output, cache_dir, fps):
    records, size = _source_records(shots, fps)
    parts, reused = _cached_segments(shots, records, fps, cache_dir)
    frames = sum(shot.frames for shot in shots)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".picture-", dir=output.parent) as temporary:
        staging = Path(temporary)
        target = staging / "picture.mp4"
        _concatenate(parts, target, staging)
        _verify_video(target, frames, fps, size)
        if any(_hash(shot.path) != record["sha256"] for shot, record in zip(shots, records, strict=True)):
            raise ValueError("Shot changed during picture assembly; previous delivery retained")
        target.replace(output)
    return {"frames": frames, "fps": fps, "shots": records, "sha256": _hash(output),
            "segments_reused": reused, "segments_rendered": len(parts) - reused,
            "dissolves": sum(bool(shot.tail_frames) for shot in shots)}


def assemble_picture(shots, output, *, cache_dir, fps=30):
    """Join encoded shots without moving picture boundaries or narration timing.

    Args:
        shots: Ordered PictureShot values; matching adjacent handles form a
            dissolve twice the handle length, centered on the nominal cut.
        output: Local silent MP4 destination, atomically replaced after checks.
        cache_dir: Local directory for verified, reusable picture segments.
        fps: Positive integer frame rate shared by all supplied shots.

    Returns:
        Frame counts, source hashes, output hash and segment cache statistics.

    Raises:
        ValueError: Invalid timing, mismatched media, or changing source files.
        OSError: Media or executables cannot be read or written.
        subprocess.CalledProcessError: FFmpeg or ffprobe fails.
    """
    shots, output, cache_dir = tuple(shots), Path(output), Path(cache_dir)
    _validate_shots(shots, fps, output)
    with _render_lock(cache_dir):
        return _assemble(shots, output, cache_dir, fps)
