"""Pair each rendered narration cue with final frames and an explicit semantic review."""

import argparse
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from edit import _project
from film_cache import _fingerprint, _hash
from film_player import _caption_cues
from film_review_judge import _assess, _judge, _pending_assessment
from film_review_page import _write_review_page


def _video_info(path):
    data = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    video = next(
        (stream for stream in data["streams"] if stream["codec_type"] == "video"), None
    )
    if video is None or not any(
        stream["codec_type"] == "audio" for stream in data["streams"]
    ):
        raise ValueError("Narration review needs a final video with an audio stream")
    duration = float(data["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Review video needs a finite positive duration")
    return duration


def _shots(path, duration):
    if path is None:
        return []
    shots = json.loads(path.read_text())["shots"]
    if not isinstance(shots, list) or not shots:
        raise ValueError("Review shot list must contain shots")
    previous = 0
    result = []
    for shot in shots:
        start, end = shot["timeline_start"], shot["timeline_end"]
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in (start, end)
        ):
            raise ValueError("Shot boundaries must be finite seconds")
        if (
            not math.isclose(start, previous, abs_tol=0.001)
            or not start < end <= duration
        ):
            raise ValueError(
                "Review shots must cover the final timeline without gaps or overlaps"
            )
        result.append(
            {
                "id": shot["id"],
                "start": start,
                "end": end,
                "role": shot.get("role", ""),
                "label": shot.get("label", ""),
                "evidence_type": shot.get("evidence_type", "unspecified"),
                "limitations": shot.get("review_limitations", ""),
            }
        )
        previous = end
    if not math.isclose(previous, duration, abs_tol=0.05):
        raise ValueError("Review shots must cover the complete final video")
    return result


def _sample_times(cue, shots):
    start, end = cue["start"], cue["end"]
    times = {start + (end - start) * fraction for fraction in (0.15, 0.5, 0.85)}
    for shot in shots:
        left, right = max(start, shot["start"]), min(end, shot["end"])
        if left < right:
            times.add((left + right) / 2)
    return sorted(round(value, 6) for value in times)


def _extract_cue(video, cue, shots, directory, index):
    identifier = f"cue-{index + 1:03d}"
    samples = []
    for number, seconds in enumerate(_sample_times(cue, shots)):
        name = f"{identifier}-{number + 1:02d}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-y",
                "-ss",
                str(seconds),
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=960:-2,format=yuvj420p",
                str(directory / name),
            ],
            check=True,
        )
        samples.append(
            {"file": name, "seconds": seconds, "sha256": _hash(directory / name)}
        )
    overlaps = [
        {
            **shot,
            "overlap_seconds": round(
                min(cue["end"], shot["end"]) - max(cue["start"], shot["start"]), 6
            ),
        }
        for shot in shots
        if shot["end"] > cue["start"] and shot["start"] < cue["end"]
    ]
    return {
        "id": identifier,
        **cue,
        "text": html.unescape(cue["text"]),
        "shots": overlaps,
        "frames": samples,
    }


def _packet(video, captions, shot_list, directory):
    duration = _video_info(video)
    inputs = _input_hashes(video, captions, shot_list)
    shots = _shots(shot_list, duration)
    cues = _caption_cues(captions.read_text().replace("\r\n", "\n"), duration)
    if (
        not cues
        or any(not cue["text"].strip() for cue in cues)
        or any(right["start"] < left["end"] for left, right in zip(cues, cues[1:]))
    ):
        raise ValueError(
            "Review needs nonempty, ordered, nonoverlapping narration cues"
        )
    records = [
        _extract_cue(video, cue, shots, directory, index)
        for index, cue in enumerate(cues)
    ]
    shutil.copyfile(video, directory / "film.mp4")
    shutil.copyfile(captions, directory / "film.srt")
    current = _input_hashes(video, captions, shot_list)
    if (
        current != inputs
        or _hash(directory / "film.mp4") != inputs["video_sha256"]
        or _hash(directory / "film.srt") != inputs["captions_sha256"]
    ):
        raise ValueError("Film inputs changed during review; regenerate the packet")
    packet = {
        "schema_version": 1,
        **inputs,
        "duration_seconds": duration,
        "cues": records,
        "scope": "Sampled final frames and supplied transcript; no automated audio transcription, "
        "continuous audiovisual viewing, or independent validation of backend execution.",
    }
    return {**packet, "packet_sha256": _fingerprint(packet)}


def _input_hashes(video, captions, shot_list):
    return {
        "video_sha256": _hash(video),
        "captions_sha256": _hash(captions),
        "shot_list_sha256": _hash(shot_list) if shot_list else None,
    }


def _prepare(video, captions, shot_list, output):
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".review-", dir=output) as temporary:
        staging = Path(temporary)
        packet = _packet(video, captions, shot_list, staging)
        destination = output / packet["packet_sha256"]
        if destination.exists():
            for path in staging.iterdir():
                path.replace(destination / path.name)
        else:
            staging.rename(destination)
        (destination / "packet.json").write_text(json.dumps(packet, indent=2) + "\n")
    return packet, destination


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--video",
        type=Path,
        help="Final MP4; defaults to the project's renders/final/film.mp4.",
    )
    parser.add_argument(
        "--captions",
        type=Path,
        help="Final SRT; defaults to the selected video's .srt sibling.",
    )
    parser.add_argument(
        "--shot-list",
        type=Path,
        help="Optional edit.json with timed shots and declared visual context.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Review archive root; defaults to renders/review.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--assessment",
        type=Path,
        help="Import a complete assessment bound to this packet hash.",
    )
    source.add_argument(
        "--judge",
        choices=["token-factory"],
        help="Send sampled frames and cue context to hosted vision inference.",
    )
    parser.add_argument(
        "--model",
        default="MiniMaxAI/MiniMax-M3",
        help="Vision model for the explicit hosted review.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 unless all cues are reviewed without mismatches or uncertainty.",
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the offline audiovisual review page."
    )
    return parser


def _arguments():
    return _parser().parse_args()


def _main():
    args = _arguments()
    project = _project(args.project.resolve())
    video = (args.video or project["output_dir"] / "final/film.mp4").resolve(
        strict=True
    )
    captions = (args.captions or video.with_suffix(".srt")).resolve(strict=True)
    output = (args.output_dir or project["output_dir"] / "review").resolve()
    packet, directory = _prepare(video, captions, args.shot_list, output)
    assessment = _selected_assessment(args, packet, directory)
    report = _save_review(packet, assessment, directory)
    if args.judge:
        assessment = _judge(packet, directory, args.model)
        if any(
            packet[key] != value
            for key, value in _input_hashes(video, captions, args.shot_list).items()
        ):
            raise ValueError(
                "Film inputs changed during model review; regenerate the packet"
            )
        report = _save_review(packet, assessment, directory)
    print(json.dumps({**report, "review_directory": str(directory)}, indent=2))
    if args.open:
        subprocess.run(
            [
                "open" if sys.platform == "darwin" else "xdg-open",
                str(directory / "review.html"),
            ],
            check=True,
        )
    if args.strict and report["status"] != "reviewed":
        raise SystemExit(1)


def _save_review(packet, assessment, directory):
    report = _assess(packet, assessment)
    (directory / "assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")
    (directory / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_review_page(packet, report, directory)
    return report


def _selected_assessment(args, packet, directory):
    if args.assessment:
        return json.loads(args.assessment.read_text())
    existing = directory / "assessment.json"
    if not args.judge and existing.is_file():
        return json.loads(existing.read_text())
    return _pending_assessment(packet)


if __name__ == "__main__":
    try:
        _main()
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(f"Film review stopped: {error}") from None
