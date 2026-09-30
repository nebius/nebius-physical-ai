"""Export a self-contained interactive replay and a factual four-condition film."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from npa.adapter.isaac_lab_lerobot import _encode_video
from npa.workflows.lerobot_transfer_data import file_sha256, write_json
from npa.workflows.physical_augmentation_contract import accepted_steps

LABELS = {
    "nominal": "Baseline",
    "displaced": "Position shift",
    "heavy": "Double mass",
    "slippery": "Low friction",
}
ORDER = tuple(LABELS)


def _telemetry(root: Path, recipe: dict, attempt: dict) -> dict:
    names = ("actions", "state", "object", "next_object", "next_tcp", "next_velocity")
    arrays = {name: np.load(root / f"{name}.npy", allow_pickle=False) for name in names}
    initial = np.asarray(attempt["initial_object_m"])
    accepted = accepted_steps(arrays, initial, recipe["success"])
    streak, holds = 0, []
    for valid in accepted:
        streak = streak + 1 if valid else 0
        holds.append(streak)
    return {
        "lift_cm": ((arrays["next_object"][:, 2] - initial[2]) * 100).tolist(),
        "speed_cm_s": (np.linalg.norm(arrays["next_velocity"], axis=1) * 100).tolist(),
        "hold_steps": holds,
        "actions": arrays["actions"].tolist(),
        "object_xyz": arrays["next_object"].tolist(),
        "tcp_xyz": arrays["next_tcp"].tolist(),
    }


def _verify_video(path: Path, attempt: dict, presentation: dict, fps: int) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    if (
        int(stream["nb_read_frames"]),
        stream["width"],
        stream["height"],
        stream["r_frame_rate"],
    ) != (attempt["length"], presentation["width"], presentation["height"], f"{fps}/1"):
        raise ValueError(
            f"Demo video differs from the verified native recording: {stream}"
        )
    return {**stream, "sha256": file_sha256(path)}


def _trial(source, output, recipe, attempt, fps):
    root = source / attempt["source"]
    relative = f"demo/videos/{attempt['condition']}-{attempt['attempt']:03d}.mp4"
    video = output / relative
    frames = np.load(root / "rgb.npy", mmap_mode="r", allow_pickle=False)
    _encode_video(frames, video, fps=fps)
    metadata = _verify_video(video, attempt, recipe["presentation"], fps)
    return {
        "condition": attempt["condition"],
        "attempt": attempt["attempt"],
        "seed": attempt["seed"],
        "success": attempt["success"],
        "frames": attempt["length"],
        "video_path": relative,
        "video": "data:video/mp4;base64,"
        + base64.b64encode(video.read_bytes()).decode(),
        "video_evidence": metadata,
        "telemetry": _telemetry(root, recipe, attempt),
    }


def _html(payload: dict, output: Path) -> None:
    template = Path(__file__).with_name("physical_augmentation_demo.html").read_text()
    # JSON is data even when a user-supplied run identifier contains HTML delimiters.
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    document = template.replace("__NPA_DEMO_DATA__", encoded)
    if document.count('id="demo-data"') != 1:
        raise ValueError("Demo template lacks its single data boundary")
    output.write_text(document)


def _font_path() -> str:
    import matplotlib

    return str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")


def _film_plate(trials: list, recipe: dict, path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    plate = Image.new("RGB", (1920, 1280), "#07111c")
    draw = ImageDraw.Draw(plate)
    heading = ImageFont.truetype(_font_path(), 30)
    label = ImageFont.truetype(_font_path(), 20)
    note = ImageFont.truetype(_font_path(), 18)
    draw.text(
        (28, 27),
        "PHYSICS IN ACTION / ONE TASK. FOUR PHYSICAL CONDITIONS.",
        fill="white",
        font=heading,
    )
    for index, trial in enumerate(trials):
        x, y = 12 + (index % 2) * 960, 92 + (index // 2) * 588
        draw.rounded_rectangle((x, y, x + 935, y + 563), radius=12, fill="#111e2b")
        case = recipe["conditions"][trial["condition"]]
        result = "ACCEPTED" if trial["success"] else "REJECTED"
        title = f"{LABELS[trial['condition']]} | {case['mass_kg']:g} kg | friction {case['friction']:g} | {result}"
        draw.text((x + 16, y + 13), title, fill="#dce9f5", font=label)
    footer = f"Actual Isaac RTX recordings | Paired seed {trials[0]['seed']} | Replay holds final frames when a trial ends"
    draw.text((28, 1250), footer, fill="#90a5ba", font=note)
    plate.save(path)


def _film_filter(duration: float) -> str:
    filters = []
    for index in range(4):
        filters.append(
            f"[{index}:v]fps=30,scale=896:504,"
            f"tpad=stop_mode=clone:stop_duration={duration}[v{index}]"
        )
        parent = "4:v" if index == 0 else f"card{index - 1}"
        x, y = 32 + (index % 2) * 960, 140 + (index // 2) * 588
        filters.append(
            f"[{parent}][v{index}]overlay=x={x}:y={y}:shortest=1[card{index}]"
        )
    return ";".join(filters)


def _film(trials: list, output: Path, recipe: dict, fps: int) -> dict:
    selected = [
        next(
            trial
            for trial in trials
            if trial["condition"] == case and trial["attempt"] == 0
        )
        for case in ORDER
    ]
    duration = max(trial["frames"] / fps for trial in selected)
    plate = output / "demo/film-plate.png"
    _film_plate(selected, recipe, plate)
    command = ["ffmpeg", "-v", "error", "-y"]
    for trial in selected:
        command += ["-i", trial["video_path"]]
    command += _film_output_args(duration)
    subprocess.run(command, cwd=output, check=True, capture_output=True)
    return _verify_video(
        output / "demo.mp4",
        {"length": math.ceil(duration * 30)},
        {"width": 1920, "height": 1280},
        30,
    )


def _film_output_args(duration):
    return [
        "-loop",
        "1",
        "-framerate",
        "30",
        "-i",
        "demo/film-plate.png",
        "-filter_complex",
        _film_filter(duration),
        "-map",
        "[card3]",
        "-frames:v",
        str(math.ceil(duration * 30)),
        "-r",
        "30",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "demo.mp4",
    ]


def _payload(source, recipe, trials, identity):
    return {
        "schema": "npa.physical-augmentation.demo.v1",
        "fps": round(1 / identity["control_dt"]),
        "conditions": recipe["conditions"],
        "criteria": recipe["success"],
        "trials": trials,
        "runtime": identity["runtime_version"],
        "action_semantics": recipe["action_semantics"],
        "outcome_time_offset_s": identity["control_dt"],
        "recipe_sha256": file_sha256(source / "recipe.json"),
    }


def _evidence(output, trials, film):
    return {
        "schema": "npa.physical-augmentation.demo.v1",
        "attempted": len(trials),
        "accepted": sum(trial["success"] for trial in trials),
        "videos": [
            {key: trial[key] for key in ("condition", "attempt", "video_evidence")}
            for trial in trials
        ],
        "film": film,
        "html_sha256": file_sha256(output / "demo.html"),
        "mp4_sha256": file_sha256(output / "demo.mp4"),
        "template_sha256": hashlib.sha256(
            Path(__file__).with_name("physical_augmentation_demo.html").read_bytes()
        ).hexdigest(),
    }


def build_demo(
    source: Path, output: Path, recipe: dict, attempts: list, identity: dict
) -> dict:
    """Build replay artifacts from every verified attempt, including failures.

    Args:
        source: Checksum-verified raw recordings.
        output: Report directory for the HTML, MP4 and video audit.
        recipe: Sealed experiment and presentation contract.
        attempts: All independently verified measured outcomes.
        identity: Native timestep and runtime identity.
    Returns:
        Content hashes and decoded-video evidence for the generated demo.
    Raises:
        ValueError: Encoded media or JSON differs from its source contract.
        OSError: Required files or encoders are unavailable.
        subprocess.CalledProcessError: Media encoding or inspection fails.
    """
    fps = round(1 / identity["control_dt"])
    trials = [_trial(source, output, recipe, attempt, fps) for attempt in attempts]
    payload = _payload(source, recipe, trials, identity)
    _html(payload, output / "demo.html")
    film = _film(trials, output, recipe, fps)
    evidence = _evidence(output, trials, film)
    write_json(output / "demo-validation.json", evidence)
    return evidence
