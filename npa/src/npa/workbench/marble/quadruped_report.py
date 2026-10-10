"""Package verified Go1 collection videos and telemetry into one offline HTML file."""

import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .api import MarbleError


def _video(root, directory, frequency):
    import imageio_ffmpeg

    output = root / f"{directory}.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(frequency),
            "-i",
            str(root / directory / "%04d.jpg"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    return "data:video/mp4;base64," + base64.b64encode(output.read_bytes()).decode()


def _data(root, result):
    physics = result["metrics"]["physics"]
    actor = result["metrics"]["observer"]["actor"]
    timings = actor["frame_wall_seconds"]
    if (
        actor["device_type"] != "CUDA"
        or actor["cpu_render_fallback"]
        or not actor["devices"]
    ):
        raise MarbleError("Quadruped report requires an actual CUDA robot render")
    if len(timings) != result["frames"] or any(
        not math.isfinite(t) or t <= 0 for t in timings
    ):
        raise MarbleError(
            "Quadruped report requires measured rendering for every frame"
        )
    trajectory = json.loads((root / "trajectory.json").read_text())
    if len(trajectory) != result["frames"] or physics["distance_m"] < 1:
        raise MarbleError("Quadruped report requires complete supported walking")
    return {
        "frames": result["frames"],
        "width": result["width"],
        "height": result["height"],
        "gpu": result["gpu"],
        "physics": physics,
        "actor": actor,
        "trajectory": trajectory,
        "world_model": result["world"]["model"],
        "limitations": result["limitations"],
        "world_hashes": result["world"]["files"],
        "robot_assets": result["robot_assets"],
    }


def write_quadruped_report(root, result):
    """Encode actual recorded frames and embed them with their measured telemetry.

    Args: Hash-verified local result bundle and decoded result manifest.
    Returns: The standalone HTML path.
    Raises: MarbleError or CalledProcessError for invalid evidence or video encoding.
    """
    data = _data(root, result)
    data["videos"] = {
        name: _video(root, name, data["physics"]["sensor_hz"])
        for name in ("observer", "frames", "depth-frames")
    }
    notices = []
    for entry in data["robot_assets"]:
        if entry["path"] == "LICENSE":
            notices.append(
                {
                    "repository": entry["repository"],
                    "text": (root / entry["file"]).read_text(),
                }
            )
    data["notices"] = notices
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    template = Path(__file__).with_name("quadruped-report.html").read_text()
    page = template.replace("__CAPTURE_DATA__", payload)
    script = page.split('<script id="player">', 1)[1].split("</script>", 1)[0]
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    page = page.replace("__SCRIPT_SHA256__", digest)
    output = root / "index.html"
    output.write_text(page)
    return output
