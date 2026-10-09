"""Build a standalone replay exclusively from verified rover collection artifacts."""

import base64
import hashlib
import json
import math
from pathlib import Path

from .api import MarbleError


def _validated_frames(root, result, directory):
    frames = []
    for index in range(result["frames"]):
        name = f"{directory}/{index:04d}.jpg"
        payload = (root / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != result["files"].get(name, {}).get(
            "sha256"
        ):
            raise MarbleError("Rover report frame hash mismatch")
        frames.append("data:image/jpeg;base64," + base64.b64encode(payload).decode())
    return frames


def _evidence(root, result):
    payload = (root / "trajectory.json").read_bytes()
    if (
        hashlib.sha256(payload).hexdigest()
        != result["files"]["trajectory.json"]["sha256"]
    ):
        raise MarbleError("Rover trajectory hash mismatch")
    trajectory = json.loads(payload)
    metrics = result["metrics"]
    for component in [metrics["rgb"], metrics["depth"], metrics["observer"]["render"]]:
        timings = component["cuda_frame_ms"]
        if len(timings) != result["frames"] or any(
            not math.isfinite(value) or value <= 0 for value in timings
        ):
            raise MarbleError(
                "Rover report requires measured CUDA work for every sensor frame"
            )
    if len(trajectory) != result["frames"] or metrics["physics"]["distance_m"] < 1:
        raise MarbleError("Rover report requires a complete moving trajectory")
    return {
        "gpu": result["gpu"],
        "metrics": metrics,
        "frames": result["frames"],
        "width": result["width"],
        "height": result["height"],
        "trajectory": trajectory,
        "limitations": result["limitations"],
        "world_model": result["world"]["model"],
        "world_generated": result["world"]["generated_this_run"],
        "asset_hashes": result["world"]["files"],
        "wall_seconds": result["wall_seconds"],
    }


def write_rover_report(root: Path, result):
    """Embed completed sensor frames, measured state, and sanitized provenance.

    Args: Local verified result directory and decoded result manifest.
    Returns: The standalone HTML path.
    Raises: MarbleError when frames, timing, or embodied evidence are incomplete.
    """
    data = _evidence(root, result)
    data["images"] = {
        name: _validated_frames(root, result, directory)
        for name, directory in [
            ("rgb", "frames"),
            ("depth", "depth-frames"),
            ("observer", "observer"),
        ]
    }
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = Path(__file__).with_name("rover-report.html").read_text()
    output = template.replace("__COLLECTION_DATA__", payload)
    script = output.split('<script id="replay">', 1)[1].split("</script>", 1)[0]
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    output = output.replace("__SCRIPT_SHA256__", digest)
    path = root / "index.html"
    path.write_text(output)
    return path
