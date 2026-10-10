"""Combine real Cycles CUDA robot renders with CUDA-rendered Marble observations."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np
from PIL import Image

from .acquisition import _download
from .api import MarbleError
from .gpu import _capture
from .navigation_geometry import collision_geometry

BLENDER_VERSION = "4.5.14"
BLENDER_ARCHIVE_SHA256 = (
    "9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3"
)


def _blender(directory):
    name = f"blender-{BLENDER_VERSION}-linux-x64"
    executable = directory / name / "blender"
    if executable.is_file():
        return executable
    directory.mkdir(exist_ok=True)
    payload = _download(
        f"https://download.blender.org/release/Blender4.5/{name}.tar.xz"
    )
    if hashlib.sha256(payload).hexdigest() != BLENDER_ARCHIVE_SHA256:
        raise MarbleError("Official Blender archive failed its pinned SHA-256")
    archive = directory / "blender.tar.xz"
    archive.write_bytes(payload)
    with tarfile.open(archive) as bundle:
        bundle.extractall(directory, filter="data")
    archive.unlink()
    return executable


def _render_part(executable, root, start, stop):
    script = Path(__file__).with_name("quadruped_blender.py")
    log_path = root / f"actor-render-{start:04d}.log"
    evidence_path = log_path.with_suffix(".json")
    with log_path.open("w") as log:
        completed = subprocess.run(
            [
                str(executable),
                "--background",
                "--factory-startup",
                "--threads",
                "4",
                "--python-exit-code",
                "1",
                "--python",
                str(script),
                "--",
                str(root),
                str(start),
                str(stop),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode or not evidence_path.is_file():
        diagnostic = log_path.read_text()[-6000:]
        raise MarbleError(f"Cycles GPU rendering failed: {diagnostic}")
    return json.loads(evidence_path.read_text())


def _merge_parts(parts, count):
    result, timings, cursor = dict(parts[0]), [], 0
    for part in parts:
        end = part["frame_stop"]
        if (
            part["frame_start"] != cursor
            or end <= cursor
            or len(part["frame_wall_seconds"]) != end - cursor
            or part["device_type"] != "CUDA"
            or part["cpu_render_fallback"]
            or part["devices"] != result["devices"]
        ):
            raise MarbleError(
                "CUDA renderer batches have incomplete or inconsistent evidence"
            )
        timings.extend(part["frame_wall_seconds"])
        cursor = end
    if cursor != count:
        raise MarbleError("CUDA renderer batches did not cover every frame")
    result.pop("frame_start")
    result.pop("frame_stop")
    result.update(frame_wall_seconds=timings, renderer_processes=len(parts))
    return result


def _render_actor(root):
    executable = _blender(root.parent / "blender-runtime")
    count = len(json.loads((root / "trajectory.json").read_text()))
    spans = [(0, count // 2), (count // 2, count)]
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(_render_part, executable, root, start, stop)
            for start, stop in spans
        ]
        result = _merge_parts([future.result() for future in futures], count)
    (root / "actor-render.json").write_text(json.dumps(result))
    (root / "actor-render.log").write_text(
        "\n".join(
            (root / f"actor-render-{start:04d}.log").read_text() for start, _ in spans
        )
    )
    return result


def _actor(root, world, request):
    vertices, faces = collision_geometry(root, world)
    (root / "render-warehouse.json").write_text(
        json.dumps({"vertices": vertices.tolist(), "faces": faces.tolist()})
    )
    settings = {
        "width": request.width,
        "height": request.height,
        "samples": request.samples,
        "field_of_view": 60,
    }
    (root / "render-settings.json").write_text(json.dumps(settings))
    evidence = _render_actor(root)
    if (
        evidence["device_type"] != "CUDA"
        or evidence["cpu_render_fallback"]
        or len(evidence["frame_wall_seconds"]) != request.frames
    ):
        raise MarbleError("Robot rendering did not provide complete CUDA GPU evidence")
    return evidence


def _linear(pixels):
    return np.where(
        pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4
    )


def _composite(root, count):
    for index in range(count):
        path = root / "observer" / f"{index:04d}.jpg"
        background = np.asarray(Image.open(path).convert("RGB"), dtype=float) / 255
        actor = (
            np.asarray(
                Image.open(root / "actors" / f"{index:04d}.png").convert("RGBA"),
                dtype=float,
            )
            / 255
        )
        alpha = actor[..., 3:4]
        blended = _linear(actor[..., :3]) * alpha + _linear(background) * (1 - alpha)
        pixels = np.where(
            blended <= 0.0031308, blended * 12.92, 1.055 * blended ** (1 / 2.4) - 0.055
        )
        Image.fromarray(np.uint8(np.clip(pixels * 255, 0, 255))).save(path, quality=95)


def render_observer(root, world, request, poses, intrinsic, torch):
    """Render a high fidelity articulated observer stream on the assigned GPU.

    Args: World/trajectory directory, world, request, calibrated cameras, CUDA runtime.
    Returns: Measured gsplat and Cycles renderer evidence.
    Raises: MarbleError or CalledProcessError on missing GPU or incomplete frames.
    """
    actor = _actor(root, world, request)
    splats = _capture(root, world, request, poses, intrinsic, torch, "observer")
    _composite(root, request.frames)
    shutil.rmtree(root / "actors")
    return {
        "render": splats,
        "actor": actor,
        "composition": "gsplat CUDA world with Cycles CUDA articulated robot, original-mesh occlusion and shadow catcher; linear-light alpha composition",
    }
