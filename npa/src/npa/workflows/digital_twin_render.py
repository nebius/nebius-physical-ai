"""Run pinned Blender CUDA rendering and publish a verified, offline digital-twin preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile
import tempfile

from PIL import Image
import requests

from npa.workbench.nurec.render_evidence import RenderTelemetry
from npa.workflows.navigation.artifacts import publish
from npa.workflows.preview_html import image_preview, write_preview

BLENDER_VERSION = "4.5.3"
BLENDER_ARCHIVE = "blender-4.5.3-linux-x64"
BLENDER_URL = (
    f"https://download.blender.org/release/Blender4.5/{BLENDER_ARCHIVE}.tar.xz"
)
BLENDER_SHA256 = "975c58fcb244273838534bba771e64ad87739216b0f9b39a888531a49a72d845"
SCHEMA = "npa.digital-twin.cuda-render.v1"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blender(root):
    archive = root / "blender.tar.xz"
    with requests.get(BLENDER_URL, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with archive.open("xb") as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    if _sha256(archive) != BLENDER_SHA256:
        raise ValueError("Blender archive differs from its official pinned SHA-256")
    with tarfile.open(archive, "r:xz") as handle:
        handle.extractall(root, filter="data")
    executable = root / BLENDER_ARCHIVE / "blender"
    if not executable.is_file():
        raise ValueError("Blender archive did not contain its native executable")
    return executable


def _gpu_label(name):
    if "B200" in name:
        return "NVIDIA B200"
    if "RTX" in name and "6000" in name and "PRO" in name:
        return "NVIDIA RTX PRO 6000"
    return "NVIDIA CUDA GPU"


def _native_receipt(root, views):
    native = json.loads((root / "native-render.json").read_text())
    devices = native.get("devices", [])
    if (
        native.get("backend") != "Blender Cycles CUDA"
        or native.get("version") != BLENDER_VERSION
        or native.get("cpu_rendering") is not False
        or not devices
        or any(device.get("type") != "CUDA" for device in devices)
    ):
        raise ValueError("Native receipt must prove the pinned CUDA-only renderer")
    frames = sorted(root.glob("frame-*.png"))
    if len(frames) != views or len(native.get("cameras", [])) != views:
        raise ValueError("Native renderer did not produce all requested viewpoints")
    if type(native.get("samples")) is not int or native["samples"] < 1:
        raise ValueError("Path-tracing samples must be a positive integer")
    if native.get("resolution") != [1280, 720]:
        raise ValueError("Native resolution differs from the reference contract")
    return native


def _receipt(root, telemetry, views):
    native = _native_receipt(root, views)
    devices = native["devices"]
    frames = sorted(root.glob("frame-*.png"))
    files = frames + [
        root / "scene.usdc",
        root / "scene.glb",
        root / "native-render.json",
    ]
    record = {
        "schema": SCHEMA,
        "backend": native["backend"],
        "version": BLENDER_VERSION,
        "scene": "Repository-authored factory cell; not a captured facility",
        "gpu_models": sorted({_gpu_label(device["name"]) for device in devices}),
        "cpu_rendering": False,
        "frame_count": views,
        "resolution": native["resolution"],
        "samples": native["samples"],
        "blender_archive_sha256": BLENDER_SHA256,
        "scene_script_sha256": _sha256(
            Path(__file__).with_name("digital_twin_scene.py")
        ),
        "files": {path.name: _sha256(path) for path in files},
        "telemetry": telemetry,
    }
    (root / "render-evidence.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def verify_render(root: Path) -> dict:
    """Validate a completed CUDA run against its bound scene and rendered bytes.

    Args:
        root: Materialized render output directory.
    Returns:
        The validated evidence record, including device-wide observations.
    Raises:
        ValueError: Renderer, media, scene, or receipt integrity checks fail.
        OSError: Required artifacts cannot be read.
    """
    record = json.loads((root / "render-evidence.json").read_text())
    if (
        record.get("schema") != SCHEMA
        or record.get("backend") != "Blender Cycles CUDA"
        or record.get("version") != BLENDER_VERSION
        or record.get("cpu_rendering") is not False
    ):
        raise ValueError("Unrecognized CUDA render evidence")
    count = record.get("frame_count")
    if type(count) is not int or count < 2:
        raise ValueError("At least two real viewpoints are required")
    expected = {f"frame-{index:03d}.png" for index in range(count)}
    if {path.name for path in root.glob("frame-*.png")} != expected:
        raise ValueError("Rendered viewpoint inventory changed")
    expected |= {"scene.usdc", "scene.glb", "native-render.json"}
    if set(record.get("files", {})) != expected:
        raise ValueError("Scene and native renderer evidence must be bound")
    for name, digest in record["files"].items():
        path = root / name
        if path.is_symlink() or _sha256(path) != digest:
            raise ValueError("Rendered artifact hash mismatch")
    if (root / "scene.glb").read_bytes()[:4] != b"glTF":
        raise ValueError("The portable scene is not a binary glTF")
    if (root / "scene.usdc").read_bytes()[:8] != b"PXR-USDC":
        raise ValueError("The OpenUSD scene is not a USD crate")
    _verify_measurements(root, record)
    return record


def _verify_measurements(root, record):
    native = _native_receipt(root, record["frame_count"])
    models = sorted({_gpu_label(device["name"]) for device in native["devices"]})
    if record.get("gpu_models") != models or record.get("samples") != native["samples"]:
        raise ValueError("Rendering measurements disagree with the native receipt")
    telemetry = record.get("telemetry", {})
    for key in ("sample_count", "peak_utilization_percent", "elapsed_seconds"):
        value = telemetry.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(
                "Rendering telemetry must contain finite numeric measurements"
            )
    if telemetry["peak_utilization_percent"] > 100:
        raise ValueError("GPU utilization cannot exceed 100 percent")


def _preview_frames(root):
    frames = []
    for index, path in enumerate(sorted(root.glob("frame-*.png"))):
        with Image.open(path) as image:
            frames.append(
                {
                    "label": f"Camera viewpoint {index + 1}",
                    "images": [
                        {
                            "label": "Cycles CUDA path-traced render",
                            "data": image_preview(image, width=1280),
                        }
                    ],
                }
            )
    return frames


def _preview(root):
    record = verify_render(root)
    frames = _preview_frames(root)
    telemetry = record["telemetry"]
    details = {
        key: record[key]
        for key in ("backend", "version", "frame_count", "samples", "files")
    }
    details["telemetry_scope"] = (
        "Allocated-device observations; not process-level hardware attestation."
    )
    write_preview(
        root / "index.html",
        title="Digital twin · factory cell",
        summary="A repository-authored industrial reference environment, rendered on a real GPU. "
        "Scrub or play the recorded camera viewpoints. The scene is an authored demonstrator, "
        "not a scan of a physical facility, a live simulation, or an XR stream.",
        metrics={
            "Renderer": "Blender Cycles · CUDA",
            "GPU": ", ".join(record["gpu_models"]),
            "Viewpoints": len(frames),
            "Resolution": "1280 × 720",
            "Peak GPU activity": f"{telemetry['peak_utilization_percent']}%"
            if telemetry["sample_count"]
            else "Unavailable",
            "Render duration": f"{telemetry['elapsed_seconds']:.1f} s",
        },
        groups=[
            {
                "title": "Rendered inspection views",
                "note": "Real GPU path tracing; all media embedded for offline viewing.",
                "frames": frames,
            }
        ],
        details=details,
    )


def _run(output, views, samples):
    if views < 2 or samples < 1:
        raise ValueError(
            "Rendering requires at least two views and one path-tracing sample"
        )
    with tempfile.TemporaryDirectory(prefix="npa-digital-twin-") as temporary:
        root = Path(temporary)
        executable = _blender(root)
        scene_script = Path(__file__).with_name("digital_twin_scene.py")
        rendered = root / "rendered"
        command = [
            str(executable),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python",
            str(scene_script),
            "--",
            "--output-path",
            str(rendered),
            "--views",
            str(views),
            "--samples",
            str(samples),
        ]
        with RenderTelemetry() as telemetry:
            subprocess.run(command, check=True)
        measurements = telemetry.summary()
        measurements["scope"] = (
            "Allocated-device telemetry during Blender execution; not process attestation."
        )
        measurements["gpu_models"] = sorted(
            {_gpu_label(name) for name in measurements["gpu_models"]}
        )
        _receipt(rendered, measurements, views)
        _preview(rendered)
        publish(rendered, output)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    _run(args.output_path, args.views, args.samples)


if __name__ == "__main__":
    _main()
