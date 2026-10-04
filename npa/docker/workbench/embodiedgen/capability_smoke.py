#!/usr/bin/env python3
"""Run EmbodiedGen's upstream TRELLIS path and validate its generated URDF in PyBullet."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pybullet as bullet
import pybullet_data
import trimesh
from PIL import Image

CAPABILITY = "img3d-cli_trellis_image_to_urdf_pybullet"
DEFAULT_INPUT = (
    "https://raw.githubusercontent.com/HorizonRobotics/EmbodiedGen/"
    "f0124197888c2b733e4eaa65acd81ad9cfda3b79/apps/assets/example_image/sample_00.jpg"
)
DEFAULT_INPUT_SHA256 = (
    "d60272ce039e4a230cb2654b5ecb74021b1e3048a82e4d27b62e2c027a109dcd"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def fetch_input(uri: str, destination: Path) -> None:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(uri, timeout=60) as response:
            destination.write_bytes(response.read())
        return
    if parsed.scheme == "s3":
        import boto3

        boto3.client("s3").download_file(
            parsed.netloc, parsed.path.lstrip("/"), str(destination)
        )
        return
    raise ValueError("input URI must use https or s3")


def require_input(output: Path) -> tuple[Path, str]:
    uri = os.environ.get("NPA_EMBODIEDGEN_INPUT_URI", DEFAULT_INPUT).strip()
    path = output / "input.jpg"
    fetch_input(uri, path)
    observed = sha256(path)
    if uri == DEFAULT_INPUT and observed != DEFAULT_INPUT_SHA256:
        raise RuntimeError("pinned upstream sample image hash mismatch")
    with Image.open(path) as image:
        image.verify()
    return path, observed


def run_upstream(source: Path, input_path: Path, generated: Path) -> None:
    command = [
        str(Path(sys.executable).parent / "img3d-cli"),
        "--image_path",
        str(input_path),
        "--output_root",
        str(generated),
        "--image3d_model",
        "TRELLIS",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{source}:{source / 'thirdparty' / 'TRELLIS'}"
    env["SPCONV_ALGO"] = "native"
    subprocess.run(command, cwd=source, env=env, check=True)


def locate_urdf(generated: Path) -> Path:
    candidates = sorted(generated.rglob("*.urdf"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one generated URDF, found {len(candidates)}")
    return candidates[0]


def xml_properties(urdf: Path) -> dict[str, str]:
    root = ET.parse(urdf).getroot()
    values = {
        node.tag: (node.text or "").strip() for node in root.findall(".//extra_info/*")
    }
    mass = root.find(".//inertial/mass")
    if mass is not None:
        values["urdf_mass_kg"] = mass.attrib.get("value", "")
    return values


def collision_meshes(urdf: Path) -> list[dict[str, Any]]:
    root = ET.parse(urdf).getroot()
    records: list[dict[str, Any]] = []
    for node in root.findall(".//collision/geometry/mesh"):
        relative = node.attrib.get("filename", "")
        mesh_path = (urdf.parent / relative).resolve()
        if (
            not relative
            or not mesh_path.is_file()
            or not mesh_path.is_relative_to(urdf.parent.resolve())
        ):
            raise RuntimeError(f"missing or unsafe collision mesh: {relative}")
        mesh = trimesh.load(mesh_path, force="mesh")
        if (
            not isinstance(mesh, trimesh.Trimesh)
            or len(mesh.vertices) < 4
            or len(mesh.faces) < 4
        ):
            raise RuntimeError(f"invalid collision mesh: {relative}")
        bounds = mesh.bounds.tolist()
        if not np.isfinite(np.asarray(bounds)).all() or np.any(
            np.subtract(bounds[1], bounds[0]) <= 0
        ):
            raise RuntimeError(f"degenerate collision mesh: {relative}")
        records.append(
            {
                "path": relative,
                "sha256": sha256(mesh_path),
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "bounds": bounds,
            }
        )
    if not records:
        raise RuntimeError("generated URDF contains no collision mesh")
    return records


def convert_to_mjcf(urdf: Path, output: Path) -> dict[str, Any]:
    """Exercise EmbodiedGen's documented URDF-to-MJCF converter.

    PyBullet consumes the exact generated URDF below. The separate MJCF export
    is an upstream-supported handoff for MuJoCo/Genesis; it is deliberately not
    described as a MuJoCo physics result.
    """
    from embodied_gen.data.asset_converter import cvt_embodiedgen_asset_to_anysim
    from embodied_gen.utils.enum import AssetType

    target_dir = output / "mjcf"
    converted = cvt_embodiedgen_asset_to_anysim(
        urdf_files=[str(urdf)],
        target_dirs=[str(target_dir)],
        target_type=AssetType.MJCF,
        source_type=AssetType.URDF,
        overwrite=True,
    )
    converted_path = Path(converted[str(urdf)]).resolve()
    if not converted_path.is_file():
        raise RuntimeError("EmbodiedGen MJCF converter did not write its output")
    root = ET.parse(converted_path).getroot()
    meshes = root.findall("./asset/mesh")
    geoms = root.findall(".//geom[@type='mesh']")
    if root.tag != "mujoco" or not meshes or not geoms:
        raise RuntimeError("EmbodiedGen MJCF output has no mesh simulation geometry")
    for mesh in meshes:
        relative = mesh.attrib.get("file", "")
        path = (converted_path.parent / relative).resolve()
        if (
            not relative
            or not path.is_file()
            or not path.is_relative_to(converted_path.parent)
        ):
            raise RuntimeError(f"invalid MJCF mesh reference: {relative}")
    return {
        "target": "MuJoCo/Genesis MJCF",
        "path": str(converted_path.relative_to(output)),
        "sha256": sha256(converted_path),
        "mesh_assets": len(meshes),
        "mesh_geoms": len(geoms),
    }


def gpu_evidence() -> dict[str, Any]:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("GPU capability evidence requires nvidia-smi") from error
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS run did not receive a CUDA GPU")
    return {
        "nvidia_smi": text.splitlines(),
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def require_rigid_body_settle(body: int) -> dict[str, Any]:
    """Prove the simulated generated body is no longer moving materially."""
    positions = []
    for step in range(120):
        bullet.stepSimulation()
        if step % 12 == 0:
            positions.append(np.asarray(bullet.getBasePositionAndOrientation(body)[0]))
    linear, angular = bullet.getBaseVelocity(body)
    linear_speed = float(np.linalg.norm(linear))
    angular_speed = float(np.linalg.norm(angular))
    drift = float(np.linalg.norm(positions[-1] - positions[0]))
    if not all(math.isfinite(value) for value in (linear_speed, angular_speed, drift)):
        raise RuntimeError("generated URDF settle measurements are not finite")
    if linear_speed > 0.05 or angular_speed > 0.1 or drift > 0.01:
        raise RuntimeError("generated URDF did not settle under rigid-body physics")
    return {
        "settle_steps": 120,
        "linear_speed_m_per_s": linear_speed,
        "angular_speed_rad_per_s": angular_speed,
        "late_window_drift_m": drift,
    }


def pybullet_validation(urdf: Path, output: Path) -> dict[str, Any]:
    client = bullet.connect(bullet.DIRECT)
    try:
        bullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = bullet.loadURDF("plane.urdf")
        bullet.setGravity(0, 0, -9.81)
        body = bullet.loadURDF(str(urdf), basePosition=[0, 0, 1.0], useFixedBase=False)
        initial = bullet.getBasePositionAndOrientation(body)[0]
        frames: list[np.ndarray] = []
        projection = bullet.computeProjectionMatrixFOV(55, 4 / 3, 0.05, 5.0)
        view = bullet.computeViewMatrix([2.0, -2.0, 1.5], [0, 0, 0.35], [0, 0, 1])
        for step in range(480):
            bullet.stepSimulation()
            if step % 8 == 0:
                pixels = bullet.getCameraImage(
                    640, 480, view, projection, renderer=bullet.ER_TINY_RENDERER
                )[2]
                frames.append(np.asarray(pixels, dtype=np.uint8)[..., :3])
        settle = require_rigid_body_settle(body)
        final = bullet.getBasePositionAndOrientation(body)[0]
        contacts = bullet.getContactPoints(bodyA=body, bodyB=plane)
        if not contacts or not all(
            math.isfinite(value) for value in (*initial, *final)
        ):
            raise RuntimeError("generated URDF did not make stable rigid-body contact")
        if final[2] >= initial[2] - 0.1:
            raise RuntimeError("generated URDF did not fall under gravity")
        view_png, view_mp4 = (
            output / "pybullet_view.png",
            output / "pybullet_settle.mp4",
        )
        iio.imwrite(view_png, frames[-1])
        iio.imwrite(view_mp4, np.stack(frames), fps=30)
        decoded = sum(1 for _ in iio.imiter(view_mp4))
        if decoded != len(frames):
            raise RuntimeError("PyBullet video did not decode completely")
        return {
            "simulator": "PyBullet DIRECT",
            "steps": 600,
            "initial_position_m": initial,
            "final_position_m": final,
            "contact_points": len(contacts),
            "settle": settle,
            "view_png": view_png.name,
            "view_mp4": view_mp4.name,
            "decoded_video_frames": decoded,
        }
    finally:
        bullet.disconnect(client)


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    output = Path(os.environ["NPA_SMOKE_OUTPUT_DIR"]).resolve()
    source = Path(os.environ["NPA_EMBODIEDGEN_SOURCE_ROOT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_path, input_hash = require_input(output)
    generated = output / "generated"
    started = time.time()
    run_upstream(source, input_path, generated)
    urdf = locate_urdf(generated)
    collisions = collision_meshes(urdf)
    mjcf = convert_to_mjcf(urdf, output)
    physics = pybullet_validation(urdf, output)
    artifacts = [
        file_record(path, output)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    report = {
        "schema": "npa.embodiedgen.image-to-rigid-object.v1",
        "status": "success",
        "solution": "embodiedgen",
        "capability": CAPABILITY,
        "capabilities_exercised": [
            "trellis_image_to_mesh_generation",
            "embodiedgen_urdf_export",
            "embodiedgen_urdf_to_mjcf_conversion",
            "collision_geometry_validation",
            "pybullet_rigid_body_settle",
            "pybullet_viewable_result",
        ],
        "source_revision": "f0124197888c2b733e4eaa65acd81ad9cfda3b79",
        "trellis_revision": "55a8e8164b195bbf927e0978f00e76c835e6011f",
        "trellis_model_revision": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
        "runtime_receipt_sha256": sha256(
            Path(os.environ["NPA_EMBODIEDGEN_RUNTIME_RECEIPT"])
        ),
        "input": {
            "sha256": input_hash,
            "source": os.environ.get("NPA_EMBODIEDGEN_INPUT_URI", DEFAULT_INPUT),
        },
        "image_reference": os.environ.get("BYOF_IMAGE", ""),
        "gpu": gpu_evidence(),
        "urdf": {
            "path": str(urdf.relative_to(output)),
            "sha256": sha256(urdf),
            "properties": xml_properties(urdf),
            "physical_property_semantics": "VLM_estimated_not_calibrated_ground_truth",
        },
        "mjcf_conversion": mjcf,
        "collision_geometry": collisions,
        "physics": physics,
        "artifacts": artifacts,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_json(output / "embodiedgen_image_to_rigid_object.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
