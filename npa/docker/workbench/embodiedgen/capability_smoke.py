#!/usr/bin/env python3
"""Run EmbodiedGen's upstream TRELLIS path and validate its generated URDF in PyBullet."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any

from asset_bundle import archive_asset_tree
from defusedxml import ElementTree as ET
import imageio.v3 as iio
import numpy as np
import pybullet as bullet
import pybullet_data
import trimesh
from PIL import Image

CAPABILITY = "img3d-cli_trellis_image_to_urdf_pybullet"
CAPABILITIES_EXERCISED = [
    "trellis_image_to_mesh_generation",
    "embodiedgen_urdf_export",
    "embodiedgen_urdf_to_mjcf_conversion",
    "collision_geometry_validation",
    "pybullet_rigid_body_settle",
    "pybullet_viewable_result",
]
DEFAULT_INPUT = (
    "https://raw.githubusercontent.com/HorizonRobotics/EmbodiedGen/"
    "f0124197888c2b733e4eaa65acd81ad9cfda3b79/apps/assets/example_image/sample_00.jpg"
)
DEFAULT_INPUT_SHA256 = (
    "d60272ce039e4a230cb2654b5ecb74021b1e3048a82e4d27b62e2c027a109dcd"
)
PHYSICS_STEP_HZ = 240
PHYSICS_OBSERVATION_SECONDS = 40
PHYSICS_OBSERVATION_STEPS = PHYSICS_STEP_HZ * PHYSICS_OBSERVATION_SECONDS
PHYSICS_SETTLE_WINDOW_STEPS = 120
PHYSICS_CAPTURE_INTERVAL_STEPS = 60
PHYSICS_CAPTURE_FPS = PHYSICS_STEP_HZ // PHYSICS_CAPTURE_INTERVAL_STEPS


class RigidBodySettleError(RuntimeError):
    """A failed settle gate that retains the measured late-window metrics."""

    def __init__(self, measurements: dict[str, Any]) -> None:
        super().__init__("generated URDF did not settle under rigid-body physics")
        self.measurements = measurements


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
    if parsed.scheme == "https":
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 443}
            or parsed.fragment
        ):
            raise ValueError("input HTTPS URI has an unsafe authority or fragment")
        request_path = parsed.path or "/"
        if parsed.query:
            request_path = f"{request_path}?{parsed.query}"
        connection = http.client.HTTPSConnection(parsed.hostname, 443, timeout=60)
        temporary = destination.with_name(f".{destination.name}.download")
        try:
            connection.request(
                "GET", request_path, headers={"User-Agent": "npa-embodiedgen"}
            )
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"input HTTPS request returned {response.status}")
            with temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        finally:
            connection.close()
        temporary.replace(destination)
        return
    if parsed.scheme == "s3":
        if not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValueError("input S3 URI must name a bucket and object")
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


def _validate_mjcf(converted_path: Path) -> tuple[int, int]:
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
    return len(meshes), len(geoms)


def convert_to_mjcf(urdf: Path, output: Path) -> dict[str, Any]:
    """Exercise EmbodiedGen's documented URDF-to-MJCF converter.

    PyBullet consumes the exact generated URDF below. The separate MJCF export
    is an upstream-supported handoff for MuJoCo/Genesis; it is deliberately not
    described as a MuJoCo physics result.
    """
    from embodied_gen.data.asset_converter import cvt_embodiedgen_asset_to_anysim
    from embodied_gen.utils.enum import AssetType

    converted = cvt_embodiedgen_asset_to_anysim(
        urdf_files=[str(urdf)],
        target_dirs=[str(output / "mjcf")],
        target_type=AssetType.MJCF,
        source_type=AssetType.URDF,
        overwrite=True,
    )
    converted_path = Path(converted[str(urdf)]).resolve()
    if not converted_path.is_file():
        raise RuntimeError("EmbodiedGen MJCF converter did not write its output")
    mesh_assets, mesh_geoms = _validate_mjcf(converted_path)
    return {
        "target": "MuJoCo/Genesis MJCF",
        "path": str(converted_path.relative_to(output)),
        "sha256": sha256(converted_path),
        "mesh_assets": mesh_assets,
        "mesh_geoms": mesh_geoms,
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
    for step in range(PHYSICS_SETTLE_WINDOW_STEPS):
        bullet.stepSimulation()
        if step % 12 == 0:
            positions.append(np.asarray(bullet.getBasePositionAndOrientation(body)[0]))
    linear, angular = bullet.getBaseVelocity(body)
    linear_speed = float(np.linalg.norm(linear))
    angular_speed = float(np.linalg.norm(angular))
    drift = float(np.linalg.norm(positions[-1] - positions[0]))
    measurements = {
        "settle_steps": PHYSICS_SETTLE_WINDOW_STEPS,
        "settle_window_seconds": PHYSICS_SETTLE_WINDOW_STEPS / PHYSICS_STEP_HZ,
        "linear_speed_m_per_s": linear_speed,
        "angular_speed_rad_per_s": angular_speed,
        "late_window_drift_m": drift,
    }
    if not all(math.isfinite(value) for value in (linear_speed, angular_speed, drift)):
        raise RuntimeError("generated URDF settle measurements are not finite")
    if linear_speed > 0.05 or angular_speed > 0.1 or drift > 0.01:
        raise RigidBodySettleError(measurements)
    return measurements


def _pybullet_camera(body: int) -> tuple[Any, Any, dict[str, Any]]:
    lower, upper = (np.asarray(bound, dtype=float) for bound in bullet.getAABB(body))
    extent = float(np.max(upper - lower))
    if not math.isfinite(extent) or extent <= 0:
        raise RuntimeError("generated URDF has invalid camera bounds")
    center = (lower + upper) / 2
    distance = max(0.25, extent * 2.5)
    direction = np.asarray([1.0, -1.0, 0.75])
    direction /= np.linalg.norm(direction)
    eye = center + direction * distance
    near, far = max(0.01, distance / 50), max(2.0, distance * 6)
    projection = bullet.computeProjectionMatrixFOV(55, 4 / 3, near, far)
    view = bullet.computeViewMatrix(eye.tolist(), center.tolist(), [0, 0, 1])
    return (
        view,
        projection,
        {
            "target_m": center.tolist(),
            "eye_m": eye.tolist(),
            "distance_m": distance,
            "near_m": near,
            "far_m": far,
        },
    )


def _capture_pybullet_frames(body: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    frames = []
    first_camera = None
    last_camera = None
    for step in range(PHYSICS_OBSERVATION_STEPS):
        bullet.stepSimulation()
        if (step + 1) % PHYSICS_CAPTURE_INTERVAL_STEPS == 0:
            view, projection, camera = _pybullet_camera(body)
            if first_camera is None:
                first_camera = camera
            last_camera = camera
            pixels = bullet.getCameraImage(
                640, 480, view, projection, renderer=bullet.ER_TINY_RENDERER
            )[2]
            frames.append(np.asarray(pixels, dtype=np.uint8)[..., :3])
    if not frames:
        raise RuntimeError("PyBullet observation did not capture any frames")
    return frames, {
        "mode": "body_aabb_tracking",
        "first_capture": first_camera,
        "last_capture": last_camera,
    }


def _write_decoded_view(
    output: Path,
    frames: list[np.ndarray],
    *,
    png_name: str = "pybullet_view.png",
    mp4_name: str = "pybullet_settle.mp4",
) -> tuple[Path, Path, int]:
    view_png, view_mp4 = output / png_name, output / mp4_name
    iio.imwrite(view_png, frames[-1])
    iio.imwrite(view_mp4, np.stack(frames), fps=PHYSICS_CAPTURE_FPS)
    decoded = sum(1 for _ in iio.imiter(view_mp4))
    if decoded != len(frames):
        raise RuntimeError("PyBullet video did not decode completely")
    return view_png, view_mp4, decoded


def _validate_pybullet_body(
    body: int, plane: int, initial: Any
) -> tuple[Any, Any, Any]:
    settle = require_rigid_body_settle(body)
    final = bullet.getBasePositionAndOrientation(body)[0]
    contacts = bullet.getContactPoints(bodyA=body, bodyB=plane)
    if not contacts or not all(math.isfinite(value) for value in (*initial, *final)):
        raise RuntimeError("generated URDF did not make stable rigid-body contact")
    if final[2] >= initial[2] - 0.1:
        raise RuntimeError("generated URDF did not fall under gravity")
    return final, contacts, settle


def _write_pybullet_failure_evidence(
    output: Path,
    frames: list[np.ndarray],
    camera: dict[str, Any],
    body: int,
    plane: int,
    initial: Any,
    error: RuntimeError,
) -> None:
    """Keep actual simulator observations when a generated asset is rejected."""

    view_png, view_mp4, decoded = _write_decoded_view(
        output,
        frames,
        png_name="pybullet_failure_view.png",
        mp4_name="pybullet_failure_settle.mp4",
    )
    final = bullet.getBasePositionAndOrientation(body)[0]
    contacts = bullet.getContactPoints(bodyA=body, bodyB=plane)
    atomic_json(
        output / "pybullet_validation_failure.json",
        {
            "schema": "npa.embodiedgen.pybullet-validation-failure.v1",
            "status": "failed",
            "reason": str(error),
            "observation_steps": PHYSICS_OBSERVATION_STEPS,
            "observation_seconds": PHYSICS_OBSERVATION_SECONDS,
            "capture_interval_steps": PHYSICS_CAPTURE_INTERVAL_STEPS,
            "capture_fps": PHYSICS_CAPTURE_FPS,
            "captured_frames": len(frames),
            "decoded_video_frames": decoded,
            "initial_position_m": initial,
            "final_position_m": final,
            "contact_points": len(contacts),
            "camera": camera,
            "settle": getattr(error, "measurements", None),
            "view_png": view_png.name,
            "view_mp4": view_mp4.name,
        },
    )


def pybullet_validation(urdf: Path, output: Path) -> dict[str, Any]:
    client = bullet.connect(bullet.DIRECT)
    try:
        bullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = bullet.loadURDF("plane.urdf")
        bullet.setGravity(0, 0, -9.81)
        body = bullet.loadURDF(str(urdf), basePosition=[0, 0, 1.0], useFixedBase=False)
        initial = bullet.getBasePositionAndOrientation(body)[0]
        frames, camera = _capture_pybullet_frames(body)
        try:
            final, contacts, settle = _validate_pybullet_body(body, plane, initial)
        except RuntimeError as error:
            _write_pybullet_failure_evidence(
                output, frames, camera, body, plane, initial, error
            )
            raise
        view_png, view_mp4, decoded = _write_decoded_view(output, frames)
        return {
            "simulator": "PyBullet DIRECT",
            "observation_steps": PHYSICS_OBSERVATION_STEPS,
            "observation_seconds": PHYSICS_OBSERVATION_SECONDS,
            "settle_steps": PHYSICS_SETTLE_WINDOW_STEPS,
            "total_steps": PHYSICS_OBSERVATION_STEPS + PHYSICS_SETTLE_WINDOW_STEPS,
            "capture_interval_steps": PHYSICS_CAPTURE_INTERVAL_STEPS,
            "capture_fps": PHYSICS_CAPTURE_FPS,
            "captured_frames": len(frames),
            "initial_position_m": initial,
            "final_position_m": final,
            "contact_points": len(contacts),
            "settle": settle,
            "camera": camera,
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


def _generate_and_validate(
    output: Path, source: Path
) -> tuple[
    str,
    Path,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    float,
]:
    generated = output / "generated"
    with tempfile.TemporaryDirectory(prefix="npa-embodiedgen-input-") as staging:
        input_path, input_hash = require_input(Path(staging))
        started = time.time()
        run_upstream(source, input_path, generated)
    urdf = locate_urdf(generated)
    collisions = collision_meshes(urdf)
    mjcf = convert_to_mjcf(urdf, output)
    generated_bundle = archive_asset_tree(generated, output / "generated_asset.tar.gz")
    mjcf["bundle"] = archive_asset_tree(output / "mjcf", output / "mjcf_asset.tar.gz")
    physics = pybullet_validation(urdf, output)
    return (
        input_hash,
        urdf,
        collisions,
        mjcf,
        generated_bundle,
        physics,
        time.time() - started,
    )


def _report_base(input_hash: str) -> dict[str, Any]:
    return {
        "schema": "npa.embodiedgen.image-to-rigid-object.v1",
        "status": "success",
        "solution": "embodiedgen",
        "capability": CAPABILITY,
        "capabilities_exercised": CAPABILITIES_EXERCISED,
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
    }


def _report(
    output: Path,
    input_hash: str,
    urdf: Path,
    details: tuple[
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        float,
    ],
) -> dict[str, Any]:
    collisions, mjcf, generated_bundle, physics, elapsed = details
    artifacts = [
        file_record(path, output)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    report = _report_base(input_hash)
    report.update(
        {
            "urdf": {
                "path": str(urdf.relative_to(output)),
                "sha256": sha256(urdf),
                "properties": xml_properties(urdf),
                "physical_property_semantics": "VLM_estimated_not_calibrated_ground_truth",
            },
            "mjcf_conversion": mjcf,
            "generated_asset_bundle": generated_bundle,
            "collision_geometry": collisions,
            "physics": physics,
            "artifacts": artifacts,
            "elapsed_seconds": round(elapsed, 3),
        }
    )
    return report


def main() -> int:
    output = Path(os.environ["NPA_SMOKE_OUTPUT_DIR"]).resolve()
    source = Path(os.environ["NPA_EMBODIEDGEN_SOURCE_ROOT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_hash, urdf, collisions, mjcf, generated_bundle, physics, elapsed = (
        _generate_and_validate(output, source)
    )
    report = _report(
        output,
        input_hash,
        urdf,
        (collisions, mjcf, generated_bundle, physics, elapsed),
    )
    atomic_json(output / "embodiedgen_image_to_rigid_object.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
