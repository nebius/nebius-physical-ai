"""Fuse calibrated metric RGB-D scans into measured colored triangle surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from npa.workbench.nurec.navigation_assets import materialize, publish, sha256
from npa.workbench.nurec.navigation_capture import read_capture, read_images
from npa.workbench.nurec.navigation_depth_validation import validate_depth


def _integrate(root: Path, capture: dict):
    import open3d as o3d

    calibration = capture["intrinsics"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        *[calibration[key] for key in ("width", "height", "fx", "fy", "cx", "cy")]
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=capture["voxel_size_m"],
        sdf_trunc=capture["sdf_trunc_m"],
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for frame in capture["frames"]:
        if frame["split"] != "integration":
            continue
        rgb, depth = read_images(root, capture, frame)
        observation = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_scale=capture["depth_units_per_meter"],
            depth_trunc=capture["depth_max_m"],
            convert_rgb_to_intensity=False,
        )
        volume.integrate(
            observation, intrinsic, np.linalg.inv(frame["camera_to_world"])
        )
    return _clean_mesh(volume.extract_triangle_mesh())


def _clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise ValueError(
            "metric scan reconstruction produced no colored triangle surface"
        )
    points, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    vertices = points[triangles]
    areas = np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=1,
    )
    if not np.isfinite(points).all() or np.any(areas <= 0):
        raise ValueError("reconstructed surface is nonfinite or degenerate")
    return mesh


def _save_surface(output: Path, mesh) -> None:
    points = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    np.savez_compressed(
        output / "surface.npz", points=points, triangles=triangles, colors=colors
    )


def _report(root, output, capture, mesh, validation, probes, elapsed) -> dict:
    import open3d as o3d

    points = np.asarray(mesh.vertices)
    return {
        "schema": "npa.navigation.rgbd_reconstruction.v1",
        "engine": "open3d.pipelines.integration.ScalableTSDFVolume",
        "open3d_version": o3d.__version__,
        "device": "CPU",
        "capture_manifest_sha256": sha256(root / "capture.json"),
        "surface_sha256": sha256(output / "surface.npz"),
        "integration_frames": sum(
            frame["split"] == "integration" for frame in capture["frames"]
        ),
        "validation_frames": len(validation["frames"]),
        "vertices": len(points),
        "triangles": len(mesh.triangles),
        "bounds_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "world": capture["world"],
        "elapsed_seconds": elapsed,
        "voxel_size_m": capture["voxel_size_m"],
        "sdf_trunc_m": capture["sdf_trunc_m"],
        "depth_validation": validation,
        "ray_probes": probes,
        "physics_validated": False,
        "visual_render_validated": False,
        "geometry_scope": "observed surfaces only; unseen space remains unknown",
    }


def reconstruct_capture(input_path: str, output_path: str) -> dict:
    """Reconstruct and immutably publish actual scan geometry and held-out evidence.

    Args:
        input_path: Local directory or S3 prefix with calibrated capture.json.
        output_path: Fresh local directory or S3 prefix for reconstructed surfaces.
    Returns:
        Measured surface geometry, engine identity, and held-out depth errors.
    Raises:
        ValueError: Capture, reconstruction, or held-out quality fails.
        ImportError: Open3D or Pillow is unavailable.
        OSError: Local input or output cannot be read or written.
        StorageError: S3 input or immutable publication fails.
    """
    with tempfile.TemporaryDirectory(prefix="npa-rgbd-reconstruct-") as temporary:
        work = Path(temporary)
        root = materialize(input_path, work / "input")
        capture = read_capture(root)
        output = work / "output"
        output.mkdir()
        started = time.monotonic()
        mesh = _integrate(root, capture)
        validation, probes = validate_depth(root, capture, mesh)
        _save_surface(output, mesh)
        report = _report(
            root, output, capture, mesh, validation, probes, time.monotonic() - started
        )
        shutil.copyfile(root / "capture.json", output / "capture.json")
        (output / "reconstruction.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        publish(output, output_path)
        return report


def main() -> None:
    """Run the metric scan reconstruction stage from workflow arguments.

    Args:
        None; arguments come from the command line.
    Returns:
        None.
    Raises:
        SystemExit: Command-line arguments are invalid.
        ValueError: Capture or reconstruction fails validation.
        ImportError: Open3D or Pillow is unavailable.
        OSError: Local input or output fails.
        StorageError: S3 input or publication fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    reconstruct_capture(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
