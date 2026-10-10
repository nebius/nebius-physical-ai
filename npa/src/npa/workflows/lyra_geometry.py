"""Calibrate Lyra predictions and extract collision surfaces without inventing hidden geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from npa.workflows.lerobot_transfer_data import file_sha256, tree_hashes, write_json
from npa.workflows.lyra_depth_validation import measured_depth_report


def align_camera_centers(predicted: np.ndarray, measured: np.ndarray):
    """Fit one similarity transform from predicted to calibrated camera centers.

    Args:
        predicted: Nx3 camera centers in reconstruction coordinates.
        measured: Matching Nx3 camera centers in metric world coordinates.
    Returns:
        Positive scale, proper rotation, translation and per-camera errors.
    Raises:
        ValueError: Shapes, values or motion cannot determine a transform.
    """
    if (
        predicted.shape != measured.shape
        or predicted.ndim != 2
        or predicted.shape[1] != 3
    ):
        raise ValueError("Calibration requires matching Nx3 camera centers")
    if len(predicted) < 4 or not np.isfinite([predicted, measured]).all():
        raise ValueError("Calibration requires at least four finite camera centers")
    left, right = predicted - predicted.mean(0), measured - measured.mean(0)
    if np.linalg.matrix_rank(left, tol=1e-4) < 2:
        raise ValueError("Capture motion is too degenerate to calibrate the scene")
    u, singular, vt = np.linalg.svd(right.T @ left)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.sum(singular * signs) / np.sum(left**2))
    if scale <= 0:
        raise ValueError("Scene calibration requires a positive scale")
    translation = measured.mean(0) - scale * rotation @ predicted.mean(0)
    residual = np.linalg.norm(
        scale * predicted @ rotation.T + translation - measured, axis=1
    )
    return scale, rotation, translation, residual


def _cameras(geometry, capture, indices):
    if "frames" not in capture:
        raise ValueError(
            "Metric manipulation requires calibrated camera poses; an uncalibrated video cannot establish collision scale"
        )
    extrinsics = geometry["extrinsics"]
    if extrinsics.shape[-2:] == (3, 4):
        row = np.broadcast_to([0, 0, 0, 1], (len(extrinsics), 1, 4))
        extrinsics = np.concatenate([extrinsics, row], axis=1)
    predicted = np.linalg.inv(extrinsics)
    measured = np.array([capture["frames"][int(i)]["camera_to_world"] for i in indices])
    scale, rotation, translation, errors = align_camera_centers(
        predicted[:, :3, 3], measured[:, :3, 3]
    )
    if np.sqrt(np.mean(errors**2)) > 0.10:
        raise ValueError(
            "Camera calibration exceeds 10 cm RMS; scene is not ready for manipulation"
        )
    transformed = predicted.copy()
    transformed[:, :3, :3] = rotation @ predicted[:, :3, :3]
    transformed[:, :3, 3] = scale * predicted[:, :3, 3] @ rotation.T + translation
    record = {
        "scale_to_meters": scale,
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "camera_rms_m": float(np.sqrt(np.mean(errors**2))),
        "camera_max_m": float(errors.max()),
        "calibration_cameras": len(indices),
        "method": "similarity fit to independently calibrated camera centers",
    }
    return scale, transformed, record


def _fuse(geometry, scale, cameras):
    import open3d as o3d

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.0075,
        sdf_trunc=0.03,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i, camera in enumerate(cameras):
        depth = geometry["depth"][i].astype(np.float32) * scale
        confidence = geometry["conf"][i]
        depth[confidence < np.percentile(confidence, 20)] = 0
        color = np.asarray(geometry["processed_images"][i], dtype=np.uint8)
        height, width = depth.shape
        intrinsic = geometry["intrinsics"][i]
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, pinhole, np.linalg.inv(camera))
    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_vertices().remove_duplicated_triangles()
    mesh.remove_degenerate_triangles().remove_unreferenced_vertices()
    if not mesh.has_triangles() or not mesh.has_vertex_colors():
        raise ValueError("Lyra depth did not produce a colored collision surface")
    return mesh


def _write_mesh(mesh, output):
    import open3d as o3d

    o3d.io.write_triangle_mesh(str(output / "collision.ply"), mesh)
    np.savez_compressed(
        output / "surface.npz",
        points=np.asarray(mesh.vertices),
        triangles=np.asarray(mesh.triangles),
        colors=np.asarray(mesh.vertex_colors),
    )


def _publish_geometry(root, output, mesh, capture, calibration):
    _write_mesh(mesh, output)
    write_json(
        output / "geometry.json",
        {
            "schema": "npa.lyra-collision-candidate.v1",
            "calibration": calibration,
            "source_geometry_sha256": file_sha256(root / "geometry.npz"),
            "surface_sha256": file_sha256(output / "surface.npz"),
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.triangles),
            "physics_validated": False,
            "geometry_scope": "estimated observed surfaces only",
            "measured_depth_validation": measured_depth_report(root, capture, mesh),
        },
    )
    write_json(output / "checksums.json", tree_hashes(output))


def main():
    """Extract metric collision candidates from real Lyra predictions.

    Args:
        None; reads the command line.
    Returns:
        None. Writes the surface and calibration evidence.
    Raises:
        ValueError: Calibration or reconstructed geometry is invalid.
        OSError: An artifact cannot be accessed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.input_path, args.output_path
    output.mkdir(parents=True, exist_ok=False)
    geometry = np.load(root / "geometry.npz", allow_pickle=False)
    capture = json.loads((root / "capture.json").read_text())
    indices = np.load(root / "cameras.npz", allow_pickle=False)["indices_da3"]
    scale, cameras, calibration = _cameras(geometry, capture, indices)
    mesh = _fuse(geometry, scale, cameras)
    _publish_geometry(root, output, mesh, capture, calibration)


if __name__ == "__main__":
    main()
