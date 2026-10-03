"""Measure reconstructed surfaces against held-out metric depth observations."""

from __future__ import annotations

import numpy as np

from npa.workbench.nurec.navigation_capture import read_images


def _rays(capture: dict, frame: dict, depth: np.ndarray):
    intrinsic = capture["intrinsics"]
    stride = capture["validation"]["pixel_stride"]
    rows, columns = np.mgrid[
        stride // 2 : depth.shape[0] : stride, stride // 2 : depth.shape[1] : stride
    ]
    distances = depth[rows, columns].ravel() / capture["depth_units_per_meter"]
    valid = (distances > 0) & (distances < capture["depth_max_m"])
    optical = np.column_stack(
        (
            (columns.ravel() - intrinsic["cx"]) / intrinsic["fx"],
            (rows.ravel() - intrinsic["cy"]) / intrinsic["fy"],
            np.ones(rows.size),
        )
    )[valid]
    norm = np.linalg.norm(optical, axis=1)
    pose = np.asarray(frame["camera_to_world"])
    directions = (optical / norm[:, None]) @ pose[:3, :3].T
    origins = np.broadcast_to(pose[:3, 3], directions.shape)
    return np.column_stack((origins, directions)).astype(np.float32), distances[
        valid
    ] * norm


def _measure(scene, rays, measured, tolerance):
    import open3d as o3d

    hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    present = np.isfinite(hits)
    errors = np.abs(hits[present] - measured[present])
    inliers = present & (np.abs(hits - measured) <= tolerance)
    return hits, errors, inliers


def _probe(rays, measured, inliers, tolerance) -> dict | None:
    indices = np.flatnonzero(inliers)
    if not len(indices):
        return None
    index = indices[len(indices) // 2]
    return {
        "origin": rays[index, :3].tolist(),
        "direction": rays[index, 3:].tolist(),
        "min_distance": max(0, float(measured[index] - tolerance)),
        "max_distance": float(measured[index] + tolerance),
    }


def validate_depth(root, capture: dict, mesh) -> tuple[dict, list[dict]]:
    """Raycast the real TSDF mesh against excluded depth frames and enforce quality.

    Args:
        root: Capture bundle directory.
        capture: Validated metric capture with explicit quality thresholds.
        mesh: Real Open3D legacy TriangleMesh produced by integration frames.
    Returns:
        Aggregate and per-frame measured errors plus measured-depth PhysX probes.
    Raises:
        ValueError: Held-out surface coverage or error thresholds fail.
        ImportError: Open3D or Pillow is unavailable.
        OSError: A held-out depth image cannot be read.
    """
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return _validate_frames(root, capture, scene)


def _validate_frames(root, capture: dict, scene) -> tuple[dict, list[dict]]:
    results, probes, errors = [], [], []
    quality = capture["validation"]
    for frame in capture["frames"]:
        if frame["split"] != "validation":
            continue
        _, depth = read_images(root, capture, frame)
        rays, measured = _rays(capture, frame, depth)
        hits, deviations, inliers = _measure(
            scene, rays, measured, quality["distance_tolerance_m"]
        )
        results.append(
            {
                "frame_id": frame["id"],
                "observed_rays": len(rays),
                "surface_hits": int(np.isfinite(hits).sum()),
                "inliers": int(inliers.sum()),
            }
        )
        errors.extend(deviations.tolist())
        probe = _probe(rays, measured, inliers, quality["distance_tolerance_m"])
        if probe is not None:
            probes.append(probe)
    report = _quality_report(results, errors, quality)
    if not probes:
        raise ValueError("held-out data produced no measured-depth physics probes")
    return report, probes


def _quality_report(results: list[dict], errors: list, quality: dict) -> dict:
    rays = sum(record["observed_rays"] for record in results)
    hits = sum(record["surface_hits"] for record in results)
    inliers = sum(record["inliers"] for record in results)
    if not rays or not hits:
        raise ValueError(
            "held-out depth validation has no observations or surface hits"
        )
    report = {
        "schema": "npa.navigation.depth_validation.v1",
        "frames": results,
        "observed_rays": rays,
        "surface_hits": hits,
        "inliers": inliers,
        "coverage": hits / rays,
        "inlier_fraction": inliers / rays,
        "mean_absolute_error_m": float(np.mean(errors)),
        "p95_absolute_error_m": float(np.quantile(errors, 0.95)),
        "thresholds": quality,
        "physics_probe_selection": "one median-index depth inlier per held-out frame",
    }
    if (
        report["coverage"] < quality["min_coverage"]
        or report["inlier_fraction"] < quality["min_inlier_fraction"]
        or report["mean_absolute_error_m"] > quality["max_mean_error_m"]
    ):
        raise ValueError(f"held-out depth quality failed: {report}")
    return report
