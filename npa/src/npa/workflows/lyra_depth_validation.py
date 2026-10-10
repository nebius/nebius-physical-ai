"""Compare Lyra collision candidates with metric depth never supplied to inference."""

from __future__ import annotations

import numpy as np

from npa.workbench.nurec.navigation_depth_validation import _rays, _measure


def measured_depth_report(root, capture: dict, mesh) -> dict | None:
    """Measure surface accuracy independently of predicted depth and appearance.

    Args:
        root: Reconstruction directory with optional measured-depth.npz.
        capture: Metric capture manifest and its preselected quality thresholds.
        mesh: Reconstructed Open3D triangle mesh in calibrated world coordinates.
    Returns:
        Explicit passing or failing measurements; None without measured depth.
    Raises:
        ValueError: Depth observations or their frame indices are malformed.
        OSError: Measured depth cannot be read.
        ImportError: Open3D is unavailable.
    """
    import open3d as o3d

    path = root / "measured-depth.npz"
    if not path.is_file():
        return None
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    results, errors = [], []
    with np.load(path, allow_pickle=False) as observations:
        for index, depth in zip(
            observations["indices"], observations["depth"], strict=True
        ):
            frame = capture["frames"][int(index)]
            rays, distances = _rays(capture, frame, depth)
            hits, deviations, inliers = _measure(
                scene, rays, distances, capture["validation"]["distance_tolerance_m"]
            )
            results.append(
                (len(rays), int(np.isfinite(hits).sum()), int(inliers.sum()))
            )
            errors.extend(deviations.tolist())
    return _report(results, errors, capture["validation"])


def _report(results, errors, thresholds):
    rays, hits, inliers = np.sum(results, axis=0).astype(int).tolist()
    coverage = hits / rays if rays else 0
    inlier_fraction = inliers / rays if rays else 0
    mean_error = float(np.mean(errors)) if errors else None
    passed = bool(
        rays
        and hits
        and coverage >= thresholds["min_coverage"]
        and inlier_fraction >= thresholds["min_inlier_fraction"]
        and mean_error <= thresholds["max_mean_error_m"]
    )
    return {
        "schema": "npa.lyra-measured-depth.v1",
        "scope": "independent measured depth; RGB views are not held out",
        "frames": len(results),
        "observed_rays": rays,
        "surface_hits": hits,
        "inliers": inliers,
        "coverage": coverage,
        "inlier_fraction": inlier_fraction,
        "mean_absolute_error_m": mean_error,
        "p95_absolute_error_m": float(np.quantile(errors, 0.95)) if errors else None,
        "thresholds": thresholds,
        "passed": passed,
        "native_physics_validated": False,
    }
