"""Measure whether each Rerun view's camera frames the geometry that view shows.

The operator's complaint was a clipped scene. The product solves its camera
distance rather than guessing it, so the interesting question is not "is there a
fit" but "what is it fitted against". This projects the geometry of every view
through that view's own camera and reports the worst point in each, in units of
the frame half-extent: 1.0 is exactly the edge, above 1.0 is off-screen.

Reads the delivered PLY bytes. No native Open3D calls beyond the readers, no
network, no Rerun.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _load_points(path: Path) -> np.ndarray:
    import open3d as o3d

    head = path.read_bytes()[:2048].decode("latin-1")
    if "element face" in head and "\nelement face 0\n" not in head:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) > 0:
            return np.asarray(mesh.vertices)
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def worst_reach(points: np.ndarray, camera: dict, aspect: float) -> dict:
    """Largest frame-relative coordinate over the points. >1.0 means clipped."""

    eye = np.asarray(camera["eye"], float)
    target = np.asarray(camera["look_target"], float)
    up = np.asarray(camera["up"], float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    half = np.tan(np.radians(camera["fov_degrees"]) / 2.0)

    local = points - eye
    depth = local @ forward
    ahead = depth > 1e-6
    lateral = (local @ right)[ahead] / (half * aspect * depth[ahead])
    vertical = (local @ true_up)[ahead] / (half * depth[ahead])
    return {
        "points": int(len(points)),
        "points_behind_camera": int((~ahead).sum()),
        "worst_horizontal_reach": float(np.abs(lateral).max()),
        "worst_vertical_reach": float(np.abs(vertical).max()),
        "fraction_outside_frame": float(
            ((np.abs(lateral) > 1.0) | (np.abs(vertical) > 1.0)).mean()
        ),
    }


def main() -> int:
    artifacts = Path(sys.argv[1])
    out = Path(sys.argv[2])

    sys.path.insert(0, str(Path(sys.argv[3]) / "src"))
    from npa.workbench.open3d.runner import (
        CAMERA_ASPECT,
        REMOVED_VIEW,
        SCAN_VIEW,
        SCENE_VIEW,
        SURFACE_VIEW,
        VIEW_GEOMETRY,
        _camera,
        _view_cameras,
    )

    fused = _load_points(artifacts / "fused.ply")
    mesh = _load_points(artifacts / "mesh.ply")
    uncropped_name = "mesh_uncropped.ply"
    uncropped = artifacts / uncropped_name
    unsupported = _load_points(uncropped) if uncropped.is_file() else None

    class _Cloud:
        """Only what _camera touches, so this needs no real point cloud."""

        def __init__(self, pts):
            self.points = pts

        def get_axis_aligned_bounding_box(self):
            pts = np.asarray(self.points)

            class _Box:
                def get_center(self):
                    return (pts.min(axis=0) + pts.max(axis=0)) / 2.0

                def get_extent(self):
                    return pts.max(axis=0) - pts.min(axis=0)

            return _Box()

    up = [0.0, 1.0, 0.0]
    geometry = {"fused": fused, "mesh": mesh, "unsupported": unsupported}

    # Before: one camera fitted to the fused cloud, shared by every view.
    shared = _camera(fused, up)
    # After: one camera per view, fitted to that view's own contents.
    per_view = _view_cameras(geometry, up)

    shown = {
        name: np.vstack(
            [geometry[key] for key in keys if geometry.get(key) is not None]
        )
        for name, keys in VIEW_GEOMETRY.items()
        if geometry.get(keys[0]) is not None
    }

    def measure(cameras):
        rows = {
            name: worst_reach(pts, cameras[name], CAMERA_ASPECT)
            for name, pts in shown.items()
            if name in cameras
        }
        return {
            "views": rows,
            "clipped_views": sorted(
                name
                for name, row in rows.items()
                if max(row["worst_horizontal_reach"], row["worst_vertical_reach"]) > 1.0
            ),
        }

    report = {
        "what": (
            "Projects each view's own geometry through the camera that view uses, "
            "in frame half-extents: 1.0 is exactly the edge and above 1.0 is off "
            "screen. 'before' shares one camera fitted to the fused cloud across "
            "all views, which is what shipped; 'after' fits one camera per view."
        ),
        "aspect": CAMERA_ASPECT,
        "view_names": {
            "scene": SCENE_VIEW,
            "scan": SCAN_VIEW,
            "surface": SURFACE_VIEW,
            "removed": REMOVED_VIEW,
        },
        "before": {
            "camera_fitted_against": "fused cloud only, shared by every view",
            "camera": shared,
            **measure(dict.fromkeys(shown, shared)),
        },
        "after": {
            "camera_fitted_against": "each view's own contents",
            "cameras": per_view,
            **measure(per_view),
        },
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    for phase in ("before", "after"):
        print(f"--- {phase}")
        for name, row in report[phase]["views"].items():
            worst = max(row["worst_horizontal_reach"], row["worst_vertical_reach"])
            print(
                f"  {worst:6.3f}  {row['fraction_outside_frame'] * 100:5.2f}% outside"
                f"  {name}"
            )
        print("  clipped:", report[phase]["clipped_views"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
