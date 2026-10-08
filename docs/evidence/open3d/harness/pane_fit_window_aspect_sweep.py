"""Where does the pane fit start clipping as the real window shape moves?"""
import json, numpy as np, open3d as o3d
from npa.workbench.open3d.runner import (
    _camera, CAMERA_FOV_DEGREES, CAMERA_WINDOW_ASPECT,
    CAMERA_SCENE_SHARE, CAMERA_TAB_SHARE,
)

fused = np.asarray(o3d.io.read_point_cloud("/a/fused.ply").points)
mesh = o3d.io.read_triangle_mesh("/a/mesh.ply")
verts = np.asarray(mesh.vertices)
scene = np.vstack([fused, verts])
UP = [0.0, 0.0, 1.0]

def reach(points, camera, aspect):
    """Largest normalized frame coordinate: >1 means the geometry leaves the pane."""
    eye = np.asarray(camera["eye"]); centre = np.asarray(camera["look_target"])
    up = np.asarray(camera["up"])
    forward = centre - eye; forward /= np.linalg.norm(forward)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rel = np.asarray(points) - eye
    depth = np.maximum(rel @ forward, 1e-6)
    ha = np.tan(np.radians(CAMERA_FOV_DEGREES) / 2.0)
    return float(np.abs(np.stack([(rel @ right) / ha / aspect, (rel @ true_up) / ha]) / depth).max())

# Panes as shipped: the fit assumes CAMERA_WINDOW_ASPECT.
panes = {"scene pane (2/3 width)": (scene, CAMERA_SCENE_SHARE),
         "tab pane (1/3 width)": (verts, CAMERA_TAB_SHARE)}
rows = []
for label, (pts, share) in panes.items():
    fitted_boundary = None
    fitted = _camera(pts, UP, aspect=CAMERA_WINDOW_ASPECT * share)
    for name, wa in [("4:3 (1.333)", 4/3), ("3:2 (1.500)", 1.5),
                     ("16:10 assumed (1.600)", 16/10), ("16:9 (1.778)", 16/9),
                     ("ultrawide 21:9 (2.333)", 21/9), ("tall/split 1:1", 1.0)]:
        r = reach(pts, fitted, wa * share)
        rows.append({"pane": label, "window_aspect": name,
                     "frame_half_extent": round(r, 4),
                     "clipped": bool(r > 1.0),
                     "margin_percent": round((1.0 - r) * 100, 1)})
    at_fit = reach(pts, fitted, CAMERA_WINDOW_ASPECT * share)
    rows.append({"pane": label, "window_aspect": "CLIPPING BOUNDARY",
                 "frame_half_extent": 1.0, "clipped": False,
                 "narrowest_safe_window_aspect": round(at_fit * CAMERA_WINDOW_ASPECT, 4)})
print(json.dumps(rows, indent=2))
