"""Author registered OpenUSD visual and collision geometry from a reconstructed scan."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from npa.workbench.nurec.navigation_assets import contained_file, sha256
from npa.workbench.nurec.navigation_publication import verify_publication


def _read_surface(root: Path, report: dict):
    path = contained_file(root, "surface.npz")
    if sha256(path) != report.get("surface_sha256"):
        raise ValueError("reconstructed surface hash differs from its report")
    with np.load(path, allow_pickle=False) as archive:
        points, triangles, colors = [
            archive[key] for key in ("points", "triangles", "colors")
        ]
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("surface requires finite XYZ points")
    if (
        colors.shape != points.shape
        or not np.isfinite(colors).all()
        or np.any((colors < 0) | (colors > 1))
    ):
        raise ValueError("surface requires finite RGB colors in [0, 1]")
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or not len(triangles)
        or not np.issubdtype(triangles.dtype, np.integer)
        or triangles.min() < 0
        or triangles.max() >= len(points)
    ):
        raise ValueError("surface triangle indices are invalid")
    if len(points) != report.get("vertices") or len(triangles) != report.get(
        "triangles"
    ):
        raise ValueError("surface counts differ from its reconstruction report")
    return points, triangles, colors


def _author(path: Path, points, triangles, colors) -> None:
    from pxr import Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Capture").GetPrim())
    surface = UsdGeom.Mesh.Define(stage, "/Capture/Surface")
    surface.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    surface.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
    )
    surface.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(triangles.astype(np.int32).ravel())
    )
    surface.CreateSubdivisionSchemeAttr("none")
    surface.CreateDoubleSidedAttr(True)
    surface.CreateDisplayColorPrimvar("vertex").Set(
        Vt.Vec3fArray.FromNumpy(colors.astype(np.float32))
    )
    if not stage.GetRootLayer().Save():
        raise ValueError("reconstructed USD surface could not be saved")


def surface_bundle(root: Path, destination: Path) -> dict:
    """Convert sealed scan surfaces to one shared visual/collision scene input.

    Args:
        root: Sealed output directory from the metric RGB-D reconstruction stage.
        destination: Fresh directory for the normal navigation scene assembler.
    Returns:
        Verified reconstruction report with actual geometry and depth evidence.
    Raises:
        ValueError: Publication, source hashes, units, or geometry are invalid.
        OSError: Surface or manifest cannot be read or written.
        ImportError: OpenUSD is unavailable.
    """
    verify_publication(root)
    report = json.loads(contained_file(root, "reconstruction.json").read_text())
    if report.get("schema") != "npa.navigation.rgbd_reconstruction.v1":
        raise ValueError("unsupported metric reconstruction schema")
    if report.get("world") != {"meters_per_unit": 1, "up_axis": "Z"}:
        raise ValueError("reconstruction must use metric Z-up world coordinates")
    if sha256(contained_file(root, "capture.json")) != report.get(
        "capture_manifest_sha256"
    ):
        raise ValueError(
            "source capture manifest differs from reconstruction provenance"
        )
    _write_bundle(root, destination, report)
    return report


def _write_bundle(root: Path, destination: Path, report: dict) -> None:
    points, triangles, colors = _read_surface(root, report)
    destination.mkdir()
    _author(destination / "surface.usdc", points, triangles, colors)
    contract = {
        "schema": "npa.nurec.navigation_input.v1",
        "visual_asset": "surface.usdc",
        "collision_asset": "surface.usdc",
        "visual_to_world": np.eye(4).tolist(),
        "collision_to_world": np.eye(4).tolist(),
        "ray_probes": report["ray_probes"],
        "capture_provenance": {"sha256": report["capture_manifest_sha256"]},
    }
    (destination / "scene.json").write_text(json.dumps(contract, indent=2) + "\n")
    for name in ("capture.json", "reconstruction.json"):
        shutil.copyfile(root / name, destination / name)
