"""Package reconstructed triangles in an explicitly calibrated robot task frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from npa.workflows.lerobot_transfer_data import file_sha256, tree_hashes, write_json
from npa.workflows.physical_augmentation_contract import make_recipe, read_recipe


def _author(points, triangles, colors, destination):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Scene").GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Scene/ReconstructedSurface")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorPrimvar("vertex").Set(
        Vt.Vec3fArray.FromNumpy(colors.astype(np.float32))
    )
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    bounds = np.array([points.min(0), points.max(0)])
    mesh.CreateExtentAttr([Gf.Vec3f(*row) for row in bounds])
    if not stage.GetRootLayer().Save():
        raise ValueError("Reconstructed USD scene could not be saved")


def _task_transform(binding):
    transform = np.asarray(binding["world_to_task"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("world_to_task must be a finite 4x4 rigid transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1
    ):
        raise ValueError("world_to_task rotation must be proper and orthonormal")
    if not np.array_equal(transform[3], [0, 0, 0, 1]):
        raise ValueError("world_to_task requires a homogeneous final row")
    return transform


def _recipe(args, output, geometry):
    recipe = make_recipe(args.run_id, 42, 3, 600)
    recipe["presentation"]["scene"] = "lyra-reconstructed-surface-v1"
    recipe["presentation"].update(wrist_width=320, wrist_height=240)
    recipe["scene_binding"] = {
        "schema": "npa.lyra-scene-binding.v1",
        "asset": "scene.usdc",
        "asset_sha256": file_sha256(output / "scene.usdc"),
        "geometry_sha256": geometry["surface_sha256"],
        "robot": "fixed-base Franka; operator-calibrated mounting pose",
        "object": "inserted 5 cm rigid cube; mass and friction assigned explicitly",
        "visual": "colored mesh fused from Lyra predicted depth and source RGB",
        "collision": "same reconstructed triangles; unseen regions remain empty",
    }
    write_json(output / "recipe.json", recipe)
    read_recipe(output / "recipe.json")
    write_json(output / "checksums.json", tree_hashes(output))


def main():
    """Assemble a portable scene with an explicit calibrated mounting transform.

    Args:
        None; reads the command line.
    Returns:
        None. Writes a self-contained scene and sealed physical augmentation recipe.
    Raises:
        ValueError: Geometry identity or the mounting transform is invalid.
        OSError: Scene inputs or outputs cannot be accessed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--binding-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    geometry = json.loads((args.input_path / "geometry.json").read_text())
    quality = geometry.get("measured_depth_validation")
    if quality is not None and quality["passed"] is not True:
        raise ValueError("Measured depth rejected this collision candidate")
    if file_sha256(args.input_path / "surface.npz") != geometry["surface_sha256"]:
        raise ValueError("Reconstructed surface no longer matches its provenance")
    transform = _task_transform(json.loads(args.binding_path.read_text()))
    with np.load(args.input_path / "surface.npz", allow_pickle=False) as surface:
        points = surface["points"] @ transform[:3, :3].T + transform[:3, 3]
        args.output_path.mkdir(parents=True, exist_ok=False)
        _author(
            points,
            surface["triangles"],
            surface["colors"],
            args.output_path / "scene.usdc",
        )
    shutil.copy2(args.binding_path, args.output_path / "mounting.json")
    shutil.copy2(args.input_path / "geometry.json", args.output_path / "geometry.json")
    _recipe(args, args.output_path, geometry)


if __name__ == "__main__":
    main()
