"""Compose supplied USD or measured RGB-D surfaces into portable navigation scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from npa.workbench.nurec.navigation_assets import (
    audit_asset,
    contained_file,
    materialize,
    publish,
    sha256,
)
from npa.workbench.nurec.navigation_geometry import (
    author_colliders,
    open_static_stage,
    rigid_transform,
    validate_probes,
)


def _read_contract(root: Path) -> dict:
    contract = json.loads(contained_file(root, "scene.json").read_text())
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != "npa.nurec.navigation_input.v1"
    ):
        raise ValueError("scene.json requires schema npa.nurec.navigation_input.v1")
    capture = contract.get("capture_provenance", {})
    if not isinstance(capture, dict) or not re.fullmatch(
        "[0-9a-f]{64}", str(capture.get("sha256", ""))
    ):
        raise ValueError(
            "capture_provenance.sha256 requires the source capture manifest digest"
        )
    for name in ("visual_to_world", "collision_to_world"):
        rigid_transform(contract.get(name))
    validate_probes(contract.get("ray_probes"))
    return contract


def _source(root: Path, contract: dict, role: str):
    path = contained_file(root, contract.get(f"{role}_asset", ""))
    if path.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError(f"{role}_asset must be a USD or USDZ asset")
    dependencies = audit_asset(path, root)
    stage = open_static_stage(path)
    if role == "visual":
        _require_visual_content(stage)
    return path, stage, dependencies


def _require_visual_content(stage) -> None:
    from pxr import Sdf, Usd, UsdGeom

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Gprim) and prim.GetTypeName() != "Volume":
            points = prim.GetAttribute("points")
            if not points or points.Get():
                return
        if prim.GetTypeName() == "Volume" and any(
            attr.GetTypeName()
            in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray)
            and attr.Get()
            for child in Usd.PrimRange(prim)
            for attr in child.GetAttributes()
        ):
            return
    raise ValueError("visual scene contains no geometry or neural Volume")


def _compose(visual, collision, contract: dict, path: Path) -> list[dict]:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    parent = UsdGeom.Xform.Define(stage, "/World/Visual")
    scale = np.eye(4)
    scale[:3, :3] *= UsdGeom.GetStageMetersPerUnit(visual)
    matrix = scale @ rigid_transform(contract["visual_to_world"])
    parent.AddTransformOp().Set(Gf.Matrix4d(matrix.tolist()))
    source = stage.DefinePrim("/World/Visual/Source")
    source.GetReferences().AddReference(visual.GetRootLayer().identifier)
    UsdGeom.Xform.Define(stage, "/World/Collision")
    colliders = author_colliders(
        collision, stage, rigid_transform(contract["collision_to_world"])
    )
    physics = UsdPhysics.Scene.Define(stage, "/World/Physics")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    if stage.GetCompositionErrors() or not stage.GetRootLayer().Save():
        raise ValueError("assembled scene could not be composed and saved")
    return colliders


def _package(source: Path, output: Path) -> None:
    from pxr import Sdf, UsdUtils

    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source)), str(output)):
        raise ValueError("OpenUSD failed to package the complete scene")
    audit_asset(output, output.parent)


def verify_scene(path: Path, expected: list[dict]) -> None:
    """Reopen a portable USDZ and verify registered collider geometry and schemas.

    Args:
        path: Packaged scene file.
        expected: Collider records from assembly.
    Returns:
        None.
    Raises:
        ValueError: Export loses dependencies, colliders, transforms, or topology.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    audit_asset(path, path.parent)
    stage = Usd.Stage.Open(str(path))
    if not stage or stage.GetCompositionErrors():
        raise ValueError("packaged scene has composition errors")
    if (
        UsdGeom.GetStageMetersPerUnit(stage) != 1
        or UsdGeom.GetStageUpAxis(stage) != "Z"
    ):
        raise ValueError("packaged scene must use meters and Z up")
    colliders = [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if {str(prim.GetPath()) for prim in colliders} != {
        item["path"] for item in expected
    }:
        raise ValueError("packaged collider inventory differs from assembly")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("navigation colliders must be static without rigid bodies")
    for item in expected:
        _verify_collider(stage, item)


def _verify_collider(stage, item: dict) -> None:
    from pxr import UsdGeom, UsdPhysics
    from npa.workbench.nurec.navigation_geometry import _geometry_sha256, _triangles

    prim = stage.GetPrimAtPath(item["path"])
    if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
        raise ValueError("packaged collider is disabled")
    if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "none":
        raise ValueError("packaged collider must preserve exact triangle geometry")
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    if not np.array_equal(matrix, np.eye(4)):
        raise ValueError("packaged collider must retain its baked world coordinates")
    points, triangles = _triangles(UsdGeom.Mesh(prim))
    if _geometry_sha256(points, triangles) != item["geometry_sha256"]:
        raise ValueError("packaged mesh differs from registered geometry")
    bounds = [points.min(axis=0), points.max(axis=0)]
    if len(triangles) != item["triangles"] or not np.allclose(bounds, item["bounds_m"]):
        raise ValueError("packaged geometry differs from registered source")


def _build(root: Path, output: Path, work: Path) -> dict:
    contract = _read_contract(root)
    visual_path, visual, visual_files = _source(root, contract, "visual")
    collision_path, collision, collision_files = _source(root, contract, "collision")
    colliders = _compose(visual, collision, contract, work / "assembled.usda")
    scene = output / "scene.usdz"
    _package(work / "assembled.usda", scene)
    verify_scene(scene, colliders)
    report = _provenance(
        root,
        contract,
        scene,
        colliders,
        {
            "visual": (visual_path, visual, visual_files),
            "collision": (collision_path, collision, collision_files),
        },
    )
    _reconstruction_evidence(root, output, report)
    (output / "provenance.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def _reconstruction_evidence(root: Path, output: Path, report: dict) -> None:
    if not (root / "reconstruction.json").is_file():
        return
    for name in ("capture.json", "reconstruction.json"):
        shutil.copyfile(root / name, output / name)
    report["reconstruction_report_sha256"] = sha256(output / "reconstruction.json")
    report["reconstruction_engine"] = "open3d.pipelines.integration.ScalableTSDFVolume"


def _provenance(root, contract, scene, colliders, sources) -> dict:
    from pxr import Usd, UsdGeom

    return {
        "schema": "npa.nurec.navigation_scene.v1",
        "scene_sha256": sha256(scene),
        "colliders": colliders,
        "collider_count": len(colliders),
        "triangle_count": sum(item["triangles"] for item in colliders),
        "capture_manifest_sha256": contract["capture_provenance"]["sha256"],
        "input_contract_sha256": sha256(root / "scene.json"),
        "inputs": {role: sha256(value[0]) for role, value in sources.items()},
        "dependency_sha256": sorted(
            {sha256(path) for value in sources.values() for path in value[2]}
        ),
        "registration": {
            key: contract[key] for key in ("visual_to_world", "collision_to_world")
        },
        "source_units": {
            role: UsdGeom.GetStageMetersPerUnit(value[1])
            for role, value in sources.items()
        },
        "source_up_axis": {
            role: str(UsdGeom.GetStageUpAxis(value[1]))
            for role, value in sources.items()
        },
        "world": {"meters_per_unit": 1, "up_axis": "Z"},
        "ray_probes": contract["ray_probes"],
        "usd_version": list(Usd.GetVersion()),
        "physics_validated": False,
        "visual_render_validated": False,
    }


def prepare_scene(input_path: str, output_path: str) -> dict:
    """Assemble a portable scene from supplied USD or sealed metric scan surfaces.

    Args:
        input_path: Directory or S3 prefix containing USD inputs or reconstructed surfaces.
        output_path: New local directory or run-scoped S3 output prefix.
    Returns:
        Measured geometry and provenance; native physics remains unverified.
    Raises:
        ValueError: Contract, geometry, dependencies, or export is invalid.
        ImportError: OpenUSD runtime is unavailable.
        StorageError: Input download or conditional output publication fails.
        ClientError: The S3 provider rejects an artifact write.
        OSError: Input, output, or temporary storage fails.
    """
    with tempfile.TemporaryDirectory(prefix="npa-navigation-") as temporary:
        work = Path(temporary).resolve()
        root = materialize(input_path, work / "input")
        if (root / "reconstruction.json").is_file():
            from npa.workbench.nurec.navigation_surface import surface_bundle

            surface_bundle(root, work / "surface")
            root = work / "surface"
        output = work / "output"
        output.mkdir()
        report = _build(root, output, work)
        publish(output, output_path)
        return report


def main() -> None:
    """Execute the stateless scene adapter for a workflow stage.

    Args:
        None; arguments are parsed from the command line.
    Returns:
        None.
    Raises:
        SystemExit: Command-line arguments are invalid.
        ValueError: Scene input validation fails.
        ImportError: OpenUSD runtime is unavailable.
        StorageError: Input download or conditional output publication fails.
        ClientError: The S3 provider rejects an artifact write.
        OSError: Input, output, or temporary storage fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    prepare_scene(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
