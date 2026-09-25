"""Verify a prepared navigation scene using real Isaac PhysX mesh ray queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

from npa.workbench.nurec.navigation_assets import (
    contained_file,
    materialize,
    publish,
    sha256,
)
from npa.workbench.nurec.navigation_geometry import validate_probes


def check_hit(hit: dict, probe: dict, collider_paths: set[str]) -> dict:
    """Require a native query hit on an assembled collider within its expected range.

    Args:
        hit: Native PhysX raycast_closest result.
        probe: Validated expected intersection in meters.
        collider_paths: Verified static mesh paths.
    Returns:
        Actual hit geometry and distance, suitable for the evidence report.
    Raises:
        ValueError: Query misses, is nonfinite, hits another body, or misses the interval.
    """
    distance = hit.get("distance")
    collision = hit.get("collision")
    if not hit.get("hit") or collision not in collider_paths:
        raise ValueError("PhysX probe did not hit a prepared collision mesh")
    if type(distance) not in (int, float) or not math.isfinite(distance):
        raise ValueError("PhysX probe returned a nonfinite distance")
    if not probe["min_distance"] <= distance <= probe["max_distance"]:
        raise ValueError(
            "PhysX probe distance is outside the operator's expected interval"
        )
    return {"collision": collision, "distance_m": distance, "expected": probe}


def _load_bundle(root: Path):
    from npa.workbench.nurec.navigation_scene import verify_scene
    from npa.workbench.nurec.navigation_publication import verify_publication

    verify_publication(root)
    scene = contained_file(root, "scene.usdz")
    provenance_bytes = contained_file(root, "provenance.json").read_bytes()
    provenance = json.loads(provenance_bytes)
    if provenance.get("schema") != "npa.nurec.navigation_scene.v1":
        raise ValueError("input requires prepared navigation scene provenance")
    if sha256(scene) != provenance.get("scene_sha256"):
        raise ValueError("scene hash differs from assembly provenance")
    if not provenance.get("colliders"):
        raise ValueError("scene provenance contains no collision meshes")
    validate_probes(provenance.get("ray_probes"))
    verify_scene(scene, provenance["colliders"])
    return scene, provenance, hashlib.sha256(provenance_bytes).hexdigest()


def _declared_image(reference: str) -> dict:
    repository = r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
    registry = r"[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?"
    pattern = rf"{registry}/(?:{repository}/)*{repository}@sha256:[0-9a-f]{{64}}"
    if not re.fullmatch(pattern, reference):
        raise ValueError(
            "--runtime-image requires an immutable registry/repository@sha256 digest; "
            "set --var isaac_image=<registry>/<image>@sha256:<64-hex-digest>"
        )
    return {
        "reference": reference,
        "attestation_scope": "operator-declared workload image reference",
        "running_image_identity_verified": False,
    }


def _physx_version(manager) -> dict:
    extension_id = manager.get_enabled_extension_id("omni.physx")
    if not extension_id:
        return {
            "available": False,
            "reason": "omni.physx extension metadata unavailable",
        }
    metadata = manager.get_extension_dict(extension_id)
    if metadata is not None and hasattr(metadata, "get_dict"):
        metadata = metadata.get_dict()
    version = (metadata or {}).get("package", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        return {"available": False, "reason": "omni.physx package version unavailable"}
    return {"available": True, "extension_id": extension_id, "version": version}


def _runtime_versions() -> dict:
    from isaacsim.core.version import get_version
    import omni.kit.app

    version = get_version()
    if not version or not isinstance(version[0], str) or not version[0].strip():
        raise ValueError("Isaac Sim runtime version is unavailable")
    manager = omni.kit.app.get_app().get_extension_manager()
    return {
        "isaac_sim": {"version": version[0], "version_components": list(version)},
        "physx": _physx_version(manager),
    }


def _query_scene(scene: Path, provenance: dict) -> list[dict]:
    import carb
    import omni.usd
    from omni.physx import get_physx_scene_query_interface
    from isaacsim.core.api import SimulationContext

    context = omni.usd.get_context()
    if not context.open_stage(str(scene)):
        raise ValueError("Isaac could not open the prepared scene")
    if context.get_stage().GetCompositionErrors():
        raise ValueError("Isaac scene has unresolved composition errors")
    simulation = SimulationContext(
        physics_prim_path="/World/Physics", stage_units_in_meters=1.0
    )
    simulation.initialize_physics()
    simulation.play()
    simulation.step(render=False)
    query = get_physx_scene_query_interface()
    colliders = {record["path"] for record in provenance["colliders"]}
    results = []
    for probe in provenance["ray_probes"]:
        hit = query.raycast_closest(
            carb.Float3(*probe["origin"]),
            carb.Float3(*probe["direction"]),
            probe["max_distance"],
        )
        results.append(check_hit(hit, probe, colliders))
    simulation.stop()
    return results


def _verify_and_publish(input_path: str, output_path: str, runtime_image: str) -> dict:
    from pxr import Usd

    declared_image = _declared_image(runtime_image)
    with tempfile.TemporaryDirectory(prefix="npa-navigation-physics-") as temporary:
        work = Path(temporary).resolve()
        root = materialize(input_path, work / "input")
        scene, provenance, provenance_sha256 = _load_bundle(root)
        probes = _query_scene(scene, provenance)
        if sha256(contained_file(root, "provenance.json")) != provenance_sha256:
            raise ValueError("assembly provenance changed during physics verification")
        report = {
            "schema": "npa.nurec.navigation_physics.v1",
            "scene_sha256": provenance["scene_sha256"],
            "assembly_provenance_sha256": provenance_sha256,
            "physics_validated": True,
            "visual_render_validated": False,
            "engine": "Isaac Sim PhysX",
            "usd_version": list(Usd.GetVersion()),
            "runtime_versions": _runtime_versions(),
            "runtime_image": declared_image,
            "probes": probes,
        }
        output = work / "output"
        output.mkdir()
        (output / "physics_validation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        publish(output, output_path)
        return report


def main() -> None:
    """Start Isaac, validate actual scene queries, and publish physics evidence.

    Args:
        None; input/output paths are parsed from the command line.
    Returns:
        None.
    Raises:
        SystemExit: Command-line arguments are invalid.
        ImportError: Not running through the Isaac runtime interpreter.
        ValueError: Image declaration, bundle, runtime metadata, or a query is invalid.
        StorageError: Input download or conditional output publication fails.
        ClientError: The S3 provider rejects an artifact write.
        OSError: Storage fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--runtime-image", required=True)
    args = parser.parse_args()
    _declared_image(args.runtime_image)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    _verify_and_publish(args.input_path, args.output_path, args.runtime_image)
    # Isaac can terminate its interpreter during close; publish only after checks.
    app.close()


if __name__ == "__main__":
    main()
