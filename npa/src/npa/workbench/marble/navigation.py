"""Prepare verified Marble geometry for the existing native Isaac navigation trainer."""

import hashlib
from pathlib import Path
import tempfile

from npa.workflows.navigation.artifacts import write_json
from npa.workflows.navigation.reference_bundle import build_bundle
from npa.workflows.navigation.stages import prepare

from .navigation_cases import build_cases
from .navigation_geometry import collision_geometry, write_scene
from .runtime import _materialize, _paths


def navigation_prepare(request):
    """Seal a real Marble collider and measured cases as native navigation inputs.

    Args: NavigationRequest with world/output paths and exact native training configuration.
    Returns: Native preparation summary plus source lineage and measured case counts.
    Raises: ValueError or MarbleError for invalid geometry, cases, runtime identity, or storage.
    """
    _paths(request)
    with tempfile.TemporaryDirectory(prefix="npa-marble-navigation-") as directory:
        work = Path(directory)
        source = work / "world"
        source.mkdir()
        world = _materialize(request.input_path, "world.json", source)
        lineage = _lineage(source, world, request)
        bundle = _native_bundle(work, source, world, request, lineage)
        result = prepare(str(bundle), request.output_path, request.image)
    return {
        **result,
        "run_id": request.run_id,
        "marble_source": lineage,
        "train_cases": request.num_envs,
        "eval_cases": request.num_envs,
    }


def _native_bundle(work, source, world, request, lineage):
    vertices, faces = collision_geometry(source, world)
    cases, support = build_cases(
        vertices,
        faces,
        count=request.num_envs,
        seed=request.seed,
        radius=request.sampling_radius_m,
    )
    write_scene(work / "scene.usdz", vertices, faces, lineage)
    write_json(work / "cases.json", cases)
    bundle = work / "bundle"
    build_bundle(
        bundle,
        image=request.image,
        iterations=request.iterations,
        episode_steps=request.episode_steps,
        num_envs=request.num_envs,
        scene_file=work / "scene.usdz",
        cases_file=work / "cases.json",
    )
    write_json(bundle / "marble-source.json", lineage)
    write_json(bundle / "marble-support.json", support)
    return bundle


def _lineage(source, world, request):
    return {
        "schema": "npa.marble.navigation_source.v1",
        "run_id": request.run_id,
        "world_manifest_sha256": hashlib.sha256(
            (source / "world.json").read_bytes()
        ).hexdigest(),
        "collider_sha256": world["files"]["collider.glb"]["sha256"],
        "source_kind": "world-api",
        "mesh_transform": world["mesh_transform"],
        "navigation_from_world": "[world.x, -world.z, world.y]",
        "units": "provider-estimated meters; not surveyed site calibration",
        "generated_in_training_run": False,
        "robot": "ANYmal-C",
        "sensor_mode": "static_raycast",
    }
