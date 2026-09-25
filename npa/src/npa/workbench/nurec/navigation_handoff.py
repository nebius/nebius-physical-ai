"""Seal qualified scan geometry and measured cases for the native navigation workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from npa.workbench.nurec.navigation_assets import materialize, sha256
from npa.workbench.nurec.navigation_publication import verify_publication


def _qualified_scan(scene, physics):
    verify_publication(scene)
    verify_publication(physics)
    assembly = json.loads((scene / "provenance.json").read_text())
    reconstruction = json.loads((scene / "reconstruction.json").read_text())
    report = json.loads((physics / "physics_validation.json").read_text())
    hashes = {
        name: sha256(scene / name)
        for name in (
            "scene.usdz",
            "capture.json",
            "reconstruction.json",
            "provenance.json",
        )
    }
    _verify_hashes(assembly, reconstruction, report, hashes)
    if report.get("physics_validated") is not True or not report.get("probes"):
        raise ValueError("scan handoff requires completed native PhysX probes")
    if [probe.get("expected") for probe in report["probes"]] != assembly.get(
        "ray_probes"
    ):
        raise ValueError("native physics report does not cover the assembled probes")
    return {
        **hashes,
        "physics_validation.json": sha256(physics / "physics_validation.json"),
    }


def _verify_hashes(assembly, reconstruction, report, hashes):
    expected = (
        (assembly.get("schema"), "npa.nurec.navigation_scene.v1"),
        (reconstruction.get("schema"), "npa.navigation.rgbd_reconstruction.v1"),
        (report.get("schema"), "npa.nurec.navigation_physics.v1"),
        (assembly.get("scene_sha256"), hashes["scene.usdz"]),
        (report.get("scene_sha256"), hashes["scene.usdz"]),
        (report.get("assembly_provenance_sha256"), hashes["provenance.json"]),
        (assembly.get("reconstruction_report_sha256"), hashes["reconstruction.json"]),
        (assembly.get("capture_manifest_sha256"), hashes["capture.json"]),
        (reconstruction.get("capture_manifest_sha256"), hashes["capture.json"]),
    )
    if any(actual != wanted for actual, wanted in expected):
        raise ValueError("scan, reconstruction and native physics lineage disagree")


def _cases(source, destination):
    payload = Path(source).read_bytes()
    values = json.loads(payload)
    if not isinstance(values, dict) or set(values) != {
        "train_cases",
        "eval_cases",
        "probe",
    }:
        raise ValueError(
            "measured cases must contain only train_cases, eval_cases and probe"
        )
    # Snapshot once; route inputs cannot override the builder's scene/image/source binding.
    destination.write_bytes(payload)
    return destination


def _retain_lineage(bundle, scene, physics, cases, hashes):
    evidence = bundle / "scan"
    evidence.mkdir()
    for name in ("capture.json", "reconstruction.json", "provenance.json"):
        shutil.copyfile(scene / name, evidence / name)
    shutil.copyfile(
        physics / "physics_validation.json", evidence / "physics_validation.json"
    )
    shutil.copyfile(cases, evidence / "cases.json")
    record = {
        "schema": "npa.navigation.scan_handoff.v1",
        "source_sha256": {**hashes, "cases.json": sha256(cases)},
        "recipe_sha256": sha256(bundle / "recipe.json"),
        "navigation_learning_verified": False,
    }
    (bundle / "scan-lineage.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _workflow_input(destination):
    if not destination.startswith("s3://"):
        return destination
    from npa.clients.storage import StorageClient
    from npa.workflows.navigation.publication import _record, _validate_record

    storage = StorageClient.from_environment()
    base = destination.rstrip("/")
    completion = _record(storage, base + "/completion.json")
    _validate_record(completion)
    if completion != _record(storage, base + "/claim.json"):
        raise ValueError("navigation handoff publication changed before completion")
    # The workflow's prepare stage accepts a raw bundle, not a logical stage prefix.
    return base + "/" + completion["attempt"] + "/"


def prepare_navigation_input(
    input_path: str,
    physics_path: str,
    cases_file: str,
    output_path: str,
    image: str,
    num_envs: int,
    iterations: int,
    episode_steps: int,
) -> dict:
    """Bind a qualified scan to the existing typed native navigation recipe.

    Args:
        input_path: Published RGB-D assembly directory or S3 prefix.
        physics_path: Matching published native PhysX report directory or prefix.
        cases_file: Local measured train/eval/probe JSON for this exact scene.
        output_path: Fresh local directory or immutable navigation S3 prefix.
        image: Exact Isaac runtime image digest.
        num_envs: Concurrent robot count and required held-out case count.
        iterations: Explicit native PPO learning iterations.
        episode_steps: Explicit native held-out control-step horizon.
    Returns:
        Hash lineage and raw input location for shared-scene-navigation.yaml.
    Raises:
        ImportError: The companion navigation implementation is not installed.
        ValueError: Scene lineage, measured cases, recipe or publication is invalid.
        OSError: Required inputs or artifact storage cannot be accessed.
    """
    experiment = {"image": image, "num_envs": num_envs}
    experiment.update(iterations=iterations, episode_steps=episode_steps)
    with tempfile.TemporaryDirectory(prefix="npa-scan-handoff-") as temporary:
        return _prepare(
            Path(temporary),
            input_path,
            physics_path,
            cases_file,
            output_path,
            **experiment,
        )


def _prepare(work, input_path, physics_path, cases_file, output_path, **settings):
    from npa.workflows.navigation.artifacts import publish
    from npa.workflows.navigation.reference_bundle import build_bundle

    scene = materialize(input_path, work / "scene")
    physics = materialize(physics_path, work / "physics")
    hashes = _qualified_scan(scene, physics)
    cases = _cases(cases_file, work / "cases.json")
    bundle = work / "navigation"
    build_bundle(bundle, scene_file=scene / "scene.usdz", cases_file=cases, **settings)
    lineage = _retain_lineage(bundle, scene, physics, cases, hashes)
    publish(bundle, output_path)
    return {**lineage, "workflow_input_uri": _workflow_input(output_path)}


def main(argv=None):
    """Prepare explicit scan-to-navigation inputs without starting a GPU job.

    Args:
        argv: Optional command-line argument list.
    Returns:
        Zero after verified input publication.
    Raises:
        ValueError: A required input or its provenance is invalid.
        ImportError: The companion native navigation package is unavailable.
        OSError: Artifact staging or publication fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("input-path", "physics-path", "cases-file", "output-path", "image"):
        parser.add_argument("--" + flag, required=True)
    for flag in ("num-envs", "iterations", "episode-steps"):
        parser.add_argument("--" + flag, required=True, type=int)
    result = prepare_navigation_input(**vars(parser.parse_args(argv)))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
