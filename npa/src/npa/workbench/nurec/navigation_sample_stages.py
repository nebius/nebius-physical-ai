"""Connect sealed public scan inputs to measured cases and native navigation preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from npa.workbench.nurec.navigation_assets import materialize, publish, sha256
from npa.workbench.nurec.navigation_publication import verify_publication
from npa.workflows.navigation.artifacts import write_json


def reconstruct_sample(input_path: str, output_path: str) -> dict:
    """Resolve verified sample publication and reconstruct the real metric surface.

    Args:
        input_path: Sealed public sample publication from navigation_sample.
        output_path: Fresh local directory or S3 reconstruction prefix.
    Returns:
        Full TSDF and held-out depth validation report.
    Raises:
        ValueError: Sample integrity, reconstruction or depth quality fails.
        OSError: Required inputs or artifacts cannot be accessed.
        ImportError: Reconstruction dependencies are unavailable.
    """
    from npa.workbench.nurec.navigation_reconstruction import reconstruct_capture
    from npa.workflows.navigation.artifacts import materialize as sealed_input

    with tempfile.TemporaryDirectory(prefix="npa-tum-reconstruct-") as temporary:
        root = sealed_input(input_path, Path(temporary) / "capture")
        return reconstruct_capture(str(root), output_path)


def prepare_cases(input_path: str, output_path: str, num_envs: int = 4000) -> dict:
    """Measure the reconstructed sample and publish disjoint supported reset cases.

    Args:
        input_path: Completed full public sample reconstruction.
        output_path: Fresh local directory or immutable S3 cases prefix.
        num_envs: Number of train cases and equally sized evaluation cohort.
    Returns:
        Measured support report with surface and capture lineage.
    Raises:
        ValueError: Reconstruction lineage or supported reset constraints fail.
        OSError: Artifact staging or publication fails.
        ImportError: Open3D or SciPy is unavailable.
    """
    from npa.workbench.nurec.navigation_sample_cases import build_cases

    with tempfile.TemporaryDirectory(prefix="npa-tum-cases-") as temporary:
        work = Path(temporary)
        root = materialize(input_path, work / "reconstructed")
        verify_publication(root)
        reconstruction = json.loads((root / "reconstruction.json").read_text())
        capture = json.loads((root / "capture.json").read_text())
        _sample_lineage(root, reconstruction, capture)
        cases, report = build_cases(root / "surface.npz", num_envs)
        report["capture_sha256"] = sha256(root / "capture.json")
        output = work / "cases"
        write_json(output / "cases.json", cases)
        write_json(output / "support.json", report)
        publish(output, output_path)
        return report


def _sample_lineage(root, reconstruction, capture):
    from npa.workbench.nurec.navigation_sample import ARCHIVE_SHA256

    valid = (
        capture.get("source", {}).get("archive_sha256") == ARCHIVE_SHA256
        and reconstruction.get("capture_manifest_sha256")
        == sha256(root / "capture.json")
        and reconstruction.get("surface_sha256") == sha256(root / "surface.npz")
        and reconstruction.get("integration_frames") == 1984
        and reconstruction.get("validation_frames") == 501
    )
    if not valid:
        raise ValueError(
            "measured office preset requires the complete pinned TUM sample"
        )


def prepare_policy(
    input_path: str,
    physics_path: str,
    cases_path: str,
    output_path: str,
    image: str,
    num_envs: int = 4000,
    iterations: int = 500,
    episode_steps: int = 300,
) -> dict:
    """Seal physically qualified scan and measured cases as native training inputs.

    Args:
        input_path: Qualified scan assembly prefix.
        physics_path: Matching native PhysX validation prefix.
        cases_path: Measured cases publication from prepare_cases.
        output_path: Fresh prepared navigation publication prefix.
        image: Exact native runtime digest.
        num_envs: Concurrent robots and held-out case count.
        iterations: Explicit PPO training iterations.
        episode_steps: Evaluation control-step horizon.
    Returns:
        Prepared native input record with scan lineage retained in its bundle.
    Raises:
        ValueError: Scene, measured cases or native provenance differ.
        OSError: Artifact reads or publication fail.
    """
    settings = dict(
        image=image,
        num_envs=num_envs,
        iterations=iterations,
        episode_steps=episode_steps,
    )
    with tempfile.TemporaryDirectory(prefix="npa-tum-policy-") as temporary:
        return _policy_handoff(
            Path(temporary), input_path, physics_path, cases_path, output_path, settings
        )


def _policy_handoff(work, input_path, physics_path, cases_path, output_path, settings):
    from npa.workbench.nurec.navigation_handoff import prepare_navigation_input
    from npa.workflows.navigation.stages import prepare

    cases = materialize(cases_path, work / "cases")
    verify_publication(cases)
    bundle = work / "bundle"
    prepare_navigation_input(
        input_path, physics_path, str(cases / "cases.json"), str(bundle), **settings
    )
    _bind_support(bundle, cases)
    return prepare(str(bundle), output_path, settings["image"])


def _bind_support(bundle, cases):
    support = json.loads((cases / "support.json").read_text())
    capture_digest = sha256(bundle / "scan/capture.json")
    reconstruction = json.loads((bundle / "scan/reconstruction.json").read_text())
    if (
        support["capture_sha256"] != capture_digest
        or support["surface_sha256"] != reconstruction["surface_sha256"]
    ):
        raise ValueError("measured reset cases belong to another scan")
    write_json(bundle / "scan/support.json", support)


def main(argv=None):
    """Run one preparation operation declared by the sample workflow graph.

    Args:
        argv: Optional stage-specific command-line arguments.
    Returns:
        Zero after successful stage publication.
    Raises:
        ValueError: Stage input, quality or provenance validation fails.
        OSError: Required input or output cannot be accessed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("reconstruct", "cases", "policy"):
        stage = commands.add_parser(name)
        stage.add_argument("--input-path", required=True)
        stage.add_argument("--output-path", required=True)
        if name in {"cases", "policy"}:
            stage.add_argument("--num-envs", type=int, default=4000)
        if name == "policy":
            for option in ("physics-path", "cases-path", "image"):
                stage.add_argument("--" + option, required=True)
            stage.add_argument("--iterations", type=int, default=500)
            stage.add_argument("--episode-steps", type=int, default=300)
    arguments = vars(parser.parse_args(argv))
    operation = arguments.pop("operation")
    handler = {
        "reconstruct": reconstruct_sample,
        "cases": prepare_cases,
        "policy": prepare_policy,
    }[operation]
    print(json.dumps(handler(**arguments), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
