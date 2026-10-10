"""Build a private offline HTML review from real Lyra and Isaac artifacts."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from npa.workflows.lerobot_transfer_data import file_sha256


def _video(path):
    if not path or not path.is_file():
        return None
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()


def _mesh(root):
    if root is None:
        return None
    with np.load(root / "surface.npz", allow_pickle=False) as surface:
        points = surface["points"].astype("<f4")
        colors = surface["colors"].astype("<f4")
        triangles = surface["triangles"].astype("<u4")
    center = (points.max(0) + points.min(0)) / 2
    radius = float(np.linalg.norm(points - center, axis=1).max())
    arrays = {"position": points, "color": colors, "triangles": triangles}
    return {
        **{
            name: base64.b64encode(value.tobytes()).decode()
            for name, value in arrays.items()
        },
        "center": center.tolist(),
        "radius": radius,
    }


def _json(root, name):
    return json.loads((root / name).read_text()) if root else None


def _trials(root, geometry):
    if root is None:
        return []
    from npa.workflows.physical_augmentation_demo import LABELS

    html = (root / "demo.html").read_text()
    marker = '<script id="demo-data" type="application/json">'
    if marker not in html:
        raise ValueError("Physical replay lacks its evidence data boundary")
    payload = json.loads(html.split(marker, 1)[1].split("</script>", 1)[0])
    binding = payload.get("scene_binding")
    if (
        not geometry
        or not binding
        or binding["geometry_sha256"] != geometry["surface_sha256"]
    ):
        raise ValueError("Robot replay is not bound to this reconstructed scene")
    trials = [trial for trial in payload["trials"] if trial["attempt"] == 0]
    for trial in trials:
        case = payload["conditions"][trial["condition"]]
        trial.update(
            label=LABELS[trial["condition"]],
            fps=payload["fps"],
            description=f"Paired seed {trial['seed']} · {case['mass_kg']} kg · friction {case['friction']} · position offset {case['offset_xy_m']} m",
        )
    return trials


def _statistics(reconstruction, geometry, actions):
    calibration = geometry["calibration"] if geometry else None
    return [
        [
            str(reconstruction["views"]) if reconstruction else "Pending",
            "Lyra reconstruction views",
        ],
        [
            f"{geometry['triangles']:,}" if geometry else "Pending",
            "reconstructed collision triangles",
        ],
        [
            f"{calibration['camera_rms_m'] * 100:.1f} cm" if calibration else "Pending",
            "camera calibration RMS",
        ],
        [
            f"{actions['accepted']} / {actions['attempted']}" if actions else "Pending",
            "accepted native robot attempts",
        ],
    ]


def _payload(args):
    reconstruction = _json(args.reconstruction_path, "reconstruction.json")
    geometry = _json(args.geometry_path, "geometry.json")
    actions = _json(args.actions_path, "report.json")
    _verify_lineage(args, reconstruction, geometry)
    return {
        "scope": "Recorded input shown below. Reconstruction, collision candidates and action results are reported only when their matching artifacts are supplied. This review does not establish wrist-camera fidelity or transfer to a physical robot.",
        "attribution": (args.input_path / "ATTRIBUTION.txt").read_text(),
        "source_video": _video(args.input_path / "capture.mp4"),
        "reconstruction_video": _video(args.reconstruction_path / "gs_trajectory.mp4")
        if reconstruction
        else None,
        "mesh": _mesh(args.geometry_path),
        "trials": _trials(args.actions_path, geometry),
        "stats": _statistics(reconstruction, geometry, actions),
        "collision_quality": _collision_quality(geometry),
        "stages": _stages(reconstruction, geometry, actions),
        "evidence": {
            "reconstruction": reconstruction,
            "geometry": geometry,
            "actions": actions,
            "customer_scene_validated": False,
            "customer_wrist_quality_validated": False,
        },
        "commands": _commands(),
    }


def _collision_quality(geometry):
    if not geometry:
        return "No reconstructed collision surface has been evaluated yet."
    quality = geometry.get("measured_depth_validation")
    if quality is None:
        return "Collision candidate only: independent measured depth was not supplied."
    error = quality["mean_absolute_error_m"]
    measured = f"{error * 100:.1f} cm" if error is not None else "unavailable"
    status = "Passed" if quality["passed"] else "Rejected"
    return (
        f"{status} by measured-depth checks: {quality['coverage']:.1%} coverage, "
        f"{quality['inlier_fraction']:.1%} inliers, {measured} mean absolute error. "
        "These RGB views were not held out. Native collision validation is separate."
    )


def _stages(reconstruction, geometry, actions):
    return [
        ["01  Captured video", "Recorded input available"],
        [
            "02  Lyra reconstruction",
            "Native output available" if reconstruction else "Awaiting GPU execution",
        ],
        [
            "03  Calibrated colliders",
            "Surface extracted" if geometry else "Awaiting reconstruction",
        ],
        [
            "04  Executed actions",
            f"{actions['accepted']} / {actions['attempted']} accepted"
            if actions
            else "Awaiting native validation",
        ],
    ]


def _verify_lineage(args, reconstruction, geometry):
    if reconstruction:
        if (
            file_sha256(args.input_path / "capture.mp4")
            != reconstruction["input_sha256"]
        ):
            raise ValueError("Lyra output belongs to another input video")
        for name, digest in reconstruction["checksums"].items():
            if (
                Path(name).name != name
                or file_sha256(args.reconstruction_path / name) != digest
            ):
                raise ValueError("Lyra output failed its native checksum")
    if geometry:
        if (
            not reconstruction
            or geometry["source_geometry_sha256"]
            != reconstruction["checksums"]["geometry.npz"]
        ):
            raise ValueError("Collision surface is not derived from this Lyra run")
        if (
            file_sha256(args.geometry_path / "surface.npz")
            != geometry["surface_sha256"]
        ):
            raise ValueError("Collision surface no longer matches its provenance")


def _commands():
    return """# Reconstruction: use your checksummed capture.mp4 bundle.
npa workbench workflow submit workflows/testing/lyra-reconstruction.yaml \\
  --project <project-alias> --infra k8s/<rtx-context> --stage-src \\
  --var bucket=<bucket> --var input_uri=s3://<bucket>/<input-prefix>/ \\
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY

# Metric camera poses and intrinsics accompany the reconstruction.
npa/.venv/bin/python -m npa.workflows.lyra_geometry \\
  --input-path <reconstruction-directory> --output-path <geometry-directory>

# Bind the robot mount with an explicit world_to_task rigid transform.
npa/.venv/bin/python -m npa.workflows.lyra_scene_assembly \\
  --input-path <geometry-directory> --binding-path <mounting.json> \\
  --output-path <prepared-scene> --run-id <run-id>

# Publish the sealed scene and geometry bundles, then execute and build HTML.
npa workbench workflow submit workflows/testing/lyra-scene-actions.yaml \\
  --project <project-alias> --infra k8s/<rtx-context> --stage-src \\
  --var bucket=<bucket> --var prepared_uri=<private-prepared-prefix> \\
  --var reconstruction_uri=<private-reconstruction-prefix> \\
  --var geometry_uri=<private-geometry-prefix> \\
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY"""


def _require_complete(payload):
    evidence = payload["evidence"]
    missing = [
        name for name in ("reconstruction", "geometry", "actions") if not evidence[name]
    ]
    if missing:
        raise ValueError(
            "Complete review requires artifacts for: " + ", ".join(missing)
        )
    if not payload["source_video"] or not payload["reconstruction_video"]:
        raise ValueError(
            "Complete review requires both recorded and reconstructed video"
        )
    if not payload["mesh"] or not payload["trials"]:
        raise ValueError(
            "Complete review requires a collision mesh and executed trials"
        )
    if evidence["actions"]["attempted"] <= 0:
        raise ValueError("Complete review requires at least one native action attempt")


def main():
    """Generate an offline HTML file without inventing unavailable stage results.

    Args:
        None; reads paths from the command line.
    Returns:
        None after writing the self-contained review.
    Raises:
        ValueError: Evidence or embedded replay is malformed.
        OSError: Input artifacts cannot be read or output cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--reconstruction-path", type=Path)
    parser.add_argument("--geometry-path", type=Path)
    parser.add_argument("--actions-path", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    data = _payload(args)
    if args.require_complete:
        _require_complete(data)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = Path(__file__).with_suffix(".html").read_text()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(template.replace("__LYRA_DEMO_DATA__", payload))


if __name__ == "__main__":
    main()
