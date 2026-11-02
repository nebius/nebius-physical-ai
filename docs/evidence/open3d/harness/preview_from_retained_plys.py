"""Render a provenance-honest Rerun preview from the retained fused cloud and mesh.

This exists because the first attempt at this preview was not honest. It fed `run_visualize`
a 64-zero `fused_sha256` and a single fragment named `f0` carrying an identity pose, built
from the already-fused cloud -- so the recording asserted a fragment identity, a pose and a
content hash that were invented to satisfy required fields. The pixels were real; the
provenance claims were not. That script and its outputs are preserved under
`viewer-ui/superseded-preview/` and must not be cited as provenance for anything.

What this writes instead is an aggregate fused-cloud view: the fused cloud and the
reconstructed mesh, with the real SHA256 of the fused file, and no fragments at all. Zero
fragments is the accurate statement here, because the individual fragment clouds of that run
were not retained locally -- only `fused.ply` and `mesh.ply` were. The scan view fits its
camera against the fused cloud rather than the fragments, so it still draws real geometry.

This is a preview regenerated from one earlier run's retained outputs. It is deliberately
*not* a workflow artifact: it was not produced by a workflow, and the shipped recording of
that run is a separate file with its own hash. Nothing here should be presented as a new
run of the pipeline.

Usage:
    preview_from_retained_plys.py <fused.ply> <mesh.ply> <output-dir> <provenance.json>
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import numpy as np  # noqa: F401  (imported for the runner's own use)
    import rerun

    from npa.workbench.open3d import runner as runner_module
    from npa.workbench.open3d.runner import run_visualize

    fused, mesh, out, provenance = (Path(a) for a in sys.argv[1:5])
    out.mkdir(parents=True, exist_ok=True)

    # Every hash here is computed from the bytes actually read. No field is filled in to
    # satisfy a schema: the pose graph carries no nodes because no fragment poses are being
    # claimed, and `fused_sha256` is the real digest of the file being logged.
    payload = {
        "pose_graph": {"nodes": [], "fused_sha256": sha256(fused)},
        "fragments": {},
        "fused_path": str(fused),
        "mesh_path": str(mesh),
        "voxel_size": 0.02,
    }
    result = run_visualize(payload, out, "preview-aggregate-fused")

    recording = out / "point_cloud.rrd"
    runner_source = Path(runner_module.__file__)
    record = {
        "what_this_is": (
            "An aggregate fused-cloud Rerun preview rendered from the retained fused cloud "
            "and mesh of an earlier completed run, for viewer inspection only."
        ),
        "what_this_is_not": [
            "Not a workflow artifact and not a new pipeline run.",
            "Not the shipped recording of the run whose PLYs it reads; that file has its "
            "own hash and remains separate.",
            "Makes no fragment, pose, or registration claim. Zero fragments are logged "
            "because the fragment clouds were not retained, and inventing one would be the "
            "defect this script exists to correct.",
        ],
        "inputs": {
            "fused_ply": {"path": fused.name, "sha256": payload["pose_graph"]["fused_sha256"]},
            "mesh_ply": {"path": mesh.name, "sha256": sha256(mesh)},
        },
        "output": {"recording": recording.name, "sha256": sha256(recording)},
        "generator": {
            "script": Path(__file__).name,
            "script_sha256": sha256(Path(__file__)),
            "runner_module_sha256": sha256(runner_source),
        },
        "camera": result.get("view_cameras") or result.get("camera"),
        "rerun_version": rerun.__version__,
        "open3d_version": __import__("open3d").__version__,
        "python": platform.python_version(),
        "limits": [
            "The reconstruction has real holes and partial surfaces; they are visible in "
            "the preview and are not defects of the rendering.",
            "No claim is made about watertightness, collision fitness, or mesh readiness.",
        ],
    }
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"recording_sha256": record["output"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
