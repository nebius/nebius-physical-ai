#!/usr/bin/env python3
"""Run the product's own reconstruct stage and record the crop margin it now reports.

Not a mirror of the stage: this calls `run_multiway` and `run_reconstruct`, the same two
entry points the workflow calls, and reads `crop_margin` out of the returned summary. The
point is that the figure is available to a caller rather than computable in principle.

Two support factors, because the margin is a property of where the cut lands relative to
this scan's geometry, and a reader should see it move with the parameter rather than
take one number on trust.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

FACTORS = (1.0, 0.87, 0.0)
VOXEL = 0.05


def _fragment(index: int, path: str) -> dict:
    payload = Path(path).read_bytes()
    return {
        "id": f"f{index}",
        "uri": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def main() -> int:
    import open3d as o3d
    from npa.workbench.open3d.runner import (
        POISSON_VERTEX_DISPLACEMENT_BOUND_M,
        POISSON_VERTEX_DISPLACEMENT_ROUTINE_M,
        run_multiway,
        run_reconstruct,
    )

    data = o3d.data.DemoICPPointClouds()
    observed = {}

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        registration = root / "registration"
        registration.mkdir()
        run_multiway(
            {
                "manifest": {
                    "fragments": [
                        _fragment(index, path) for index, path in enumerate(data.paths)
                    ],
                    "voxel_size": VOXEL,
                },
                "fragments": {
                    f"f{index}": path for index, path in enumerate(data.paths)
                },
            },
            registration,
            "crop-margin-live",
        )
        fused = next(registration.rglob("fused.ply"))

        for factor in FACTORS:
            output = root / f"reconstruct-{factor}"
            output.mkdir()
            summary = run_reconstruct(
                {
                    "request": {
                        "input_path": str(fused),
                        "output_path": str(output),
                        "run_id": f"crop-margin-live-{factor}",
                        "poisson_depth": 9,
                        "density_quantile": 0.02,
                        "support_distance_factor": factor,
                    },
                    "fused_path": str(fused),
                    "voxel_size": VOXEL,
                },
                output,
                f"crop-margin-live-{factor}",
            )
            observed[f"support_distance_factor_{factor}"] = {
                "unsupported_vertices_removed": summary["unsupported_vertices_removed"],
                "crop_margin": summary["crop_margin"],
            }

    record = {
        "what_this_measures": (
            "That the support crop's decision margin is reported by the product's own "
            "reconstruct stage, and what it says on the record's own input. The figure "
            "existed inside the crop before this and was discarded, so a user could "
            "not tell whether the published reproducibility applied to their scan."
        ),
        "why_it_matters": (
            "The crop is a threshold on a per-vertex distance and the reconstruction "
            "under it is not reproducible. Whether its verdicts reproduce is therefore "
            "a fact about the caller's geometry, not a property of the rule, and "
            "crop-decision-stability.json says so explicitly: the condition to "
            "re-check elsewhere is the ratio, not the verdict."
        ),
        "executed": (
            "npa.workbench.open3d.runner.run_multiway then run_reconstruct, the entry "
            "points the workflow calls; no mirrored copy of the stage"
        ),
        "input": "open3d.data DemoICPPointClouds, three fragments, voxel 0.05",
        "displacement_scales_the_verdict_uses": {
            "routine_m": POISSON_VERTEX_DISPLACEMENT_ROUTINE_M,
            "worst_recorded_m": POISSON_VERTEX_DISPLACEMENT_BOUND_M,
        },
        "per_support_factor": observed,
        "what_the_result_shows": (
            "On this input the verdict is not_guaranteed at both enabled factors, from "
            "a handful of vertices inside the routine displacement out of eighteen "
            "thousand. That is consistent with the 46 pair comparisons that found no "
            "cell changing its verdict: a vertex inside the displacement is able to "
            "cross, not likely to. The first version of this field had a single "
            "threshold and reported the same scan as not_guaranteed on the strength of "
            "the unreproduced outlier alone, which is a warning a reader would learn "
            "to ignore."
        ),
        "what_this_does_not_establish": (
            "Anything about whether the reconstruction is correct. Every figure here "
            "concerns whether a rerun would decide differently, and a reproducible "
            "verdict on an invented surface is still a verdict on an invented surface."
        ),
        "reproducibility": {
            "command": (
                "npa/.venv/bin/python docs/evidence/open3d/harness/crop_margin_live.py"
            ),
            "note": (
                "Registration and reconstruction are both nondeterministic, so the "
                "counts move between runs. The verdict and the orders of magnitude are "
                "what reproduce."
            ),
        },
        "provenance": {
            "harness": "harness/crop_margin_live.py",
            "harness_sha256": "recomputed below",
            "path_is_relative_to": "this evidence directory",
        },
    }
    record["provenance"]["harness_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    json.dump(record, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
