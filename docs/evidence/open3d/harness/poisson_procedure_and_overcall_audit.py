"""Answer the review lane's two open items on the sampling claim.

First: they asked for the Poisson-disk procedure behind the claim that a zero-error
reconstruction measures exactly 0.0 unsupported area under even sampling -- generator, radius,
oversampling, and the resulting max/median nearest-neighbour ratio. Their own attempt to
reproduce it produced a "blue noise" set *less* even than plain uniform random (max/median
4.50 against 3.76), so they correctly declined to treat it as a test of the claim. This
records what was actually run, with the evenness measured rather than assumed.

Second: they withdrew an "undecided band" ruling on the grounds that the reading never
over-calls fabrication, which makes a plain assertion above the threshold sound. That
property is load-bearing for the whole field now, so it is checked here against every
zero-error construction in this tree at once rather than trusted from one sweep.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/poisson_procedure_and_overcall_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from npa.workbench.open3d.runner import _crop_justification, _support

OUT = Path("/data/poisson-procedure-and-overcall-audit.json")

SAMPLES = 60000
INIT_FACTOR = 3


class _Mesh:
    def __init__(self, count: int) -> None:
        self.vertices = range(count)


def geometry(name: str):
    if name == "icosphere":
        mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=1.0)
        return mesh.subdivide_loop(number_of_iterations=4)
    mesh = o3d.io.read_triangle_mesh(o3d.data.ArmadilloMesh().path)
    mesh.remove_duplicated_vertices()
    return mesh


def evenness(cloud) -> dict[str, float]:
    """Max/median nearest-neighbour ratio: the review lane's own evenness measure."""
    nn = np.asarray(cloud.compute_nearest_neighbor_distance())
    median = float(np.median(nn))
    return {
        "median_nn": median,
        "max_nn": float(nn.max()),
        "p99_nn": float(np.percentile(nn, 99)),
        "max_over_median_nn": float(nn.max() / median) if median > 0 else 0.0,
        "p99_over_median_nn": float(np.percentile(nn, 99) / median)
        if median > 0
        else 0.0,
    }


def main() -> int:
    report: dict = {
        "poisson_disk_procedure": {
            "asked_by": "review lane, 2026-09-20T04:40Z",
            "why": (
                "Their reproduction attempt produced a set less even than uniform random "
                "(max/median 4.50 against 3.76) by weighted sample elimination, so it could "
                "not test the claim. This records the generator actually used."
            ),
            "generator": (
                "open3d.geometry.TriangleMesh.sample_points_poisson_disk, Open3D's own "
                "elimination-based Poisson-disk sampler. No hand-rolled blue noise."
            ),
            "radius": (
                "Not specified directly. The sampler derives the elimination radius from the "
                "requested point count and mesh surface area; only number_of_points is passed."
            ),
            "number_of_points": SAMPLES,
            "oversampling": (
                f"init_factor={INIT_FACTOR}, so the sampler draws {INIT_FACTOR}x the requested "
                "points uniformly before eliminating down to the target."
            ),
            "open3d_version": o3d.__version__,
        },
        "measured_evenness": {},
        "overcall_audit": {
            "property_under_test": (
                "A reconstruction with zero invented surface must never read as an "
                "extrapolated shell. The review lane withdrew their undecided-band ruling on "
                "this property, so it carries the weight that ruling would have."
            ),
            "cells": [],
        },
    }

    for name in ("icosphere", "armadillo"):
        mesh = geometry(name)
        mesh.compute_vertex_normals()
        for scheme in ("uniform_random", "poisson_disk"):
            cloud = (
                mesh.sample_points_uniformly(number_of_points=SAMPLES)
                if scheme == "uniform_random"
                else mesh.sample_points_poisson_disk(
                    number_of_points=SAMPLES, init_factor=INIT_FACTOR
                )
            )
            key = f"{name}/{scheme}"
            even = evenness(cloud)
            report["measured_evenness"][key] = even

            # Zero error by construction: the surface scored is the surface sampled.
            for multiple in (1.0, 2.0, 3.0):
                voxel = even["median_nn"] * multiple
                support = _support(o3d, mesh, cloud, voxel)
                reading = _crop_justification(support, 0, _Mesh(len(mesh.vertices)))
                report["overcall_audit"]["cells"].append(
                    {
                        "case": key,
                        "voxel_over_median_nn": multiple,
                        "unsupported_area_fraction": support[
                            "unsupported_area_fraction"
                        ],
                        "share_beyond_3_voxels": reading[
                            "unsupported_area_share_beyond_3_voxels"
                        ],
                        "reads_as": reading["removed_surface_reads_as"],
                        "over_called": reading["removed_surface_reads_as"]
                        == "extrapolated shell",
                    }
                )

    cells = report["overcall_audit"]["cells"]
    report["overcall_audit"]["over_calls"] = sum(1 for c in cells if c["over_called"])
    report["overcall_audit"]["cells_checked"] = len(cells)

    # Does even sampling actually zero the headline fraction, and is it more even at all?
    even_pairs = {}
    for name in ("icosphere", "armadillo"):
        u = report["measured_evenness"][f"{name}/uniform_random"]["max_over_median_nn"]
        p = report["measured_evenness"][f"{name}/poisson_disk"]["max_over_median_nn"]
        rows = {
            c["voxel_over_median_nn"]: c
            for c in cells
            if c["case"] == f"{name}/poisson_disk"
        }
        uniform_rows = {
            c["voxel_over_median_nn"]: c
            for c in cells
            if c["case"] == f"{name}/uniform_random"
        }
        even_pairs[name] = {
            "uniform_max_over_median_nn": u,
            "poisson_max_over_median_nn": p,
            "poisson_is_more_even": p < u,
            "unsupported_at_1x_uniform": uniform_rows[1.0]["unsupported_area_fraction"],
            "unsupported_at_1x_poisson": rows[1.0]["unsupported_area_fraction"],
            "unsupported_at_2x_uniform": uniform_rows[2.0]["unsupported_area_fraction"],
            "unsupported_at_2x_poisson": rows[2.0]["unsupported_area_fraction"],
        }
    report["sampling_comparison"] = even_pairs

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))

    print("=== evenness (review lane's max/median nearest-neighbour measure) ===")
    for key, even in report["measured_evenness"].items():
        print(f"{key:28} max/median {even['max_over_median_nn']:6.2f}")
    print("\n=== does even sampling zero the headline fraction? ===")
    for name, block in even_pairs.items():
        print(
            f"{name:12} more even: {block['poisson_is_more_even']!s:5}  "
            f"unsup@1x {block['unsupported_at_1x_uniform']:.5f} -> "
            f"{block['unsupported_at_1x_poisson']:.5f}   "
            f"unsup@2x {block['unsupported_at_2x_uniform']:.5f} -> "
            f"{block['unsupported_at_2x_poisson']:.5f}"
        )
    print("\n=== over-call audit on zero-error reconstructions ===")
    for cell in cells:
        print(
            f"{cell['case']:28} voxel {cell['voxel_over_median_nn']:.1f}x  "
            f"share {cell['share_beyond_3_voxels']:.5f}  {cell['reads_as']}"
        )
    print(
        f"\nover-calls: {report['overcall_audit']['over_calls']} of "
        f"{report['overcall_audit']['cells_checked']} cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
