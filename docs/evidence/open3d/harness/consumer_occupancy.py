"""Occupancy-mapping consumer: does a cell called occupied have any observation?

This is the script that produced `downstream-consumer.json`. It was not retained
alongside its output on the first pass, which the review lane correctly called the
one artifact in the chain that could not be audited. It was recovered by
enumerating candidate occupancy definitions against the recorded counts, and it
reproduces every published figure exactly, including the two the reviewer could
not reach by surface sampling: 2684 baseline and 1948 candidate occupied cells.

The definition is deliberately symmetric, and that symmetry is the point. A cell
is *occupied* when its centre lies within one support radius of the reconstructed
surface, and *supported* when its centre lies within the same radius of an
observed sample. Both sides are a distance-to-nearest query at the same radius, so
neither the surface nor the observations get a more generous test than the other.

Why a distance band rather than the enclosed volume: `compute_occupancy` asks
whether a point is inside a closed surface, and one of the two meshes being
compared is deliberately not closed. Scoring an open mesh with an inside/outside
test would compare the two variants on incomparable terms and would flatter the
closed baseline with thousands of interior cells (5817 against 4474 when tried).
A band around the surface is the question a mapping consumer actually asks: is
there structure in this cell.

What this measures, and only this: fabricated obstacle volume. Not collision
safety, not navigation readiness.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/consumer_occupancy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from harness_io import output_path, read_cloud, read_mesh

BUNDLE = Path("/data/final-bundle")
DEFAULT_OUT = "/data/downstream-consumer-reproduced.json"

CELL = 0.1
#: Support radius as a multiple of the cell. 0.87 is the rounded value of the
#: cell's circumradius factor sqrt(3)/2 = 0.8660254, a cell-centre-to-corner
#: reach, so a cell counts as touched when anything falls anywhere inside it.
#: It is the rounded constant that ran, not the exact factor. An earlier version of
#: this comment credited a 0.5-to-2.0 sweep to the review lane; they searched their
#: own dispatches and phases for it, found nothing, and said plainly that they never
#: ran it and never claimed to. The sweep is now this lane's own, over cell sizes
#: 0.05/0.1/0.2 against factors 0.5/0.87/1.5/2.0, in
#: `occupancy-instrument-controls.json` under `parameter_sweep`.
SUPPORT_RADIUS_FACTOR = 0.87

VARIANTS = {"baseline": "mesh-baseline/mesh.ply", "candidate": "mesh/mesh.ply"}


def probe_grid(cloud, cell: float):
    """The cell centres this instrument scores, and the grid they came from.

    Extracted from `main()` so a control can score synthetic geometry with the same
    instrument instead of a second copy of it. No line here changed in the move; the
    reproduction check at the end of `main()` is what proves that, because it compares
    the complete recomputed report against the published record and exits non-zero on
    any difference.
    """
    box = cloud.get_axis_aligned_bounding_box()
    origin = np.asarray(box.get_min_bound())
    dims = np.round(np.asarray(box.get_extent()) / cell).astype(int)
    index = (
        np.array(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"))
        .reshape(3, -1)
        .T
    )
    return origin + (index + 0.5) * cell, dims


def support_mask(cloud, centres, radius: float):
    """Which cells hold an observation: centre within `radius` of a sample."""
    import open3d as o3d

    tree = o3d.geometry.KDTreeFlann(cloud)
    return np.array(
        [bool(tree.search_radius_vector_3d(centre, radius)[0]) for centre in centres]
    )


def score_variant(mesh, centres, supported, radius: float, cell: float) -> dict:
    """Occupancy of one surface against one support mask, at one radius.

    `occupied` and `supported` are the same distance-to-nearest query at the same
    radius, which is the symmetry the module docstring argues for: neither the surface
    nor the observations get the more generous test.
    """
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distance = scene.compute_distance(
        o3d.core.Tensor(centres.astype(np.float32))
    ).numpy()
    return score_from_distance(distance, supported, radius, cell)


def score_from_distance(distance, supported, radius: float, cell: float) -> dict:
    """The definitions themselves, with the Open3D query already done.

    Split out so they can be exercised where Open3D is not installed, which is every
    CI shard: `open3d` appears in `npa/pyproject.toml` only as a container image tag,
    is in no optional-dependency extra, and is not in `npa/ci/requirements.txt`. A
    reviewer pointed out that the guard protecting this instrument from being quietly
    replaced by a copy was the one test that could never run there.

    Pure NumPy and pure arithmetic over two boolean masks, so a reader can check that a
    published set of counts and rates is internally what these definitions produce.
    """
    occupied = distance <= radius
    phantom = occupied & ~supported
    return {
        "occupied_cells": int(occupied.sum()),
        "phantom_obstacle_cells": int(phantom.sum()),
        "phantom_obstacle_rate": float(phantom.sum() / occupied.sum()),
        "phantom_volume_m3": float(phantom.sum() * cell**3),
        "precision_wrt_observation": float(1 - phantom.sum() / occupied.sum()),
        "recall_of_observed_cells": float(
            (occupied & supported).sum() / supported.sum()
        ),
        "missed_structure_cells": int((supported & ~occupied).sum()),
    }


def differences(expected, actual, path: str = "") -> list[tuple[str, object, object]]:
    """Return every structural or value difference between two report objects."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else key
            if key not in expected or key not in actual:
                mismatches.append((child, expected.get(key), actual.get(key)))
            else:
                mismatches.extend(differences(expected[key], actual[key], child))
        return mismatches
    if isinstance(expected, float) or isinstance(actual, float):
        if not isinstance(expected, (int, float)) or not isinstance(
            actual, (int, float)
        ):
            return [(path, expected, actual)]
        return (
            []
            if abs(float(expected) - float(actual)) <= 1e-9
            else [(path, expected, actual)]
        )
    return [] if expected == actual else [(path, expected, actual)]


def main() -> int:
    out = output_path(DEFAULT_OUT)
    cloud = read_cloud(BUNDLE / "multiway/fused.ply")
    centres, dims = probe_grid(cloud, CELL)
    radius = SUPPORT_RADIUS_FACTOR * CELL
    supported = support_mask(cloud, centres, radius)

    report = {
        "cell_size": CELL,
        "cells": int(np.prod(dims)),
        "cells_with_samples": int(supported.sum()),
        "variants": {},
    }
    for name, relative in VARIANTS.items():
        mesh = read_mesh(BUNDLE / relative)
        report["variants"][name] = {
            "artifact": relative,
            **score_variant(mesh, centres, supported, radius, CELL),
        }

    base, cand = report["variants"]["baseline"], report["variants"]["candidate"]
    report["improvement"] = {
        "phantom_cells_removed": base["phantom_obstacle_cells"]
        - cand["phantom_obstacle_cells"],
        "phantom_volume_removed_m3": base["phantom_volume_m3"]
        - cand["phantom_volume_m3"],
        "phantom_rate_baseline": base["phantom_obstacle_rate"],
        "phantom_rate_candidate": cand["phantom_obstacle_rate"],
        "recall_delta": cand["recall_of_observed_cells"]
        - base["recall_of_observed_cells"],
    }

    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    original = json.loads(Path("/data/downstream-consumer.json").read_text())
    mismatches = differences(original, report)
    print(f"grid {tuple(int(d) for d in dims)} = {report['cells']} cells")
    print(f"cells_with_samples {report['cells_with_samples']}")
    for variant in VARIANTS:
        row = report["variants"][variant]
        print(
            f"{variant:10} occupied {row['occupied_cells']:5d}  "
            f"phantom {row['phantom_obstacle_cells']:4d}  "
            f"precision {row['precision_wrt_observation']:.10f}  "
            f"recall {row['recall_of_observed_cells']:.10f}"
        )
    if mismatches:
        print(f"\n{len(mismatches)} field(s) differ from the published record:")
        for path, was, now in mismatches:
            print(f"  {path}: published {was} -> recomputed {now}")
    else:
        print("\nevery published field reproduced exactly")
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
