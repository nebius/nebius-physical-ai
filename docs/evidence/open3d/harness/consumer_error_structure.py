"""What are the occupancy consumer's residual errors, spatially?

`downstream-consumer.json` reports the mapping consumer's residual errors as two
counts: 138 phantom cells and 74 missed-structure cells on the cropped
candidate. Counts cannot distinguish the two cases a mapping consumer cares
about. Seventy-four single cells scattered along a scan boundary is the
resolution limit of a 0.1 m grid. One contiguous seventy-four-cell region is an
unmapped wall, and a consumer trusting the map would drive through it.

This script decomposes both error classes into 6-connected components and
reports every component size, so the distinction is readable rather than
inferred. It also reports how far beyond the support radius each missed cell
sits: a miss at 1.01 radii is a quantisation boundary, a miss at five radii is
a hole.

Connectivity is 6 (face-adjacent) and not 26. Two blocks touching at a single
corner are not one obstacle, and 26-connectivity would merge them and
understate how fragmented the error is. The labeller is validated against that
exact case before it is used on the run, in `validate_labeller`, because an
unvalidated labeller would produce component sizes that look like measurement.

The occupancy definition is the one in `consumer_occupancy.py`, unchanged and
deliberately so: this script must reproduce that script's published figures
before it reports anything new, and it exits non-zero if it does not.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/consumer_error_structure.py

Or against a local bundle, with Open3D importable:
    NPA_OPEN3D_BUNDLE=<bundle-dir> python consumer_error_structure.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage

BUNDLE = Path(os.environ.get("NPA_OPEN3D_BUNDLE", "/data/final-bundle"))
PUBLISHED = Path(
    os.environ.get("NPA_OPEN3D_PUBLISHED", "/data/downstream-consumer.json")
)
OUT = Path(os.environ.get("NPA_OPEN3D_OUT", "/data/consumer-error-structure.json"))

CELL = 0.1
SUPPORT_RADIUS_FACTOR = 0.87
VARIANTS = {"baseline": "mesh-baseline/mesh.ply", "candidate": "mesh/mesh.ply"}

#: A residual error component spanning this many cells or more is "structured".
#: At 0.1 m that is a 0.2 m cube's worth of volume -- object scale rather than
#: sampling noise. Frozen in the claim before any component size was known.
STRUCTURED_CELLS = 8

#: Face adjacency only. See the module docstring on why not 26.
SIX_CONNECTED = ndimage.generate_binary_structure(3, 1)


def components_with(mask: np.ndarray, structure: np.ndarray) -> list[int]:
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return []
    return sorted((int(s) for s in np.bincount(labels.ravel())[1:]), reverse=True)


def components(mask: np.ndarray) -> list[int]:
    """Sizes of the 6-connected components of a boolean occupancy mask."""
    return components_with(mask, SIX_CONNECTED)


def validate_labeller() -> dict[str, object]:
    """Prove the labeller separates what it must before trusting its sizes.

    The corner case is the one that matters: two 1-cell blocks touching only at
    a corner are two obstacles, and 26-connectivity reports them as one. If this
    control passed under either connectivity it would not be a control.
    """
    corner = np.zeros((3, 3, 3), dtype=bool)
    corner[0, 0, 0] = True
    corner[1, 1, 1] = True
    corner_six = components(corner)
    corner_twentysix = components_with(corner, ndimage.generate_binary_structure(3, 3))

    bar = np.zeros((6, 6, 6), dtype=bool)
    bar[1, 1, 1:5] = True  # one 4-cell face-adjacent run
    bar[4, 4, 4] = True  # and one isolated cell, far from it

    empty = np.zeros((4, 4, 4), dtype=bool)
    full = np.ones((2, 3, 4), dtype=bool)

    checks = {
        "corner_touching_blocks_stay_separate": corner_six == [1, 1],
        "corner_case_would_merge_under_26_connectivity": corner_twentysix == [2],
        "a_run_and_an_isolated_cell_are_sized_correctly": components(bar) == [4, 1],
        "an_empty_mask_has_no_components": components(empty) == [],
        "a_full_mask_is_one_component_of_every_cell": components(full) == [24],
    }
    return {
        "checks": checks,
        "labeller_is_valid": all(checks.values()),
        "connectivity": "6 (face-adjacent)",
    }


def classify(mask: np.ndarray) -> dict[str, object]:
    """Describe an error mask by its 6-connected components.

    The largest component's bounding box is reported alongside its size because
    the two readings answer different questions. Eight cells arranged 2x2x2 is a
    lump of invented solid; eight cells arranged 1x2x4 is a one-cell-thick rind
    clinging to a real surface, which is what a distance-band occupancy test
    produces at the edge of a correct mesh. The size alone cannot tell them
    apart, and the frozen threshold is a size threshold, so the shape is
    reported next to it rather than folded into it.
    """
    labels, count = ndimage.label(mask, structure=SIX_CONNECTED)
    sizes_desc: list[int] = []
    extent: list[int] = []
    if count:
        sizes = np.bincount(labels.ravel())[1:]
        sizes_desc = sorted((int(s) for s in sizes), reverse=True)
        coords = np.argwhere(labels == int(np.argmax(sizes)) + 1)
        extent = (coords.max(axis=0) - coords.min(axis=0) + 1).tolist()

    structured = [s for s in sizes_desc if s >= STRUCTURED_CELLS]
    return {
        "cells_total": sum(sizes_desc),
        "components": len(sizes_desc),
        "component_sizes": sizes_desc,
        "largest_component_cells": max(sizes_desc, default=0),
        "largest_component_volume_m3": max(sizes_desc, default=0) * CELL**3,
        "largest_component_extent_cells": extent,
        "largest_component_min_thickness_cells": min(extent, default=0),
        "single_cell_components": sum(1 for s in sizes_desc if s == 1),
        "cells_in_structured_components": sum(structured),
        "structured_components": len(structured),
        "is_structured": bool(structured),
    }


def main() -> int:
    cloud = o3d.io.read_point_cloud(str(BUNDLE / "multiway/fused.ply"))
    box = cloud.get_axis_aligned_bounding_box()
    origin = np.asarray(box.get_min_bound())
    dims = np.round(np.asarray(box.get_extent()) / CELL).astype(int)
    index = (
        np.array(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"))
        .reshape(3, -1)
        .T
    )
    centres = origin + (index + 0.5) * CELL
    radius = SUPPORT_RADIUS_FACTOR * CELL

    tree = o3d.geometry.KDTreeFlann(cloud)
    supported_flat = np.array(
        [bool(tree.search_radius_vector_3d(centre, radius)[0]) for centre in centres]
    )

    report: dict[str, object] = {
        "question": (
            "The mapping consumer's residual errors are reported as counts. Are "
            "they scattered single cells at the grid's resolution limit, or "
            "contiguous regions a consumer would be wrong about?"
        ),
        "cell_size": CELL,
        "support_radius_factor": SUPPORT_RADIUS_FACTOR,
        "structured_threshold_cells": STRUCTURED_CELLS,
        "structured_threshold_was_frozen_before_measuring": True,
        "labeller_validation": validate_labeller(),
        "variants": {},
    }

    mismatches: list[str] = []
    published = json.loads(PUBLISHED.read_text())

    for name, relative in VARIANTS.items():
        mesh = o3d.io.read_triangle_mesh(str(BUNDLE / relative))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        distance = scene.compute_distance(
            o3d.core.Tensor(centres.astype(np.float32))
        ).numpy()
        occupied_flat = distance <= radius
        phantom_flat = occupied_flat & ~supported_flat
        missed_flat = supported_flat & ~occupied_flat

        shape = tuple(int(d) for d in dims)
        phantom = classify(phantom_flat.reshape(shape))
        missed = classify(missed_flat.reshape(shape))

        # How far beyond the radius does each missed cell's nearest surface sit?
        missed_excess = (distance[missed_flat] / radius).tolist()

        row = {
            "artifact": relative,
            "occupied_cells": int(occupied_flat.sum()),
            "phantom_obstacle_cells": int(phantom_flat.sum()),
            "missed_structure_cells": int(missed_flat.sum()),
            "phantom_obstacle_structure": phantom,
            "missed_structure_structure": missed,
            "missed_distance_in_radii": {
                "min": float(min(missed_excess, default=0.0)),
                "median": float(np.median(missed_excess)) if missed_excess else 0.0,
                "max": float(max(missed_excess, default=0.0)),
                "within_two_radii": sum(1 for e in missed_excess if e <= 2.0),
                "beyond_five_radii": sum(1 for e in missed_excess if e > 5.0),
            },
        }
        report["variants"][name] = row

        # Acceptance gate 1: the same occupancy definition must reproduce the
        # published counts, or this script is measuring a different thing.
        for key in (
            "occupied_cells",
            "phantom_obstacle_cells",
            "missed_structure_cells",
        ):
            was = published["variants"][name][key]
            if row[key] != was:
                mismatches.append(f"{name}.{key}: published {was} -> here {row[key]}")

    report["reproduces_published_counts"] = not mismatches
    report["mismatches_against_published"] = mismatches

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"grid {tuple(int(d) for d in dims)}, radius {radius:.4f} m")
    print(f"labeller valid: {report['labeller_validation']['labeller_is_valid']}")
    for name in VARIANTS:
        row = report["variants"][name]
        for kind in ("phantom_obstacle", "missed_structure"):
            block = row[f"{kind}_structure"]
            print(
                f"{name:10} {kind:17} {block['cells_total']:4d} cells in "
                f"{block['components']:3d} components, largest "
                f"{block['largest_component_cells']:4d} "
                f"extent {block['largest_component_extent_cells']} "
                f"({'structured' if block['is_structured'] else 'noise'})"
            )
        d = row["missed_distance_in_radii"]
        print(
            f"{name:10} missed distance    median {d['median']:.2f} radii, "
            f"max {d['max']:.2f}, {d['beyond_five_radii']} beyond 5"
        )
    if mismatches:
        print("\nDOES NOT reproduce the published consumer counts:")
        for line in mismatches:
            print(f"  {line}")
        return 1
    print("\nreproduces every published count it re-derives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
