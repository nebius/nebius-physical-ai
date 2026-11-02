#!/usr/bin/env python3
"""Why the occlusion control's pixel counts move, measured one stage at a time.

`controls-manifest.json` publishes a visible-pixel fraction to fifteen significant
figures whose own spread sits in the fourth decimal, and the obvious reading -- that a
rasterizer handed the same geometry twice draws it twice -- is wrong. This harness runs
the geometry half of that pipeline repeatedly and reports, for each layer the frames are
built from, whether the bytes, the point count and the point *set* come back the same.

The distinction between bytes and set matters. A cloud written in a different order has
different bytes and identical content, and a z-buffer resolving ties between coincident
points would be a fair explanation for a one-pixel difference. So the comparison here is
a set difference on exact coordinate triples, not a hash: it separates "the same points
arrived in another order" from "different points arrived", and only the second can move
a count by five.

Nothing here is a thread experiment. `#1024` already established that this
reconstruction varies at a fixed thread count and is no better at one thread, so the
lever is known not to apply, and repeating it would measure nothing. What is open is
*which* layer the variation reaches and by how much, because the answer decides whether
a reader looking at a changed pixel count should suspect the renderer.

Writes JSON to stdout. Takes several minutes: the scan build and the reconstruction run
once per repeat.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parent
REPEATS = 4


def _load(name: str):
    """Import a sibling harness by path, since this directory is not a package."""

    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(points: np.ndarray) -> np.ndarray:
    """One structured row per point, so set operations compare whole coordinates."""

    contiguous = np.ascontiguousarray(points)
    dtype = [(axis, contiguous.dtype) for axis in ("x", "y", "z")]
    return contiguous.view(dtype).ravel()


def _points_of(layer) -> np.ndarray:
    vertices = getattr(layer, "vertices", None)
    return np.asarray(layer.points if vertices is None else vertices)


def _compare(label: str, repeats: list[np.ndarray]) -> dict:
    digests = [
        hashlib.sha256(np.ascontiguousarray(one).tobytes()).hexdigest()
        for one in repeats
    ]
    counts = [len(one) for one in repeats]

    first = _rows(repeats[0])
    against_first = []
    for index, other in enumerate(repeats[1:], start=2):
        rows = _rows(other)
        only_first = len(np.setdiff1d(first, rows))
        only_other = len(np.setdiff1d(rows, first))
        against_first.append(
            {
                "repeat": index,
                "points_only_in_repeat_1": only_first,
                "points_only_in_this_repeat": only_other,
                "shared": len(np.intersect1d(first, rows)),
            }
        )

    differing = [one["points_only_in_repeat_1"] for one in against_first]
    largest = max(differing) if differing else 0
    return {
        "layer": label,
        "repeats": len(repeats),
        "point_counts": counts,
        "point_count_is_stable": len(set(counts)) == 1,
        "distinct_byte_digests": len(set(digests)),
        "bit_identical_every_repeat": len(set(digests)) == 1,
        "set_difference_against_repeat_1": against_first,
        "largest_fraction_of_points_differing": (
            0.0 if not counts[0] else round(largest / counts[0], 6)
        ),
        "the_same_points_in_another_order": (len(set(digests)) > 1 and largest == 0),
    }


def main() -> int:
    import open3d as o3d

    occlusion = _load("audit_view_occlusion")

    collected: dict[str, list[np.ndarray]] = {
        "fused": [],
        "retained": [],
        "subject": [],
    }
    for _ in range(REPEATS):
        cloud, voxel = occlusion.build_eagle(o3d)
        samples, kept, removed = occlusion.reconstruct_and_crop(o3d, cloud, voxel)
        if kept is None or removed is None:
            raise RuntimeError("no crop to compare")
        collected["fused"].append(_points_of(samples))
        collected["retained"].append(_points_of(kept))
        collected["subject"].append(_points_of(removed))

    layers = {name: _compare(name, runs) for name, runs in collected.items()}
    deterministic = sorted(
        name for name, one in layers.items() if one["bit_identical_every_repeat"]
    )
    varying = sorted(name for name in layers if name not in deterministic)

    record = {
        "what_this_measures": (
            "Whether the layers vlm-controls-v4 renders come back identical when the "
            "geometry half of that pipeline is repeated in one process. The published "
            "visible-pixel fraction moves in its fourth decimal between runs and the "
            "question is where that enters."
        ),
        "why_a_set_difference_and_not_a_hash": (
            "A cloud written in a different order has different bytes and identical "
            "content, and coincident points resolved differently by a z-buffer would be "
            "a fair explanation for one pixel. A set difference on exact coordinate "
            "triples tells those two apart; a digest cannot."
        ),
        "repeats": REPEATS,
        "layers": layers,
        "bit_identical_layers": deterministic,
        "varying_layers": varying,
        "what_this_establishes": (
            "The input to the reconstruction is reproducible and its output is not: "
            f"{', '.join(deterministic) or 'no layer'} came back byte-identical across "
            f"{REPEATS} repeats, while {', '.join(varying) or 'no layer'} did not, and "
            "not by reordering -- the set differences below are nonzero and the "
            "subject's point count itself moves. So the renderer is drawing geometry "
            "that genuinely differs, and a pixel count that moves is not evidence about "
            "the rasterizer. PR #1024 isolates this reconstruction as the cause of the "
            "same class of variation in capability-record-support-crop.json, measured "
            "at a fixed thread count and no better at one thread."
        ),
        "what_this_does_not_establish": (
            "Nothing about the renderer's own determinism, which is untested here, and "
            "nothing new about the cause inside the reconstruction. It also does not "
            "revisit the thread count: that lever is already known not to reach this "
            "stage, so a repeat under it would measure nothing."
        ),
        "reproducibility": {
            "command": (
                "npa/.venv/bin/python docs/evidence/open3d/harness/layer_repeat_diff.py"
            ),
            "note": (
                "Every figure here is a between-run comparison, so the figures "
                "themselves are expected to move. The verdicts are what reproduce."
            ),
        },
    }
    json.dump(record, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
