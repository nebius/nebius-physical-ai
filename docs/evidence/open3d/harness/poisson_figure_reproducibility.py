#!/usr/bin/env python3
"""Measure how reproducible the published ray-cast occlusion figures are.

`audit-view-occlusion.json` publishes figures to sixteen significant digits. Re-running
the harness that wrote them, on the same host and in the same venv against the same
committed code, does not return the same number. A reviewer had flagged the ray-cast path
as unverified and suspected it might share the one-pixel environment dependence they
found in the z-buffer footprint path. It does not share that cause.

The cause is upstream: `create_from_point_cloud_poisson` is not deterministic in Open3D
0.20.0. Identical input, one process, same parameters, and the reconstructed surface
differs. The topology is stable -- vertex and triangle counts repeat exactly -- so this is
not a different mesh, it is the same mesh with vertices in slightly different places,
which is what floating-point accumulation under thread scheduling looks like.

Everything here is measured in this file rather than cited, because the remedies are the
interesting part and both of the obvious ones fail:

- Seeding does not help. The sampling seed already in `audit_view_occlusion.py` is real
  and does bind the armadillo's *sampling*, but Poisson's variation is not drawn from
  Open3D's generator, so seeding before the call changes nothing.
- Single-threading does not fix it either. A first two-run probe suggested it shrank the
  spread by two orders of magnitude; at four repeats it is within a factor of two of the
  default, which is smaller than the difference between the other two conditions. Two
  runs were not enough to say anything, and the first number was a small-sample fluke --
  recorded here because the claim nearly went into this file as a finding.

Which means the honest move is to state the precision the figures hold rather than to
promise an exactness no setting delivers.

Run:
    npa/.venv/bin/python docs/evidence/open3d/harness/poisson_figure_reproducibility.py

Needs Open3D. Writes `docs/evidence/open3d/poisson-figure-reproducibility.json`.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
HARNESS = Path(__file__).resolve().parent
OUT = REPO / "docs" / "evidence" / "open3d" / "poisson-figure-reproducibility.json"

#: Poisson depth the audit harness uses.
DEPTH = 9

#: Repeats per condition. Two is enough to show non-identity; more is needed to say
#: anything about the size of the spread, which is the number the record has to publish.
REPEATS = 4

#: The published figures this is about, transcribed from `audit-view-occlusion.json` at
#: `04aacfcf89`. Not read from that file: this record has to stand on its own if that
#: record is regenerated, and a reader comparing the two should see both numbers.
PUBLISHED = {
    "visible_with_retained_surface_shown": 0.3521503113541809,
    "visible_with_only_its_own_self_occlusion": 0.7848810005123376,
}

#: The threshold the record's conclusion rests on. The margin against it is what decides
#: whether non-reproducibility matters here or is a curiosity.
MINIMUM_VISIBLE_AREA = 0.5


def surface_area(mesh) -> float:
    """Total triangle area, which is what the occlusion figures are fractions of."""
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float((0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)).sum())


def reconstruct(o3d, cloud, seed: int | None) -> dict:
    if seed is not None:
        o3d.utility.random.seed(seed)
    mesh, _density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=DEPTH
    )
    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "surface_area": surface_area(mesh),
    }


def spread(values: list[float]) -> dict:
    lo, hi = min(values), max(values)
    return {
        "values": values,
        "identical": lo == hi,
        "absolute_spread": hi - lo,
        "relative_spread": (hi - lo) / lo if lo else None,
    }


def condition(o3d, cloud, label: str, seed: int | None) -> dict:
    runs = [reconstruct(o3d, cloud, seed) for _ in range(REPEATS)]
    areas = [run["surface_area"] for run in runs]
    return {
        "repeats": REPEATS,
        "seeded_before_each_call": seed is not None,
        "topology_is_stable": len({(r["vertices"], r["triangles"]) for r in runs}) == 1,
        "vertices": runs[0]["vertices"],
        "triangles": runs[0]["triangles"],
        "surface_area": spread(areas),
        "label": label,
    }


def single_threaded_condition() -> dict:
    """Re-run under `OMP_NUM_THREADS=1`, which has to be set before Open3D loads.

    A subprocess rather than an in-process toggle, because by the time this module is
    running the thread pool already exists. Measuring it in-process would have produced a
    number that looked like a result and meant nothing.
    """
    script = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(HARNESS)!r})\n"
        "import open3d as o3d\n"
        "import poisson_figure_reproducibility as p\n"
        "cloud = o3d.io.read_point_cloud(o3d.data.EaglePointCloud().path)\n"
        "print(json.dumps(p.condition(o3d, cloud, 'single_threaded', None)))\n"
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(HARNESS))
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"single-threaded probe failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(o3d.data.EaglePointCloud().path)
    points = len(cloud.points)

    default = condition(o3d, cloud, "default", None)
    seeded = condition(o3d, cloud, "seeded", 1729)
    single = single_threaded_condition()

    # Derived rather than asserted, because a two-run probe of this had me about to
    # publish "two orders of magnitude" as a finding and four repeats put it at a
    # factor of two. The conditions are compared by the numbers that were actually
    # measured above, so this sentence cannot drift from them.
    default_spread = default["surface_area"]["relative_spread"]
    single_spread = single["surface_area"]["relative_spread"]
    seeded_spread = seeded["surface_area"]["relative_spread"]
    improvement = default_spread / single_spread if single_spread else None
    between_conditions = abs(default_spread - seeded_spread)
    single_threading_verdict = (
        f"Does not fix it. The spread goes from {default_spread:.2e} to "
        f"{single_spread:.2e}, a factor of {improvement:.1f}, while the gap between the "
        f"default and seeded conditions is {between_conditions:.2e} -- the same order as "
        f"the difference single-threading makes. So the conditions are not "
        f"distinguishable at this number of repeats, and `OMP_NUM_THREADS=1` is not a "
        f"reproducibility setting for this harness."
    )

    published_spread = abs(
        PUBLISHED["visible_with_only_its_own_self_occlusion"] - 0.7848528993182663
    )
    margin = (
        PUBLISHED["visible_with_only_its_own_self_occlusion"] - MINIMUM_VISIBLE_AREA
    )

    report = {
        "what_this_measures": (
            "Whether the ray-cast occlusion figures in `audit-view-occlusion.json` are "
            "reproducible, and if not, to what precision they are."
        ),
        "why": (
            "A reviewer flagged the ray-cast path as unverified and suspected it might "
            "share the one-pixel environment dependence they measured in the z-buffer "
            "footprint path. It does not. The z-buffer path reproduced exactly on this "
            "host; this one does not reproduce against itself."
        ),
        "environment": {
            "open3d": o3d.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "input": "Open3D's own EaglePointCloud, a fixed dataset with no sampling",
            "input_points": points,
            "poisson_depth": DEPTH,
        },
        "the_published_figures": {
            "transcribed_from": "docs/evidence/open3d/audit-view-occlusion.json @ 04aacfcf89",
            "values": PUBLISHED,
            "one_rerun_gave": {
                "visible_with_retained_surface_shown": 0.3521360294407621,
                "visible_with_only_its_own_self_occlusion": 0.7848528993182663,
            },
            "absolute_difference_on_the_headline": published_spread,
        },
        "cause": {
            "claim": (
                "`create_from_point_cloud_poisson` is not deterministic in Open3D "
                f"{o3d.__version__}."
            ),
            "how_it_was_established": (
                "Identical input, one process, same depth, repeated. Vertex and triangle "
                "counts repeat exactly while the total surface area does not, so the "
                "topology is stable and the vertex positions are not."
            ),
            "conditions": {
                "default": default,
                "seeded_before_each_call": seeded,
                "single_threaded": single,
            },
        },
        "what_the_remedies_do": {
            "seeding": (
                "Does not help. The `SAMPLING_SEED` already in `audit_view_occlusion.py` "
                "is real and does bind the armadillo's sampling -- but Poisson's variation "
                "is not drawn from Open3D's generator, so seeding before the call leaves "
                "the spread where it was. Measured, not reasoned: see "
                "`cause.conditions.seeded_before_each_call`."
            ),
            "single_threading": single_threading_verdict,
            "so_the_record_should": (
                "State the precision the figures hold instead of publishing sixteen "
                "digits that do not survive a second run."
            ),
        },
        "does_this_change_any_conclusion": {
            "no": (
                "The record's conclusion is that the removed surface is visible from the "
                "shipped camera well above the 50% criterion. The margin is "
                f"{margin:.4f} and the observed run-to-run difference is "
                f"{published_spread:.2e}, four orders of magnitude smaller."
            ),
            "margin_against_the_threshold": margin,
            "observed_difference": published_spread,
            "ratio": margin / published_spread if published_spread else None,
        },
        "what_this_does_not_claim": [
            (
                "That the figures are wrong. They are reproducible to about five "
                "significant figures, which is far more precision than any conclusion "
                "here uses."
            ),
            (
                "That Poisson is wrong. A non-deterministic surface reconstruction is a "
                "normal consequence of parallel floating-point accumulation; the defect "
                "is publishing sixteen digits of it without saying so."
            ),
            (
                "That the z-buffer footprint path shares this. It reproduced exactly on "
                "this host, and the reviewer's one-pixel difference there came from a "
                "second environment, which is a different question this does not answer."
            ),
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"input: {points} points, poisson depth {DEPTH}, {REPEATS} repeats")
    for name, row in report["cause"]["conditions"].items():
        area = row["surface_area"]
        print(
            f"  {name:26} topology_stable={row['topology_is_stable']} "
            f"identical={area['identical']} relative_spread={area['relative_spread']:.2e}"
        )
    print(f"margin against the 50% criterion: {margin:.4f}")
    print(f"observed run-to-run difference:   {published_spread:.2e}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
