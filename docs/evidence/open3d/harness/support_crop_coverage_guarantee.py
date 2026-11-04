"""Is the support crop's coverage claim a guarantee, or did it hold on one dataset?

`capability-record-support-crop.json` lists among its supported claims:

    Surface with no observed sample within one voxel can be removed without moving the
    observed samples off the surface: coverage within a voxel is bit-identical at
    0.9927456237186564 and sample-to-surface RMSE changes by 4.7e-05 m.

The measurement is not in dispute. The wording reads as a property of the rule -- remove
surface no sample is within one voxel of, and the samples stay on the surface -- and that
is what this checks, because a consumer who reads it that way will not look again.

It is not a property of the rule. The crop tests *vertices* against
`voxel * support_distance_factor` and removes them with
`remove_vertices_by_mask`, which deletes every triangle touching a masked vertex.
Coverage asks about sample-to-*surface* distance. Two different questions, and the gap
between them has two separate openings:

1. A factor below 1.0 makes the vertex test tighter than the coverage question, so a
   vertex between `factor * voxel` and `voxel` from its nearest sample is removed while
   that sample is still within one voxel of it. The schema default of 1.0 closes this
   band. The run behind the record used the 1.0 default, which closes it.

2. A triangle dies for any one unsupported vertex, so two vertices a fraction of a voxel
   from a sample lose their triangle to a third vertex far away. No factor closes this
   one, because the distant vertex is removed at every factor below its own distance.

Each is shown on geometry small enough to read: a flat strip plus one patch placed where
the mechanism bites. The strip is there so the mesh survives the crop; `remove_vertices_
by_mask` raising on an emptied mesh would hide the effect behind an exception.

Run:
    python docs/evidence/open3d/harness/support_crop_coverage_guarantee.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

VOXEL = 1.0
#: Factors probed per mechanism. The first list straddles 1.0 so the band appears and
#: disappears; the second goes above it to show that going further does not help.
BAND_FACTORS = (1.0, 0.95, 0.87, 0.5)
ASYMMETRY_FACTORS = (1.0, 1.5, 2.0, 4.0)
RECORDED_CLAIM = (
    "Surface with no observed sample within one voxel can be removed without moving "
    "the observed samples off the surface: coverage within a voxel is bit-identical at "
    "0.9927456237186564 and sample-to-surface RMSE changes by 4.7e-05 m."
)
RECORDED_FACTOR = 1.0
SCHEMA_DEFAULT_FACTOR = 1.0


def _strip(verts: list[list[float]], tris: list[list[int]]) -> None:
    """A flat two-row strip along x, densely observed, well clear of either patch."""

    for x in (0.0, 1.0, 2.0):
        verts += [[x, 0.0, 0.0], [x, 1.0, 0.0]]
    for i in range(2):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        tris += [[a, b, c], [b, d, c]]


def band_geometry(o3d):
    """A patch whose every vertex is in the band: past 0.87 voxels, inside 1.0."""

    import numpy as np

    verts: list[list[float]] = []
    tris: list[list[int]] = []
    _strip(verts, tris)
    base = len(verts)
    verts += [[10.9, 0.0, 0.0], [10.9, 0.05, 0.0], [10.9, 0.0, 0.05]]
    tris += [[base, base + 1, base + 2]]
    points = [[x, 0.5, 0.0] for x in np.arange(0.0, 2.01, 0.25)] + [[10.0, 0.0, 0.0]]
    return _build(o3d, verts, tris, points)


def asymmetry_geometry(o3d):
    """One triangle with two vertices close to a sample and a third far from everything."""

    import numpy as np

    verts: list[list[float]] = []
    tris: list[list[int]] = []
    _strip(verts, tris)
    base = len(verts)
    verts += [[10.0, 0.0, 0.0], [10.0, 0.6, 0.0], [10.0, 0.3, 4.0]]
    tris += [[base, base + 1, base + 2]]
    points = [[x, 0.5, 0.0] for x in np.arange(0.0, 2.01, 0.25)] + [[10.0, 0.3, 0.1]]
    return _build(o3d, verts, tris, points)


def _build(o3d, verts: list[list[float]], tris: list[list[int]], points: list):
    import numpy as np

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array(verts, dtype=float)),
        o3d.utility.Vector3iVector(np.array(tris, dtype=np.int32)),
    )
    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.array(points, dtype=float))
    )
    return mesh, cloud


def vertex_distances(o3d, cloud, vertices):
    """What the crop tests: each vertex's distance to the nearest observed sample.

    This is the product's own rule, re-stated here rather than imported, so the
    demonstration does not depend on the module it is making a claim about.
    """

    import numpy as np

    tree = o3d.geometry.KDTreeFlann(cloud)
    return np.array(
        [float(np.sqrt(tree.search_knn_vector_3d(v, 1)[2][0])) for v in vertices]
    )


def surface_distances(o3d, mesh, cloud):
    """What coverage asks: each sample's distance to the nearest point of the surface."""

    import numpy as np

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    points = np.asarray(cloud.points).astype(np.float32)
    return scene.compute_distance(o3d.core.Tensor(points)).numpy()


def crop_at(o3d, mesh, cloud, factor: float) -> dict[str, Any]:
    """Apply the product's crop at one factor and report what coverage did."""

    import numpy as np

    work = o3d.geometry.TriangleMesh(mesh.vertices, mesh.triangles)
    before = surface_distances(o3d, work, cloud)
    triangles_before = len(work.triangles)
    mask = vertex_distances(o3d, cloud, np.asarray(work.vertices)) > VOXEL * factor
    work.remove_vertices_by_mask(mask)
    after = surface_distances(o3d, work, cloud)
    covered_before = before < VOXEL
    covered_after = after < VOXEL
    abandoned = covered_before & ~covered_after
    return {
        "support_distance_factor": factor,
        "vertices_removed": int(mask.sum()),
        "triangles_before": triangles_before,
        "triangles_after": len(work.triangles),
        "coverage_within_voxel_before": float(covered_before.mean()),
        "coverage_within_voxel_after": float(covered_after.mean()),
        "coverage_is_bit_identical": float(covered_before.mean())
        == float(covered_after.mean()),
        "samples_abandoned": int(abandoned.sum()),
        "worst_abandoned_voxels_from_surface": float(after[abandoned].max() / VOXEL)
        if abandoned.any()
        else None,
    }


def mechanism(o3d, label: str, geometry, factors) -> dict[str, Any]:
    import numpy as np

    mesh, cloud = geometry(o3d)
    patch = vertex_distances(o3d, cloud, np.asarray(mesh.vertices))[-3:]
    rows = [crop_at(o3d, mesh, cloud, f) for f in factors]
    return {
        "label": label,
        "patch_vertex_distances_to_nearest_sample_in_voxels": [
            round(float(d) / VOXEL, 4) for d in patch
        ],
        "factors": rows,
        "factors_where_coverage_fell": [
            r["support_distance_factor"]
            for r in rows
            if not r["coverage_is_bit_identical"]
        ],
        # A factor above every vertex's own distance removes nothing, so coverage
        # survives because there was no crop. That is not the rule being safe, and
        # counting it as such would be the flattering reading of this measurement.
        "factors_where_coverage_held_after_a_real_crop": [
            r["support_distance_factor"]
            for r in rows
            if r["coverage_is_bit_identical"] and r["vertices_removed"] > 0
        ],
        "factors_where_nothing_was_removed": [
            r["support_distance_factor"] for r in rows if r["vertices_removed"] == 0
        ],
    }


def repository_provenance() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    clean = subprocess.run(["git", "diff", "--quiet"], check=False).returncode == 0
    return {"repository_commit": commit, "repository_clean": clean}


def main() -> int:
    import open3d as o3d

    band = mechanism(o3d, "factor_below_one_opens_a_band", band_geometry, BAND_FACTORS)
    asym = mechanism(
        o3d,
        "a_triangle_dies_for_one_vertex",
        asymmetry_geometry,
        ASYMMETRY_FACTORS,
    )

    worst = max(
        (
            r["worst_abandoned_voxels_from_surface"]
            for r in asym["factors"]
            if r["worst_abandoned_voxels_from_surface"] is not None
        ),
        default=None,
    )

    record = {
        "schema": "npa.evidence.open3d.support-crop-coverage-guarantee/v0.0.1",
        "question": (
            "The support-crop capability record lists a bit-identical coverage result "
            "among its supported claims, worded as a property of the crop rule. Is it "
            "one?"
        ),
        "answer": (
            "No. It is a measurement that held on that dataset. Surface within one "
            "voxel of an observed sample can be removed, by two mechanisms, and one of "
            "them is open at every factor including the schema default."
        ),
        "the_claim_examined": {
            "source": "docs/evidence/open3d/capability-record-support-crop.json",
            "field": "claim_scope.supported_claims[1]",
            "text": RECORDED_CLAIM,
            "what_is_not_disputed": (
                "That coverage was bit-identical on that run, and that the RMSE moved "
                "by 4.7e-05 m. Nothing here re-measures that dataset; the inputs are "
                "not on this host. What is disputed is reading it as a guarantee."
            ),
            "the_run_used_factor": RECORDED_FACTOR,
            "the_schema_default_is": SCHEMA_DEFAULT_FACTOR,
        },
        "why_the_two_questions_differ": (
            "The crop masks vertices whose distance to the nearest sample exceeds "
            "`voxel * support_distance_factor`, and `remove_vertices_by_mask` deletes "
            "every triangle touching a masked vertex. Coverage measures each sample's "
            "distance to the nearest point of the remaining *surface*. A vertex test "
            "and a surface measurement do not have to agree, and where they disagree "
            "the crop removes surface a sample is within one voxel of."
        ),
        "mechanisms": {
            "factor_below_one_opens_a_band": {
                **band,
                "reading": (
                    f"Coverage fell at {band['factors_where_coverage_fell']} and the "
                    f"patch survived untouched at "
                    f"{band['factors_where_nothing_was_removed']}. That patch's "
                    "vertices all lie further than 0.87 voxels from the nearest "
                    "sample and all closer than 1.0, so at a factor of 1.0 the "
                    "vertex test and the coverage question ask the same thing and "
                    "the crop leaves it alone. This mechanism really is closed at "
                    f"the schema default of {SCHEMA_DEFAULT_FACTOR}. The run behind "
                    f"the record used {RECORDED_FACTOR}."
                ),
            },
            "a_triangle_dies_for_one_vertex": {
                **asym,
                "reading": (
                    f"Coverage fell at {asym['factors_where_coverage_fell']}, which "
                    "includes the schema default and two factors above it. Two of "
                    "that triangle's vertices are 0.32 voxels from a sample and are "
                    "retained; the third is 3.9 voxels out and is not, and the "
                    "triangle goes with it. The sample those two vertices sat under "
                    f"ends {worst:.1f} voxels from any surface. Coverage did hold at "
                    f"{asym['factors_where_nothing_was_removed']}, and that is worth "
                    "stating precisely rather than counting as a win: at that factor "
                    "the crop removed zero vertices, because the factor exceeds the "
                    "distant vertex's own 3.9. Coverage survived because there was "
                    "no crop. No factor both crops this geometry and keeps coverage "
                    f"-- {asym['factors_where_coverage_held_after_a_real_crop']} is "
                    "the set that would have to be non-empty for raising the factor "
                    "to be the remedy."
                )
                if worst is not None
                else "no abandonment observed",
                "closed_only_by": (
                    "Not cropping. `--support-distance-factor 0` disables the crop "
                    "explicitly; a factor above the mesh's worst vertex distance does "
                    "the same thing by accident, which is what the untouched row "
                    "above is. The capability record's own limitations already offer "
                    "factor 0 as the escape for consumers needing closed geometry; "
                    "what it does not say is that coverage needs the same escape."
                ),
            },
        },
        "what_changed_in_the_product": {
            "the_report_now_publishes": [
                (
                    "coverage_before_crop, so the before/after pair the claim "
                    "compares is in the artifact rather than computed by hand "
                    "beside it"
                ),
                (
                    "coverage_cost_of_crop, with the count of samples left without "
                    "surface within one voxel and how far the worst of them ended up"
                ),
            ],
            "why_not_change_the_crop": (
                "The crop is not wrong. A conservative crop that errs toward removing "
                "is what an honest-holes reconstruction wants, and the asymmetry is "
                "inherent to removing triangles by vertex mask. What was missing is the "
                "measurement, because without it a consumer assumes the cost was zero "
                "-- and on the recorded dataset it was, which is the most convincing "
                "way to be wrong about the next one."
            ),
            "the_advice_corrected": (
                "`_crop_justification` told a reader to raise the factor to 1.5 or 2.0. "
                "That keeps more surface, but this measurement shows it does not make "
                "the crop safe for coverage at any factor, so the sentence now says so "
                "and points at the new field."
            ),
        },
        "recorded_run_observation": {
            "support_distance_factor": RECORDED_FACTOR,
            "coverage_within_voxel_before": 0.9927456237186564,
            "coverage_within_voxel_after": 0.9927456237186564,
            "reading": (
                "The retained real-data record measured unchanged observed-sample "
                "coverage at factor 1.0. That one observation is not a guarantee and "
                "does not establish frequency on other scans."
            ),
            "source": "docs/evidence/open3d/capability-record-support-crop.json",
        },
        "what_this_does_not_claim": [
            (
                "That the recorded dataset's coverage result is wrong. It was not "
                "re-measured and the inputs are not reachable from this host."
            ),
            (
                "That the crop removes correct geometry on real data. These are "
                "constructed meshes chosen to isolate a mechanism, not samples of "
                "anything."
            ),
            (
                "That abandonment is common. Its frequency on real scans is "
                "unmeasured; the new field is what would measure it."
            ),
            (
                "That a factor of 1.0 is sufficient. It closes the band and leaves "
                "the asymmetry open."
            ),
        ],
        "generator": {
            "harness": "docs/evidence/open3d/harness/support_crop_coverage_guarantee.py",
            "open3d_version": o3d.__version__,
            "python": sys.version.split()[0],
            **repository_provenance(),
            "voxel": VOXEL,
        },
    }

    out = Path(
        os.environ.get(
            "NPA_CROP_COVERAGE_OUT",
            Path(__file__).resolve().parents[1]
            / "support-crop-coverage-guarantee.json",
        )
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    for name, block in record["mechanisms"].items():
        print(f"\n{name}")
        for row in block["factors"]:
            verdict = "held" if row["coverage_is_bit_identical"] else "FELL"
            print(
                f"  factor {row['support_distance_factor']:<4} {verdict:4}  "
                f"coverage {row['coverage_within_voxel_before']:.4f} -> "
                f"{row['coverage_within_voxel_after']:.4f}  "
                f"abandoned {row['samples_abandoned']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
