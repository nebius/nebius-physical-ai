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
   band. The run behind the record used 0.87, which does not.

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
RECORDED_FACTOR = 0.87
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


def real_reconstruction_control(o3d) -> dict[str, Any]:
    """Run the actual product path on a real cloud, and report that it costs nothing.

    Publishing only the constructed failures would leave a reader thinking this happens
    routinely. It does not happen here at all. A sphere sampled densely everywhere but
    one cap gives Poisson something to extrapolate and the crop something to remove, and
    at every factor tried -- including one aggressive enough to delete 1516 vertices --
    no sample lost its surface. The field added to the report will usually read zero.
    That is the useful thing about a measured zero: it is a zero rather than an
    assumption.
    """

    import tempfile

    import numpy as np
    from npa.workbench.open3d.artifacts import validate_support
    from npa.workbench.open3d.runner import run_reconstruct

    rng = np.random.default_rng(7)
    points = rng.normal(size=(24000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points = points[points[:, 2] < 0.65]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fused = root / "fused.ply"
        o3d.io.write_point_cloud(str(fused), cloud)
        for factor in (1.0, 0.87, 0.5, 0.0):
            out = root / f"factor-{factor}"
            out.mkdir()
            report = run_reconstruct(
                {
                    "request": {
                        "input_path": str(fused),
                        "output_path": str(out),
                        "run_id": f"crop-coverage-control-{factor}",
                        "poisson_depth": 7,
                        "support_distance_factor": factor,
                    },
                    "fused_path": str(fused),
                    "voxel_size": 0.05,
                },
                out,
                f"crop-coverage-control-{factor}",
            )
            # The report has to satisfy the guard that now reads these fields, or the
            # product publishes a pair its own validator rejects.
            validate_support(report, voxel_size=0.05)
            cost = report["coverage_cost_of_crop"]
            rows.append(
                {
                    "support_distance_factor": factor,
                    "unsupported_vertices_removed": report[
                        "unsupported_vertices_removed"
                    ],
                    "coverage_within_voxel_before": report["coverage_before_crop"][
                        "fraction_within_voxel"
                    ],
                    "coverage_within_voxel_after": report["coverage"][
                        "fraction_within_voxel"
                    ],
                    "samples_abandoned_by_crop": cost["samples_abandoned_by_crop"],
                    "coverage_is_bit_identical": cost["coverage_is_bit_identical"],
                    "accepted_by_validate_support": True,
                }
            )
    worst = max(r["samples_abandoned_by_crop"] for r in rows)
    return {
        "input": (
            "24k points on a unit sphere with the cap above z=0.65 left unobserved, "
            "voxel 0.05, Poisson depth 7"
        ),
        "factors": rows,
        "reading": (
            f"Every factor cost coverage nothing: the worst row abandoned {worst} "
            "samples, and that includes a factor of 0.5 which removed "
            f"{max(r['unsupported_vertices_removed'] for r in rows)} vertices. The "
            "mechanisms above need surface whose vertices straddle the threshold or a "
            "triangle stretched between a supported and an unsupported region, and a "
            "densely sampled sphere has neither. This is what the recorded dataset's "
            "bit-identical result looks like from the inside, and why reading it as a "
            "guarantee is so easy."
        ),
        "every_report_was_accepted_by_the_validator": all(
            r["accepted_by_validate_support"] for r in rows
        ),
    }


def repository_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


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
        "on_real_reconstruction_data_it_costs_nothing": real_reconstruction_control(
            o3d
        ),
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
            "repository_commit": repository_commit(),
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
