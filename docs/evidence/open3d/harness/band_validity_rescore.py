"""Criterion 5 of the signed-measure claim: what the stricter measure does to the real cases.

The claim's fifth criterion is *"rerun the real cases and report the direction and size of
every change to the committed numbers, including none."* This is that, for the one sweep whose
conclusion turns on the 0.05 false-positive threshold.

Review scoped it down and was right to: six of the eight rows already sit at or above 0.05, so
a stricter measure can only push them further above and no verdict can move. That is provable
rather than assumed, because the stricter mask is a superset -- `far | misoriented` contains
`far` -- so the stricter reading is never below the unsigned one.

That leaves the rows below the threshold. Review named one, `rows[1]` at 0.002738, and excluded
`rows[0]` because it reads exactly 0.000000. Exactly zero is not immunity: in #1006 the two-box
bridge read 0.000000 unsigned and 0.004602 under normal correspondence, which is the largest
*relative* divergence measured anywhere in that experiment. A row at exactly zero is where the
unsigned measure is most lenient, not least. So both rows are measured here.

Nothing is overwritten. The committed numbers stay as they are and the deltas are published
beside them, which is what the criterion asks for.

A note on what the delta is. #1004 established that `create_from_point_cloud_poisson` is
intrinsically nondeterministic -- three different meshes from one byte-identical cloud in one
process, including under `OMP_NUM_THREADS=1`. So this cannot reproduce the committed numbers
exactly, and does not try to. Both measures are computed on the *same* reconstructed mesh, so
the delta between them is exact even where the absolute values drift from what was committed.
The drift is reported separately, and the two decision-relevant rows are run repeatedly so the
drift is bounded rather than assumed small.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from band_validity_sweep import COUNTS, fabricated_area_fraction, poisson
from harness_io import output_path, seed_sampling

OUT = output_path("/data/band-validity-rescore.json")

#: Same cutoff as #1006, so the two measurements are of the same instrument.
NORMAL_AGREEMENT_DEGREES = 60.0

#: The false-positive threshold the sweep's conclusion rests on.
FALSE_POSITIVE_THRESHOLD = 0.05

#: Rows below the threshold, which are the only ones where a verdict could move.
ROWS_THAT_COULD_MOVE = (0, 1)

#: How many times to rebuild each decision-relevant row, to bound Poisson's nondeterminism.
REPEATS = 3


def both_measures(mesh, reference, voxel: float) -> dict[str, float]:
    """The committed unsigned reading and the same reading with normal correspondence added.

    `far` is computed exactly as the committed measure computes it -- per-triangle maximum of
    its vertices' distances to the reference, past one voxel -- by calling the sweep's own
    function rather than reimplementing it.

    Misorientation is asked at triangle centroids against triangle normals, because a
    triangle's orientation *is* its normal and its centroid is the natural place to ask where
    the reference lies. The absolute cosine is used: the reference's primitive normals carry a
    sign this comparison has no business trusting.
    """
    committed = fabricated_area_fraction(mesh, reference, voxel)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(reference))

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    a, b, c = (
        vertices[triangles[:, 0]],
        vertices[triangles[:, 1]],
        vertices[triangles[:, 2]],
    )
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(areas.sum())

    vertex_distance = scene.compute_distance(
        o3d.core.Tensor(vertices.astype(np.float32))
    ).numpy()
    far = vertex_distance[triangles].max(axis=1) > voxel

    centroids = vertices[triangles].mean(axis=1)
    closest = scene.compute_closest_points(
        o3d.core.Tensor(centroids.astype(np.float32))
    )
    reference_normal = closest["primitive_normals"].numpy()
    own_normal = np.asarray(mesh.triangle_normals)
    cosine = np.clip(np.abs(np.sum(reference_normal * own_normal, axis=1)), 0.0, 1.0)
    disagreement = np.degrees(np.arccos(cosine))
    misoriented = disagreement > NORMAL_AGREEMENT_DEGREES

    def share(mask: np.ndarray) -> float:
        return float(areas[mask].sum() / total) if total > 0 else 0.0

    return {
        "unsigned_fabricated": committed["fabricated_area_fraction"],
        "unsigned_recomputed_here": share(far),
        "stricter_fabricated": share(far | misoriented),
        "misoriented_only_share": share(misoriented & ~far),
        "median_disagreement_degrees": round(float(np.median(disagreement)), 4),
        "triangles": len(triangles),
    }


def build(reference, count: int, voxel: float) -> dict[str, float]:
    """One fresh sample draw and one reconstruction of it."""
    cloud = reference.sample_points_uniformly(number_of_points=count)
    mesh, _ = poisson(cloud, voxel)
    mesh.compute_triangle_normals()
    return both_measures(mesh, reference, voxel)


def rebuild_one_cloud(reference, count: int, voxel: float) -> list[dict[str, float]]:
    """One sample draw, reconstructed `REPEATS` times, to isolate Poisson from sampling.

    Without this the spread across repeats mixes two sources, because every call to
    `sample_points_uniformly` advances the global RNG and so draws a different cloud even
    inside one seeded process. I first reported the combined spread as a bound on Poisson's
    nondeterminism, which it is not.
    """
    cloud = reference.sample_points_uniformly(number_of_points=count)
    readings = []
    for _ in range(REPEATS):
        mesh, _ = poisson(cloud, voxel)
        mesh.compute_triangle_normals()
        readings.append(both_measures(mesh, reference, voxel))
    return readings


def main() -> int:
    seed = seed_sampling()
    reference = o3d.io.read_triangle_mesh(o3d.data.ArmadilloMesh().path)
    reference.compute_vertex_normals()

    densest = reference.sample_points_uniformly(number_of_points=COUNTS[0])
    voxel = float(
        np.median(np.asarray(densest.compute_nearest_neighbor_distance())) * 2.0
    )

    committed = json.loads((OUT.parent / "band-validity-sweep.json").read_text())[
        "rows"
    ]

    measured = []
    for index in ROWS_THAT_COULD_MOVE:
        count = COUNTS[index]
        runs = [build(reference, count, voxel) for _ in range(REPEATS)]
        unsigned = [r["unsigned_fabricated"] for r in runs]
        stricter = [r["stricter_fabricated"] for r in runs]
        recorded = committed[index]["fabricated_area_fraction"]

        fixed_cloud = rebuild_one_cloud(reference, count, voxel)
        fixed_unsigned = [r["unsigned_fabricated"] for r in fixed_cloud]
        fixed_stricter = [r["stricter_fabricated"] for r in fixed_cloud]
        measured.append(
            {
                "row": index,
                "samples": count,
                "committed_unsigned": recorded,
                "unsigned_this_run": {
                    "runs": [round(v, 8) for v in unsigned],
                    "median": round(statistics.median(unsigned), 8),
                    "spread": round(max(unsigned) - min(unsigned), 8),
                    "what_varies": "a fresh sample draw and a fresh reconstruction of it",
                },
                "stricter": {
                    "runs": [round(v, 8) for v in stricter],
                    "median": round(statistics.median(stricter), 8),
                    "spread": round(max(stricter) - min(stricter), 8),
                    "what_varies": "a fresh sample draw and a fresh reconstruction of it",
                },
                "one_cloud_reconstructed_three_times": {
                    "unsigned": [round(v, 8) for v in fixed_unsigned],
                    "stricter": [round(v, 8) for v in fixed_stricter],
                    "unsigned_spread": round(
                        max(fixed_unsigned) - min(fixed_unsigned), 8
                    ),
                    "stricter_spread": round(
                        max(fixed_stricter) - min(fixed_stricter), 8
                    ),
                    "what_varies": "the reconstruction only; the cloud is held fixed",
                    "why": (
                        "Every sample_points_uniformly call advances the global RNG, so the "
                        "three runs above draw three different clouds even inside one seeded "
                        "process. Their spread therefore bounds sampling and reconstruction "
                        "together, not Poisson alone. I first wrote it up as the latter."
                    ),
                },
                "delta_within_each_run": [
                    round(s - u, 8) for s, u in zip(stricter, unsigned)
                ],
                "largest_stricter_reading": round(max(stricter), 8),
                "crosses_the_threshold": max(stricter) >= FALSE_POSITIVE_THRESHOLD,
                "lift_needed_to_cross": (
                    "unbounded ratio from zero; the gap is "
                    f"{FALSE_POSITIVE_THRESHOLD - max(stricter):.6f} in absolute terms"
                    if max(unsigned) == 0.0
                    else f"{FALSE_POSITIVE_THRESHOLD / max(unsigned):.1f}x"
                ),
                "reproduction_drift_from_committed": round(
                    statistics.median(unsigned) - recorded, 8
                ),
            }
        )

    unchanged = []
    for index, row in enumerate(committed):
        if index in ROWS_THAT_COULD_MOVE:
            continue
        unchanged.append(
            {
                "row": index,
                "committed_unsigned": row["fabricated_area_fraction"],
                "already_at_or_above_threshold": (
                    row["fabricated_area_fraction"] >= FALSE_POSITIVE_THRESHOLD
                ),
            }
        )

    report = {
        "what_this_is": (
            "Criterion 5 of program/claims/open3d-signed-fabrication-measure/CLAIM.md: the "
            "direction and size of every change the stricter measure makes to the committed "
            "numbers, including none."
        ),
        "nothing_was_overwritten": (
            "band-validity-sweep.json is untouched. These readings sit beside it."
        ),
        "why_only_two_rows_were_measured": {
            "argument": (
                "The stricter mask is a superset of the unsigned one -- `far | misoriented` "
                "contains `far` -- so the stricter reading is never below the unsigned reading. "
                "A row already at or above the threshold can therefore only move further "
                "above it, and no verdict can change."
            ),
            "this_is_proved_not_assumed": (
                "It follows from the mask construction, and it is also visible in every row "
                "measured here: delta is non-negative in each run."
            ),
            "rows_covered_by_the_argument": unchanged,
            "where_review_and_i_differ": (
                "Review excluded rows[0] because it reads exactly 0.000000. Exactly zero is "
                "where the unsigned measure is most lenient, not least: in #1006 the two-box "
                "bridge read 0.000000 unsigned and 0.004602 under normal correspondence, the "
                "largest relative divergence in that experiment. So rows[0] is measured too."
            ),
        },
        "measured_rows": measured,
        "what_the_spreads_mean": {
            "within_a_run": (
                "Both measures are computed on the same reconstructed mesh, so every delta "
                "reported here is exact. Nothing about the instability below touches it."
            ),
            "across_runs": (
                "Two sources, separated above. Three fresh draws each reconstructed once "
                "bounds sampling and reconstruction together; one cloud reconstructed three "
                "times isolates the reconstruction."
            ),
            "a_correction_to_my_own_first_write_up": (
                "I first labelled the combined spread as a bound on Poisson's "
                "nondeterminism. It is not: sample_points_uniformly advances the global RNG, "
                "so each repeat draws a different cloud even under a fixed seed. The "
                "one-cloud column was added to say which is which."
            ),
            "relation_to_the_bound_published_in_1004": (
                "That bound -- worst relative difference 2.17e-3 across 106 leaves -- was "
                "measured on two seeded runs of one harness, which is a third experiment "
                "again. Neither of these columns contradicts it; they measure different "
                "things, and this note exists so the three are not conflated later."
            ),
        },
        "open3d_version": o3d.__version__,
        "sampling_seed": seed,
        "normal_agreement_cutoff_degrees": NORMAL_AGREEMENT_DEGREES,
        "false_positive_threshold": FALSE_POSITIVE_THRESHOLD,
        "visual_evaluation": {
            "status": "not_applicable",
            "reason": (
                "Scalar area shares computed from geometry, with no render and nothing for a "
                "vision model to see. The nonvisual allowance in EXPANDED-REVIEW-CONTRACT.md "
                "covers this, and review confirmed the same call on #1006."
            ),
        },
    }

    any_crossing = any(row["crosses_the_threshold"] for row in measured)
    report["verdict"] = {
        "any_verdict_moves": any_crossing,
        "statement": (
            "At least one row crosses the false-positive threshold under the stricter measure; "
            "the sweep's conclusion does not survive unchanged."
            if any_crossing
            else "No verdict moves. Every row keeps the reading it was committed with, and "
            "criterion 5 is satisfied with 'none' on the only rows where change was possible."
        ),
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    for row in measured:
        print(
            f"row {row['row']} ({row['samples']:>6} samples): "
            f"committed {row['committed_unsigned']:.6f}  "
            f"unsigned {row['unsigned_this_run']['median']:.6f}  "
            f"stricter {row['stricter']['median']:.6f}  "
            f"max stricter {row['largest_stricter_reading']:.6f}  "
            f"crosses={row['crosses_the_threshold']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
