#!/usr/bin/env python3
"""Does the support crop's *decision* reproduce, or only its average?

`capability-record-support-crop.json` reports its reproducibility as "numeric to ~4e-06
relative on surface area". That is an aggregate and it is true. The same record's
`claim_scope.supported_claims[3]` publishes a per-cell downstream result -- 733 fewer
phantom obstacle cells, precision rising from 0.675 to 0.929 -- and a user asking the
crop *which parts of my mesh are unsupported* is reading a per-element answer that no
aggregate figure describes.

That gap matters because the partition is known to move. PR #1024 established that the
Poisson reconstruction underneath returns different geometry from identical input, with
no seed or thread lever reaching it, and PR #1015 measured the consequence one stage
later: around 1% of the retained layer's points and 2% of the subject's differ between
repeats. So the vertices the crop rule is applied to are not the same vertices twice,
and the rule itself -- `distance_to_cloud > voxel * support_distance_factor`, a
threshold on a per-vertex quantity -- has nothing to stabilise them with.

This measures the consequence a user sees. The fused cloud is built once and re-read for
every repeat, so registration noise cannot contribute and whatever moves comes from the
reconstruct-and-crop stage alone.

Comparison is spatial, not per-vertex. Poisson returns a different vertex set each run,
so "this vertex changed its mind" is not a question that can be asked; a downstream
consumer does not ask it either. Space is voxelised at the scene's own voxel size and
each cell is scored kept, removed or absent, which is the form the published occupancy
consumer already works in.

Writes JSON to stdout. Several minutes: one registration plus one reconstruction per
repeat.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType

import numpy as np

#: Raised from 4 after an independent lane showed why 4 is not enough. The margin is
#: computed from the worst per-pair displacement, and that statistic has a tail: their
#: sweep hit one pair three orders of magnitude above its neighbours and then failed to
#: reproduce it three times on the byte-identical file. Four repeats give three pairs,
#: which can only describe the middle of a distribution like that. 24 give 23.
REPEATS = 24

#: Rebuilds of the fused cloud, for the arm that lets registration vary too. Fewer than
#: REPEATS because each one re-runs multiway registration, and the question it answers
#: is coarser: does the published figure move at all.
INVOCATIONS = 5

#: The tail event itself, kept rather than dropped, because the claim it bounds is the
#: one this record exists to make. Measured by an independent review lane running this
#: module's `_one_crop` on a 53x denser input of the same scene; not reproduced in three
#: further attempts on a coordinate-identical file, so it is reported as an observation
#: with an unknown rate rather than as a property. The ratio below is taken against
#: whichever displacement is worse, this or anything measured here.
TAIL_OBSERVED_ELSEWHERE = MappingProxyType(
    {
        "displacement_m": 2.187e-03,
        "cut_m": 0.05,
        # Published under the name `margin_factor` before 2026-11-02. Same formula,
        # renamed: `cut_m / displacement_m` compares two lengths and is not a factor
        # by which a decision clears anything. See
        # `what_the_cut_to_separation_ratio_is_not` below.
        "cut_over_displacement": 0.05 / 2.187e-03,
        "cut_over_displacement_is": "cut_m / displacement_m",
        "cells_that_flipped_there": 0,
        "reproduced_in_three_further_attempts": False,
        "measured_by": (
            "an independent review lane, on a 53x denser input of the same scene, "
            "driving this module's _one_crop with the product modules stubbed"
        ),
        "why_it_is_kept": (
            "Dropping an unreproduced outlier because it is inconvenient is how a "
            "stated bound becomes a median. It is three orders of magnitude from "
            "everything else and still 20x smaller than the cut, and no cell changed "
            "its verdict where it was observed -- so it lowers the cut-to-separation "
            "ratio this record reports without contradicting any verdict that was "
            "actually checked."
        ),
    }
)

#: The record's own input and voxel: three upstream Redwood fragments at 0.05.
#: Two densities. A margin measured on a sparse cloud says nothing about a real
#: capture unless the wobble is shown not to grow with the geometry.
VOXELS = (0.05, 0.02)
VOXEL = VOXELS[0]
POISSON_DEPTH = 9
DENSITY_QUANTILE = 0.02
SUPPORT_DISTANCE_FACTOR = 1.0

#: The published consumer's grid, copied from consumer_occupancy.py so the downstream
#: figures here are the same quantity that record reports rather than a near relative.
CELL = 0.1
SUPPORT_RADIUS_FACTOR = 0.87


def _fused_cloud(o3d, scratch: Path):
    """One fused cloud through the product's own multiway registration, written once.

    Held fixed for every repeat on purpose. Registration is itself not reproducible --
    that is #1033's finding -- and letting it vary would leave the result unattributable
    between two candidate causes.
    """

    from npa.workbench.open3d.runner import run_multiway

    paths = o3d.data.DemoICPPointClouds().paths
    fragments = {f"cloud_bin_{index}": path for index, path in enumerate(paths)}
    payload = {
        "manifest": {
            "schema_version": "npa.open3d.registration.v1",
            # The manifest identifies fragments by content, so the digests are the
            # product's own requirement rather than decoration.
            "fragments": [
                {
                    "id": key,
                    "uri": value,
                    "sha256": hashlib.sha256(Path(value).read_bytes()).hexdigest(),
                    "bytes": Path(value).stat().st_size,
                }
                for key, value in fragments.items()
            ],
            "voxel_size": VOXEL,
        },
        "fragments": fragments,
    }
    run_multiway(payload, scratch, "crop-decision-stability")
    fused = scratch / "fused.ply"
    if not fused.exists():
        candidates = sorted(one.name for one in scratch.rglob("*.ply"))
        raise RuntimeError(f"no fused cloud at {fused}; wrote {candidates}")
    return fused


def _one_crop(o3d, fused: Path) -> dict:
    """The product's reconstruct-and-crop, mirrored from `runner.py`.

    Mirrored rather than imported because the product entry point writes a run
    directory and returns a summary; what is needed here is the intermediate vertex
    sets, which it does not hand back. Every parameter and every call is the same, and
    the crop rule is the same comparison on the same quantity.
    """

    cloud = o3d.io.read_point_cloud(str(fused))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 2.0, max_nn=30)
    )
    cloud.orient_normals_consistent_tangent_plane(k=30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=POISSON_DEPTH
    )
    density = np.asarray(densities)
    threshold = float(np.quantile(density, DENSITY_QUANTILE))
    mesh.remove_vertices_by_mask(density < threshold)
    mesh.compute_vertex_normals()

    uncropped = o3d.geometry.TriangleMesh(mesh)

    vertices = np.asarray(mesh.vertices)
    tree = o3d.geometry.KDTreeFlann(cloud)
    distances = np.array(
        [
            float(np.sqrt(tree.search_knn_vector_3d(vertex, 1)[2][0]))
            for vertex in vertices
        ]
    )
    limit = VOXEL * SUPPORT_DISTANCE_FACTOR
    crop = distances > limit

    kept_points = vertices[~crop]
    removed_points = vertices[crop]
    cropped = o3d.geometry.TriangleMesh(mesh)
    cropped.remove_vertices_by_mask(crop)
    cropped.compute_vertex_normals()

    return {
        "limit": limit,
        # The control. If the reconstruction came back identical, a stable decision
        # downstream is not evidence that the decision is robust -- it is evidence that
        # nothing was asked of it. Without this the whole measurement is unreadable.
        "mesh_digest": hashlib.sha256(
            np.ascontiguousarray(np.asarray(uncropped.vertices)).tobytes()
        ).hexdigest(),
        "cloud_digest": hashlib.sha256(
            np.ascontiguousarray(np.asarray(cloud.points)).tobytes()
        ).hexdigest(),
        "normals_digest": hashlib.sha256(
            np.ascontiguousarray(np.asarray(cloud.normals)).tobytes()
        ).hexdigest(),
        "vertices": len(vertices),
        "removed": int(crop.sum()),
        "removed_fraction": float(crop.sum() / len(vertices)),
        "kept_points": kept_points,
        "removed_points": removed_points,
        "distances": distances,
        "uncropped_mesh": uncropped,
        "cropped_mesh": cropped,
        "cloud": cloud,
    }


def _cells(points: np.ndarray, origin: np.ndarray, size: float) -> set:
    index = np.floor((points - origin) / size).astype(np.int64)
    return set(map(tuple, index))


def _grid(cloud) -> tuple[np.ndarray, float]:
    """The consumer's cell centres and support radius for one cloud.

    One function because both arms need it and the first version of the second arm
    open-coded it slightly differently -- truncating where this rounds -- which moved
    the figure by about 50 cells and briefly looked like a finding about registration.
    """

    box = cloud.get_axis_aligned_bounding_box()
    origin = np.asarray(box.get_min_bound())
    dims = np.round(np.asarray(box.get_extent()) / CELL).astype(int)
    index = (
        np.array(np.meshgrid(*[np.arange(one) for one in dims], indexing="ij"))
        .reshape(3, -1)
        .T
    )
    return origin + (index + 0.5) * CELL, SUPPORT_RADIUS_FACTOR * CELL


def _occupancy(o3d, mesh, cloud, centres: np.ndarray, radius: float) -> dict:
    """The published consumer's own computation, on one repeat's mesh."""

    tree = o3d.geometry.KDTreeFlann(cloud)
    supported = np.array(
        [bool(tree.search_radius_vector_3d(centre, radius)[0]) for centre in centres]
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distance = scene.compute_distance(
        o3d.core.Tensor(centres.astype(np.float32))
    ).numpy()
    occupied = distance <= radius
    phantom = occupied & ~supported
    return {
        "occupied_cells": int(occupied.sum()),
        "phantom_obstacle_cells": int(phantom.sum()),
        "precision_wrt_observation": float(1 - phantom.sum() / occupied.sum()),
        "recall_of_observed_cells": float(
            (occupied & supported).sum() / supported.sum()
        ),
    }


def _how_the_meshes_differ(runs: list[dict]) -> dict:
    """Different vertices, or the same vertices in another order?

    A digest cannot tell those apart and they are not the same finding. Reordering is
    harmless to a threshold applied per vertex; genuinely different positions are the
    thing that could move a vertex across the cut. The crop's stability means something
    quite different depending on which happened, so it is measured rather than assumed.
    """

    def rows(mesh) -> np.ndarray:
        points = np.ascontiguousarray(np.asarray(mesh.vertices))
        dtype = [(axis, points.dtype) for axis in ("x", "y", "z")]
        return points.view(dtype).ravel()

    first = rows(runs[0]["uncropped_mesh"])
    differences = []
    for number, one in enumerate(runs[1:], start=2):
        other = rows(one["uncropped_mesh"])
        differences.append(
            {
                "repeat": number,
                "vertices_in_repeat_1_only": len(np.setdiff1d(first, other)),
                "vertices_in_this_repeat_only": len(np.setdiff1d(other, first)),
                "vertex_counts": [len(first), len(other)],
            }
        )

    worst = max(one["vertices_in_repeat_1_only"] for one in differences)
    return {
        "per_repeat": differences,
        "largest_set_difference": worst,
        "verdict": (
            "the same vertices in a different order"
            if worst == 0
            else "genuinely different vertex positions"
        ),
    }


def _how_far_the_movers_move(o3d, runs: list[dict], limit: float) -> dict:
    """How big is the displacement, against the threshold it would have to cross?

    This is what gives a reassuring result a scale. "The verdicts agreed" says more
    when the movement behind it is sub-micron against a 5 cm cut than when it is the
    same order as the cut -- but only as a comparison of two lengths. It does not say
    the decision had that much room: a vertex already sitting a micron inside the cut
    flips under a movement far smaller than the cut, so the cut is not the clearance.
    An earlier version of this docstring treated a sub-micron wobble as self-evidently
    safe because of its ratio to the cut, which is the inference this function's own
    output now refutes in `what_the_cut_to_separation_ratio_is_not`. What supports the verdict is
    the no-flip count, which is reported separately and belongs to these repeats.

    Measured as the nearest-neighbour distance between two repeats' vertex sets, in
    both directions. That is a *lower bound* on how far any individual vertex moved,
    not the movement itself: the two sets are not in correspondence, so a vertex that
    travelled can be charged the distance to whichever other-repeat vertex happens to
    sit nearest. The generated sentences used to call it "the largest distance any
    vertex moved", which is a stronger quantity than this measures, in a record whose
    value is the precision of its margin language. It was also computed in one
    direction only, which misses a vertex present in one repeat and absent from the
    other.
    """

    import numpy as np

    def one_way(source, target) -> np.ndarray:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(target)
        tree = o3d.geometry.KDTreeFlann(cloud)
        return np.array(
            [
                float(np.sqrt(tree.search_knn_vector_3d(vertex, 1)[2][0]))
                for vertex in source
            ]
        )

    first = np.asarray(runs[0]["uncropped_mesh"].vertices)
    worst = 0.0
    per_repeat = []
    for number, one in enumerate(runs[1:], start=2):
        vertices = np.asarray(one["uncropped_mesh"].vertices)
        forward = one_way(vertices, first)
        backward = one_way(first, vertices)
        nearest = np.concatenate([forward, backward])
        worst = max(worst, float(nearest.max()))
        per_repeat.append(
            {
                "repeat": number,
                "median_displacement_m": float(np.median(nearest)),
                "p99_displacement_m": float(np.percentile(nearest, 99)),
                "max_displacement_m": float(nearest.max()),
                "measured": (
                    "symmetric nearest-neighbour distance between this repeat's "
                    "vertex set and repeat 1's, a lower bound on per-vertex movement"
                ),
            }
        )

    pair_worsts = np.array([one["max_displacement_m"] for one in per_repeat])
    # Against *this* case's cut. `TAIL_OBSERVED_ELSEWHERE["cut_over_displacement"]` is
    # a ratio computed against the 5 cm cut, and taking the smaller of the two ratios
    # carried that cut into cases with a different one: at the 2 cm cut it published
    # 22.9x where 0.02 / 2.187e-03 = 9.1x is what the formula gives. The displacement
    # is the portable quantity; the ratio is not. Correcting the arithmetic does not
    # make the result a decision margin -- see the ratio's own disclaimer below.
    worst_known = max(worst, TAIL_OBSERVED_ELSEWHERE["displacement_m"])
    ratio = limit / worst_known

    return {
        "per_repeat": per_repeat,
        "the_distribution_of_the_per_pair_worst": {
            "why_a_distribution_and_not_one_number": (
                "The reported ratio is computed from `worst`, which makes `worst` the "
                "quantity it rests on -- and it is a tail statistic. A handful of "
                "repeats describes the middle of its distribution, not the case the "
                "margin has to survive. An independent lane hit one per-pair worst "
                "three orders of magnitude above its neighbours and could not "
                "reproduce it in three further attempts on the identical file, which "
                "is exactly the shape a four-repeat run cannot see. So the pairs are "
                "reported as a distribution and the headline ratio is taken against "
                "the worst value known anywhere rather than the worst seen here."
            ),
            "pairs_compared": int(pair_worsts.size),
            "min_m": float(pair_worsts.min()),
            "median_m": float(np.median(pair_worsts)),
            "p90_m": float(np.percentile(pair_worsts, 90)),
            "max_m": float(pair_worsts.max()),
            "spread_max_over_median": float(pair_worsts.max() / np.median(pair_worsts))
            if np.median(pair_worsts)
            else None,
        },
        "worst_displacement_m": worst,
        "the_threshold_it_would_have_to_cross_m": limit,
        # Both ratios were published as `margin_factor` and
        # `margin_at_the_worst_displacement_known_anywhere` before 2026-11-02. Same
        # formulas, renamed: see `what_the_cut_to_separation_ratio_is_not`.
        "cut_over_the_worst_separation_here": float(limit / worst) if worst else None,
        "the_tail_observed_elsewhere": dict(TAIL_OBSERVED_ELSEWHERE),
        "the_worst_displacement_known_anywhere_m": float(worst_known),
        "cut_over_the_worst_separation_known_anywhere": float(ratio),
        "how_that_ratio_is_computed": (
            "the_threshold_it_would_have_to_cross_m divided by "
            "the_worst_displacement_known_anywhere_m"
        ),
        "what_the_cut_to_separation_ratio_is_not": {
            "it_is_not_a_decision_margin": (
                "The crop keeps a vertex when its distance to the cloud is at most the "
                "cut, so what decides a verdict is the distance from that vertex to "
                "the cut -- |distance - cut| -- and not the cut itself. A vertex at "
                f"0.049999 m against a {TAIL_OBSERVED_ELSEWHERE['cut_m']} m cut is one "
                "micron from flipping, and the "
                f"{TAIL_OBSERVED_ELSEWHERE['displacement_m']:.3g} m observation above "
                "flips it, even though the cut is "
                f"{TAIL_OBSERVED_ELSEWHERE['cut_over_displacement']:.1f}x that "
                "movement. So this ratio does not say a decision clears its boundary "
                "by that factor, and no figure in this record says how close to the "
                "cut the vertices that matter actually sit."
            ),
            "what_it_is": (
                "A comparison of two lengths: how large this case's cut is relative to "
                "the largest separation observed between two repeats' vertex sets. It "
                "is useful for the scale it gives and it is not a safety factor."
            ),
            "what_does_support_the_verdict": (
                "The finite observation, which is in "
                "`the_per_cell_decision_a_user_reads`: across the pairs compared here, "
                "no cell changed its verdict. That is a count of what happened on "
                "these repeats, not a guarantee about another run."
            ),
            "what_would_be_needed_for_a_real_margin": (
                "Per-vertex distance-to-boundary clearance under a stated vertex "
                "correspondence, with the cardinality assumption declared. Neither is "
                "measured here, and this record does not substitute the ratio for it."
            ),
        },
        "what_that_means": (
            "A vertex only changes the answer by crossing the support threshold or "
            "landing in another cell. Across "
            f"{pair_worsts.size} pairs here the largest nearest-neighbour distance "
            f"between two repeats' vertex sets was {worst:.3g} m against a {limit} m "
            "cut, and no cell changed its verdict in any of those pairs. Taking the "
            "worst displacement known anywhere, including the unreproduced observation "
            f"above, which is {worst_known:.3g} m, this case's cut is {ratio:.1f}x that "
            "separation. That ratio is a comparison of two lengths and not a clearance: "
            "a vertex already sitting a micron inside the cut flips under a movement "
            "far smaller than the cut, so what is established here is the no-flip "
            "count above and the scale of the movement relative to the cut -- not that "
            "a rerun would decide the same way."
        ),
    }


def _what_a_user_may_rely_on(moved: float, agreement: list[dict]) -> str:
    """Written from the outcome, because both outcomes were possible when this ran."""

    worst = min(one["jaccard"] for one in agreement)
    if moved == 0:
        return (
            "The per-cell unsupported mask, not only the aggregate. Every repeat "
            "agreed on every cell -- Jaccard 1.0, no cell changed its verdict -- so "
            "the answer to 'which parts of my mesh are unsupported' is reproducible "
            "on this input even though the reconstruction it is computed from is not."
        )
    return (
        "The aggregate removed fraction to the precision its own spread shows. The "
        f"per-cell mask only to a Jaccard of {worst:.4f}: on average {moved:.0f} cells "
        "disagreed between two repeats about whether they are unsupported, so a "
        "per-region answer read off one run is not the answer another run gives."
    )


def _whether_that_is_benign(moved: float, straddling: float, displacement: dict) -> str:
    """The mechanism, which is what tells a reader when this stops holding."""

    worst = displacement["worst_displacement_m"]
    limit = displacement["the_threshold_it_would_have_to_cross_m"]
    pairs = displacement["the_distribution_of_the_per_pair_worst"]["pairs_compared"]
    ratio = displacement["cut_over_the_worst_separation_known_anywhere"]
    if moved == 0:
        return (
            f"Yes, on every comparison made here: across {pairs} pairs no cell "
            f"changed its verdict. Around {straddling:.0f} vertices per repeat sit "
            "within 10% of the cut, which is where a sharp threshold on a continuous "
            "quantity is vulnerable, and none of them crossed. That is the result -- a "
            "count of what happened on these repeats, not a measurement of how much "
            "room the decision had. Over the same pairs the largest nearest-neighbour "
            "distance "
            f"between two repeats' vertex sets was {worst:.3g} m against a {limit} m "
            "cut -- a lower bound on how far any vertex moved, since the two sets are "
            "not in correspondence -- and no cell changed its verdict in any of them. "
            "The ratio this record reports is taken against the worst displacement "
            "known anywhere rather than that, because the statistic has a tail an "
            f"earlier four-repeat run could not see: the cut is {ratio:.1f}x that "
            "displacement. Read that as scale and not as clearance. The decision "
            "boundary is at |distance - cut|, which nothing here measures, so a vertex "
            "already near the cut can flip under a movement far smaller than it. What "
            "transfers to another scan is the method: re-read one fused cloud, compare "
            "the per-cell verdicts pairwise, and report the separation next to the cut. "
            f"The {pairs}-pair no-flip count does not transfer. It was measured on this "
            "capture, at these repeats, with this cut, and a different scan has to "
            "produce its own."
        )
    return (
        f"Partly. Around {straddling:.0f} vertices per repeat sit within 10% of the "
        f"cut and the largest nearest-neighbour distance between repeats is "
        f"{worst:.3g} m against a {limit} m cut. A flip count of the same order as that band is a threshold "
        "behaving as a threshold; one much larger is a different problem."
    )


SCALING_TOLERANCE = 0.1
"""How close a ratio must sit to 1.0, or to the vertex ratio, to be called either.

Two densities give one ratio, so this cannot be a fitted confidence interval; it is a
declared band. It exists because the alternative is a two-way branch, and a two-way
branch has to hand the whole interval between its two cases to one of them. Ten percent
is wide enough that the band boundaries are not doing the work of the conclusion and
narrow enough that "proportional" still means proportional.
"""


def _how_displacement_scales(vertex_ratio: float, displacement_ratio: float) -> str:
    """Which of four things the two ratios support, or that they support none of them.

    Previously two branches split at `displacement_ratio >= vertex_ratio`, which gave
    the entire open interval between 1.0 and the vertex ratio to "fixed-magnitude
    artefact". At a displacement ratio of 4.2 against a vertex ratio of 4.29 that
    published "fixed-magnitude" for near-perfect proportional scaling. A band that wide
    is not a classification.
    """

    near_one = displacement_ratio <= 1.0 + SCALING_TOLERANCE
    near_vertices = (
        abs(displacement_ratio - vertex_ratio) <= SCALING_TOLERANCE * vertex_ratio
    )

    if near_one and near_vertices:
        # Only when the capture barely changed, and then neither name means anything.
        return "indeterminate"
    if near_one:
        return "shrinks" if displacement_ratio < 1.0 - SCALING_TOLERANCE else "fixed"
    if near_vertices:
        return "proportional"
    if displacement_ratio > vertex_ratio:
        return "faster_than_the_capture"
    return "partial"


_SCALING_READINGS = MappingProxyType(
    {
        "shrinks": (
            "The displacement fell as the capture got denser, so it is not something "
            "that scales with the cloud. The margin is therefore set by the cut, which "
            "a caller chooses, and not by the capture, which they do not."
        ),
        "fixed": (
            "The displacement held roughly constant as the capture got denser, so it "
            "is a fixed-magnitude artefact of the reconstruction. The margin is "
            "therefore set by the cut, which a caller chooses, and not by the capture, "
            "which they do not."
        ),
        "proportional": (
            "The displacement scaled with the capture, so the margin measured on one "
            "cloud does not transfer to a denser one and this record cannot "
            "generalise it."
        ),
        "faster_than_the_capture": (
            "The displacement grew faster than the capture did, so a denser cloud is "
            "worse than proportionally worse and this record cannot generalise to one."
        ),
        "partial": (
            "The displacement grew, but by appreciably less than the capture did. Two "
            "densities cannot separate partial scaling from a fixed artefact plus "
            "noise, so neither reading is supported and this record does not claim "
            "one. A third density would not settle it either: see the retained ratios "
            "below, where repeat runs of these same two settings have landed in "
            "different bands, so the quantity varies between runs and not only "
            "between densities."
        ),
        "indeterminate": (
            "The two densities are too close together for their ratio to establish "
            "anything about scaling. Treat this as one density measured twice."
        ),
    }
)


#: The same figure from earlier runs of this harness, carried for the same reason as
#: the displacement ratios below and with the same limitation: this harness does not
#: append to it, so it is the history someone wrote down rather than a log. It exists
#: because the figure is constant across every repeat *within* a run and still moved
#: between runs, and a record whose headline is "exactly reproducible under the crop"
#: has to say which of those two things it measured.
PHANTOM_FIGURE_ON_EARLIER_RUNS = MappingProxyType(
    {
        # Retained across separate runs: the scene voxel has read 733 and 734, the
        # fine voxel 920, 919 and 900. Within each run every repeat agreed. The 900 was
        # measured at 2026-11-02T20:59Z and is the widest, which is why the fine case's
        # retained span is 20 cells against the scene case's 1 -- a figure that is one
        # number inside a run is not one number across runs, and nothing here claims
        # the retained span is the whole of it.
        0.05: (733, 734),
        0.02: (920, 919, 900),
    }
)


#: The placeholder every provenance block carries until `main` substitutes the real
#: digest. Named rather than written twice, because two of the three blocks used to
#: ship with the literal still in them: the audit only inspects 64-hex values, so it
#: neither verified nor counted the placeholder, and the binding test read the
#: top-level block alone.
_TO_BE_FILLED_IN = "recomputed below"


RATIOS_OBSERVED_ON_EARLIER_RUNS = MappingProxyType(
    {
        # 0.888889 is attested by README.md at this commit ("0.8889 on the next two")
        # and was dropped from this constant rather than re-measured, which is the
        # hazard the docstring below describes. Its vertices ratio was not recorded.
        #
        # 2.121212 is the uncomfortable one, and it is retained for that reason: a run
        # during this repair measured 1.67e-05 m at the fine voxel against 7.87e-06 m
        # at the scene voxel, which is the denser case moving *further* and lands in a
        # different band from every other run. Dropping it would have left the record
        # concluding "every retained run lands in fixed/shrinks" over a list curated to
        # make that true. The margins are unaffected -- the tail observed elsewhere,
        # 2.187e-03 m, still dominates both cuts -- so what it costs is the scaling
        # scaling conclusion, not the observed verdicts.
        #
        # 0.717391, 1.220588 and 2.142857 are the three runs this record published
        # before this one: each was the current run's value at the time, and each would
        # have been dropped by the next regeneration, since this harness overwrites the
        # current value and never appends. They are added here by hand for that reason.
        # Without that step a correction that only touches prose silently shortens the
        # history the record publishes, which is the curation this constant exists to
        # prevent. An author who regenerates this record should append the value it
        # published before doing so.
        #
        # 2.142857 also matters on its own: it is a second reading in the band 2.121212
        # sits in, so the denser case moving further is no longer a single run.
        "displacement_denser_over_sparser": (
            0.548387,
            0.717391,
            0.888889,
            1.220588,
            2.121212,
            2.142857,
        ),
        "vertices_denser_over_sparser": (
            4.293651,
            4.267058,
            None,
            4.284116,
            4.262256,
            4.262260,
        ),
        "band_each_time": (
            "shrinks",
            "shrinks",
            "shrinks",
            "partial",
            "partial",
            "partial",
        ),
        "measured_at": (
            "commit 9c492ad33a of this harness, same host, same inputs",
            "2026-11-02T20:17Z, harness df8b7cb4fa, the record this one supersedes",
            "attested in README.md, not re-measured here",
            "2026-11-02T20:59Z, harness ffaed600ec, same host and inputs",
            "2026-11-02T19:50Z on this host, same inputs, REPEATS=24",
            "2026-11-02T21:05Z, harness 34cda72243, same host and inputs",
        ),
    }
)
"""The same quantity from an earlier run of this harness, kept rather than overwritten.

It is a ratio of two maxima of 23-pair distributions, so it inherits their tails twice
over and moves between harness runs far more than the figures it is built from. Quoting
one value for it as though it reproduced would repeat, one level up, exactly the defect
that made the margin a median dressed as a bound.
"""


def _is_the_ratio_reproducible(vertex_ratio: float, displacement_ratio: float) -> dict:
    """What a reader may rely on from this comparison, and what they may not.

    The band reproduced across runs; the ratio that selects it did not. Both facts
    belong next to the figure, because a reader who quotes the number is quoting noise
    and a reader who quotes the conclusion is not.
    """

    earlier = RATIOS_OBSERVED_ON_EARLIER_RUNS["displacement_denser_over_sparser"]
    band = _how_displacement_scales(vertex_ratio, displacement_ratio)
    bands = set(RATIOS_OBSERVED_ON_EARLIER_RUNS["band_each_time"]) | {band}
    every = (*earlier, displacement_ratio)

    return {
        "the_ratio_is_not": (
            # "Every ... this harness has produced" over a frozen literal the harness
            # never appends to. Each run replaced the current value and discarded the
            # previous one unless a human edited the constant, so the sentence was an
            # absolute claim over a curated list -- the same shape as the defect the
            # update section says it is fixing, one level up. It says what it is now.
            f"The {len(every)} displacement ratios this record retains, of however "
            "many runs there have been: "
            + ", ".join(f"{one:.4f}" for one in sorted(every))
            + ". The earlier ones are carried in a constant this harness does not "
            "append to, so this is the history someone wrote down, not a log. It is "
            "a ratio of two maxima, so it carries both tails and moves "
            "between runs by more than its own distance from the band edges. Do not "
            "quote it as a measurement of how displacement scales with density."
        ),
        "the_band_is": (
            f"Every retained run lands in {'/'.join(sorted(bands))}, and the readings for "
            "shrinks and fixed support the same conclusion -- displacement that does "
            "not grow with the capture, so the margin is set by the cut. That is what "
            "this comparison establishes."
            if bands <= {"shrinks", "fixed"}
            else "Runs have landed in different bands with different readings: "
            f"{'/'.join(sorted(bands))}. This comparison establishes nothing until "
            "that is resolved."
        ),
        "the_displacement_ratios_this_record_retains": [
            float(one) for one in sorted(every)
        ],
        "runs_this_is_drawn_from": len(every),
        "the_history_is_not_append_only": (
            "this harness writes the current run's ratio and carries the earlier ones "
            "from a hand-maintained constant, so a run whose ratio is not added to "
            "that constant leaves no trace here"
        ),
    }


def _what_density_establishes(
    vertex_ratio: float, displacement_ratio: float, worst: float
) -> str:
    """Generated from the comparison, because the direction is the whole claim.

    This sentence previously said "further" while its own two numbers said the denser
    mesh moved its vertices less far. Writing a directional claim next to the figures
    rather than deriving it from them is how that survives review of the figures.
    """

    # The reading below is one run's. Where the retained runs disagree about which
    # band they are in, this sentence has to say so rather than state the conclusion
    # that `is_this_ratio_itself_reproducible` withdraws two fields later -- a reader
    # who stops at the first of those two has been told the opposite of the record's
    # own verdict.
    retained_bands = {
        _how_displacement_scales(vertex_ratio, one)
        for one in RATIOS_OBSERVED_ON_EARLIER_RUNS["displacement_denser_over_sparser"]
    } | {_how_displacement_scales(vertex_ratio, displacement_ratio)}

    if displacement_ratio > 1.0:
        direction = f"{displacement_ratio:.2f}x further"
    elif displacement_ratio < 1.0:
        direction = f"{1.0 / displacement_ratio:.2f}x less far"
    else:
        direction = "exactly as far"

    band = _how_displacement_scales(vertex_ratio, displacement_ratio)

    reading = _SCALING_READINGS[band]
    if not retained_bands <= {"shrinks", "fixed"}:
        reading = (
            f"{reading} That is this run's reading only: the retained runs do not "
            f"agree on which band this is -- {'/'.join(sorted(retained_bands))} -- so "
            "`is_this_ratio_itself_reproducible` withdraws the conclusion and this "
            "sentence cannot be read as establishing it. What survives every run is "
            "the margin, which is quoted against the worst displacement known anywhere "
            "rather than against either of these."
        )

    return (
        f"A {vertex_ratio:.1f}x denser mesh moved its vertices {direction}. "
        f"{reading} "
        "Both meshes come from the same capture at two downsample voxels, and the cut "
        "and the normal radius move with the voxel, so this is two operating points "
        "rather than a density axis -- it does not establish that density is what "
        "changed the displacement. "
        f"A reader can compute their own margin: voxel * support_distance_factor over "
        f"about {worst:.1g} m."
    )


def _tail_bound(cases: dict) -> dict:
    """What 46 pairs can and cannot say about an event seen once somewhere else.

    The honest answer has two halves. The tail did not reproduce, which bounds its rate
    from above and is worth stating as a number. And a bound is not an absence: the
    event happened, so the published margin stays the conservative one regardless of
    what this run saw.
    """

    pairs = sum(
        one["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"][
            "the_distribution_of_the_per_pair_worst"
        ]["pairs_compared"]
        for one in cases.values()
    )
    here = max(
        one["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"][
            "the_distribution_of_the_per_pair_worst"
        ]["max_m"]
        for one in cases.values()
    )
    there = TAIL_OBSERVED_ELSEWHERE["displacement_m"]

    # One-sided 95% upper bound on a per-pair rate given zero events, 1 - 0.05 ** (1/n).
    # Stated as the method rather than a bare number so a reader can check it.
    rate_bound = 1.0 - 0.05 ** (1.0 / pairs) if pairs else None

    return {
        "pairs_compared_here": pairs,
        "worst_displacement_here_m": here,
        "the_event_being_looked_for_m": there,
        "how_many_times_larger_that_event_is": float(there / here),
        "did_it_recur": False,
        "upper_bound_on_its_per_pair_rate": rate_bound,
        "how_that_bound_was_computed": (
            f"Zero occurrences in {pairs} pairs. One-sided 95% upper bound on a "
            "per-pair rate is 1 - 0.05 ** (1 / n), which assumes the pairs are "
            "independent draws -- they share a fused cloud within a case, so treat it "
            "as indicative rather than exact."
        ),
        "what_this_does_and_does_not_settle": (
            f"The distribution here is tight: the worst of {pairs} pairs is "
            f"{here:.3g} m and the event looked for is {there / here:.0f}x larger than "
            "that, so whatever produced it is rare rather than routine. It does not "
            "settle that it cannot happen, and a bound is not an absence -- it was "
            "observed once under this same rule. The published margin therefore stays "
            "the conservative one. Raising repeats further would describe that tail "
            "better; it would not turn the cut-to-separation ratio into a clearance."
        ),
    }


def _spread(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "values": [float(one) for one in array],
        "distinct": len(set(array.tolist())),
        "spread": float(array.max() - array.min()),
        "relative_spread": (
            float((array.max() - array.min()) / abs(array.mean()))
            if array.mean()
            else 0.0
        ),
    }


def _two_arms(fixed: dict, rebuilt: dict, voxel: float) -> str:
    """Attribute the residual, which neither arm can do alone.

    The fixed-cloud arm varies Poisson only; the rebuilt arm varies registration too.
    The difference between their spreads is what says which stage the movement belongs
    to, and it is the sentence a reader of the published figure actually needs.
    """

    moving = rebuilt["phantom_cells_removed"]
    here = fixed["values"][0] if fixed["values"] else float("nan")
    observed = sorted({here, *PHANTOM_FIGURE_ON_EARLIER_RUNS.get(voxel, ())})
    if len(observed) > 1:
        across_runs = (
            f" Across the {len(observed)} values this record retains from separate "
            f"harness runs the figure is not one number -- "
            f"{', '.join(f'{one:.0f}' for one in observed)}, a span of "
            f"{max(observed) - min(observed):.0f} cell(s) -- so the constancy below is "
            "a within-run property, and two of these records compared against each "
            "other should be expected to differ by about that much."
        )
    else:
        across_runs = (
            f" The runs this record retains all read {here:.0f}, though it does not "
            "retain every run there has been."
        )
    if fixed["distinct"] == 1 and moving["spread"] > 0:
        return (
            "This harness's own phantom-cells-removed figure is exactly reproducible "
            "across repeats under the crop and is not exactly reproducible end to end. It is not the "
            "published number re-measured: this harness fuses its own cloud, and "
            "`what_this_does_not_establish` in this same case says the figures are not "
            "comparable to the record's published values -- so whether it lands on the "
            "733 in downstream-consumer.json is a coincidence to note rather than a "
            f"reproduction of it.{across_runs} Holding the fused cloud fixed, all "
            f"{len(fixed['values'])} repeats return {fixed['values'][0]:.0f} -- one "
            f"value, despite every repeat reconstructing a different mesh. Rebuilding "
            f"it, the figure moves across {moving['distinct']} values spanning "
            f"{moving['spread']:.0f} cell(s). So the residual belongs to the "
            "registration upstream, which is separately known not to be reproducible, "
            "and not to the crop or the reconstruction under it. A reader quoting the "
            "figure should quote it with that spread; a reader comparing two crop "
            "settings on one fused cloud can quote it exactly."
        )
    if fixed["distinct"] == 1 and moving["spread"] == 0:
        return (
            "The published figure reproduces under both arms, so neither the "
            "reconstruction nor the registration moves it on this input."
        )
    return (
        f"The figure is not exactly reproducible even with the fused cloud held fixed: "
        f"{fixed['distinct']} values spanning {fixed['spread']:.0f} cell(s) across "
        f"{len(fixed['values'])} repeats, against {moving['spread']:.0f} when "
        "registration varies too. The reconstruction alone moves it, so the crop's "
        "per-cell agreement does not carry all the way to this number."
    )


def _across_invocations(o3d, voxel: float, times: int) -> dict:
    """The same question with the fused cloud rebuilt, which is the user's situation.

    Everything else here holds the fused cloud fixed so that Poisson is the only thing
    varying -- without that the result is unattributable. But a user does not re-read
    one fused cloud; they run the stage, and registration is itself not reproducible
    (#1033). So the published figure has to be checked the second way too, or the record
    would answer a question nobody asks.

    This exists because the first version of this record claimed the downstream figure
    came back "exactly 733 every run" on the strength of the fixed-cloud arm, and a
    later invocation returned 734.
    """

    figures, clouds, vertices = [], set(), []
    for _ in range(times):
        with tempfile.TemporaryDirectory() as scratch:
            fused = _fused_cloud(o3d, Path(scratch))
            clouds.add(hashlib.sha256(fused.read_bytes()).hexdigest())
            run = _one_crop(o3d, fused)

        cloud = run["cloud"]
        centres, radius = _grid(cloud)

        before = _occupancy(o3d, run["uncropped_mesh"], cloud, centres, radius)
        after = _occupancy(o3d, run["cropped_mesh"], cloud, centres, radius)
        figures.append(
            before["phantom_obstacle_cells"] - after["phantom_obstacle_cells"]
        )
        vertices.append(run["vertices"])

    spread = _spread([float(one) for one in figures])
    return {
        "why_this_arm_exists": (
            "The fixed-cloud arm isolates Poisson, which is what makes its result "
            "attributable. It is not what a user runs. Here the fused cloud is rebuilt "
            "every time, so registration varies as well, and the question is whether "
            "the published per-cell figure survives that."
        ),
        "invocations": times,
        "distinct_fused_clouds": len(clouds),
        "vertices_per_invocation": vertices,
        "phantom_cells_removed": spread,
        "what_it_shows": (
            f"{spread['distinct']} distinct value(s) across {times} invocations, "
            f"{sorted({int(one) for one in figures})}, a spread of "
            f"{spread['spread']:.0f} cell(s) on a figure of about "
            f"{np.median(figures):.0f} -- {spread['spread'] / np.median(figures):.2%} "
            "of it. This arm lets registration vary as well, so it does not attribute "
            "that spread to anything on its own; compare it with the fixed-cloud arm "
            "above."
        ),
    }


def _one_case(o3d, voxel: float) -> dict:
    """Everything for one voxel size: the control, the decision, the consumer."""

    with tempfile.TemporaryDirectory() as scratch:
        fused = _fused_cloud(o3d, Path(scratch))
        fused_digest = hashlib.sha256(fused.read_bytes()).hexdigest()
        runs = [_one_crop(o3d, fused) for _ in range(REPEATS)]

        cloud = runs[0]["cloud"]
        origin = np.asarray(cloud.get_axis_aligned_bounding_box().get_min_bound())
        centres, radius = _grid(cloud)

        consumer = [
            {
                "repeat": number,
                "baseline_uncropped": _occupancy(
                    o3d, one["uncropped_mesh"], one["cloud"], centres, radius
                ),
                "candidate_cropped": _occupancy(
                    o3d, one["cropped_mesh"], one["cloud"], centres, radius
                ),
            }
            for number, one in enumerate(runs, start=1)
        ]

        fixed_arm = _spread(
            [
                one["baseline_uncropped"]["phantom_obstacle_cells"]
                - one["candidate_cropped"]["phantom_obstacle_cells"]
                for one in consumer
            ]
        )
        rebuilt = _across_invocations(o3d, voxel, INVOCATIONS)

        # Per-cell verdicts, at the scene voxel, which is the form the question is
        # actually asked in: is *this part* of my mesh supported.
        removed_cells = [_cells(one["removed_points"], origin, VOXEL) for one in runs]
        kept_cells = [_cells(one["kept_points"], origin, VOXEL) for one in runs]

        agreement = []
        for number in range(1, REPEATS):
            first_removed, other_removed = removed_cells[0], removed_cells[number]
            union = first_removed | other_removed
            shared = first_removed & other_removed
            # A cell whose verdict genuinely flipped: removed in one run, kept and not
            # removed in the other. Cells that simply vanish are a different defect.
            flipped = (first_removed - other_removed) & kept_cells[number] | (
                other_removed - first_removed
            ) & kept_cells[0]
            agreement.append(
                {
                    "repeat": number + 1,
                    "removed_cells_in_repeat_1": len(first_removed),
                    "removed_cells_here": len(other_removed),
                    "cells_both_called_unsupported": len(shared),
                    "jaccard": float(len(shared) / len(union)) if union else 1.0,
                    "cells_whose_verdict_flipped": len(flipped),
                }
            )

        # Are the movers sitting on the threshold? If so the instability is the
        # expected behaviour of a sharp cut through a continuous quantity, and saying
        # so is part of the answer; if not, it is a different and worse problem.
        limit = runs[0]["limit"]
        near = []
        for one in runs:
            distances = one["distances"]
            band = np.abs(distances - limit) <= 0.1 * limit
            near.append(
                {
                    "vertices_within_10pc_of_the_limit": int(band.sum()),
                    "as_fraction_of_all_vertices": float(band.sum() / len(distances)),
                    "removed_vertices": one["removed"],
                }
            )
        displacement = _how_far_the_movers_move(o3d, runs, limit)
        moved = float(
            np.mean([one["cells_whose_verdict_flipped"] for one in agreement])
        )
        straddling = float(
            np.mean([one["vertices_within_10pc_of_the_limit"] for one in near])
        )

        case = {
            "what_this_measures": (
                "Whether the support crop returns the same decision twice, as opposed "
                "to the same average. The fused cloud is built once and re-read for "
                "every repeat, so registration cannot contribute."
            ),
            "why_cells_and_not_vertices": (
                "Poisson returns a different vertex set each run, so no vertex can be "
                "followed between repeats and 'this vertex changed its mind' is not a "
                "question that can be asked. A consumer does not ask it either: it "
                "asks whether a region of space is occupied. Cells at the scene voxel "
                "are that question, and are the form the published occupancy consumer "
                "already works in."
            ),
            "input": {
                "dataset": "open3d.data DemoICPPointClouds, the record's own input",
                "voxel_size": VOXEL,
                "fused_cloud_sha256": fused_digest,
                "fused_once_and_reused": True,
            },
            "crop_rule": (
                f"a vertex is unsupported when its distance to the fused cloud exceeds "
                f"voxel * {SUPPORT_DISTANCE_FACTOR} = {limit}"
            ),
            "repeats": REPEATS,
            "did_the_reconstruction_vary_at_all": {
                "why_this_is_first": (
                    "A stable decision means nothing if nothing upstream moved. The "
                    "same mistake sank a thread experiment on this record: the lever "
                    "never engaged, so the result measured nothing. Read this block "
                    "before any figure below it."
                ),
                "distinct_fused_clouds": len({one["cloud_digest"] for one in runs}),
                "distinct_normal_sets": len({one["normals_digest"] for one in runs}),
                "distinct_poisson_meshes": len({one["mesh_digest"] for one in runs}),
                "poisson_varied_between_repeats": (
                    len({one["mesh_digest"] for one in runs}) > 1
                ),
                "and_whether_that_is_only_reordering": _how_the_meshes_differ(runs),
                "and_how_far_it_moved": displacement,
            },
            "the_aggregate_the_record_reports": {
                "removed_vertex_count": _spread([one["removed"] for one in runs]),
                "removed_fraction": _spread([one["removed_fraction"] for one in runs]),
                "total_vertices": _spread([one["vertices"] for one in runs]),
            },
            "the_per_cell_decision_a_user_reads": agreement,
            "where_the_movers_sit": near,
            "the_published_downstream_figure": {
                "per_repeat": consumer,
                "phantom_cells_removed_by_the_crop": fixed_arm,
                "precision_after_the_crop": _spread(
                    [
                        one["candidate_cropped"]["precision_wrt_observation"]
                        for one in consumer
                    ]
                ),
                "with_the_fused_cloud_rebuilt_instead": rebuilt,
                "the_same_figure_on_retained_earlier_runs": [
                    float(one) for one in PHANTOM_FIGURE_ON_EARLIER_RUNS.get(voxel, ())
                ],
                "what_the_two_arms_together_show": _two_arms(fixed_arm, rebuilt, voxel),
            },
            "what_a_user_may_rely_on": _what_a_user_may_rely_on(moved, agreement),
            "whether_that_is_benign": _whether_that_is_benign(
                moved, straddling, displacement
            ),
            "what_this_does_not_establish": (
                "Nothing about the cause inside Poisson, which #1024 covers, and "
                "nothing about whether the crop rule is the right rule. The figures "
                "are not comparable to the record's published values: this harness "
                "mirrors the reconstruct-and-crop stage rather than running the full "
                "six-stage workflow, and its fused cloud is its own."
            ),
            "reproducibility": {
                "command": (
                    "npa/.venv/bin/python "
                    "docs/evidence/open3d/harness/crop_decision_stability.py"
                ),
                "note": (
                    "Every figure here is a between-run comparison, so the figures are "
                    "expected to move. The verdicts are what reproduce."
                ),
            },
            "provenance": {
                "harness": "harness/crop_decision_stability.py",
                "harness_sha256": _TO_BE_FILLED_IN,
                "path_is_relative_to": "this evidence directory",
            },
        }
    return case


def main() -> int:
    import open3d as o3d

    cases = {}
    for voxel in VOXELS:
        global VOXEL
        VOXEL = voxel
        cases[f"voxel_{voxel}"] = _one_case(o3d, voxel)

    displacements = {
        key: one["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"][
            "worst_displacement_m"
        ]
        for key, one in cases.items()
    }
    densities = {
        key: one["the_aggregate_the_record_reports"]["total_vertices"]["values"][0]
        for key, one in cases.items()
    }
    # Both ratios are the denser case over the sparser case. They were previously
    # max/min of their own values, which silently ran the displacement ratio in the
    # opposite direction to the vertex ratio -- the two numbers disagreed about which
    # case was the numerator, and the sentence below them said "further" when the
    # denser mesh in fact moved its vertices less far.
    sparser, denser = sorted(densities, key=densities.get)

    vertex_ratio = float(densities[denser] / densities[sparser])
    displacement_ratio = float(displacements[denser] / displacements[sparser])

    record = {
        "what_this_measures": (
            "Whether the support crop returns the same decision twice, as opposed to "
            "the same average, and whether that survives a denser capture. The fused "
            "cloud is built once per case and re-read for every repeat, so "
            "registration cannot contribute."
        ),
        "cases": cases,
        "does_the_tail_reproduce_here": _tail_bound(cases),
        "does_the_margin_survive_a_denser_capture": {
            "why_this_is_asked": (
                "The first case is a sparse fused cloud, and a margin measured there is "
                "worth nothing if the wobble grows with the geometry. The denser case "
                "exists to find out, because that is the first thing a reader with a "
                "real capture would ask."
            ),
            "what_differs_between_the_two_cases": (
                "The same capture, not a second one. Both cases fuse the same three "
                f"upstream Redwood fragments; the density difference comes from "
                f"`voxel_down_sample` at {VOXELS[0]} against {VOXELS[1]}, which is a "
                "downsample parameter. And changing that voxel moves four things at "
                "once: the fused-cloud density, the normal-estimation radius "
                "(voxel * 2.0), the crop threshold itself (voxel * "
                f"{SUPPORT_DISTANCE_FACTOR}, so {VOXELS[0] * SUPPORT_DISTANCE_FACTOR} "
                f"against {VOXELS[1] * SUPPORT_DISTANCE_FACTOR} m) and the cell size "
                "the per-cell verdicts are compared on. So this bounds the margin at "
                "two operating points of the product's own configuration. It does not "
                "isolate capture density, and the displacement ratio below cannot be "
                "attributed to density alone. Isolating it would need the voxel held "
                "while the input changes, which is a different experiment."
            ),
            "both_ratios_are": f"{denser} divided by {sparser}, the denser over the sparser",
            "vertices_per_case": densities,
            "worst_displacement_per_case_m": displacements,
            "vertices_denser_over_sparser": vertex_ratio,
            "displacement_denser_over_sparser": displacement_ratio,
            "how_the_displacement_scales": _how_displacement_scales(
                vertex_ratio, displacement_ratio
            ),
            "scaling_band_tolerance": SCALING_TOLERANCE,
            "is_this_ratio_itself_reproducible": _is_the_ratio_reproducible(
                vertex_ratio, displacement_ratio
            ),
            "what_that_establishes": _what_density_establishes(
                vertex_ratio, displacement_ratio, max(displacements.values())
            ),
        },
        "why_cells_and_not_vertices": (
            "Poisson returns a different vertex set each run, so no vertex can be "
            "followed between repeats and 'this vertex changed its mind' is not a "
            "question that can be asked. A consumer does not ask it either: it asks "
            "whether a region of space is occupied. Cells at the scene voxel are that "
            "question, and are the form the published occupancy consumer works in."
        ),
        "reproducibility": {
            "command": (
                "npa/.venv/bin/python "
                "docs/evidence/open3d/harness/crop_decision_stability.py"
            ),
            "note": (
                "Every figure here is a between-run comparison, so the figures are "
                "expected to move. The verdicts are what reproduce."
            ),
        },
        "provenance": {
            "harness": "harness/crop_decision_stability.py",
            "harness_sha256": _TO_BE_FILLED_IN,
            "path_is_relative_to": "this evidence directory",
        },
    }

    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def fill(node: object) -> None:
        """Substitute every provenance block, not only the top-level one."""

        if isinstance(node, dict):
            if node.get("harness_sha256") == _TO_BE_FILLED_IN:
                node["harness_sha256"] = digest
            for value in node.values():
                fill(value)
        elif isinstance(node, list):
            for value in node:
                fill(value)

    fill(record)
    assert _TO_BE_FILLED_IN not in json.dumps(record), (
        "a provenance block was left unsubstituted, which publishes a placeholder "
        "where a binding should be"
    )
    json.dump(record, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
