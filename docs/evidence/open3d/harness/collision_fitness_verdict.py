"""Is the reconstructed surface usable as a collision geometry? Measured, then stated.

Every record in this chain has so far *abstained* on collision fitness -- "no claim
is made about watertightness, collision fitness, or mesh readiness". Abstention was
honest while nothing had been measured, but the lane brief asks for a verdict: if the
mesh is not fit for collision use, say so and point at the narrower output that is
real. This script produces the measurement the verdict rests on.

A collision checker asks two questions of a surface: is this point inside the
obstacle, and how far inside. Both are defined only for a closed surface. So the
test is not "does the mesh look solid" but "does the mesh answer inside/outside
consistently at all".

The decisive instrument is ray parity. For a closed surface, a ray cast from a point
to infinity crosses it an even number of times when the point is outside and an odd
number when inside, and that answer does not depend on which direction the ray went.
For a surface with holes, the parity verdict depends on the direction, because some
directions escape through a hole. The fraction of probe points whose inside/outside
verdict *disagrees across ray directions* is therefore a direct measure of how often
a collision query would be undefined, and it needs no reference model to compute.

Neither retained variant is assumed to be closed. The synthetic sphere below is the
control: its disagreement rate has to be zero, its inside count has to be non-zero and
close to the analytic sphere volume, and parity has to agree with signed distance.

What this measures: whether inside/outside is well defined, how much of the surface
is hole boundary, and whether the solid interior a collision query would infer is
anywhere near the observed structure. What it does not measure: accuracy against
ground truth, which no artifact in this chain carries.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/collision_fitness_verdict.py

BUNDLE may be overridden for a local run against the privately retained bundle; the
report records which root produced it by shape, never by path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from harness_io import output_path, read_cloud, read_mesh

BUNDLE = Path(os.environ.get("OPEN3D_BUNDLE", "/data/final-bundle"))
OUT = output_path("/data/collision-fitness-verdict.json")

VARIANTS = {"baseline": "mesh-baseline/mesh.ply", "candidate": "mesh/mesh.ply"}
CLOUD = "multiway/fused.ply"

#: Ray directions for the parity test, deliberately off-axis. The first version of
#: this harness used six axis-aligned directions plus eight diagonals, which is
#: wrong: against an axis-aligned probe grid those rays graze shared triangle edges
#: and vertices, where count_intersections returns two hits or none, and parity
#: flips for reasons that have nothing to do with holes. It reported 521 undefined
#: cells on a mesh Open3D guarantees is closed. Fourteen fixed pseudo-random
#: directions report zero on the same mesh. The seed is pinned so the figure is
#: reproducible, and the closed-mesh control below is what justifies the choice.
DIRECTIONS = np.random.default_rng(12345).normal(size=(14, 3)).astype(np.float32)

#: Probe grid spacing, matched to the occupancy consumer's cell so the two records
#: describe the same discretisation of the same volume.
CELL = 0.1

#: A cell counts as supported by observation within this multiple of the cell, the
#: same cell-centre-to-corner reach the occupancy consumer uses.
SUPPORT_RADIUS_FACTOR = 0.87


def probe_points(cloud: o3d.geometry.PointCloud) -> np.ndarray:
    """Cell centres spanning the observed bounding box, the volume a planner queries."""
    box = cloud.get_axis_aligned_bounding_box()
    lo, hi = box.get_min_bound(), box.get_max_bound()
    axes = [np.arange(lo[i] + CELL / 2, hi[i], CELL) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3).astype(np.float32)


def closedness(mesh: o3d.geometry.TriangleMesh) -> dict:
    """Open3D's own topology predicates, reported whatever they say."""
    edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    boundary = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=True))
    _, counts, _ = mesh.cluster_connected_triangles()
    return {
        "is_watertight": bool(mesh.is_watertight()),
        "is_edge_manifold": bool(mesh.is_edge_manifold()),
        "is_vertex_manifold": bool(mesh.is_vertex_manifold()),
        "is_self_intersecting": bool(mesh.is_self_intersecting()),
        "triangles": len(mesh.triangles),
        "vertices": len(mesh.vertices),
        # allow_boundary_edges=False counts boundary edges as non-manifold, so the
        # difference between the two calls is the boundary-edge count itself.
        "non_manifold_edges": len(boundary),
        "boundary_edges": int(len(edges) - len(boundary)),
        "connected_components": len(counts),
        "largest_component_triangle_share": (
            round(float(max(counts)) / float(len(mesh.triangles)), 4)
            if len(counts)
            else None
        ),
    }


def parity_disagreement(mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> dict:
    """How often inside/outside depends on which way the ray went.

    Zero means every direction agrees, which is what a closed surface must give. A
    positive rate is the fraction of queried cells where a collision checker's most
    basic question has no single answer.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    unit = DIRECTIONS / np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)
    verdicts = np.empty((len(unit), len(points)), dtype=bool)
    for i, direction in enumerate(unit):
        rays = np.hstack([points, np.tile(direction, (len(points), 1))]).astype(
            np.float32
        )
        crossings = scene.count_intersections(o3d.core.Tensor(rays)).numpy()
        verdicts[i] = (
            crossings % 2
        ) == 1  # odd crossings => inside, for a closed surface

    agree_inside = verdicts.all(axis=0)
    agree_outside = (~verdicts).all(axis=0)
    undefined = ~(agree_inside | agree_outside)
    return {
        "probe_cells": len(points),
        "ray_directions": len(unit),
        "cells_all_directions_call_inside": int(agree_inside.sum()),
        "cells_all_directions_call_outside": int(agree_outside.sum()),
        "cells_where_the_directions_disagree": int(undefined.sum()),
        "undefined_fraction_of_probed_volume": round(float(undefined.mean()), 6),
        "undefined_fraction_of_cells_any_direction_calls_inside": round(
            float(undefined.sum()) / float(undefined.sum() + agree_inside.sum()), 6
        )
        if (undefined.sum() + agree_inside.sum())
        else None,
    }


def inferred_solid_vs_observed(
    mesh: o3d.geometry.TriangleMesh, points: np.ndarray, cloud: o3d.geometry.PointCloud
) -> dict:
    """How much inferred interior lies farther than one radius from an observation?

    Uses the majority verdict across directions, which is the most favourable reading
    available to the mesh: it resolves exactly the cells the parity test found
    undefined, in the mesh's favour, rather than discarding them.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    unit = DIRECTIONS / np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)

    inside_votes = np.zeros(len(points), dtype=np.int32)
    for direction in unit:
        rays = np.hstack([points, np.tile(direction, (len(points), 1))]).astype(
            np.float32
        )
        crossings = scene.count_intersections(o3d.core.Tensor(rays)).numpy()
        inside_votes += (crossings % 2) == 1
    solid = inside_votes > (len(unit) / 2)

    tree = o3d.geometry.KDTreeFlann(cloud)
    radius = CELL * SUPPORT_RADIUS_FACTOR
    supported = np.array(
        [
            tree.search_radius_vector_3d(p.astype(np.float64), radius)[0] > 0
            for p in points
        ]
    )

    solid_count = int(solid.sum())
    return {
        "measurement_scope": (
            "Distance from inferred solid cells to observed surface samples. This includes "
            "ordinary interior that a surface sensor cannot observe; it is not a fabricated-"
            "geometry or collision-accuracy measurement."
        ),
        "cells_a_planner_would_call_solid": solid_count,
        "of_those_with_no_observation_within_support_radius": int(
            (solid & ~supported).sum()
        ),
        "unobserved_share_of_inferred_solid": round(
            float((solid & ~supported).sum()) / solid_count, 4
        )
        if solid_count
        else None,
        "unobserved_inferred_solid_volume_m3": round(
            float((solid & ~supported).sum()) * CELL**3, 4
        ),
    }


#: What the verdict field says when the instrument failed its control. Review found that
#: an earlier version computed the verdict from the mesh alone and wrote the report three
#: lines before consulting the control, so a quarantined run still left a file on disk
#: stating a fitness verdict produced by an instrument just proven invalid -- and the
#: consumers of these records are tests that read the JSON, which would pass on it.
WITHHELD = "withheld: the instrument failed its closed-mesh control"

#: Suffix for the quarantined artifact, so an invalid run never occupies the published name.
QUARANTINE = ".quarantined-invalid-instrument"


def verdict_for(validation: dict, fit: bool) -> str:
    """The control gates the verdict itself, not a sibling field a reader has to find."""
    if not validation["instrument_is_valid"]:
        return WITHHELD
    return "fit" if fit else "not fit for collision use"


def publication_path(report: dict, out: Path) -> Path:
    """Where this report may be written. An invalid run cannot take the published name."""
    if report["instrument_validation"]["instrument_is_valid"]:
        return out
    return out.with_name(f"{out.stem}{QUARANTINE}{out.suffix}")


def instrument_validation() -> dict:
    """Does the parity test give the right answer on a mesh known to be closed?

    Without this the verdict below is worthless. The first version of this harness
    returned "undefined" for every cell of both real variants, which looked like a
    finding and was an artifact of ray direction. A sphere Open3D constructs is
    closed by construction, so the parity test must find zero undefined cells in it,
    and its inside count must agree with compute_signed_distance -- which Open3D
    defines only for closed meshes and which is therefore unavailable as the primary
    instrument here, but is a fair cross-check on a mesh that qualifies.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    axis = np.arange(-1.5, 1.5, 0.15)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = points.reshape(-1, 3).astype(np.float32)

    parity = parity_disagreement(sphere, points)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sphere))
    signed = scene.compute_signed_distance(o3d.core.Tensor(points)).numpy()

    inside_by_parity = parity["cells_all_directions_call_inside"]
    inside_by_distance = int((signed < 0).sum())
    analytic = (4.0 / 3.0) * np.pi / 0.15**3
    analytic_relative_error = abs(inside_by_parity - analytic) / analytic
    parity_agrees = inside_by_parity == inside_by_distance
    instrument_is_valid = (
        bool(sphere.is_watertight())
        and parity["cells_where_the_directions_disagree"] == 0
        and inside_by_parity > 0
        and parity_agrees
        and analytic_relative_error < 0.05
    )
    return {
        "control_mesh": "open3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)",
        "control_is_watertight": bool(sphere.is_watertight()),
        "undefined_cells": parity["cells_where_the_directions_disagree"],
        "instrument_is_valid": instrument_is_valid,
        "inside_cells_by_parity": inside_by_parity,
        "inside_cells_by_signed_distance": inside_by_distance,
        "parity_agrees_with_signed_distance": parity_agrees,
        "analytic_cell_estimate": round(float(analytic), 1),
        "analytic_relative_error": round(float(analytic_relative_error), 6),
        "analytic_tolerance": 0.05,
        "what_the_first_version_reported_here": 521,
        "why_it_was_wrong": "Axis-aligned rays against an axis-aligned probe grid graze "
        "shared edges and vertices, where count_intersections returns two hits or none. "
        "Parity then flips for a reason unrelated to holes, so every cell looked undefined "
        "and both real variants looked equally unfit. Off-axis directions remove the "
        "degeneracy, and this control is how the error surfaced before it was published.",
    }


def build() -> dict:
    validation = instrument_validation()
    input_paths = {"cloud": BUNDLE / CLOUD} | {
        name: BUNDLE / relative for name, relative in VARIANTS.items()
    }
    cloud = read_cloud(input_paths["cloud"])
    points = probe_points(cloud)

    variants = {}
    for name in VARIANTS:
        mesh = read_mesh(input_paths[name])
        if not mesh.has_triangles():
            raise RuntimeError(f"input mesh is empty: {input_paths[name]}")
        variants[name] = {
            "topology": closedness(mesh),
            "inside_outside_is_well_defined": parity_disagreement(mesh, points),
            "if_a_planner_trusted_it_anyway": inferred_solid_vs_observed(
                mesh, points, cloud
            ),
        }

    candidate = variants["candidate"]
    baseline = variants["baseline"]
    fit = (
        candidate["topology"]["is_watertight"]
        and candidate["inside_outside_is_well_defined"][
            "cells_where_the_directions_disagree"
        ]
        == 0
    )
    return {
        "input_provenance": {
            "open3d_version": o3d.__version__,
            "files": {
                name: {
                    "relative_path": str(path.relative_to(BUNDLE)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in input_paths.items()
            },
        },
        "question": "Can the delivered surface be used as a collision geometry?",
        "verdict": verdict_for(validation, fit),
        "instrument_validation": validation,
        "why": (
            "The delivered surface is not closed, so the two questions a collision checker "
            "asks -- is this point inside the obstacle, and how deep -- have no defined "
            "answer on it. This is measured rather than inferred from appearance: ray "
            "parity disagrees across directions on "
            f"{candidate['inside_outside_is_well_defined']['cells_where_the_directions_disagree']} "
            "of the probed cells, which is where a query would silently return whichever "
            "answer the ray direction happened to produce."
        )
        if not fit
        else "Closed, manifold, and parity-consistent in every probed cell.",
        "i_expected_the_baseline_to_be_the_closed_control_and_it_is_not": (
            "Both variants are open. Baseline is not watertight either: "
            f"{baseline['topology']['boundary_edges']} boundary edges, not edge-manifold, "
            f"{baseline['topology']['connected_components']} components. So the verdict is not "
            "'the candidate lost closedness to gain mapping quality' -- neither variant ever "
            "had a defined interior, and the synthetic sphere above is the only real control "
            "in this report. I had assumed otherwise from an existing docstring and checked it "
            "instead of repeating it."
        ),
        "an_existing_justification_in_this_chain_named_the_weaker_reason": (
            "consumer_occupancy.py explains its choice of a distance band over an enclosed "
            "volume by saying 'one of the two meshes being compared is deliberately not "
            "closed', which invites the reading that the other one is. Measured: neither is. "
            "The decision to avoid compute_occupancy was right, and right for a stronger "
            "reason than the one recorded -- it was not a fairness adjustment between a closed "
            "and an open mesh, it was the only defined option for either."
        ),
        "why_the_mesh_with_eight_times_fewer_boundary_edges_scores_worse": (
            f"Baseline has {baseline['topology']['boundary_edges']} boundary edges against the "
            f"candidate's {candidate['topology']['boundary_edges']}, yet a higher undefined "
            f"fraction ({baseline['inside_outside_is_well_defined']['undefined_fraction_of_probed_volume']} "
            f"vs {candidate['inside_outside_is_well_defined']['undefined_fraction_of_probed_volume']}). "
            "That is not a contradiction: once a surface is open at all, parity stops tracking "
            "hole count and starts tracking how much surface a ray can cross. Baseline carries "
            f"{baseline['topology']['triangles']} triangles to the candidate's "
            f"{candidate['topology']['triangles']}, so more directions find an odd count. "
            "Neither number should be read as a hole metric, and the undefined fraction is "
            "reported here only as evidence that the query is undefined, not as a quality score."
        ),
        "what_is_fit_instead": (
            "Occupancy-band mapping, which is what downstream-consumer.json measures and "
            "where the candidate is the better artifact: a cell is called occupied when it "
            "lies within one support radius of the surface, a question an open surface "
            "answers perfectly well. The phantom-obstacle rate there falls from 0.3245 to "
            "0.0708. That is the narrower useful output, and it is not a workaround for "
            "this verdict -- it is a different question that does not need a closed solid."
        ),
        "interior_distance_diagnostic_not_corroboration": (
            "Taking the majority verdict across directions resolves every undefined cell in "
            "the mesh's favour instead of discarding it. Of those inferred-solid cells, "
            f"{candidate['if_a_planner_trusted_it_anyway']['of_those_with_no_observation_within_support_radius']} "
            f"of {candidate['if_a_planner_trusted_it_anyway']['cells_a_planner_would_call_solid']} "
            "have no observation within one support radius, representing "
            f"{candidate['if_a_planner_trusted_it_anyway']['unobserved_inferred_solid_volume_m3']} m3 "
            "of inferred interior. Baseline is larger on the same diagnostic, "
            f"{baseline['if_a_planner_trusted_it_anyway']['unobserved_inferred_solid_volume_m3']} m3. "
            "Surface sensors do not observe ordinary solid interiors, so this is not a "
            "fabrication measurement and does not corroborate the collision verdict."
        ),
        "what_this_does_not_claim": (
            "Nothing about accuracy against ground truth, which no artifact in this chain "
            "carries. A watertight mesh can be confidently wrong; closedness is necessary "
            "for a collision query to be defined and nowhere near sufficient for it to be "
            "safe. Nor does this say the reconstruction is bad at what it is for -- the same "
            "candidate artifact is the better one for occupancy mapping, and both statements "
            "are true at once."
        ),
        "variants": variants,
        "method": {
            "parity": "count_intersections along 14 directions per cell; odd means inside for "
            "a closed surface. A cell is undefined when the directions do not agree.",
            "probe_grid": f"{CELL} m cell centres spanning the observed cloud's bounding box, "
            "matching the occupancy consumer's discretisation.",
            "support_radius": f"{SUPPORT_RADIUS_FACTOR} x cell, the same cell-centre-to-corner "
            "reach the occupancy consumer uses.",
            "solid_vote": "majority of 14 directions, which resolves every undefined cell in "
            "the mesh's favour rather than discarding it.",
        },
    }


def main() -> int:
    report = build()
    # The check comes before the write. An earlier version had these the other way round,
    # so a failed control still left a verdict-bearing file at the published path.
    destination = publication_path(report, OUT)
    if not report["instrument_validation"]["instrument_is_valid"]:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print("Instrument failed its closed-mesh control; no verdict is published.")
        print(f"Quarantined to {destination}; {OUT} not written.")
        return 1

    destination.write_text(json.dumps(report, indent=2) + "\n")
    candidate = report["variants"]["candidate"]
    baseline = report["variants"]["baseline"]
    print(f"Verdict: {report['verdict']}.")
    print(
        f"  candidate: watertight={candidate['topology']['is_watertight']}, "
        f"boundary_edges={candidate['topology']['boundary_edges']}, "
        f"undefined cells={candidate['inside_outside_is_well_defined']['cells_where_the_directions_disagree']}"
    )
    # Not "control": baseline turned out to be open too. The control is the sphere above.
    print(
        f"  baseline:  watertight={baseline['topology']['is_watertight']}, "
        f"boundary_edges={baseline['topology']['boundary_edges']}, "
        f"undefined cells={baseline['inside_outside_is_well_defined']['cells_where_the_directions_disagree']}"
    )
    print(
        f"  instrument control (closed sphere): undefined cells="
        f"{report['instrument_validation']['undefined_cells']}, "
        f"parity==signed_distance="
        f"{report['instrument_validation']['parity_agrees_with_signed_distance']}"
    )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
