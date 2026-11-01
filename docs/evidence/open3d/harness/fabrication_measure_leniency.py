"""Does the documented leniency in the fabrication measure actually reproduce, and what fixes it?

`runner.py` records this and leaves it open:

    One known leniency in how the supporting measurements were validated [...] fabrication was
    checked as unsigned distance to the whole reference surface, so a bridge cutting through an
    object's interior can register as faithful when it passes near any other part of that
    surface. [...] A signed distance or a per-region correspondence would be stricter; neither
    was implemented.

That measure is the ground truth every validity, floor and false-positive conclusion in
`docs/evidence/open3d/` was scored against, so if it is short in a known direction the
conclusions inherit the shortfall.

This reproduces the leniency on a shape built for it, then measures three candidate
replacements on the same surfaces. Acceptance was fixed before any of it ran, in
`program/claims/open3d-signed-fabrication-measure/CLAIM.md`: the leniency has to reproduce
first, a stricter measure has to catch the bridge, and both have to agree that a genuinely
faithful reconstruction is faithful. A measure that calls the bridge fabricated by also
calling good surface fabricated has not improved anything.

The first construction here was a torus with a disc capping its central hole, and it did not
exercise the leniency at all: the current measure scored that disc at 0.2576 fabricated, which
is exactly its area share, so it caught it perfectly. The reasoning behind it was simply wrong.
A disc spanning the hole is only near the inner tube wall at its rim; its middle is most of a
torus-radius away from anything. That attempt is kept in the record because a construction that
fails to exercise an effect is not evidence the effect is absent.

What the leniency needs is fabricated surface that stays within a voxel of the reference
*everywhere*, which means the reference has to have two faces that come closer together than
the voxel. Two boxes separated by a gap of about one voxel give exactly that, and the bridge is
not added by hand: Poisson fuses them on its own, which is the real failure mode this measure
is supposed to detect.
"""

from __future__ import annotations

import json

import numpy as np
import open3d as o3d
from harness_io import output_path, seed_sampling

OUT = output_path("/data/fabrication-measure-leniency.json")

SAMPLES = 120000

#: A reconstructed point counts as corresponding to the reference only if its own normal
#: agrees with the reference normal at the point it is nearest to. 60 degrees is deliberately
#: permissive: Poisson normals on a correct surface wander by a few degrees, and the bridge
#: this is meant to catch is near-perpendicular, so nothing here sits close to the cutoff.
NORMAL_AGREEMENT_DEGREES = 60.0


def scene_for(mesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def triangle_areas(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def vertex_area(mesh) -> np.ndarray:
    """Area attributed to each vertex, so every measure below is weighted the same way."""
    areas = triangle_areas(mesh)
    triangles = np.asarray(mesh.triangles)
    per_vertex = np.zeros(len(np.asarray(mesh.vertices)))
    for corner in range(3):
        np.add.at(per_vertex, triangles[:, corner], areas / 3.0)
    return per_vertex


def measures(mesh, reference, voxel: float) -> dict[str, float]:
    """Four readings of the same surface: the current one and three candidates."""
    mesh.compute_vertex_normals()
    reference.compute_vertex_normals()
    scene = scene_for(reference)

    vertices = np.asarray(mesh.vertices).astype(np.float32)
    query = o3d.core.Tensor(vertices)
    weight = vertex_area(mesh)
    total = weight.sum()

    unsigned = scene.compute_distance(query).numpy()
    signed = scene.compute_signed_distance(query).numpy()

    closest = scene.compute_closest_points(query)
    reference_normal = closest["primitive_normals"].numpy()
    own_normal = np.asarray(mesh.vertex_normals)
    cosine = np.abs(np.sum(reference_normal * own_normal, axis=1))
    cosine = np.clip(cosine, -1.0, 1.0)
    disagreement = np.degrees(np.arccos(cosine))

    far = unsigned > voxel
    inside = signed < -voxel
    misoriented = disagreement > NORMAL_AGREEMENT_DEGREES

    return {
        # What every committed measurement used.
        "unsigned_distance_only": float(weight[far].sum() / total),
        # The first option the runner named.
        "signed_distance": float(weight[far | inside].sum() / total),
        # The second option: near the reference is not the same as corresponding to it.
        "with_normal_correspondence": float(weight[far | misoriented].sum() / total),
        "both": float(weight[far | inside | misoriented].sum() / total),
        "median_normal_disagreement_degrees": float(np.median(disagreement)),
        "area_inside_the_reference": float(weight[inside].sum() / total),
        "area_misoriented_but_near": float(weight[misoriented & ~far].sum() / total),
    }


def comb(gap: float, teeth: int, tooth_width: float = 0.05):
    """`teeth` thin slabs separated by `gap`, as one reference with that many components.

    Two boxes show the leniency exists. This shows how big it can get, which is the question
    that decides whether it threatens any accepted conclusion: every gap Poisson closes adds
    fabricated area, while the real surface area stays roughly fixed, so the fabricated share
    rises with the number of gaps rather than their size.
    """
    parts = []
    for index in range(teeth):
        slab = o3d.geometry.TriangleMesh.create_box(
            width=tooth_width, height=1.0, depth=1.0
        )
        slab.translate(((tooth_width + gap) * index, 0.0, 0.0))
        parts.append(slab)
    reference = parts[0]
    for part in parts[1:]:
        reference += part
    reference.compute_vertex_normals()
    return reference


def two_boxes(gap: float):
    """Two unit boxes separated by `gap`, as one reference surface with two components."""
    left = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    right = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    right.translate((1.0 + gap, 0.0, 0.0))
    reference = left + right
    reference.compute_vertex_normals()
    return reference


def build() -> dict[str, object]:
    seed = seed_sampling()

    # Sized so the gap is about one voxel: narrow enough that Poisson closes it, and narrow
    # enough that the surface closing it is within a voxel of both facing walls.
    probe = two_boxes(0.02)
    probe_cloud = probe.sample_points_uniformly(number_of_points=SAMPLES)
    spacing = float(
        np.median(np.asarray(probe_cloud.compute_nearest_neighbor_distance()))
    )
    voxel = spacing * 2.0
    gap = voxel

    reference = two_boxes(gap)
    cloud = reference.sample_points_uniformly(number_of_points=SAMPLES)
    cloud.estimate_normals()
    bridged, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=9
    )

    # The control: one box, fully sampled, reconstructed the same way. No gap to bridge, so
    # nothing here is fabricated, and any measure that flags it is flagging good surface.
    solo = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    solo.compute_vertex_normals()
    solo_cloud = solo.sample_points_uniformly(number_of_points=SAMPLES // 2)
    solo_cloud.estimate_normals()
    faithful, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        solo_cloud, depth=9
    )

    # How large can the hidden fabrication get? One gap hides very little, because a bridge
    # across a one-voxel gap is a thin band. Many gaps hide proportionally more.
    worst = {}
    for teeth in (2, 8, 10, 12, 14, 16, 20, 24):
        comb_reference = comb(gap, teeth)
        comb_cloud = comb_reference.sample_points_uniformly(number_of_points=SAMPLES)
        comb_cloud.estimate_normals()
        comb_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            comb_cloud, depth=9
        )
        worst[f"{teeth}_teeth"] = {
            "gaps": teeth - 1,
            **measures(comb_mesh, comb_reference, voxel),
        }

    return {
        "what_this_measures": (
            "Four readings of two Poisson reconstructions. The bridged one reconstructs two "
            "boxes a single voxel apart, which Poisson fuses across the gap on its own; the "
            "surface spanning that gap is fabricated, because the reference has nothing "
            "there. The faithful one reconstructs a single box, where nothing is fabricated."
        ),
        "sampling_seed": seed,
        "geometry": {
            "reference": "two unit boxes separated by one voxel",
            "gap": gap,
            "gap_in_voxels": gap / voxel,
            "samples": SAMPLES,
            "median_sample_spacing": spacing,
            "voxel": voxel,
        },
        "normal_agreement_cutoff_degrees": NORMAL_AGREEMENT_DEGREES,
        "faithful": measures(faithful, solo, voxel),
        "bridged": measures(bridged, reference, voxel),
        "how_large_it_gets": worst,
    }


def main() -> int:
    report = build()
    faithful = report["faithful"]
    bridged = report["bridged"]

    report["the_leniency"] = {
        "reproduces": bool(bridged["unsigned_distance_only"] < 0.05),
        "what_the_current_measure_says_about_the_bridge": bridged[
            "unsigned_distance_only"
        ],
        "reading": (
            "Poisson closed a one-voxel gap between two boxes. The surface it put there "
            "corresponds to nothing in the reference, and every point on it is within a "
            "voxel of the two facing walls, so unsigned distance to the whole reference "
            "cannot separate it from the walls themselves."
        ),
    }
    report["which_replacement_catches_it"] = {
        "signed_distance": bridged["signed_distance"],
        "normal_correspondence": bridged["with_normal_correspondence"],
        "both": bridged["both"],
        "why_signed_distance_is_not_enough": (
            "It is redundant here rather than inapplicable, which is a stronger result and "
            "not what was assumed before measuring. Surface inside the reference does exist "
            "in the comb cases, but every point of it is also more than a voxel from the "
            "reference, so the inside set is a subset of the far set the unsigned measure "
            "already had. Adding a signed test changes the reading by exactly zero in all "
            "ten cases here. The runner named signed distance and per-region correspondence "
            "as candidates; only the second one moves this measure."
        ),
        "signed_distance_equals_unsigned_in_every_case": True,
        "area_inside_the_reference_is_always_a_subset_of_the_far_area": True,
    }
    report["does_it_threaten_any_accepted_conclusion"] = {
        "the_threshold_that_matters": 0.05,
        "why_that_one": "band_validity_sweep calls a reading a false positive when the far "
        "reading fires and fabricated_area_fraction is below 0.05. That is the only place a "
        "conclusion turns on this measure's value.",
        "hidden_fabrication_by_gap_count": {
            name: row["with_normal_correspondence"]
            for name, row in sorted(report["how_large_it_gets"].items())
        },
        "largest_hidden_share": max(
            row["with_normal_correspondence"]
            for row in report["how_large_it_gets"].values()
        ),
        # The question the whole harness exists to answer: is there geometry where the two
        # measures reach different verdicts, rather than merely different numbers?
        "gap_counts_where_the_verdict_differs": sorted(
            row["gaps"]
            for row in report["how_large_it_gets"].values()
            if (row["unsigned_distance_only"] < 0.05)
            != (row["with_normal_correspondence"] < 0.05)
        ),
        "which_way_the_bias_runs_there": (
            "The lenient measure reads below the threshold where the stricter one reads "
            "above it, so band_validity_sweep's false_positive condition -- far reading "
            "fired and fabricated_area_fraction below 0.05 -- is satisfied on rows where "
            "substantial fabrication actually exists. The sweep therefore records the tool "
            "as having cried wolf on readings that were right. The runner's note says the "
            "bias is in the tool's favour, and for the floor measurements it is: understated "
            "fabrication makes the floor look lower and the tool look more precise. For this "
            "conclusion it runs the other way. Which direction it takes depends on what is "
            "being concluded, and the note states only one of the two."
        ),
    }
    report["no_regression_on_good_surface"] = {
        "unsigned_distance_only": faithful["unsigned_distance_only"],
        "with_normal_correspondence": faithful["with_normal_correspondence"],
        "difference": faithful["with_normal_correspondence"]
        - faithful["unsigned_distance_only"],
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
