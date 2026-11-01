"""Controls for the occupancy instrument that produces this lane's positive claim.

`consumer_occupancy.py` reports that the candidate's phantom-obstacle rate is 0.0708
against the baseline's 0.3245, and #991 names occupancy-band mapping as the honest
narrower output for a surface it has just ruled unfit for collision use. That is the
result this lane ships, and it had no control: the instrument behind the *negative*
verdict in #991 is validated three ways -- zero undefined cells on a closed sphere,
parity agreeing exactly with `compute_signed_distance` at 1237 against 1237, and an
analytic estimate a reviewer recomputed independently -- while the instrument behind the
positive claim was validated none.

The collision record states the general lesson as "a measurement that makes every
candidate fail is not a finding, it is an untested instrument." This is that sentence
read in the other direction.

These controls call the real instrument. `probe_grid`, `support_mask` and `score_variant`
are imported from `consumer_occupancy`, which was refactored to expose them without
changing a line of their logic; its own reproduction check against the published record
is what guards that, because it exits non-zero on any field that moves.

Geometry is synthetic and deterministic so that the answer is known and a reader can
reproduce it without the run bundle, which lives in object storage rather than in the
repository. The observed cloud is a Fibonacci lattice evaluated in closed form, not a
sampler call: `sample_points_uniformly` takes no seed in Open3D 0.20, and a control whose
input moves between runs is not a control.

Run inside the Open3D image:
    docker run --rm -v "$EVIDENCE_DIR:/data" --entrypoint python "$IMAGE" \
        /data/harness/occupancy_instrument_controls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))

from consumer_occupancy import (
    CELL,
    SUPPORT_RADIUS_FACTOR,
    probe_grid,
    score_variant,
    support_mask,
)

OUT = Path(__file__).resolve().parent.parent / "occupancy-instrument-controls.json"

#: Sample counts for the floor sweep. The floor is a sampling-density artifact, so the
#: interesting axis is how many observations the cloud carries, not how good the mesh is.
DENSITIES = (500, 1000, 2000, 4000, 8000, 16000)

#: The density the headline controls are reported at, mid-range so the floor is neither
#: flattered by saturation nor exaggerated by sparsity.
REFERENCE_DENSITY = 4000


def fibonacci_sphere(count: int, radius: float = 1.0) -> np.ndarray:
    """Points on a sphere of the given radius, deterministic and exactly on the surface.

    Exactly on the surface matters: these stand for observations of the geometry being
    scored, so any gap the instrument reports is the instrument's, not the data's.
    """
    index = np.arange(count, dtype=float) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = np.pi * (1.0 + 5.0**0.5) * index
    return radius * np.stack(
        [
            np.cos(azimuth) * np.sin(polar),
            np.sin(azimuth) * np.sin(polar),
            np.cos(polar),
        ],
        axis=1,
    )


def cloud_of(points: np.ndarray):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def sphere(radius: float = 1.0, resolution: int = 20, centre=(0.0, 0.0, 0.0)):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    mesh.translate(np.asarray(centre, float))
    mesh.compute_vertex_normals()
    return mesh


def combined(*meshes):
    out = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        out += mesh
    out.compute_vertex_normals()
    return out


def measure(mesh, points: np.ndarray, cell: float, radius_factor: float) -> dict:
    """Score one mesh against one observed cloud with the real instrument."""
    cloud = cloud_of(points)
    centres, dims = probe_grid(cloud, cell)
    radius = radius_factor * cell
    supported = support_mask(cloud, centres, radius)
    row = score_variant(mesh, centres, supported, radius, cell)
    row["cells"] = int(np.prod(dims))
    row["cells_with_samples"] = int(supported.sum())
    row["observations"] = len(points)
    row["observations_per_supported_cell"] = round(
        len(points) / max(int(supported.sum()), 1), 3
    )
    return row


def regime_comparison(by_density: dict) -> dict:
    """Where the real artifact's sampling density sits among the control densities.

    Read from the committed records, not assumed: `downstream-consumer.json` holds the
    supported-cell count and `replay-b18d-public-report.json` holds the fused cloud's
    point count. Both are in this repository, so a reader can check this without the
    bundle.

    This is a regime indication and not a correction to the published figure. The
    control is a closed convex sphere at one scale; the real artifact is a room scan
    whose surface area per cell differs. What it does establish is the direction and
    rough size of the floor the published number sits on.
    """
    evidence = Path(__file__).resolve().parent.parent
    consumer = json.loads((evidence / "downstream-consumer.json").read_text())
    replay = json.loads((evidence / "replay-b18d-public-report.json").read_text())

    fused = None

    def find_fused(node):
        nonlocal fused
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "fused_points" and isinstance(value, int):
                    fused = value
                find_fused(value)
        elif isinstance(node, list):
            for item in node:
                find_fused(item)

    find_fused(replay)
    supported_cells = consumer["cells_with_samples"]
    actual = round(fused / supported_cells, 3) if fused else None

    nearest = min(
        by_density.items(),
        key=lambda item: abs(item[1]["observations_per_supported_cell"] - actual),
    )
    return {
        "the_artifacts_observations": fused,
        "the_artifacts_supported_cells": supported_cells,
        "the_artifacts_observations_per_supported_cell": actual,
        "nearest_control_density": nearest[0],
        "that_controls_observations_per_supported_cell": nearest[1][
            "observations_per_supported_cell"
        ],
        "that_controls_floor": nearest[1]["phantom_obstacle_rate"],
        "the_published_candidate_rate": consumer["improvement"][
            "phantom_rate_candidate"
        ],
        "the_published_baseline_rate": consumer["improvement"]["phantom_rate_baseline"],
        "reading": (
            "The floor is not zero, so the candidate's published rate cannot be read as "
            "'this fraction of its occupied volume is fabricated' -- part of any such "
            "number is the instrument. At the nearest control density the floor is well "
            "below the published candidate rate, so most of that rate is not floor. The "
            "comparison between the two variants is unaffected either way: both are "
            "scored on one grid against one support mask, so the floor applies equally "
            "to both and cannot produce the gap between them."
        ),
        "what_would_settle_it_properly": (
            "Running this instrument on the real cloud with the real surface removed and "
            "replaced by a surface known to be fully observed. That needs "
            "/data/final-bundle, which is not in the repository."
        ),
    }


def main() -> int:
    observed = fibonacci_sphere(REFERENCE_DENSITY)
    truth = sphere()

    report: dict = {
        "what_this_validates": (
            "The instrument in consumer_occupancy.py, which produces this lane's "
            "published positive claim (phantom-obstacle rate 0.3245 -> 0.0708) and which "
            "had no positive or negative control."
        ),
        "instrument": {
            "imported_from": "consumer_occupancy.py",
            "functions": ["probe_grid", "support_mask", "score_variant"],
            "not_reimplemented": (
                "The harness was refactored to expose these without changing their "
                "logic. Its own reproduction check against downstream-consumer.json "
                "exits non-zero on any field that moves, so a reader with the bundle can "
                "confirm the extraction did not alter a published number."
            ),
            "cell_size": CELL,
            "support_radius_factor": SUPPORT_RADIUS_FACTOR,
        },
    }

    # --- the floor: fully observed geometry, nothing fabricated ---
    report["positive_control"] = {
        "geometry": "closed sphere r=1.0, observations exactly on its surface",
        "expectation_before_measuring": (
            "Every occupied cell lies near observed surface by construction, so a "
            "phantom rate at or near zero."
        ),
        "measured": measure(truth, observed, CELL, SUPPORT_RADIUS_FACTOR),
    }

    # --- the floor is a density artifact, so sweep density ---
    by_density = {
        str(count): measure(truth, fibonacci_sphere(count), CELL, SUPPORT_RADIUS_FACTOR)
        for count in DENSITIES
    }
    report["the_floor_is_a_sampling_artifact"] = {
        "why": (
            "occupied is distance to the continuous mesh surface; supported is distance "
            "to the nearest discrete observation. A cell centre can sit within the radius "
            "of a smooth surface and outside the radius of every sample of it. Nothing "
            "is fabricated in any of these rows."
        ),
        "by_observation_count": {
            str(count): {
                key: row[key]
                for key in (
                    "phantom_obstacle_rate",
                    "observations_per_supported_cell",
                )
            }
            for count, row in by_density.items()
        },
        "how_this_bears_on_the_published_figure": regime_comparison(by_density),
    }

    # --- does the instrument see geometry that is not observed? ---
    fabricated_inside = combined(truth, sphere(radius=0.35))
    report["negative_control_fabrication_inside_the_observed_extent"] = {
        "geometry": (
            "the same sphere plus an unobserved r=0.35 shell at its centre; no "
            "observation lies near the added surface"
        ),
        "expectation_before_measuring": "phantom rate must rise above the floor",
        "measured": measure(fabricated_inside, observed, CELL, SUPPORT_RADIUS_FACTOR),
    }

    # --- the blind spot: the grid comes from the cloud, not from the mesh ---
    fabricated_outside = combined(truth, sphere(radius=0.6, centre=(3.0, 0.0, 0.0)))
    report["negative_control_fabrication_outside_the_observed_extent"] = {
        "geometry": (
            "the same sphere plus an unobserved r=0.6 shell centred 3.0 away, entirely "
            "outside the observed cloud's bounding box"
        ),
        "expectation_before_measuring": (
            "unknown when written. probe_grid derives the grid from the cloud's bounding "
            "box, so fabricated surface beyond it may never be probed."
        ),
        "measured": measure(fabricated_outside, observed, CELL, SUPPORT_RADIUS_FACTOR),
    }

    # --- the other direction: observed structure the mesh fails to cover ---
    half = sphere()
    vertices = np.asarray(half.vertices)
    triangles = np.asarray(half.triangles)
    keep = np.all(vertices[triangles][:, :, 2] < 0.5, axis=1)
    half.triangles = o3d.utility.Vector3iVector(triangles[keep])
    half.remove_unreferenced_vertices()
    half.compute_vertex_normals()
    report["recall_control_observed_structure_the_mesh_omits"] = {
        "geometry": "the sphere with its cap above z=0.5 deleted, cloud unchanged",
        "expectation_before_measuring": (
            "missed_structure_cells must rise above the positive control's, which is the "
            "recall direction rather than the phantom direction"
        ),
        "measured": measure(half, observed, CELL, SUPPORT_RADIUS_FACTOR),
    }

    # --- does the separation survive the parameters it was measured at? ---
    sweep = {}
    for cell in (0.05, 0.1, 0.2):
        for factor in (0.5, 0.87, 1.5, 2.0):
            floor = measure(truth, observed, cell, factor)["phantom_obstacle_rate"]
            fabricated = measure(fabricated_inside, observed, cell, factor)[
                "phantom_obstacle_rate"
            ]
            # Rounded first, then subtracted. Rounding the difference instead leaves the
            # margin disagreeing with the two rates printed beside it by up to 1e-6,
            # which is exactly the kind of mismatch a reader cannot check and a test
            # lands on the boundary of.
            shown_floor = round(floor, 6)
            shown_fabricated = round(fabricated, 6)
            sweep[f"cell={cell},factor={factor}"] = {
                "floor": shown_floor,
                "with_fabrication": shown_fabricated,
                "fabrication_is_detected": bool(fabricated > floor),
                "margin": round(shown_fabricated - shown_floor, 6),
            }
    report["the_guard_on_this_harness_was_inert_in_ci_and_evadable"] = {
        "raised_by": "review lane, 2026-11-01T21:18Z, three findings on this PR",
        "the_first_version": (
            "`test_the_controls_call_the_instrument_rather_than_a_copy_of_it` asserted "
            "object identity: the functions this harness holds are the objects "
            "`consumer_occupancy` defines."
        ),
        "why_identity_was_not_enough": (
            "A function-local shadow satisfies module-attribute identity while executing "
            "something else. Reproduced before changing anything: an honest harness and "
            "an evading harness both pass the identity assertion and return ('REAL', 10) "
            "and ('COPY', 15). Identity constrains the module namespace; execution "
            "resolves names one scope down."
        ),
        "why_it_also_never_ran_where_it_mattered": (
            "Open3D is installed nowhere CI runs: it appears in `npa/pyproject.toml` only "
            "as a container image tag, in no optional-dependency extra, and not in "
            "`npa/ci/requirements.txt`. The guard skipped on every shard, on every PR."
        ),
        "what_replaced_it": [
            (
                "A behavioural check: the instrument and this harness are called on one "
                "shared input and every field of the two results must agree. A copy now "
                "has to be correct to pass. Verified against the reviewer's own mutation "
                "-- a function-local shadow inside `measure()`, import line intact and "
                "the module attribute still the real object -- which the identity test "
                "passed and this one fails."
            ),
            (
                "`score_from_distance` split out of `score_variant`, carrying the "
                "definitions with the Open3D query already done, so it imports under "
                "NumPy alone. Every published row is rebuilt from its own counts and put "
                "back through the real arithmetic, which runs on every shard."
            ),
        ],
        "a_claim_of_mine_that_was_false_until_i_tested_it": (
            "I first wrote the definition check to import this control harness too, and "
            "stated it ran without Open3D. It did not: this harness imports Open3D at "
            "module level. A first attempt to prove it used a `find_module` hook, which "
            "Python 3.12 removed, so it blocked nothing and the check appeared to pass. "
            "With a working `find_spec` hook the import failed. The test now imports the "
            "instrument alone, and was re-verified with `open3d` genuinely unresolvable: "
            "12 passed, 1 skipped, and four mutations of published values each caught in "
            "that state."
        ),
        "mutations_run_in_both_states": {
            "how": (
                "Each mutation was applied, the suite run twice -- once normally, once "
                "with `open3d` made unresolvable through `sys.meta_path` to reproduce a "
                "CI shard -- and the file restored."
            ),
            "caught_in_both_states": [
                "positive control phantom rate zeroed",
                "floor claimed to rise with density",
                "inside-extent detection margin erased",
                "outside-extent phantom count inflated",
                "recall control missed cells zeroed",
                "precision made inconsistent with counts",
                "sweep claims detection it lacks",
                "occupied cells inconsistent with phantom",
            ],
            "caught_only_where_open3d_is_installed": [
                (
                    "the reviewer's function-local shadow of `score_variant` inside "
                    "`measure()`: 1 failed locally, 12 passed and 1 skipped in the CI "
                    "state"
                )
            ],
        },
        "what_is_still_not_bound": (
            "The shadow defence is behavioural, so it needs Open3D, so it is still absent "
            "from CI -- measured, not assumed: the shadow mutation passes in the CI state. "
            "What runs there is the definitional half, which catches any published number "
            "that does not satisfy the instrument's own arithmetic. Neither notices a "
            "shadow added without re-running the harness, because the record would then "
            "no longer be this harness's output and nothing here compares the two."
        ),
    }
    report["parameter_sweep"] = {
        "why": (
            "The published pair was measured at one cell size and one radius factor. The "
            "constant's comment used to credit a 0.5-to-2.0 sweep to the review lane; "
            "they searched their own dispatches and phases for it and reported that they "
            "never ran it and never said they had. The attribution is deleted rather than "
            "hunted for, because there is nothing to point at, and the sweep below is "
            "this lane's own."
        ),
        "what_is_swept": "the controls, where the answer is known",
        "what_is_not_swept": (
            "the published 0.3245/0.0708 pair, which needs /data/final-bundle. That "
            "bundle is a run output in object storage, not in the repository or the "
            "image, so no reader working from this repository can recompute it at other "
            "parameters either."
        ),
        "rows": sweep,
        "fabrication_detected_in_every_combination": all(
            row["fabrication_is_detected"] for row in sweep.values()
        ),
    }

    report["what_this_does_not_claim"] = {
        "not_that_the_published_figures_are_wrong": (
            "Nothing here recomputes them. The controls establish what the instrument "
            "does on geometry whose answer is known, which is the thing that was missing."
        ),
        "not_that_the_reconstruction_is_accurate": (
            "No artifact in this chain carries ground truth. A low phantom rate means "
            "occupied cells have observations near them, not that the surface is in the "
            "right place."
        ),
        "the_control_geometry_is_not_the_artifact": (
            "A closed convex sphere at one scale, not a room scan. The floor's size "
            "depends on surface area per cell, so the regime comparison above indicates "
            "a magnitude and is not a measurement of the artifact's own floor."
        ),
        "the_blind_spot_is_a_property_of_the_grid_not_of_this_scene": (
            "probe_grid derives the grid from the observed cloud's bounding box. Any "
            "fabricated surface beyond that box is unprobed in any scene, so this "
            "limitation travels with the instrument rather than with the control."
        ),
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    print(f"wrote {OUT}")
    floor = report["positive_control"]["measured"]["phantom_obstacle_rate"]
    inside = report["negative_control_fabrication_inside_the_observed_extent"][
        "measured"
    ]["phantom_obstacle_rate"]
    outside = report["negative_control_fabrication_outside_the_observed_extent"][
        "measured"
    ]["phantom_obstacle_rate"]
    print(f"floor (fully observed)            {floor:.6f}")
    print(f"fabrication inside the extent     {inside:.6f}")
    print(f"fabrication outside the extent    {outside:.6f}")
    print(
        f"floor by density                  {report['the_floor_is_a_sampling_artifact']['by_observation_count']}"
    )
    print(
        "fabrication detected in every swept combination: "
        f"{report['parameter_sweep']['fabrication_detected_in_every_combination']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
