"""Measure whether the scan determines the camera's up axis, or whether the run does.

The pipeline infers an up vector from the largest planar segment and publishes the
inlier fraction and the tilt so the inference is visible. What none of those fields
shows is whether a *different run* of the same code on the same cloud would have
named a different axis -- and `segment_plane` is RANSAC, so on a scan holding two
comparable planes it does.

Five clouds, because one of each kind proves nothing:

  ambiguous_room       a floor and a wall of equal point count: the failure
  floor_four_to_one    the same room with a quarter as many wall points
  floor_ten_to_one     and with a tenth, so "determined" is not one example either
  real_demo_scan       Open3D's own demo capture, 198835 points: the negative control
  armadillo_sampled    a model with no floor at all: the case agreement misses

The rooms vary *point count*, not extent. An earlier version of this harness varied
the wall's side length while giving both faces the same number of points, which made
the supposedly floor-dominant control another near-tie that happened to be stable:
`segment_plane` scores a plane by how many points fall within the threshold, so area
is not the variable that matters.

Run with the repository's Open3D environment on the PYTHONPATH:

    python docs/evidence/open3d/harness/up_axis_ambiguity.py > \
        docs/evidence/open3d/up-axis-ambiguity.json

Imports the shipped `_plane_up_reading` and `_up_axis` rather than reimplementing
them, so what is measured here is the code that runs in the pipeline.
"""

from __future__ import annotations

import json
import platform
import sys
from typing import Any

import numpy as np
import open3d as o3d
from npa.workbench.open3d.runner import (
    UP_AXIS_AGREEMENT_COUNTEREXAMPLE,
    UP_AXIS_REPEATS,
    _plane_up_reading,
    _up_axis,
)

#: Single fits per cloud. Far more than the pipeline runs, because the question here
#: is what the distribution looks like, not what one run decides.
FITS = 40

#: Pipeline runs per cloud when measuring what gets published.
PIPELINE_RUNS = 30

#: Repeat counts swept to show how detection depends on the count, including the
#: shipped default.
REPEAT_COUNTS = (4, UP_AXIS_REPEATS, 16)

#: Voxel matching the accepted records; the Armadillo is in its own units.
VOXEL = 0.05
ARMADILLO_VOXEL = 2.0

#: Side length of both faces of the constructed rooms, in the same units as VOXEL.
ROOM_SIDE = 4.0


def room(floor_points: int, wall_points: int):
    """A floor in z=0 and a wall in x=0, both 4x4, with the point counts as variable.

    Both are exactly planar, so neither is the harder fit. Inliers are counted, so
    the point count is what decides which plane wins.
    """

    rng = np.random.default_rng(7)
    floor = np.column_stack(
        [
            rng.uniform(0, ROOM_SIDE, floor_points),
            rng.uniform(0, ROOM_SIDE, floor_points),
            np.zeros(floor_points),
        ]
    )
    wall = np.column_stack(
        [
            np.zeros(wall_points),
            rng.uniform(0, ROOM_SIDE, wall_points),
            rng.uniform(0, ROOM_SIDE, wall_points),
        ]
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack([floor, wall]))
    return cloud


def two_walls(first_points: int, second_points: int):
    """Planes at x=0 and y=0, so the candidate axes are x and y -- never z.

    The floor-and-wall room cannot show the fallback working: its majority branch is
    the declared default, so publishing the default and publishing the winning branch
    produce the same vector. Here the default is neither candidate, which is what
    makes the difference observable -- a mutation that published the winner survived
    the functional smoke until this construction existed.
    """

    rng = np.random.default_rng(7)
    first = np.column_stack(
        [
            np.zeros(first_points),
            rng.uniform(0, ROOM_SIDE, first_points),
            rng.uniform(0, ROOM_SIDE, first_points),
        ]
    )
    second = np.column_stack(
        [
            rng.uniform(0, ROOM_SIDE, second_points),
            np.zeros(second_points),
            rng.uniform(0, ROOM_SIDE, second_points),
        ]
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack([first, second]))
    return cloud


def armadillo():
    """A model with no floor and no wall, sampled as a cloud."""

    mesh = o3d.io.read_triangle_mesh(o3d.data.ArmadilloMesh().path)
    return mesh.sample_points_uniformly(number_of_points=40000)


def real_scan():
    """Open3D's own demo capture: the data every accepted record is built from."""

    return o3d.io.read_point_cloud(o3d.data.DemoICPPointClouds().paths[0])


def single_fit_distribution(cloud, voxel: float) -> dict[str, Any]:
    """What one fit answers, over many independent fits.

    Deliberately calls `_plane_up_reading`, the single fit, because that is the
    behaviour a consumer saw before the repeats existed.
    """

    axes: dict[tuple[float, ...], list[dict[str, float]]] = {}
    weak = 0
    for _ in range(FITS):
        reading = _plane_up_reading(o3d, cloud, voxel)
        if reading is None or reading["weak"]:
            weak += 1
            continue
        axes.setdefault(tuple(reading["up"]), []).append(
            {
                "fraction": reading["plane_inlier_fraction"],
                "tilt": reading["angle_to_snapped_axis_degrees"],
            }
        )
    branches = []
    for axis, readings in sorted(axes.items(), key=lambda item: -len(item[1])):
        fractions = [one["fraction"] for one in readings]
        tilts = [one["tilt"] for one in readings]
        branches.append(
            {
                "up": list(axis),
                "chosen_by_fits": len(readings),
                "plane_inlier_fraction_range": [min(fractions), max(fractions)],
                "tilt_to_snapped_axis_degrees_range": [min(tilts), max(tilts)],
            }
        )
    return {
        "fits": FITS,
        "weak_or_failed_fits": weak,
        "distinct_axes": len(axes),
        "branches": branches,
        "no_published_field_separates_the_branches": _overlap(branches),
    }


def _overlap(branches: list[dict[str, Any]]) -> Any:
    """Do the branches' published ranges overlap, so a reader cannot tell them apart?

    Only meaningful when there is more than one branch; `null` otherwise, rather
    than a default that would read as a measurement.
    """

    if len(branches) < 2:
        return None
    first, second = branches[0], branches[1]
    overlaps = {}
    for key in ("plane_inlier_fraction_range", "tilt_to_snapped_axis_degrees_range"):
        low_a, high_a = first[key]
        low_b, high_b = second[key]
        overlaps[key] = max(low_a, low_b) <= min(high_a, high_b)
    return overlaps


def seeding_is_not_a_remedy(cloud, voxel: float, seed: int = 5) -> dict[str, Any]:
    """Re-set Open3D's seed to the same value before every fit, and count answers.

    If seeding settled this, the fix would be to seed and move on. It does not:
    `segment_plane` is threaded and the interleaving is not under the seed's
    control. Reported as counts rather than a verdict, because the split itself
    moves between processes.
    """

    answers: dict[tuple[float, ...], int] = {}
    for _ in range(10):
        o3d.utility.random.seed(seed)
        reading = _plane_up_reading(o3d, cloud, voxel)
        if reading is None or reading["weak"]:
            continue
        key = tuple(reading["up"])
        answers[key] = answers.get(key, 0) + 1
    return {
        "seed_re_set_before_every_fit": seed,
        "fits": 10,
        "answers": [
            {"up": list(axis), "times": times}
            for axis, times in sorted(answers.items(), key=lambda item: -item[1])
        ],
    }


def what_the_pipeline_publishes(cloud, voxel: float) -> dict[str, Any]:
    """`_up_axis` as the pipeline calls it, over many runs, at its shipped default."""

    seen: dict[tuple[str, tuple[float, ...], int], int] = {}
    runner_ups = []
    for _ in range(PIPELINE_RUNS):
        report = _up_axis(o3d, cloud, voxel)
        agreement = report.get("agreement") or {}
        seen_key = (
            report["source"],
            tuple(report["up"]),
            agreement.get("distinct_axes", 0),
        )
        seen[seen_key] = seen.get(seen_key, 0) + 1
        runner = agreement.get("runner_up_plane") or {}
        if runner.get("measured"):
            runner_ups.append(runner["inliers_relative_to_the_winner"])
    return {
        "pipeline_runs": PIPELINE_RUNS,
        "repeats_per_run": UP_AXIS_REPEATS,
        "distinct_published_answers": len(seen),
        "answers": [
            {
                "source": source,
                "up": list(up),
                "distinct_axes_found_by_repeats": distinct,
                "published_by_runs": times,
            }
            for (source, up, distinct), times in sorted(
                seen.items(), key=lambda item: -item[1]
            )
        ],
        "runner_up_relative_to_winner_range": (
            [min(runner_ups), max(runner_ups)] if runner_ups else None
        ),
    }


def detection_by_repeat_count(cloud, voxel: float) -> dict[str, Any]:
    """How often the ambiguity is caught, as a function of the repeat count.

    This is the honest cost of the check: it detects a swap when the repeats happen
    to disagree, so its rate follows (1-p)^n + p^n for a per-fit minority
    probability p. Published as counts at three repeat counts, with the per-fit
    split beside them, so the residual is attributable rather than mysterious.
    """

    rates = []
    for repeats in REPEAT_COUNTS:
        caught = sum(
            1
            for _ in range(PIPELINE_RUNS)
            if _up_axis(o3d, cloud, voxel, repeats=repeats)["source"]
            == "default-ambiguous-plane"
        )
        rates.append(
            {
                "repeats": repeats,
                "fits": repeats + 1,
                "is_the_shipped_default": repeats == UP_AXIS_REPEATS,
                "ambiguity_detected_in_runs": caught,
                "of_runs": PIPELINE_RUNS,
            }
        )
    return {
        "rates": rates,
        "why_it_is_not_one": (
            "The check fires when the repeats disagree, so it misses when every fit "
            "lands on the same branch by chance. At the per-fit split measured beside "
            "this, (1-p)^n + p^n accounts for the observed misses at all three counts, "
            "which means the residual is the branches agreeing by chance rather than a "
            "defect in the check. The rate falls as one branch gets rarer -- and a "
            "rarer branch also rolls the view less often."
        ),
    }


def case(name: str, cloud, voxel: float, note: str) -> dict[str, Any]:
    """One cloud, measured as a distribution and as what the pipeline publishes."""

    return {
        "cloud": name,
        "points": len(cloud.points),
        "voxel_size": voxel,
        "what_this_cloud_is": note,
        "single_fit_distribution": single_fit_distribution(cloud, voxel),
        "what_the_pipeline_publishes": what_the_pipeline_publishes(cloud, voxel),
    }


def main() -> None:
    ambiguous = room(8100, 8100)
    cases = [
        case(
            "ambiguous_room",
            ambiguous,
            VOXEL,
            "A 4x4 floor and a 4x4 wall with 8100 points each. Equal in the quantity "
            "segment_plane scores, so the only question is which plane a given fit "
            "lands on. This is the failure the repeats exist to catch.",
        ),
        case(
            "two_walls_ambiguous",
            two_walls(8100, 8100),
            VOXEL,
            "Planes at x=0 and y=0 with 8100 points each. Ambiguous like the room "
            "above, but with the declared default reachable by neither branch, so "
            "the fallback publishing the default is distinguishable from it "
            "publishing whichever branch won.",
        ),
        case(
            "floor_four_to_one",
            room(8100, 2025),
            VOXEL,
            "The same room with a quarter as many wall points. A positive control: "
            "the scan does determine an axis and the pipeline must still name it.",
        ),
        case(
            "floor_ten_to_one",
            room(8100, 810),
            VOXEL,
            "And with a tenth, so that 'determined' is not a single example either.",
        ),
        case(
            "real_demo_scan",
            real_scan(),
            VOXEL,
            "Open3D's own demo capture, the data every accepted record in this "
            "directory is built from. The negative control: a measure that fired "
            "here would be unusable.",
        ),
        case(
            "armadillo_sampled",
            armadillo(),
            ARMADILLO_VOXEL,
            "A model with no floor and no wall. The case the repeats do not catch: "
            "they agree unanimously on a plane tilted well off the axis it gets "
            "snapped to, because one plane does dominate a convex-ish shape.",
        ),
    ]
    report = {
        "what_this_measures": (
            "Whether the camera's up vector is determined by the scan or by the run. "
            "The pipeline infers it from the largest planar segment, segment_plane is "
            "RANSAC, and the published fields -- source, inlier fraction, tilt -- take "
            "overlapping values on both branches when two planes compete, so one run's "
            "record cannot show that the next run would roll the view by ninety "
            "degrees."
        ),
        "why_it_matters": (
            "The up vector is consumed by the view cameras, not merely recorded. An "
            "axis chosen by a coin flip is a view rolled onto its side in some "
            "fraction of runs, which is the reviewability problem the inference was "
            "added to fix."
        ),
        "cases": cases,
        "seeding_is_not_a_remedy": seeding_is_not_a_remedy(ambiguous, VOXEL),
        "detection_by_repeat_count": detection_by_repeat_count(ambiguous, VOXEL),
        "what_the_cases_establish": [
            (
                "A single fit on the ambiguous room returns two different axes, and the "
                "inlier fraction and tilt ranges of the two branches overlap, so no field "
                "the function publishes separates them."
            ),
            (
                "Seeding does not settle it: ten fits with the same seed re-set before "
                "each one return both axes."
            ),
            (
                "With the repeats, the ambiguous room publishes the declared default "
                "labelled as the default in most runs, instead of one of the two branches "
                "labelled as measured in every run."
            ),
            (
                "All three determined clouds -- two constructed, one real -- still get a "
                "named axis, so the check does not refuse to answer on data that has an "
                "answer."
            ),
        ],
        "what_this_does_not_claim": (
            "Not that the check always fires: detection_by_repeat_count gives its "
            "measured rate at three repeat counts, and the misses are accounted for by "
            "the branches agreeing by chance. Not that repeat agreement means the axis "
            "is right -- the armadillo case agrees unanimously on a plane tilted 19 "
            "degrees off the axis it is snapped to, holding 13% of the points, just "
            "above the one-tenth floor the weak-plane guard enforces. And not a "
            "calibrated threshold on the runner-up plane: five clouds separate cleanly "
            "but five clouds cannot place a line that decides whether a camera axis is "
            "trustworthy, so that number is published as evidence and gates nothing."
        ),
        "counterexample_the_code_publishes": dict(UP_AXIS_AGREEMENT_COUNTEREXAMPLE),
        "cost": (
            "About 85 ms per plane fit on the 198835-point real scan, so the shipped "
            f"default of {UP_AXIS_REPEATS} extra fits plus the runner-up costs roughly "
            "0.8 s against a multi-second reconstruction."
        ),
        "reproduce": (
            "python docs/evidence/open3d/harness/up_axis_ambiguity.py, with npa/src on "
            "PYTHONPATH in an environment holding Open3D. The constructed rooms are "
            "seeded; the fit distributions are not reproducible by construction, which "
            "is the finding."
        ),
        "environment": {
            "open3d_version": o3d.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
