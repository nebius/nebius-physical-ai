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

import hashlib
import json
import platform
import sys
from pathlib import Path
from types import MappingProxyType
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
#: Equal point counts on floor and wall: the construction the ambiguity needs.
ROOM_POINTS = 8100


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


def detection_by_repeat_count(
    cloud, voxel: float, distribution: dict[str, Any]
) -> dict[str, Any]:
    """How often the ambiguity is caught, as a function of the repeat count.

    This is the honest cost of the check: it detects a swap when the repeats happen
    to disagree, so its rate follows (1-p)^n + p^n for a per-fit minority
    probability p. Published as counts at three repeat counts, with the per-fit
    split beside them, so the residual is attributable rather than mysterious.
    """

    # The per-fit minority rate from the same cloud's single-fit sweep, passed in
    # rather than re-measured so the model is checked against the split published
    # beside it instead of a second sample of the same quantity.
    branches = distribution["branches"]
    attempted = distribution["fits"]
    minority = (
        min(one["chosen_by_fits"] for one in branches) / attempted
        if len(branches) > 1
        else 0.0
    )

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
        "does_detection_rise_with_the_fit_count": rise_with_fit_count(),
        "why_it_is_not_one": (
            "The check fires when the repeats disagree, so it misses when every fit "
            "lands on the same branch. The chance-agreement account of the residual, "
            "(1-p)^n + p^n for a per-fit minority rate p over n fits, is consistent "
            "with the measurement when the measurement has the power to test it: see "
            "does_the_minority_rate_vary_between_processes, which puts 3600 checks "
            "against each process's own p and finds the predicted and observed miss "
            "counts within a couple of sigma. It is admissible, not confirmed -- the "
            "independence it assumes between fits in one run is untested. The rates "
            "above are measurements either way and are not derived from any model."
        ),
        "independence_of_the_fits": _independence_check(rates, minority),
    }


def _independence_check(rates: list[dict], minority: float) -> dict[str, Any]:
    """Test the chance-agreement explanation against the split measured beside it.

    The claim being tested is that a miss is every fit landing on the same branch
    independently, so misses should follow (1-p)^n + p^n for n fits -- one plus the
    repeats, because the first fit counts too.

    This is a within-run test on purpose. Comparing regenerations would show the
    same thing (the per-fit minority moved from 0.200 to 0.375 between two runs
    while detection at nine fits moved the wrong way, from 27 of 30 down to 25 of
    30, when a larger minority must raise detection under independence) but it would
    depend on a file that is no longer committed. Measuring p and the detection rate
    in one process needs no such reference.

    Two people checked the arithmetic of the old claim and neither of us checked
    whether the model was admissible. Review recomputed it with the repeat count as
    the exponent rather than the fit count and reported that it failed at the
    smallest count; at the right exponent that particular objection dissolves. The
    objection that survives is larger than either of us made: `segment_plane` is
    threaded and the record already shows its split moving between processes, so the
    fits are not independent draws and a fixed-p binomial is not a description of
    them. The honest residual is unexplained, and the rates stand on their own.
    """

    from math import comb

    predictions = []
    for rate in rates:
        runs = rate["of_runs"]
        misses = runs - rate["ambiguity_detected_in_runs"]
        fits = rate["fits"]
        agree = (1.0 - minority) ** fits + minority**fits
        predictions.append(
            {
                "fits": fits,
                "probability_all_fits_agree": round(agree, 6),
                "predicted_misses": round(agree * runs, 3),
                "observed_misses": misses,
                "one_sided_p_of_at_most_the_observed": round(
                    sum(
                        comb(runs, k) * agree**k * (1.0 - agree) ** (runs - k)
                        for k in range(misses + 1)
                    ),
                    4,
                ),
            }
        )
    disagreements = [
        one
        for one in predictions
        if not 0.05 <= one["one_sided_p_of_at_most_the_observed"] <= 0.95
    ]
    return {
        "model_tested": "(1-p)^n + p^n, for n fits and a fixed per-fit minority p",
        "per_fit_minority_rate_measured_in_this_process": minority,
        "exponent_is": "the number of fits, which is one plus the repeats",
        "predictions": predictions,
        "counts_where_this_process_rejects_the_model": [
            one["fits"] for one in disagreements
        ],
        "this_verdict_does_not_reproduce": (
            "and that is the point -- see across_processes below. One process is not "
            "enough to decide this, so no verdict is published from this block alone"
        ),
        "does_the_minority_rate_vary_between_processes": (
            does_the_minority_rate_vary_between_processes()
        ),
        "repeats_within_one_process": repeats_within_one_process(),
        "what_this_does_not_change": (
            "the check still detects the ambiguity in most runs at the shipped count, "
            "and the rate is published rather than claimed away. Nothing above is "
            "derived from the chance-agreement account, so whether that account holds "
            "changes the explanation of why the check misses and not the measurement "
            "of how often"
        ),
    }


HIGH_POWER_FITS = 400
HIGH_POWER_CHECKS = 300
HIGH_POWER_PROCESSES = 12


SWEEP_PROBE_FLAG = "--sweep-probe"
SWEEP_RUNS_PER_PROCESS = 25


def sweep_in_this_process() -> list[int]:
    """Detections at each repeat count over `SWEEP_RUNS_PER_PROCESS` pipeline runs."""

    cloud = room(ROOM_POINTS, ROOM_POINTS)
    return [
        sum(
            1
            for _ in range(SWEEP_RUNS_PER_PROCESS)
            if _up_axis(o3d, cloud, VOXEL, repeats=repeats)["source"]
            == "default-ambiguous-plane"
        )
        for repeats in REPEAT_COUNTS
    ]


#: Detections per configuration reported by an independent lane that ran this harness on
#: its own host, in its own virtualenv, against a fresh download of the demo meshes, at
#: the same 300 runs across 12 interpreters. Quoted, not measured here -- which is why
#: the comparison below is computed from it rather than restated, and why the record says
#: whose numbers these are. Nothing is shared between the two runs but the harness bytes.
INDEPENDENT_RUN_DETECTIONS = (233, 283, 296)


def _against_the_independent_run(totals: list[int], of_runs: int) -> dict[str, Any]:
    """Compare this run's proportions against the independently reported ones.

    Two sweeps of nine hundred runs each, sharing no interpreter, no virtualenv and no
    seed -- there is no seed anywhere in this harness -- are a stronger statement about
    these rates than either sweep alone. The reason to compute the comparison instead of
    noting that it agreed is that a regeneration moves these proportions, so the
    agreement has to be recomputed to stay true.

    The per-step result matters more than the headline. Both runs find every step
    rising, but the step from nine fits to seventeen is the soft one in both, and the
    independent run put it below three sigma. The shipped repeat count does not depend
    on that step.
    """

    compared = []
    for index, count in enumerate(REPEAT_COUNTS):
        mine = totals[index] / of_runs
        theirs = INDEPENDENT_RUN_DETECTIONS[index] / of_runs
        pooled = (totals[index] + INDEPENDENT_RUN_DETECTIONS[index]) / (2 * of_runs)
        error = (2.0 * pooled * (1.0 - pooled) / of_runs) ** 0.5
        compared.append(
            {
                "fits": count + 1,
                "this_run": round(mine, 4),
                "the_independent_run": round(theirs, 4),
                "difference_in_sigmas": round((theirs - mine) / error, 2)
                if error > 0
                else None,
                "is_the_shipped_count": count == UP_AXIS_REPEATS,
            }
        )
    widest = max(
        (one for one in compared if one["difference_in_sigmas"] is not None),
        key=lambda one: abs(one["difference_in_sigmas"]),
        default=None,
    )
    return {
        "whose": (
            "an independent review lane, on its own host and virtualenv, with a fresh "
            "download of the demo meshes and these harness bytes"
        ),
        "runs_each": of_runs,
        "detections_they_reported": list(INDEPENDENT_RUN_DETECTIONS),
        "per_configuration": compared,
        "widest_disagreement_in_sigmas": (
            abs(widest["difference_in_sigmas"]) if widest else None
        ),
        "what_this_adds": (
            "the rates themselves, not just the ordering. Two sweeps of this size "
            "sharing no interpreter, no virtualenv and no seed agreeing to within a "
            "sigma or so is what a real measurement looks like; one sweep is a "
            "measurement of one machine"
        ),
        "where_the_two_runs_are_softest": (
            "the step from nine fits to seventeen. Both runs find it positive and both "
            "find the step containing the shipped count far stronger, but the "
            "independent run put the upper step below three sigma where this one is "
            "higher. The shipped repeat count rests on the lower step, so quoting the "
            "seventeen-fit improvement on its own would be quoting the least "
            "reproducible part of this curve"
        ),
    }


def rise_with_fit_count() -> dict[str, Any]:
    """Does detection actually rise with the fit count, measured with enough runs?

    The sweep above runs thirty times per configuration, where one count's standard
    deviation is about one and a half runs. That cannot order two configurations a
    few runs apart, and on one regeneration it did not: five fits returned 27 and
    nine fits returned 26, inverting the ordering the shipped repeat count rests on.
    The inversion was noise, and so was the ordering in the runs before it.

    This pools `SWEEP_RUNS_PER_PROCESS` runs from each of `HIGH_POWER_PROCESSES`
    fresh interpreters, which is enough to separate configurations differing by a few
    percent. The comparison published is the shipped count against the next lower
    one, in sigmas of the difference, because that is the pair whose ordering
    justifies the default.
    """

    import subprocess

    runs = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), SWEEP_PROBE_FLAG],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(HIGH_POWER_PROCESSES)
    ]
    totals = [0] * len(REPEAT_COUNTS)
    for run in runs:
        out, _ = run.communicate()
        if run.returncode != 0:
            raise RuntimeError(f"a sweep probe exited {run.returncode}")
        for index, value in enumerate(int(one) for one in out.split()):
            totals[index] += value
    of_runs = SWEEP_RUNS_PER_PROCESS * HIGH_POWER_PROCESSES
    rates = [one / of_runs for one in totals]
    shipped = REPEAT_COUNTS.index(UP_AXIS_REPEATS)
    pairs = []
    for lower, upper in zip(
        range(len(REPEAT_COUNTS) - 1), range(1, len(REPEAT_COUNTS))
    ):
        difference = rates[upper] - rates[lower]
        # Standard error of a difference of two independent proportions.
        error = sum(one * (1.0 - one) / of_runs for one in (rates[lower], rates[upper]))
        error = error**0.5
        pairs.append(
            {
                "from_fits": REPEAT_COUNTS[lower] + 1,
                "to_fits": REPEAT_COUNTS[upper] + 1,
                "detection_rose_by": round(difference, 4),
                "one_sigma_of_the_difference": round(error, 4),
                "in_sigmas": round(difference / error, 2) if error > 0 else None,
                "this_pair_includes_the_shipped_count": shipped in (lower, upper),
                # A proportion of exactly 1.0 contributes nothing to the standard
                # error, so a step ending at a clean sweep gets an optimistic sigma
                # from the standard two-proportion formula. Seventeen fits returned
                # 300 of 300 on one regeneration and 296 on an independent lane's,
                # which is the same rate measured twice, not a rate of one. Flagged
                # rather than adjusted: the formula is the usual one and the reader
                # needs to know which cell it flattered.
                "an_endpoint_detected_every_run": 1.0 in (rates[lower], rates[upper]),
            }
        )
    decisive = [one for one in pairs if one["in_sigmas"] is not None]
    reproduction = _against_the_independent_run(totals, of_runs)
    return {
        "an_independent_lane_ran_this_too": reproduction,
        "processes": HIGH_POWER_PROCESSES,
        "each_process_is_a_fresh_interpreter": True,
        "runs_per_configuration": of_runs,
        "fits_per_configuration": [one + 1 for one in REPEAT_COUNTS],
        "detections": totals,
        "detection_rate": [round(one, 4) for one in rates],
        "consecutive_pairs": pairs,
        "every_step_rose": all(one["detection_rose_by"] > 0.0 for one in pairs),
        "smallest_step_in_sigmas": (
            min(one["in_sigmas"] for one in decisive) if decisive else None
        ),
        "the_step_the_shipped_count_rests_on": next(
            (
                one
                for one in pairs
                if one["to_fits"] == UP_AXIS_REPEATS + 1 and one["in_sigmas"]
            ),
            None,
        ),
        "not_every_step_is_equally_firm": (
            "the two steps are published separately because they do not reproduce "
            "equally well. The step up to the shipped count has come back above five "
            "sigma in every run including an independent lane's. The step from the "
            "shipped count to seventeen fits has come back anywhere from below three "
            "sigma to above five, and one regeneration put seventeen fits at 300 of "
            "300, where the standard two-proportion formula flatters the comparison. "
            "Nothing shipped depends on that upper step, so the record orders it and "
            "declines to put a floor under it"
        ),
        "so": (
            "detection rises with the fit count by more than the measurement's own "
            "noise at every step, which the thirty-run sweep above could not show "
            "and on one regeneration contradicted. That sweep is retained because "
            "its scatter is the subject of this record; this block is what the "
            "structural claim behind the shipped repeat count rests on"
        ),
        "what_it_does_not_say": (
            "that more fits are free. Each repeat is another RANSAC fit on the "
            "downsampled cloud, and the record does not measure the cost, so this "
            "orders the configurations by detection and not by any trade-off"
        ),
    }


MINORITY_PROBE_FLAG = "--minority-probe"


def minority_in_this_process() -> tuple[int, int, int, int]:
    """The two quantities the chance-agreement model relates, in one process.

    The minority-axis count out of `HIGH_POWER_FITS` single fits, which is the
    model's input, and the number of full pipeline checks out of
    `HIGH_POWER_CHECKS` that missed the ambiguity, which is its prediction. Both
    must come from the same process: the model's claim is that this process's
    minority rate explains this process's miss rate.

    Returns counts rather than rates so the pooling below is exact.
    """

    cloud = room(ROOM_POINTS, ROOM_POINTS)
    axes: dict[tuple[float, ...], int] = {}
    usable = 0
    for _ in range(HIGH_POWER_FITS):
        reading = _plane_up_reading(o3d, cloud, VOXEL)
        if reading is None or reading["weak"]:
            continue
        usable += 1
        key = tuple(reading["up"])
        axes[key] = axes.get(key, 0) + 1
    misses = sum(
        1
        for _ in range(HIGH_POWER_CHECKS)
        if _up_axis(o3d, cloud, VOXEL, repeats=UP_AXIS_REPEATS)["source"]
        != "default-ambiguous-plane"
    )
    return (
        (usable - max(axes.values()) if axes else 0),
        usable,
        misses,
        HIGH_POWER_CHECKS,
    )


def _spawn_minority_probes(processes: int) -> list[tuple[int, int, int, int]]:
    """Run the probe in `processes` fresh interpreters and collect their counts.

    A fresh interpreter, not a worker in a pool and not a loop in this one: the
    quantity in question is whether the rate moves between processes, and
    `segment_plane`'s thread count and seed state are properties of a process.
    """

    import subprocess

    runs = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), MINORITY_PROBE_FLAG],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(processes)
    ]
    counts = []
    for run in runs:
        out, _ = run.communicate()
        if run.returncode != 0:
            raise RuntimeError(f"a minority probe exited {run.returncode}")
        minority, usable, misses, checks = (int(one) for one in out.split())
        counts.append((minority, usable, misses, checks))
    return counts


def _model_against_the_miss_rate(
    counts: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    """Does the chance-agreement model predict how often the check actually misses?

    The model: a fit lands on the minority axis with probability p, the fits are
    independent, so all `1 + repeats` of them agree -- and the check reports a
    measured axis instead of the ambiguity -- with probability
    `(1-p)**fits + p**fits`. Each process supplies its own p and its own miss count,
    so the comparison never borrows one process's rate to explain another's misses.

    Pooled over every process this is thousands of fits and thousands of checks,
    which is the power the earlier probes lacked. A one-sigma band is published
    beside the prediction so a reader can see whether a gap matters: the previous
    verdicts turned on differences of two or three misses in thirty runs, where the
    noise alone is one and a half.
    """

    fits = UP_AXIS_REPEATS + 1
    predicted = 0.0
    variance = 0.0
    observed = 0
    checks = 0
    for minority, usable, misses, runs in counts:
        if not usable:
            continue
        rate = minority / usable
        agree = (1.0 - rate) ** fits + rate**fits
        predicted += agree * runs
        variance += runs * agree * (1.0 - agree)
        observed += misses
        checks += runs
    sigma = variance**0.5
    gap = (observed - predicted) / sigma if sigma > 0 else None
    return {
        "fits_per_check": fits,
        "checks_run": checks,
        "misses_predicted_by_each_process_own_minority_rate": round(predicted, 1),
        "misses_observed": observed,
        "one_sigma_of_the_prediction": round(sigma, 1),
        "gap_in_sigmas": round(gap, 2) if gap is not None else None,
        "model_rejected": gap is not None and abs(gap) > MODEL_REJECTION_SIGMAS,
        "so": (
            "stated as the comparison rather than as a verdict, because the quantity "
            "that decides it is a gap in sigmas and a reader should see both terms"
        ),
        "where_the_sign_of_this_gap_stands": _pooled_gap(gap),
    }


#: Gaps in sigmas from runs of `_model_against_the_miss_rate` that preceded the one
#: published beside them. Retained rather than averaged away: the first two were -2.29
#: and -1.3, which looked like a systematic 12 percent over-prediction, and the
#: response was to double the checks per process so a bias that size would clear three
#: sigma. The third came back at +0.13. Reading a figure that has not been shown to
#: reproduce is what this whole record is about, and it was about to be committed by
#: the record's author; rerunning the block cost four minutes and was the only thing
#: that caught it. The pooling below is why they are kept as a list.
EARLIER_GAPS_IN_SIGMAS = (-2.29, -1.3, 0.13, -1.04, -0.02, -2.72)


#: Where this record stops calling a gap noise, for the per-run gap and for the
#: pooled statistic alike. Named because both now use it and because the pooled
#: statistic has reached it: a threshold applied in one place and written as prose in
#: the other is how "a couple of sigma" came to sit beside -3.01.
MODEL_REJECTION_SIGMAS = 3.0


def _armadillo_tilt_range(cases: list[dict[str, Any]]) -> tuple[float, float]:
    """The tilt range the counterexample's bound is measured against."""

    case = next(one for one in cases if one["cloud"] == "armadillo_sampled")
    branches = case["single_fit_distribution"]["branches"]
    spans = [one["tilt_to_snapped_axis_degrees_range"] for one in branches]
    return min(one[0] for one in spans), max(one[1] for one in spans)


def _clearance_the_rule_justifies(low: float, high: float) -> tuple[float, float]:
    """The clearance the sweep supports here, and the most the degree line can afford.

    Deliberately the *largest* affordable level rather than the smallest sufficient
    one, matching the suite: with the smallest, loosening the tolerated crossing rate
    would loosen the gate, and no assertion about the record can catch that.
    """

    affordable = (low - MARGIN_RULE["floor_must_stay_above_degrees"]) / (high - low)
    offered = sorted(float(one) for one in MARGIN_RULE["crossing_rate_by_clearance"])
    within_reach = [one for one in offered if one <= affordable]
    if not within_reach:
        raise SystemExit(
            f"this run's fits span {high - low:.2f} degrees from a smallest {low:.2f}, "
            f"so a floor above {MARGIN_RULE['floor_must_stay_above_degrees']:g} can "
            f"carry only {affordable:.3f} of the range -- less than the smallest level "
            f"the sweep measured ({offered[0]:.2f}). The published tilt claim is no "
            "longer supportable at any margin this record justifies, and lowering the "
            "degree line to rescue it would cost the claim its content."
        )
    return within_reach[-1], affordable


def _why_not_more_margin(low: float, high: float) -> str:
    """Why the clearance stops where it does, in the numbers of the run that wrote it."""

    clearance, affordable = _clearance_the_rule_justifies(low, high)
    line = MARGIN_RULE["floor_must_stay_above_degrees"]
    span = high - low
    residual = MARGIN_RULE["crossing_rate_by_clearance"][f"{clearance:.2f}"]
    highest = low - clearance * span
    return (
        "The two requirements pull opposite ways and this record's spread leaves "
        f"little room between them: the fits span {span:.2f} degrees from a smallest "
        f"{low:.2f}, so a floor above {line:g} degrees cannot carry more than "
        f"{affordable:.3f} of the range. Half the range -- which is where the crossing "
        f"rate first reaches zero -- would put the floor at {low - 0.5 * span:.2f} and "
        f"cost the claim its content. At the {clearance:.2f} the published floor has "
        f"to carry, the residual is {residual:.1%} rather than zero, and the floor "
        f"could sit anywhere from {line:g} to {highest:.2f} degrees -- a band "
        f"{highest - line:.2f} degrees wide, which is the gap between two steps of "
        "the sweep and not a range to choose within. Successive floors picked inside "
        "it (14.0, 12.0, 10.5, 10.2) each satisfied this rule against the fits in "
        f"front of them and failed against the next, so the floor is published at "
        f"{line:g} where no regeneration can invalidate it. That makes the margin "
        "comparison an identity and moves the check to the line above it: when the "
        "smallest measured tilt comes within the sweep's own level of the degree "
        "line, no level is affordable and this record does not build."
    )


def _what_this_does_not_claim(
    cases: list[dict[str, Any]], detection: dict[str, Any]
) -> str:
    """The three limits, with their figures read off the cases rather than retyped."""

    armadillo = next((one for one in cases if "armadillo" in one["cloud"]), None)
    tilt = UP_AXIS_AGREEMENT_COUNTEREXAMPLE["plane_tilt_to_the_snapped_axis_degrees"]
    fraction = UP_AXIS_AGREEMENT_COUNTEREXAMPLE["plane_inlier_fraction_was_at_most"]
    measured_tilt = "not recorded in this run"
    if armadillo:
        spans = [
            one["tilt_to_snapped_axis_degrees_range"]
            for one in armadillo["single_fit_distribution"]["branches"]
            if one.get("tilt_to_snapped_axis_degrees_range")
        ]
        if spans:
            measured_tilt = (
                f"{min(one[0] for one in spans):.1f} to "
                f"{max(one[1] for one in spans):.1f} degrees"
            )
    ranked = [
        one
        for one in cases
        if one["what_the_pipeline_publishes"].get("runner_up_relative_to_winner_range")
    ]
    independence = detection["independence_of_the_fits"][
        "does_the_minority_rate_vary_between_processes"
    ]
    predicts = independence["and_does_it_predict_the_miss_rate"]
    pooled = predicts["where_the_sign_of_this_gap_stands"]
    # Three outcomes, and the sentence has to say which one this run got. Writing the
    # favourable one in by hand is how "the misses are accounted for" outlived the
    # measurement that stopped supporting it.
    if pooled["pooled_model_rejected"]:
        chance = (
            "the misses are fewer than independent fits would produce: pooled over "
            f"{pooled['runs_pooled']} runs the gap is {pooled['pooled_z']} sigma, "
            f"past the {pooled['rejection_line_in_sigmas']} this record rejects at, "
            "in the direction where the check misses less often than chance "
            "agreement predicts. So the rate is the measurement and chance agreement "
            "is not a mechanism this record can claim"
        )
    elif predicts["model_rejected"]:
        chance = (
            "this run's misses depart from what independent fits would produce by "
            f"{predicts['gap_in_sigmas']} sigma, which one run cannot separate from "
            "noise; the pooled figure in the same block is what to read"
        )
    else:
        chance = (
            "the misses are consistent with the branches agreeing by chance -- a "
            f"reading the independence block finds admissible at "
            f"{predicts['checks_run']} checks, not one it confirms"
        )
    return (
        "Not that the check always fires: detection_by_repeat_count gives its "
        f"measured rate at three repeat counts, and {chance}. Not that repeat "
        "agreement "
        f"means the axis is right: the armadillo case agrees unanimously on a plane "
        f"tilted past the published {tilt} degree bound off the axis it is snapped to "
        f"(this run measured {measured_tilt}), holding at most {fraction} of the "
        "points, just above the one-tenth floor the weak-plane guard enforces. And "
        f"not a calibrated threshold on the runner-up plane: across {len(ranked)} "
        "clouds the ambiguous constructions rank above the determined ones every "
        "time, but the gap is an ordering with a narrow margin rather than a clean "
        f"separation, and {len(ranked)} clouds cannot place a line that decides "
        "whether a camera axis is trustworthy, so that number is published as "
        "evidence and gates nothing."
    )


def _pooled_gap(gap: float | None) -> dict[str, Any]:
    """Every gap measured so far, pooled, so the direction is derived not asserted.

    One run cannot tell a 10 percent bias from noise and six can start to. Each gap
    is one draw of a quantity whose unit is one sigma, so under a correct model their
    mean has standard deviation 1/sqrt(n) and the pooled z is mean * sqrt(n).
    """

    gaps = list(EARLIER_GAPS_IN_SIGMAS) + ([gap] if gap is not None else [])
    mean = sum(gaps) / len(gaps)
    pooled_z = mean * len(gaps) ** 0.5
    # Derived, not written. Both the adjective and the run count were fixed strings:
    # they said "a couple of sigma" and "after six runs" while the number beside them
    # had reached -3.01 across seven gaps -- and 3.0 is this record's own rejection
    # line for exactly this quantity, applied per-run a few lines below. A regeneration
    # taking the pooled z to -5 would have shipped under the same two words. The
    # strength has to be a function of the z, and the count a function of the list.
    strength = (
        "within noise"
        if abs(pooled_z) < 2.0
        else "around two sigma"
        if abs(pooled_z) < 3.0
        else "at or past the three-sigma line this record uses to reject the model"
    )
    rejected = abs(pooled_z) > MODEL_REJECTION_SIGMAS
    return {
        "gaps_in_sigmas": [round(one, 2) for one in gaps],
        "this_runs_gap_is_the_last_of_them": gap is not None,
        "mean_gap_in_sigmas": round(mean, 2),
        "pooled_z": round(pooled_z, 2),
        "runs_pooled": len(gaps),
        # The same line the per-run gap is judged against, applied to the pooled
        # statistic and published as a field rather than left for a reader to apply.
        # It is reached: the pooled direction has got as far as the record's own
        # rejection threshold, which is a fact about this record and is why the
        # sentence below has to be derived from it.
        "rejection_line_in_sigmas": MODEL_REJECTION_SIGMAS,
        "pooled_model_rejected": rejected,
        # Emitted here because a regeneration has to carry it. This disclosure was the
        # independent reviewer's condition for publishing the pooled z at all, and it
        # had been hand-written into the record while no producer emitted it -- so the
        # next regeneration would have dropped it and left the z standing alone, with
        # the test that demands it passing only against a file nothing reproduces.
        # Found by regenerating, which is the only thing that finds this.
        "how_this_sample_was_sized": (
            "adaptively, which a reader has to know to weigh the z above. The block "
            "was run once, then again because the first two gaps looked like a bias, "
            "and the decision to keep running was made after seeing them. A z from a "
            "sample whose size depended on the data it contains is weaker than the "
            "same z from a sample sized in advance, and the direction of the bias is "
            "towards overstating the effect. Each regeneration adds one draw on the "
            "same terms"
        ),
        "negative_means": (
            "fewer misses observed than predicted, so the fits all agree less often "
            "than independence would make them and the check misses slightly less "
            "often than the model says. The direction favours the check"
        ),
        "so": (
            f"the gap has leaned negative across these {len(gaps)} runs, and pooled "
            f"it is {strength} -- enough to suspect the model over-predicts misses by "
            "around a tenth, not enough to publish that as a figure, and in the "
            "direction that makes the check better than the model rather than worse. "
            "No single run rejects the model and the record does not assert it, so "
            "this changes nothing downstream. Read it against "
            "how_this_sample_was_sized above: the sample size was decided after "
            "seeing the early gaps, which biases a pooled statistic towards "
            f"overstating the effect, so {'the pooled statistic reaching the line' if rejected else 'the direction'} "
            "is a reason to keep measuring rather than a result. It is published "
            f"because declining to look at the direction after {len(gaps)} runs would "
            "be the same error as reading it after two"
        ),
    }


def does_the_minority_rate_vary_between_processes() -> dict[str, Any]:
    """The measurement the probe below claims to make and does not.

    `independence_across_processes` is named for separate processes and runs its
    repeats in a loop in this one, so the 1.1-sigma spread it published is a
    within-process spread. It cannot see the quantity it was built to test. That
    error is mine and the probe is renamed below rather than deleted.

    This runs twelve fresh interpreters at ten times the fits per estimate, which
    divides counting noise by about three. The test is Pearson's chi-square for
    homogeneity: under one fixed rate the statistic's expectation is its degrees of
    freedom, so a value near that count means the spread is noise and a fixed-p
    model is admissible on this evidence.
    """

    counts = _spawn_minority_probes(HIGH_POWER_PROCESSES)
    rates = [one / two for one, two, _, _ in counts if two]
    pooled_minority = sum(one for one, _, _, _ in counts)
    pooled_usable = sum(two for _, two, _, _ in counts)
    pooled = pooled_minority / pooled_usable
    chi_square = 0.0
    for minority, usable, _, _ in counts:
        if not usable or not 0.0 < pooled < 1.0:
            continue
        expected = pooled * usable
        chi_square += (minority - expected) ** 2 / (expected * (1.0 - pooled))
    degrees = len([1 for _, usable, _, _ in counts if usable]) - 1
    noise = (pooled * (1.0 - pooled) / (pooled_usable / len(counts))) ** 0.5
    spread = max(rates) - min(rates)
    return {
        "and_does_it_predict_the_miss_rate": _model_against_the_miss_rate(counts),
        "processes": HIGH_POWER_PROCESSES,
        "each_process_is_a_fresh_interpreter": True,
        "fits_attempted_per_process": HIGH_POWER_FITS,
        "usable_fits_per_process": [two for _, two, _, _ in counts],
        "per_fit_minority_per_process": [round(one, 4) for one in rates],
        "per_fit_minority_range": [round(min(rates), 4), round(max(rates), 4)],
        "pooled_per_fit_minority": round(pooled, 4),
        "one_sigma_of_counting_noise_at_that_rate": round(noise, 4),
        "spread_in_sigmas": round(spread / noise, 2) if noise > 0 else None,
        "chi_square": round(chi_square, 2),
        "degrees_of_freedom": degrees,
        "chi_square_per_degree_of_freedom": (
            round(chi_square / degrees, 2) if degrees > 0 else None
        ),
        # Reject homogeneity only well past the 0.1% point, because the claim being
        # tested is the admissibility of a model and a near miss should not decide it.
        "homogeneity_rejected": degrees > 0 and chi_square > 3.0 * degrees,
        "so": (
            "at ten times the power the four-process probe had, the per-process "
            "minority rates are consistent with one fixed rate: the chi-square per "
            "degree of freedom is near one and the spread is a couple of sigma of "
            "counting noise, where a rate genuinely spanning 0.2 to 0.375 would now "
            "be eight. The variation the low-power probe saw was its own noise. This "
            "removes the reason that probe gave for withdrawing the chance-agreement "
            "model, so a fixed-p model is admissible on this evidence"
        ),
        "what_this_still_does_not_establish": (
            "that the model is correct. Admissible is not confirmed: this measures "
            "the rate's stability across processes, not whether the fits within one "
            "pipeline run are independent of each other, which is the assumption the "
            "model actually needs. The detection rates published above do not depend "
            "on it either way"
        ),
        "this_supersedes": (
            "`repeats_within_one_process` below, which is retained because its spread "
            "is what prompted this one. Two things were wrong with it and both are "
            "recorded rather than quietly fixed: it was named for separate processes "
            "and used none, so its spread was never the between-process quantity it "
            "was built to test; and at forty fits it could not have decided either "
            "way. It was right to call its own result undecidable"
        ),
    }


def _probe_one_process() -> dict[str, Any]:
    """Per-fit minority and misses at the shipped count, measured in one process."""

    cloud = room(ROOM_POINTS, ROOM_POINTS)
    distribution = single_fit_distribution(cloud, VOXEL)
    branches = distribution["branches"]
    minority = (
        min(one["chosen_by_fits"] for one in branches) / distribution["fits"]
        if len(branches) > 1
        else 0.0
    )
    fits = UP_AXIS_REPEATS + 1
    misses = sum(
        1
        for _ in range(PIPELINE_RUNS)
        if _up_axis(o3d, cloud, VOXEL, repeats=UP_AXIS_REPEATS)["source"]
        != "default-ambiguous-plane"
    )
    return {"minority": minority, "fits": fits, "misses": misses, "runs": PIPELINE_RUNS}


def repeats_within_one_process(processes: int = 4) -> dict[str, Any]:
    """The same test repeated in this process, because its verdict moves between runs.

    Named `independence_across_processes` when it was written, which was wrong: the
    loop below calls `_probe_one_process` in this interpreter. The spread it reports
    is between repeated measurements in one process, not between processes, and the
    between-process question is answered by
    `does_the_minority_rate_vary_between_processes` above.

    Run once, this test said the model held (predicted 3.03 misses against 3
    observed at nine fits, p = 0.64). Run again it rejected the model outright
    (p = 0.375 measured, 0.44 misses predicted, 5 observed). Publishing either
    verdict would be publishing a figure that does not reproduce, which is the
    defect this whole record is about, arriving one level up.

    So the verdict is the spread. `segment_plane` is threaded and the seeded split
    already moves between processes, so the per-fit minority is not a constant of
    the geometry and a fixed-p model has no fixed p to be given.
    """

    from math import comb

    observations = []
    for _ in range(processes):
        probe = _probe_one_process()
        agree = (1.0 - probe["minority"]) ** probe["fits"] + probe["minority"] ** probe[
            "fits"
        ]
        runs, misses = probe["runs"], probe["misses"]
        tail = sum(
            comb(runs, k) * agree**k * (1.0 - agree) ** (runs - k)
            for k in range(misses + 1)
        )
        observations.append(
            {
                "per_fit_minority": round(probe["minority"], 4),
                "predicted_misses": round(agree * runs, 3),
                "observed_misses": misses,
                "of_runs": runs,
                "one_sided_p": round(tail, 4),
                "model_rejected_here": not 0.05 <= tail <= 0.95,
            }
        )
    minorities = [one["per_fit_minority"] for one in observations]
    rejecting = [one for one in observations if one["model_rejected_here"]]
    # Is the spread in the minority rate more than counting noise? Each estimate is
    # FITS Bernoulli draws, so under a constant p the standard deviation is
    # sqrt(p(1-p)/FITS). Without this comparison the observed spread looks like
    # evidence that p varies, and it is not.
    pooled = sum(minorities) / len(minorities)
    noise = (pooled * (1.0 - pooled) / FITS) ** 0.5
    spread = max(minorities) - min(minorities)
    return {
        "processes": processes,
        "fits_per_run": UP_AXIS_REPEATS + 1,
        "fits_per_minority_estimate": FITS,
        "observations": observations,
        "per_fit_minority_range": [min(minorities), max(minorities)],
        "pooled_per_fit_minority": round(pooled, 4),
        "one_sigma_of_counting_noise_at_that_rate": round(noise, 4),
        "spread_in_sigmas": round(spread / noise, 2) if noise > 0 else None,
        "the_spread_exceeds_counting_noise": spread > 4.0 * noise,
        "processes_rejecting_the_model": len(rejecting),
        "so": (
            "nothing, on its own. At forty fits per estimate this probe could not "
            "separate a varying rate from counting noise in either direction, and it "
            "was measuring repeated runs in one process while reporting them as "
            "separate processes. Its verdicts are retained as observations. The "
            "between-process question it was built for is answered at ten times this "
            "power by `does_the_minority_rate_vary_between_processes` above"
        ),
        "what_would_settle_it": (
            "a few hundred fits per process across a dozen processes would separate a "
            "varying rate from counting noise. That is a question about RANSAC's "
            "threading, not about whether this check works, so it is named rather "
            "than answered here"
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


#: The rule the published bounds have to satisfy, kept here rather than in the test that
#: enforces it.
#:
#: A test cannot defend its own threshold. Both of this record's previous guards were
#: literals in the suite, and relaxing either one -- the margin fraction to a tenth, the
#: ten-degree line to zero -- left every assertion passing, because the shipped bounds
#: happen to clear the weaker version too. So the level and the line live in the record,
#: where changing them is a change to evidence and the test can notice.
#:
#: `crossing_rate_by_clearance` is not mine. A second lane ran 20 fresh regenerations,
#: pooled them with this record's 4, and measured how often a bound sitting exactly at
#: each fraction is crossed by a later run. Their caveats travel with it: 380 pairs from
#: 20 runs are not independent, so the ordering is solid and the digits are not, and the
#: inlier sweep used this record's fixed range where the tilt sweep used per-run ranges.
#: The k=0 row is the control -- symmetry demands 50% and the method returned 190/380.
MARGIN_RULE = MappingProxyType(
    {
        "what_it_is": (
            "A bound has to clear the nearest end of the range this record's own fits "
            "cover by a fraction of that range's width. Scaled per quantity, because a "
            "flat percentage asks as much of the inlier fraction, whose maximum moves "
            "0.25% between runs, as of the tilt, whose minimum moves 3%."
        ),
        "crossing_rate_by_clearance": {
            "0.00": 0.500,
            "0.10": 0.263,
            "0.20": 0.095,
            "0.30": 0.029,
            "0.50": 0.000,
        },
        "crossing_rate_measured_by": (
            "A second lane, over 20 fresh regenerations pooled with this record's 4. "
            "The 380 pairs are not independent, so the ordering is robust and the "
            "digits are not."
        ),
        "tolerated_crossing_rate": 0.05,
        "floor_must_stay_above_degrees": 10.0,
        "why_that_line": (
            "A floor of zero degrees and a ceiling of every point in the cloud would "
            "satisfy any margin rule and say nothing. The claim is that this plane is "
            "badly tilted and thinly supported, so the floor has to stay where that is "
            "still what it says."
        ),
        # Derived per run by `_why_not_more_margin`, because it is arithmetic on
        # the fits and it quoted a 9.51 span from a smallest 13.41 after the fits had
        # moved to 9.99 from 13.24 -- the same drift the bounds themselves are
        # guarded against, in the sentence explaining the guard.
        "why_not_more_margin": "filled in by main()",
    }
)


def main() -> None:
    if SWEEP_PROBE_FLAG in sys.argv[1:]:
        # One child of `rise_with_fit_count`, which needs detection counts only.
        print(*sweep_in_this_process())
        return
    if MINORITY_PROBE_FLAG in sys.argv[1:]:
        # One child of `_spawn_minority_probes`, which needs counts and nothing else.
        print(*minority_in_this_process())
        return
    ambiguous = room(ROOM_POINTS, ROOM_POINTS)
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
            room(ROOM_POINTS, 2025),
            VOXEL,
            "The same room with a quarter as many wall points. A positive control: "
            "the scan does determine an axis and the pipeline must still name it.",
        ),
        case(
            "floor_ten_to_one",
            room(ROOM_POINTS, 810),
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
    detection = detection_by_repeat_count(
        ambiguous, VOXEL, cases[0]["single_fit_distribution"]
    )
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
        "detection_by_repeat_count": detection,
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
        # Every figure in here is derived from the cases and bounds in this same
        # object. Written out, it said 19 degrees and 13% for the armadillo (no such
        # measurement is in this record, and the published bound had been lowered to
        # 10.5), said the misses "are accounted for" where the independence block
        # concludes admissible and not confirmed, and said five clouds separate
        # cleanly where there are six and the separation is an ordering with a margin
        # the runner itself records as as little as 0.05.
        "what_this_does_not_claim": _what_this_does_not_claim(cases, detection),
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
        "the_margin_rule_the_bounds_must_satisfy": {
            **MARGIN_RULE,
            "why_not_more_margin": _why_not_more_margin(*_armadillo_tilt_range(cases)),
        },
        "provenance": {
            "harness": "harness/up_axis_ambiguity.py",
            "harness_sha256": "recomputed by main()",
            "path_is_relative_to": "this evidence directory",
            "why_a_hash_when_the_figures_do_not_reproduce": (
                "Because the two say different things. The fits are not reproducible, "
                "so a reader cannot check this record by re-running it and comparing "
                "numbers -- which makes it matter more, not less, that the code that "
                "produced it is identifiable. Without this a harness can be edited "
                "and the record left stale with nothing to notice."
            ),
        },
        "environment": {
            "open3d_version": o3d.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    report["provenance"]["harness_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
