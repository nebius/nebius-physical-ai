"""The support crop reports how close it came to deciding otherwise.

The crop is a threshold on a per-vertex distance and the Poisson reconstruction under it
is not reproducible, so whether its verdicts reproduce is a fact about the user's scan
rather than a property of the rule. `docs/evidence/open3d/crop-decision-stability.json`
measured that on two captures and said so explicitly: the condition to re-check
elsewhere is the ratio, not the verdict.

`runner.py` computed the quantity that decides it and discarded it. These tests hold the
reported margin to the distances it comes from, and -- the part that makes the rest worth
having -- prove the verdict can come out negative.

Needs no Open3D: `_crop_margin` takes the distance array the crop already thresholded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from npa.workbench.open3d.runner import (
    CROP_MARGIN_BANDS,
    POISSON_VERTEX_DISPLACEMENT_BOUND_M,
    POISSON_VERTEX_DISPLACEMENT_ROUTINE_M,
    _crop_margin,
)

LIMIT = 0.05


def test_a_disabled_crop_reports_no_margin_rather_than_a_misleading_one() -> None:
    for distances in (None, np.array([])):
        margin = _crop_margin(distances, 0.0)
        assert margin["applies"] is False
        assert "disabled" in margin["why_not"]
        assert "verdict" not in margin


def test_geometry_far_from_the_cut_is_reported_reproducible_with_its_distance() -> None:
    """The comfortable case: outside even the unreproduced outlier."""
    distances = np.array([0.0, 0.004, 0.2, 0.5, 0.012])
    margin = _crop_margin(distances, LIMIT)

    assert margin["applies"] is True
    assert margin["threshold_m"] == LIMIT
    closest = margin["closest_vertex_to_the_threshold_m"]
    assert closest == pytest.approx(min(abs(one - LIMIT) for one in distances))
    assert margin["closest_as_a_fraction_of_the_threshold"] == pytest.approx(
        closest / LIMIT
    )

    reproducibility = margin["reproducibility"]
    assert reproducibility["verdict"] == "reproducible"
    assert reproducibility["vertices_within_the_routine_displacement"] == 0
    assert reproducibility["vertices_within_the_worst_recorded_displacement"] == 0
    assert "no rerun can move one across the cut" in reproducibility["what_that_means"]


def test_geometry_inside_the_displacement_bound_is_reported_not_guaranteed() -> None:
    """The negative control, without which the positive verdict means nothing.

    A gate that cannot fail is not a gate. This constructs the case the published
    result does not cover -- a vertex nearer to the cut than vertices are known to move
    between runs -- and requires the output to say so rather than reporting a margin
    and leaving the reader to compare it themselves.
    """
    inside = LIMIT + POISSON_VERTEX_DISPLACEMENT_ROUTINE_M / 2
    distances = np.array([0.0, 0.004, inside, 0.5])
    margin = _crop_margin(distances, LIMIT)

    reproducibility = margin["reproducibility"]
    assert reproducibility["verdict"] == "not_guaranteed"
    assert reproducibility["vertices_within_the_routine_displacement"] == 1

    said = reproducibility["what_that_means"]
    assert "1 vertex sits closer" in said, said
    # Able to cross is not likely to cross, and the record's own input has exactly one
    # such vertex while reproducing 24 times out of 24.
    assert "may return a different verdict" in said
    assert "May, not will" in said
    # What a user can actually do, without claiming the knob is reliably a reduction.
    assert "denser capture" in said
    assert "not reliably a reduction" in said


def test_the_three_verdicts_follow_the_two_measured_scales() -> None:
    """One bound could not express what the measurements say.

    The first version had a single threshold, and on the record's own input it called
    the scan `not_guaranteed` -- the same scan whose 46 pair comparisons all agreed.
    That is not a useful warning, and collapsing two measured scales is what produced
    it. Written against the constants so that re-measuring either re-aims the verdict
    rather than breaking this test.
    """
    routine = POISSON_VERTEX_DISPLACEMENT_ROUTINE_M
    bound = POISSON_VERTEX_DISPLACEMENT_BOUND_M
    assert 0.0 < routine < bound, (routine, bound)

    inside_routine = _crop_margin(np.array([LIMIT + routine * 0.99]), LIMIT)
    between = _crop_margin(np.array([LIMIT + bound * 0.99]), LIMIT)
    outside_both = _crop_margin(np.array([LIMIT + bound * 1.01]), LIMIT)

    assert inside_routine["reproducibility"]["verdict"] == "not_guaranteed"
    assert between["reproducibility"]["verdict"] == (
        "reproducible_at_the_routine_displacement"
    ), (
        "a vertex outside the recurring displacement but inside the outlier has no middle"
    )
    assert outside_both["reproducibility"]["verdict"] == "reproducible"

    # The middle verdict has to say both halves, or it is just a pass with extra words.
    said = between["reproducibility"]["what_that_means"]
    assert "reproducible in practice" in said
    assert "not reproducible by construction" in said
    assert "not reproduced in three further attempts" in said


def test_the_near_threshold_population_is_a_distribution_not_one_number() -> None:
    """One count answers "how many are near the cut" only if "near" is defined."""
    rng = np.random.default_rng(0)
    distances = np.concatenate(
        [
            rng.uniform(0.0, 0.1, 400),
            np.full(5, LIMIT * 1.001),
        ]
    )
    margin = _crop_margin(distances, LIMIT)
    bands = margin["vertices_within"]

    assert len(bands) == len(CROP_MARGIN_BANDS)
    counts = [
        bands[f"{share:g}_of_the_threshold"]["count"] for share in CROP_MARGIN_BANDS
    ]
    assert counts == sorted(counts), (
        f"the bands are not nested: {counts} for {CROP_MARGIN_BANDS}, so a wider band "
        "excludes a vertex a narrower one includes"
    )
    for share in CROP_MARGIN_BANDS:
        one = bands[f"{share:g}_of_the_threshold"]
        assert one["count"] == int((np.abs(distances - LIMIT) < share * LIMIT).sum())
        assert one["fraction"] == pytest.approx(one["count"] / len(distances))


def test_it_says_what_it_does_not_establish() -> None:
    """A reproducible verdict on an invented surface is still a verdict on one."""
    margin = _crop_margin(np.array([0.0, 0.5]), LIMIT)
    said = margin["what_this_does_not_establish"]
    assert "reconstruction is correct" in said
    assert "invented surface" in said


def test_the_record_from_the_real_stage_carries_the_margin_it_describes() -> None:
    """The figure has to reach a caller, not merely be computable.

    `crop-margin-live.json` is produced by calling `run_multiway` and `run_reconstruct`
    -- the entry points the workflow calls -- and reading the field out of the returned
    summary. This checks the record against the constants the verdict is derived from,
    so a record generated before a re-measurement cannot sit beside a changed bound.

    Needs no Open3D: it reads the committed record.
    """
    record = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "docs"
            / "evidence"
            / "open3d"
            / "crop-margin-live.json"
        ).read_text()
    )

    scales = record["displacement_scales_the_verdict_uses"]
    assert scales["routine_m"] == POISSON_VERTEX_DISPLACEMENT_ROUTINE_M
    assert scales["worst_recorded_m"] == POISSON_VERTEX_DISPLACEMENT_BOUND_M
    assert "run_multiway then run_reconstruct" in record["executed"]
    assert "no mirrored copy" in record["executed"]

    enabled = [
        one["crop_margin"]
        for one in record["per_support_factor"].values()
        if one["crop_margin"]["applies"]
    ]
    disabled = [
        one["crop_margin"]
        for one in record["per_support_factor"].values()
        if not one["crop_margin"]["applies"]
    ]
    assert enabled, "no factor in the record actually exercised the crop"
    assert disabled, (
        "the disabled factor is missing, so the record does not show the field "
        "declining to report a margin there is no threshold for"
    )

    for margin in enabled:
        reproducibility = margin["reproducibility"]
        assert reproducibility["verdict"] in {
            "reproducible",
            "reproducible_at_the_routine_displacement",
            "not_guaranteed",
        }
        # Nested by construction: the routine scale is inside the recorded worst.
        assert (
            reproducibility["vertices_within_the_routine_displacement"]
            <= reproducibility["vertices_within_the_worst_recorded_displacement"]
        ), reproducibility
        assert margin["closest_vertex_to_the_threshold_m"] >= 0.0
        assert margin["threshold_m"] > 0.0
