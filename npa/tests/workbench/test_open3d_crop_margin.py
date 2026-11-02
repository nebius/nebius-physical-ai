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
import re
from collections.abc import Mapping
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
    assert reproducibility["verdict"] == "reproducible_at_every_displacement_measured"
    assert reproducibility["vertices_within_the_routine_displacement"] == 0
    assert reproducibility["vertices_within_the_worst_recorded_displacement"] == 0
    said = reproducibility["what_that_means"]
    # Scoped to what was measured. This assertion previously required the sentence
    # "no rerun can move one across the cut", so the test was holding the impossibility
    # claim in place rather than catching it.
    assert "no displacement anyone has measured would move one across the cut" in said
    assert "not about reruns in general" in said


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
    assert (
        outside_both["reproducibility"]["verdict"]
        == "reproducible_at_every_displacement_measured"
    )

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
            "reproducible_at_every_displacement_measured",
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


def test_no_verdict_claims_a_displacement_cannot_happen() -> None:
    """The figures are sample maxima, so no verdict may be phrased as an impossibility.

    The strongest verdict previously read "so no rerun can move one across the cut".
    That is an impossibility claim derived from the largest value in a finite number of
    comparisons on one scene, which is the shape of claim this whole record exists to
    avoid making. Checked on every reachable verdict rather than only the strongest,
    because the wording is generated per branch.
    """
    threshold = 0.05
    cases = {
        "inside_routine": np.array([threshold + 1.0e-07, threshold + 0.03]),
        "inside_bound": np.array([threshold + 1.0e-04, threshold + 0.03]),
        "outside_both": np.array([threshold + 0.03, threshold + 0.04]),
    }

    forbidden = (
        "no rerun can",
        "cannot move",
        "can never",
        "is guaranteed",
        "guarantees",
        "will always",
        "proves",
    )
    seen = set()
    for name, distances in cases.items():
        reproducibility = _crop_margin(distances, threshold)["reproducibility"]
        seen.add(reproducibility["verdict"])
        said = reproducibility["what_that_means"].lower()
        for phrase in forbidden:
            assert phrase not in said, (name, phrase, said)

    # Stated, not recomputed: these are the three verdicts this field may emit, and the
    # two affirmative ones each name the scale they are scoped to rather than asserting
    # reproducibility outright.
    assert seen == {
        "not_guaranteed",
        "reproducible_at_the_routine_displacement",
        "reproducible_at_every_displacement_measured",
    }, seen


def test_the_record_states_the_scope_its_figures_came_from() -> None:
    """An applicability limit, because the figures are one scene's and the verdict is not.

    A caller reconstructing different geometry gets a verdict computed from displacement
    measured somewhere else. That is useful and it is not transferable by construction,
    so the output has to say which it is.
    """
    margin = _crop_margin(np.array([0.08, 0.09]), 0.05)
    limits = margin["reproducibility"]["what_no_verdict_here_establishes"]

    assert "sample maxima" in limits
    assert "not bounds" in limits
    assert "one scene" in limits

    provenance = margin["reproducibility"]["measured_by"]
    assert isinstance(provenance, Mapping), (
        "provenance is a single string, so a reader cannot tell the measurement apart "
        "from the input it was measured on"
    )
    assert "one scene" in provenance["on_what_input"]
    assert "sample maxima" in provenance["kind_of_figure"]
    assert "46 pair comparisons" in provenance["what_was_measured"]


def test_provenance_does_not_point_at_a_path_absent_from_the_tree() -> None:
    """A provenance string naming a file that is not here is not provenance.

    The field cited `docs/evidence/open3d/crop-decision-stability.json`, which is
    published with the stability study and is not on this branch. A reader following the
    citation finds nothing. Any repo-relative path this field names must resolve.
    """
    provenance = _crop_margin(np.array([0.08, 0.09]), 0.05)["reproducibility"][
        "measured_by"
    ]
    repository = Path(__file__).resolve().parents[3]

    paths = []
    for value in provenance.values():
        paths.extend(re.findall(r"[\w./-]+/[\w./-]+\.(?:json|py|md)", value))

    for candidate in paths:
        assert (repository / candidate).exists(), (
            f"provenance names {candidate}, which does not exist in this tree; cite it "
            "by content, or by a path that resolves where the reader is standing"
        )


def test_the_margin_survives_being_written_to_an_artifact() -> None:
    """Every field here is serialised into a run artifact, so it has to encode.

    The structured provenance was first emitted as the `MappingProxyType` constant
    itself. `isinstance(value, Mapping)` is true of that proxy, so the shape assertions
    above passed while the real harness died with `Object of type mappingproxy is not
    JSON serializable` and wrote a truncated record. A test that cannot fail the way the
    artifact fails is not a test of the artifact.
    """
    for distances, limit in (
        (np.array([0.08, 0.09]), 0.05),
        (np.array([0.05 + 1.0e-07, 0.08]), 0.05),
        (None, 0.0),
    ):
        margin = _crop_margin(distances, limit)
        round_tripped = json.loads(json.dumps(margin))
        assert round_tripped == margin, (
            "the margin does not survive a JSON round trip unchanged, so the artifact "
            "a caller reads is not the value this code returned"
        )
