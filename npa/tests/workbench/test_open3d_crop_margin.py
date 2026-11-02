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

import importlib.util
import json
import re
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from npa.workbench.open3d import runner
from npa.workbench.open3d.runner import (
    CROP_MARGIN_BANDS,
    POISSON_VERTEX_DISPLACEMENT_BOUND_M,
    POISSON_VERTEX_DISPLACEMENT_ROUTINE_M,
    _crop_margin,
)

LIMIT = 0.05
#: A voxel inside the range the displacement figures were measured at. Tests that want
#: one of the measured verdicts have to declare it: outside the range the field abstains,
#: which is the point of `test_a_voxel_outside_the_measured_range_abstains...`.
VOXEL = 0.05


def test_a_disabled_crop_reports_no_margin_rather_than_a_misleading_one() -> None:
    for distances in (None, np.array([])):
        margin = _crop_margin(distances, 0.0)
        assert margin["applies"] is False
        assert "disabled" in margin["why_not"]
        assert "verdict" not in margin


def test_geometry_far_from_the_cut_is_reported_reproducible_with_its_distance() -> None:
    """The comfortable case: outside even the unreproduced outlier."""
    distances = np.array([0.0, 0.004, 0.2, 0.5, 0.012])
    margin = _crop_margin(distances, LIMIT, VOXEL)

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
    margin = _crop_margin(distances, LIMIT, VOXEL)

    reproducibility = margin["reproducibility"]
    assert reproducibility["verdict"] == "not_guaranteed"
    assert reproducibility["vertices_within_the_routine_displacement"] == 1

    said = reproducibility["what_that_means"]
    assert "1 vertex sits closer" in said, said
    # Able to cross is not likely to cross. The committed record's own input exposes a
    # couple of vertices at this scale and no in-repo artifact holds a 24-repeat count;
    # this comment previously claimed both, which is a claim about a run rather than
    # about the sentence under test.
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

    inside_routine = _crop_margin(np.array([LIMIT + routine * 0.99]), LIMIT, VOXEL)
    between = _crop_margin(np.array([LIMIT + bound * 0.99]), LIMIT, VOXEL)
    outside_both = _crop_margin(np.array([LIMIT + bound * 1.01]), LIMIT, VOXEL)

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
    margin = _crop_margin(distances, LIMIT, VOXEL)
    bands = margin["vertices_within"]

    assert len(bands) == len(CROP_MARGIN_BANDS)
    counts = [
        bands[f"within_{format(share * 100, 'g')}_percent_of_the_threshold"]["count"]
        for share in CROP_MARGIN_BANDS
    ]
    assert counts == sorted(counts), (
        f"the bands are not nested: {counts} for {CROP_MARGIN_BANDS}, so a wider band "
        "excludes a vertex a narrower one includes"
    )
    for share in CROP_MARGIN_BANDS:
        one = bands[f"within_{format(share * 100, 'g')}_percent_of_the_threshold"]
        assert one["count"] == int((np.abs(distances - LIMIT) < share * LIMIT).sum())
        assert one["fraction"] == pytest.approx(one["count"] / len(distances))


def test_it_says_what_it_does_not_establish() -> None:
    """A reproducible verdict on an invented surface is still a verdict on one."""
    margin = _crop_margin(np.array([0.0, 0.5]), LIMIT, VOXEL)
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
        reproducibility = _crop_margin(distances, threshold, VOXEL)["reproducibility"]
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
    margin = _crop_margin(np.array([0.08, 0.09]), 0.05, VOXEL)
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
    provenance = _crop_margin(np.array([0.08, 0.09]), 0.05, VOXEL)["reproducibility"][
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
        margin = _crop_margin(distances, limit, VOXEL)
        round_tripped = json.loads(json.dumps(margin))
        assert round_tripped == margin, (
            "the margin does not survive a JSON round trip unchanged, so the artifact "
            "a caller reads is not the value this code returned"
        )


def _publication_tool():
    """The deny-by-default report generator, loaded without running it.

    It lives in `docs/evidence/open3d/harness/`, outside the package, and needs no
    Open3D. Loading it here is what puts its allowlist under CI: no workflow runs the
    harness, so a published field it cannot classify would otherwise stay latent until
    the next evidence publication.
    """
    path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "evidence"
        / "open3d"
        / "harness"
        / "derive_public_run_report.py"
    )
    spec = importlib.util.spec_from_file_location("derive_public_run_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leaves(node, prefix: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        yield f"{prefix}[]" if prefix else "[]"
    else:
        yield prefix


def test_every_published_margin_leaf_is_classified_for_publication() -> None:
    """The reconstruct summary is published verbatim, and the publisher denies by default.

    `runtime.reconstruct` writes this summary as `surface/result.json`, and
    `derive_public_run_report.py` raises `SystemExit` on the first leaf that matches
    neither its allowlist nor its omit list. Adding a subtree without classifying it
    breaks the only sanctioned route for publishing a real run's artifacts, on every run
    from that commit onward -- and no CI job runs the harness, so nothing would say so.

    All four shapes of the field are checked, including the two that decline to report a
    margin, because they emit different keys.
    """
    tool = _publication_tool()
    allowed = tool.ALLOWED["surface/result.json"]

    shapes = {
        "enabled, voxel inside the measured range": (
            np.array([0.08, 0.051]),
            0.05,
            0.05,
        ),
        "enabled, voxel outside it": (np.array([0.8, 0.51]), 0.5, 0.5),
        "disabled": (None, 0.0, 0.05),
        "enabled with no vertices": (np.array([]), 0.05, 0.05),
    }
    for label, args in shapes.items():
        margin = _crop_margin(*args)
        unclassified = [
            leaf
            for leaf in _leaves({"crop_margin": margin})
            if not any(tool._matches(pattern, leaf) for pattern in allowed)
        ]
        assert not unclassified, (label, unclassified)


def test_no_published_key_contains_a_dot() -> None:
    """The publisher splits leaf paths on `.`, so a dotted key silently mis-segments.

    The band keys were `0.01_of_the_threshold`, which splits into `0` and
    `01_of_the_threshold` and cannot be addressed by any pattern a reader would write.
    Checked structurally rather than against the current band values, so changing
    CROP_MARGIN_BANDS to another fraction cannot reintroduce it.
    """
    for bands in ((0.01, 0.1, 0.5), (0.005, 0.25), (1.0,)):
        with patch.object(runner, "CROP_MARGIN_BANDS", bands):
            margin = _crop_margin(np.array([0.08, 0.051]), 0.05, 0.05)
        for key in margin["vertices_within"]:
            assert "." not in key, (bands, key)
        assert len(margin["vertices_within"]) == len(bands), bands


def test_a_voxel_outside_the_measured_range_abstains_rather_than_reassures() -> None:
    """The threshold scales with the capture; the displacement figures do not.

    `limit = voxel * support_distance_factor`, so a coarser capture inflates the
    threshold while both displacement constants stay at the absolute metres they were
    measured at. Nothing in this function read the voxel, so the same 2.187e-03 m was
    0.44% of a 0.5 m threshold -- almost every vertex clears it -- and the strongest
    verdict was published for a scale regime where no displacement has been measured.
    The error ran in the unsafe direction: the coarser the capture, the more confidently
    immutable the result was declared.
    """
    low, high = runner.MEASURED_VOXEL_RANGE_M

    # Geometry a comfortable distance clear of the cut in every case, so the verdict is
    # decided by the configuration and not by the distances.
    for voxel in (high * 10.0, low / 10.0, None):
        limit = (voxel or high) * 1.0
        margin = _crop_margin(
            np.array([limit + limit * 5, limit + limit * 6]), limit, voxel
        )
        reproducibility = margin["reproducibility"]
        assert reproducibility["verdict"] == "not_measured_at_this_configuration", (
            voxel,
            reproducibility["verdict"],
        )
        assert "scales with the capture" in reproducibility["what_that_means"]
        assert "not symmetric" in reproducibility["what_that_means"]
        # The measurements it declines to apply are still reported, since the counts and
        # distances are what the crop actually decided on.
        assert margin["closest_vertex_to_the_threshold_m"] > 0.0
        assert margin["vertices_within"]

    inside = _crop_margin(np.array([high + 0.03, high + 0.04]), high, high)
    assert (
        inside["reproducibility"]["verdict"]
        == "reproducible_at_every_displacement_measured"
    )
    assert inside["reproducibility"]["measured_voxel_range_m"] == [low, high]


def test_the_count_bound_names_the_scale_it_holds_at() -> None:
    """ "Up to that many vertices" was contradicted by the field beside it.

    At the routine displacement the record's own input exposed 2 vertices while
    `vertices_within_the_worst_recorded_displacement` in the same object read 238. The
    sentence offered the smaller number as an upper bound on how far the derived counts
    can move, while the module's own second constant records a displacement 200x larger.
    A caller sizing an error bar off it understated the movement by two orders of
    magnitude with the right figure one key away.
    """
    limit = 0.05
    routine = runner.POISSON_VERTEX_DISPLACEMENT_ROUTINE_M
    bound = runner.POISSON_VERTEX_DISPLACEMENT_BOUND_M
    # One vertex inside the routine scale, several more inside the recorded outlier.
    distances = np.array(
        [limit + routine * 0.5] + [limit + bound * 0.5] * 4 + [limit + 0.03],
    )
    reproducibility = _crop_margin(distances, limit, 0.05)["reproducibility"]

    at_routine = reproducibility["vertices_within_the_routine_displacement"]
    at_bound = reproducibility["vertices_within_the_worst_recorded_displacement"]
    assert at_routine == 1 and at_bound == 5, reproducibility

    said = reproducibility["what_that_means"]
    assert f"up to {at_routine} at that displacement" in said, said
    assert f"up to {at_bound} at the largest displacement ever recorded" in said, said
    # The outer figure is the one to size an error bar from, and the sentence says so
    # rather than leaving the reader to find the sibling key.
    assert said.index(f"up to {at_routine}") < said.index(f"up to {at_bound}"), said


def test_the_field_says_a_displacement_is_not_a_tracked_vertex() -> None:
    """The precondition both the verdict and the bound reason over, stated.

    A rerun calls Poisson again and re-derives its density threshold from that run's own
    density array, so the vertex set is regenerated rather than perturbed; "how far a
    vertex moved" has no correspondence to be defined over. Three reconstructs of one
    fused cloud at voxel 0.05 produced identical removal counts here, so the decisions
    were stable -- but stability observed is not a correspondence, and the categorical
    reading of the counts needs the assumption declared rather than assumed.
    """
    reproducibility = _crop_margin(np.array([0.08, 0.051]), 0.05, 0.05)[
        "reproducibility"
    ]
    said = reproducibility["what_a_displacement_means_here"]

    assert "Not the motion of a tracked vertex" in said
    assert "regenerated rather than perturbed" in said
    assert "nearest-point distances between two" in said
    assert "not a count of vertices that can flip" in said
    # And it has to admit the failure mode the counts cannot bound at all.
    assert "introduce or drop vertices" in said


def test_an_enabled_crop_with_no_vertices_does_not_claim_to_be_disabled() -> None:
    """Two conditions shared one reason, and for one of them the reason was false.

    `if limit <= 0.0 or not len(distances)` returned "support cropping is disabled" for
    both. The existing disabled-crop test passed `limit=0.0` for every case, so Python
    short-circuited on the first clause and the empty-array branch had no coverage at
    all. `_crop_margin(None, limit)` with a positive limit also raised TypeError on
    `len(None)`.
    """
    limit = 0.05
    for distances in (np.array([]), None):
        margin = _crop_margin(distances, limit, 0.05)
        assert margin["applies"] is False
        why = margin["why_not"]
        assert "no vertices to measure" in why, why
        assert "disabled" not in why, (
            "an enabled crop reported as disabled is the mislabel this covers"
        )
        assert f"{limit:g} m threshold" in why, why

    disabled = _crop_margin(np.array([0.08]), 0.0, 0.05)
    assert disabled["applies"] is False
    assert "disabled" in disabled["why_not"]
