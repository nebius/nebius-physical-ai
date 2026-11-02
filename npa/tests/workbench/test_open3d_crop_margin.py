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
#: One of the two voxel sizes the reference figures were actually measured at, with the
#: Poisson depth that study ran. Both have to be declared for `configuration_measured`
#: to be true: there is no interval between the two measured sizes, and depth changes
#: the vertex set the crop decides on.
VOXEL = 0.05
DEPTH = 9


def test_a_disabled_crop_reports_no_margin_rather_than_a_misleading_one() -> None:
    for distances in (None, np.array([])):
        margin = _crop_margin(distances, 0.0)
        assert margin["applies"] is False
        assert "disabled" in margin["why_not"]
        assert "verdict" not in margin


def test_geometry_far_from_the_cut_reports_its_distance_and_no_vertices_near() -> None:
    """The comfortable case, reported as a measurement rather than as a verdict."""
    distances = np.array([0.0, 0.004, 0.2, 0.5, 0.012])
    margin = _crop_margin(distances, LIMIT, VOXEL, DEPTH)

    assert margin["applies"] is True
    assert margin["threshold"] == LIMIT
    closest = margin["closest_vertex_to_the_threshold"]
    assert closest == pytest.approx(min(abs(one - LIMIT) for one in distances))
    assert margin["closest_as_a_fraction_of_the_threshold"] == pytest.approx(
        closest / LIMIT
    )

    comparison = margin["reference_comparison"]
    assert comparison["description"] == (
        "no_vertices_closer_than_either_reference_displacement"
    )
    assert comparison["vertices_closer_than_the_routine_reference"] == 0
    assert comparison["vertices_closer_than_the_larger_reference"] == 0
    # The strongest reading available, and it is still only a description of this run.
    assert "Not a prediction about re-running" in comparison["what_this_is"]


def test_geometry_inside_the_displacement_bound_is_reported_not_guaranteed() -> None:
    """The negative control, without which the positive verdict means nothing.

    A gate that cannot fail is not a gate. This constructs the case the published
    result does not cover -- a vertex nearer to the cut than vertices are known to move
    between runs -- and requires the output to say so rather than reporting a margin
    and leaving the reader to compare it themselves.
    """
    inside = LIMIT + POISSON_VERTEX_DISPLACEMENT_ROUTINE_M / 2
    distances = np.array([0.0, 0.004, inside, 0.5])
    margin = _crop_margin(distances, LIMIT, VOXEL, DEPTH)

    comparison = margin["reference_comparison"]
    assert comparison["description"] == (
        "vertices_closer_than_both_reference_displacements"
    )
    assert comparison["vertices_closer_than_the_routine_reference"] == 1
    assert comparison["vertices_closer_than_the_larger_reference"] == 1

    # The count is the finding; the distance is what a caller acts on. Neither is
    # dressed up as a statement about what a rerun would do.
    assert margin["closest_vertex_to_the_threshold"] == pytest.approx(
        POISSON_VERTEX_DISPLACEMENT_ROUTINE_M / 2
    )
    assert "sample maxima" in comparison["reference_displacements"]["kind_of_figure"]


def test_the_three_descriptions_follow_the_two_reference_scales() -> None:
    """Three descriptions, because two reference figures cut the line in three.

    Written against the constants so that re-measuring either re-aims the description
    rather than breaking this test. What changed from the first version of this field
    is only what the three names claim: they now say where this run's vertices sat
    relative to two numbers, and nothing about reruns.
    """
    routine = POISSON_VERTEX_DISPLACEMENT_ROUTINE_M
    bound = POISSON_VERTEX_DISPLACEMENT_BOUND_M
    assert 0.0 < routine < bound, (routine, bound)

    cases = {
        "inside both": (LIMIT + routine * 0.99, "both_reference_displacements"),
        "inside the larger only": (
            LIMIT + bound * 0.99,
            "the_larger_reference_displacement_only",
        ),
        "outside both": (LIMIT + bound * 1.01, "either_reference_displacement"),
    }
    for label, (distance, tail) in cases.items():
        margin = _crop_margin(np.array([distance]), LIMIT, VOXEL, DEPTH)
        got = margin["reference_comparison"]["description"]
        assert got.endswith(tail), (label, got)

    # And the names are exactly these three. Stated rather than derived, so a fourth
    # cannot appear without this test being updated deliberately.
    assert {
        _crop_margin(np.array([one]), LIMIT, VOXEL, DEPTH)["reference_comparison"][
            "description"
        ]
        for one, _ in cases.values()
    } == {
        "vertices_closer_than_both_reference_displacements",
        "vertices_closer_than_the_larger_reference_displacement_only",
        "no_vertices_closer_than_either_reference_displacement",
    }


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

    scales = record["reference_displacements_the_comparison_uses"]
    assert scales["routine_m"] == POISSON_VERTEX_DISPLACEMENT_ROUTINE_M
    assert scales["largest_recorded_m"] == POISSON_VERTEX_DISPLACEMENT_BOUND_M
    assert "run_multiway then run_reconstruct" in record["executed"]
    assert "no mirrored copy" in record["executed"]

    # The interpreter and the source it imported, because these worktrees have no venv
    # of their own and a record naming a relative `npa/.venv/bin/python` cannot be
    # re-run. The runner path has to be this candidate's, not the lending worktree's.
    provenance = record["provenance"]
    assert provenance["npa_runner_imported_from"].endswith(
        "npa/src/npa/workbench/open3d/runner.py"
    )
    assert Path(provenance["npa_runner_imported_from"]).exists(), (
        "the record names a runner source that is not on this machine"
    )
    assert provenance["interpreter"].endswith("python")
    assert provenance["open3d_imported_from"]["version"]

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
        comparison = margin["reference_comparison"]
        assert comparison["description"] in {
            "no_vertices_closer_than_either_reference_displacement",
            "vertices_closer_than_the_larger_reference_displacement_only",
            "vertices_closer_than_both_reference_displacements",
        }
        # Nested by construction: the routine reference is inside the larger one.
        assert (
            comparison["vertices_closer_than_the_routine_reference"]
            <= comparison["vertices_closer_than_the_larger_reference"]
        ), comparison
        assert margin["closest_vertex_to_the_threshold"] >= 0.0
        assert margin["threshold"] > 0.0
        assert "coordinate units" in margin["distance_units"]

    # The narrative has to be derived from the rows, not maintained beside them. The
    # previous one said "a handful of vertices ... at both enabled factors" and the next
    # run returned zero at one of them.
    shows = record["what_the_result_shows"]
    for name, got in record["per_support_factor"].items():
        assert name in shows, (name, shows)
        if got["crop_margin"]["applies"]:
            closer = got["crop_margin"]["reference_comparison"][
                "vertices_closer_than_the_larger_reference"
            ]
            assert f"{closer} closer than the larger" in shows, (name, shows)
    # No prediction anywhere in the record. `what_this_does_not_establish` is the one
    # exception, and only because its job is to name the two claims that were removed;
    # the exemption is explicit so the phrases cannot drift back in under cover of it.
    disclaimer = record.pop("what_this_does_not_establish")
    assert "reproducible in practice" in disclaimer
    assert "error bar" in disclaimer
    assert "both were removed" in disclaimer
    for phrase in _PREDICTION_PHRASES:
        assert phrase not in json.dumps(record).lower(), phrase


#: Phrasings that make a claim about a run nobody has made yet. Every one of these was
#: in this field at some point. The first version said the strongest case meant "no
#: rerun can move one across the cut"; the second replaced that with "a rerun is
#: expected to return the same verdicts ... reproducible in practice" and offered the
#: near-threshold counts as the figure to size an error bar from. Both are predictions
#: drawn from sample maxima on one scene, and the second contradicted the correspondence
#: caveat printed two keys below it.
_PREDICTION_PHRASES = (
    "no rerun can",
    "cannot move",
    "can never",
    "is guaranteed",
    "guarantees",
    "will always",
    "proves",
    "a rerun is expected",
    "expected to return the same",
    "reproducible in practice",
    "by up to",
    "error bar",
    "will return",
    "would return the same",
)


def test_nothing_in_the_margin_predicts_what_a_rerun_would_do() -> None:
    """No field here may make a claim about a run that has not happened.

    Checked over the whole serialised payload rather than one sentence, in every
    reachable configuration, because the wording is generated per branch and the
    previous version's contradiction was between two keys rather than inside one.
    """
    threshold = 0.05
    cases = {
        "inside both": (
            np.array([threshold + 1.0e-07, threshold + 0.03]),
            VOXEL,
            DEPTH,
        ),
        "inside the larger": (
            np.array([threshold + 1.0e-04, threshold + 0.03]),
            VOXEL,
            DEPTH,
        ),
        "outside both": (np.array([threshold + 0.03, threshold + 0.04]), VOXEL, DEPTH),
        "unmeasured voxel": (np.array([threshold + 0.03]), 0.03, DEPTH),
        "unmeasured depth": (np.array([threshold + 0.03]), VOXEL, 11),
        "no configuration": (np.array([threshold + 0.03]), None, None),
    }

    for name, (distances, voxel, depth) in cases.items():
        payload = json.dumps(_crop_margin(distances, threshold, voxel, depth)).lower()
        for phrase in _PREDICTION_PHRASES:
            assert phrase not in payload, (name, phrase)

    # And the one sentence whose job is to say so, says so.
    comparison = _crop_margin(np.array([0.08]), threshold, VOXEL, DEPTH)[
        "reference_comparison"
    ]
    does_not = comparison["what_this_does_not_establish"]
    assert "That a rerun would decide the same way, or differently" in does_not
    assert "no count here bounds how much any count would change" in does_not


def test_the_record_states_the_scope_its_figures_came_from() -> None:
    """An applicability limit, because the figures are one scene's and the verdict is not.

    A caller reconstructing different geometry gets a verdict computed from displacement
    measured somewhere else. That is useful and it is not transferable by construction,
    so the output has to say which it is.
    """
    margin = _crop_margin(np.array([0.08, 0.09]), 0.05, VOXEL, DEPTH)
    comparison = margin["reference_comparison"]

    assert "sample maxima" in comparison["reference_displacements"]["kind_of_figure"]
    assert (
        "Neither is a bound" in comparison["reference_displacements"]["kind_of_figure"]
    )
    assert "one scene" in comparison["measured_configuration"]["scene"]

    provenance = comparison["measured_by"]
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
    provenance = _crop_margin(np.array([0.08, 0.09]), 0.05, VOXEL, DEPTH)[
        "reference_comparison"
    ]["measured_by"]
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


def test_a_setting_between_the_two_measured_ones_is_not_called_measured() -> None:
    """Two tested points are not an interval, and the field used to treat them as one.

    `MEASURED_VOXEL_RANGE_M = (0.02, 0.05)` was read as `low <= voxel <= high`, so
    voxel 0.03 -- a setting nobody has run -- received the same affirmative reading as
    the two that were measured, and Poisson depth was not considered at all. Both are
    now exact: the configuration matches or it does not, and when it does not the field
    says which part failed instead of reporting a weaker reading.
    """
    low, high = runner.MEASURED_VOXEL_SIZES
    depth = runner.REFERENCE_POISSON_DEPTH
    between = (low + high) / 2.0
    assert low < between < high, (low, between, high)

    unmeasured = {
        "between the two measured sizes": (between, depth),
        "coarser than either": (high * 10.0, depth),
        "finer than either": (low / 10.0, depth),
        "measured size, unmeasured depth": (high, depth + 2),
        "no voxel reported": (None, depth),
        "no depth reported": (high, None),
    }
    for label, (voxel, got_depth) in unmeasured.items():
        limit = (voxel or high) * 1.0
        margin = _crop_margin(
            np.array([limit + limit * 5, limit + limit * 6]), limit, voxel, got_depth
        )
        comparison = margin["reference_comparison"]
        assert comparison["configuration_measured"] is False, label
        assert comparison["why_the_configuration_does_not_match"], label
        # The measurement it declines to call comparable is still reported, because the
        # counts and distances are what the crop actually decided on.
        assert margin["closest_vertex_to_the_threshold"] > 0.0
        assert margin["vertices_within"]

    for voxel in (low, high):
        matched = _crop_margin(
            np.array([voxel + 0.03, voxel + 0.04]), voxel, voxel, depth
        )["reference_comparison"]
        assert matched["configuration_measured"] is True, voxel
        assert matched["why_the_configuration_does_not_match"] is None
        assert matched["measured_configuration"]["voxel_sizes"] == [low, high]
        assert matched["measured_configuration"]["poisson_depth"] == depth


def test_the_counts_are_reported_without_being_offered_as_a_bound() -> None:
    """The counts are real and stay; what was removed is what they were sold as.

    The field read "the counts derived from it move by up to 1 at that displacement --
    and by up to 239 at the largest displacement ever recorded, which is the figure to
    size an error bar from", two keys above a caveat stating that a rerun regenerates
    the vertex set and that "the counts above do not bound that". Both cannot be true.
    The counts are kept because they describe this run; the bound and the error bar are
    gone because nothing measured supports them.
    """
    limit = 0.05
    routine = runner.POISSON_VERTEX_DISPLACEMENT_ROUTINE_M
    bound = runner.POISSON_VERTEX_DISPLACEMENT_BOUND_M
    # One vertex inside the routine scale, several more inside the recorded outlier.
    distances = np.array(
        [limit + routine * 0.5] + [limit + bound * 0.5] * 4 + [limit + 0.03],
    )
    comparison = _crop_margin(distances, limit, VOXEL, DEPTH)["reference_comparison"]

    at_routine = comparison["vertices_closer_than_the_routine_reference"]
    at_larger = comparison["vertices_closer_than_the_larger_reference"]
    assert at_routine == 1 and at_larger == 5, comparison
    assert comparison["fraction_closer_than_the_larger_reference"] == pytest.approx(
        5 / 6
    )

    payload = json.dumps(comparison)
    for phrase in ("up to", "error bar", "at most", "no more than"):
        assert phrase not in payload, phrase
    # "bound" may appear, but only in a denial. Every occurrence has to be negated,
    # because the phrase this replaces -- "the figure to size an error bar from" --
    # would otherwise be reachable again by rewording.
    # "bound" may appear, but only inside a denial. Checked with a negator window
    # rather than a forbidden substring, because the sentence this replaces -- "the
    # figure to size an error bar from" -- would otherwise be reachable again by
    # rewording, while the honest "not a bound on what can occur" has to stay legal.
    for found in re.finditer(r"bound(?:s|ed)?\b", payload, re.IGNORECASE):
        window = payload[max(0, found.start() - 36) : found.start()].lower()
        assert re.search(r"\b(not|no|neither|never|nothing)\b", window), (
            "an occurrence of 'bound' with no negator in front of it: "
            + payload[max(0, found.start() - 90) : found.end() + 40]
        )


def test_the_field_says_a_displacement_is_not_a_tracked_vertex() -> None:
    """The precondition both the verdict and the bound reason over, stated.

    A rerun calls Poisson again and re-derives its density threshold from that run's own
    density array, so the vertex set is regenerated rather than perturbed; "how far a
    vertex moved" has no correspondence to be defined over. Three reconstructs of one
    fused cloud at voxel 0.05 produced identical removal counts here, so the decisions
    were stable -- but stability observed is not a correspondence, and the categorical
    reading of the counts needs the assumption declared rather than assumed.
    """
    comparison = _crop_margin(np.array([0.08, 0.051]), 0.05, VOXEL, DEPTH)[
        "reference_comparison"
    ]
    said = comparison["what_this_does_not_establish"]

    assert "regenerated rather than perturbed" in said
    assert "no vertex can be followed between runs" in said
    assert "nearest-point distances between two" in said
    assert "not the motion of tracked vertices" in said
    # And the failure mode no count can bound at all.
    assert "introduce or drop vertices" in said
    # Scene identity is the other thing nothing here checks.
    assert "Scene identity is also not checked" in said


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
        margin = _crop_margin(distances, limit, VOXEL, DEPTH)
        assert margin["applies"] is False
        why = margin["why_not"]
        assert "no vertices to measure" in why, why
        assert "disabled" not in why, (
            "an enabled crop reported as disabled is the mislabel this covers"
        )
        assert f"{limit:g} threshold" in why, why
        assert f"{limit:g} m threshold" not in why, (
            "the threshold is voxel * factor in the caller's own coordinates, so "
            "stamping metres on it asserts a unit this tool never learns"
        )

    disabled = _crop_margin(np.array([0.08]), 0.0, VOXEL, DEPTH)
    assert disabled["applies"] is False
    assert "disabled" in disabled["why_not"]


def test_no_distance_measured_from_the_input_is_labelled_metres() -> None:
    """The caller's coordinates are whatever the caller's scanner produced.

    `--voxel-size` is in the scan's own units, the support threshold is
    `voxel * support_distance_factor`, and the distances are from the caller's cloud.
    None of the three is known to be metres, and this tool never inspects them. The
    field used to publish `threshold_m`, `closest_vertex_to_the_threshold_m` and a
    `{limit:g} m threshold` message, which stamps metres on a scan that may be in
    millimetres -- and then compared those numbers against fixed metre-valued
    references as though the comparison were calibrated.

    The two reference figures keep their unit, because theirs is known: they were
    measured on the demo fragments. What has to be explicit is that pairing them with
    the caller's numbers assumes something nobody checked.
    """
    margin = _crop_margin(np.array([0.08, 0.09]), 0.05, VOXEL, DEPTH)

    for key in margin:
        assert not key.endswith("_m"), key
    assert "coordinate units" in margin["distance_units"]
    assert "not inspected or validated" in margin["distance_units"]

    comparison = margin["reference_comparison"]
    for key in comparison:
        assert not key.endswith("_m"), key
    # The references are metres and say so, in their own sub-object rather than by
    # suffixing a key whose value came from the caller.
    assert comparison["reference_displacements"]["units"].startswith("metres")
    precondition = comparison["unit_precondition"]
    assert "own coordinate units" in precondition
    assert "only meaningful if this run's coordinates are metres" in precondition
    assert "ignore the comparison" in precondition

    # And an empty-crop message must not stamp a unit either.
    why = _crop_margin(np.array([]), 0.05, VOXEL, DEPTH)["why_not"]
    assert "m threshold" not in why, why
