"""Hole and outlier figures, the controls behind them, and the test that was discarded.

These read committed evidence; the meshes are retained privately and the harness needs
the real bundle. What is testable here is that each figure is reported with the control
that makes it readable, that the outer patch edge is not counted as a hole, and that
the trade the hole count exposes stays in the record.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
HARNESS = EVIDENCE / "harness"

# The digest of this report's scalar projection, pinned so the bound compares values
# rather than their number. Its exact construction is in the test that asserts it.
PROSE_ONLY_EDIT_PROJECTION = (
    "e149364291607ba750082d70ed0a9df185a077b73ded1872f330b6718c3c063f"
)
REPORT = EVIDENCE / "holes-and-outliers.json"
RECORD = EVIDENCE / "capability-record-holes-and-outliers.json"


def _report() -> dict:
    return json.loads(REPORT.read_text())


def _record() -> dict:
    return json.loads(RECORD.read_text())


def scalar_leaves(node: object, path: str = "") -> list[tuple[str, object]]:
    """Every leaf that is not a string, with the path that reaches it.

    Not "numeric": one of the fifty is a boolean, so the set is 49 numbers and one flag. A
    reviewer pointed out that the non-string reading is the stronger bound, because a flipped
    flag falls inside it, and the stronger one is the one to state.

    Module level rather than nested, so the digest check and the check that the report's
    *description* of this construction matches it share one definition. A reviewer could not
    reproduce the digest from that description because the path spelling was missing from it.
    """
    if isinstance(node, dict):
        return [
            leaf
            for key, value in node.items()
            for leaf in scalar_leaves(value, f"{path}/{key}")
        ]
    if isinstance(node, list):
        return [
            leaf
            for index, value in enumerate(node)
            for leaf in scalar_leaves(value, f"{path}[{index}]")
        ]
    return [] if isinstance(node, str) else [(path, node)]


def test_the_hole_detector_passes_its_sphere_control() -> None:
    """Zero loops intact, one per removed cap, area within 2% of analytic.

    Checked first because every hole number below is worthless without it. A detector
    that finds loops in a closed sphere is finding its own bug.
    """
    validation = _report()["detector_validation"]
    assert validation["loops_found_on_the_intact_closed_sphere"] == 0, validation
    assert validation["loops_found_with_one_cap_removed"] == 1, validation
    assert validation["loops_found_with_two_caps_removed"] == 2, validation
    assert validation["area_error_fraction"] < 0.02, validation
    assert validation["detector_is_valid"] is True, validation


def test_the_outer_patch_edge_is_not_counted_as_a_hole() -> None:
    """The largest loop spans the mesh and contains every other loop's box.

    This is the difference between reporting 0.96 m2 of hole and 6.17. Pinned with the
    evidence for the claim, not just the claim: bbox faces reached, and containment of
    all the other loops.
    """
    candidate = _report()["holes"]["candidate"]
    outer = candidate["outer_boundary"]

    assert outer["mesh_bbox_faces_it_reaches"] >= 4, outer
    held, total = outer["other_loops_whose_boxes_it_contains"].split("/")
    assert held == total, outer
    assert int(total) == candidate["interior_holes"], (
        outer,
        candidate["interior_holes"],
    )
    # The outer edge dominates, which is exactly why miscounting it would be dramatic.
    assert outer["area"] > candidate["interior_hole_area"], (outer, candidate)


def test_the_candidate_has_interior_holes_and_the_baseline_has_none() -> None:
    """The trade behind the variant this chain prefers, which was written down nowhere.

    The candidate cut phantom obstacle volume and opened interior holes doing it. A
    reader choosing between the variants needs both halves.
    """
    holes = _report()["holes"]
    assert holes["candidate"]["interior_holes"] == 13, holes["candidate"]
    assert holes["baseline"]["interior_holes"] == 0, holes["baseline"]
    assert 0.0 < holes["candidate"]["interior_hole_area"] < 1.5, holes["candidate"]
    assert holes["candidate"]["interior_hole_area_as_share_of_surface"] < 0.15, holes[
        "candidate"
    ]

    trade = _record()["holes"]["the_trade_this_makes_visible"]
    assert "0.871" in trade and "0.138" in trade, trade


def test_the_outlier_conclusion_rests_on_a_random_point_control() -> None:
    """ "6.6mm from the surface" means nothing until random points are 41x further.

    Without the control the figure is unreadable: it could mean the surface is
    everywhere. The control is what licenses the word "fitted".
    """
    outliers = _report()["outliers"]
    distances = outliers["distance_to_the_reconstructed_surface"]
    verdict = outliers["were_they_rejected_or_fitted"]

    assert verdict["answer"] == "fitted", verdict
    # Random control must be far; retained points must be nearest; outliers in between.
    assert (
        distances["retained_points_median"]
        < distances["outlier_points_median"]
        < distances["random_points_in_the_same_bbox_median"]
    ), distances
    assert verdict["outliers_are_this_much_closer_than_random"] > 10, verdict
    assert verdict["and_this_much_further_than_retained_points"] < 5, verdict


def test_the_outlier_count_is_reported_as_a_swept_parameter_not_a_fact() -> None:
    """Counts move by more than 10x across standard settings, so the sweep ships.

    The conclusion is about the ratio to the random control, which is why it survives
    the parameter choice. Pinned so a single flattering setting cannot replace it.
    """
    sweep = _report()["outliers"][
        "the_count_is_a_parameter_choice_so_here_is_the_sweep"
    ]
    assert len(sweep) >= 4, sweep
    counts = [entry["outliers"] for entry in sweep.values()]
    assert max(counts) / min(counts) > 10, sweep
    # The reported headline must be one of the swept settings, not a separate choice.
    assert "nb=20,std=2.0" in sweep, sweep
    assert sweep["nb=20,std=2.0"]["outliers"] == _report()["outliers"]["outliers"]


def test_the_discarded_attribution_test_is_recorded_with_both_rates() -> None:
    """It would have read as a 91.5% causal finding. Its control says it separates nothing.

    Kept in the record because a discarded instrument is evidence about the method, and
    because it is why the fitting result is stated as correlational.
    """
    discarded = _report()["a_test_i_built_and_discarded"]
    assert discarded["verdict"] == "discarded", discarded
    # The control rate must be at least as high as the outlier rate; that is the reason.
    assert (
        discarded["rate_for_retained_points_as_a_control"]
        >= discarded["rate_for_outliers"]
    )
    assert "separates nothing" in discarded["why"], discarded
    assert "re-reconstruction" in discarded["why"], discarded


def test_both_arms_of_the_discarded_test_ask_the_same_question() -> None:
    """One raised asymmetry was retracted with a proof; the other one was real.

    Review flagged the control's rank-based self-exclusion as an upward bias, in the
    direction that props up the discard, then retracted it four minutes later with an
    exhaustive two-case proof of equivalence. The sampling asymmetry in the same code was
    real and was not retracted. The record has to keep those straight, because this test is
    what protects the conclusion in CI.
    """
    asymmetry = _report()["a_test_i_built_and_discarded"][
        "an_asymmetry_review_found_in_this_discarded_test"
    ]
    # The retraction, and that the identity change is not claimed as a bug fix.
    assert "retracted" in asymmetry["their_finding_and_their_retraction_of_it"], (
        asymmetry
    )
    not_a_fix = asymmetry["what_i_changed_and_what_i_am_not_claiming"]
    assert "not a correctness fix" in not_a_fix, not_a_fix
    assert "not claiming it fixed a defect" in not_a_fix, not_a_fix

    assert "5.9%" in asymmetry["their_premise_measured"], asymmetry
    assert asymmetry["its_effect_on_the_rate"].startswith("Nil"), asymmetry
    # The sampling asymmetry was the real one, and it is recorded as such.
    assert "every seventh" in asymmetry["what_was_actually_asymmetric"], asymmetry
    assert "vary one thing" in asymmetry["a_mistake_i_nearly_made_checking_this"], (
        asymmetry
    )
    assert asymmetry["does_the_discard_still_hold"].startswith("Yes"), asymmetry


def test_the_report_does_not_overclaim_on_either_figure() -> None:
    """Holes are not called defects and outliers are not called noise.

    Neither is decidable without ground truth, which nothing in this chain carries.
    """
    caveat = _report()["what_this_does_not_claim"]
    assert "ground truth" in caveat, caveat
    assert "not automatically a defect" in caveat, caveat
    assert "real thin structure" in caveat, caveat


def _harness():
    """Import the harness with Open3D stubbed, so this runs where the image is not.

    Everything at the harness's module scope is imports and two env-derived paths, so
    a stub is enough to reach `publication`. The alternative was asserting against the
    harness's source text, which would pass on a file that reads correctly and runs
    wrong.
    """
    import importlib.util
    import sys
    import types

    path = EVIDENCE / "harness" / "holes_and_outliers.py"
    spec = importlib.util.spec_from_file_location("_holes_harness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    stub = types.ModuleType("open3d")
    stub.utility = types.SimpleNamespace()  # type: ignore[attr-defined]
    saved = sys.modules.get("open3d")
    sys.modules["open3d"] = stub
    try:
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            del sys.modules["open3d"]
        else:
            sys.modules["open3d"] = saved
    return module


def test_a_failed_detector_does_not_reach_the_canonical_report_path() -> None:
    """The printed message has to be true at the moment it prints.

    It said "no hole counts are published" one line after writing them to the
    canonical path. The exit code was right and the record would have carried
    `detector_is_valid: false`, so a careful reader was protected -- but the path is
    what the other harnesses and this suite read, and a reader who trusts it would
    have found counts there.
    """
    harness = _harness()
    canonical = Path("/does-not-matter/holes-and-outliers.json")

    valid = {"detector_validation": {"detector_is_valid": True}}
    assert harness.publication(valid, canonical) == (canonical, True)

    invalid = {"detector_validation": {"detector_is_valid": False}}
    destination, published = harness.publication(invalid, canonical)
    assert published is False
    assert destination != canonical, (
        "a run whose detector failed its control must not land on the path every "
        "other reader treats as the published measurement"
    )
    assert destination.parent == canonical.parent, destination
    assert "invalid" in destination.name, destination


def test_the_record_binds_its_harness_and_report_bytes() -> None:
    """Same hash-binding contract the rest of this evidence directory uses."""
    bound = _record()["provenance"]["harness_that_produced_this"]
    assert len(bound) == 2, bound
    for entry in bound:
        target = EVIDENCE / entry["path"]
        assert target.exists(), entry
        assert hashlib.sha256(target.read_bytes()).hexdigest() == entry["sha256"], (
            entry["path"]
        )


def test_the_sample_asymmetry_figure_is_checkable_from_rates_in_the_record() -> None:
    """A reader must be able to recompute it. The previous figure could not be.

    I published "about 1.3 points" for the sample asymmetry. Review did the arithmetic on the
    committed bytes and found no pairing of the numbers in the record that yields it, because it
    came from an isolation run whose intermediate rates were never shipped. The replacement names
    both rates, so this test derives the claim instead of matching a literal.
    """
    report = json.loads((EVIDENCE / "holes-and-outliers.json").read_text())
    discarded = report["a_test_i_built_and_discarded"]
    asymmetry = discarded["an_asymmetry_review_found_in_this_discarded_test"]
    worth = asymmetry["what_the_sample_asymmetry_was_worth"]

    rates = [float(f"0.{part}") for part in re.findall(r"0\.(\d{3})", worth)]
    assert len(rates) == 2, worth
    points = round(abs(rates[1] - rates[0]) * 100, 1)
    assert f"{points} points" in worth, (points, worth)

    # The head rate it cites has to be the one the report actually publishes.
    assert discarded["rate_for_retained_points_as_a_control"] in rates, (
        rates,
        discarded,
    )
    # And the withdrawn figure must be named as withdrawn rather than quietly dropped.
    assert "1.3" in worth and "withdrawn" in worth, worth


def test_the_sample_finding_is_not_attributed_to_review() -> None:
    """Review raised the rank rule and never the sample; crediting them would be wrong.

    The earlier wording said the asymmetry "was real, review did not retract it", which is
    literally true and reads as though they had raised both and withdrawn one.
    """
    report = json.loads((EVIDENCE / "holes-and-outliers.json").read_text())
    asymmetry = report["a_test_i_built_and_discarded"][
        "an_asymmetry_review_found_in_this_discarded_test"
    ]

    claimed = asymmetry["what_was_actually_asymmetric"]
    assert "mine rather than review's" in claimed, claimed
    assert "review did not retract it" not in claimed, claimed

    harness = (HARNESS / "holes_and_outliers.py").read_text()
    assert "I found a second asymmetry in the same edit" in harness
    assert "review did not retract it" not in harness


def test_correcting_the_asymmetry_widened_the_gap_it_was_checking() -> None:
    """The direction matters: the error made my own conclusion look weaker, not stronger."""
    report = json.loads((EVIDENCE / "holes-and-outliers.json").read_text())
    discarded = report["a_test_i_built_and_discarded"]
    asymmetry = discarded["an_asymmetry_review_found_in_this_discarded_test"]
    direction = asymmetry["the_correction_strengthened_the_conclusion_it_was_checking"]

    gaps = [float(g) for g in re.findall(r"0\.0\d\d", direction)]
    assert len(gaps) == 2 and gaps[1] > gaps[0], direction
    published = round(
        discarded["rate_for_retained_points_as_a_control"]
        - discarded["rate_for_outliers"],
        3,
    )
    assert published == gaps[1], (published, gaps)


def test_the_hand_edit_declares_itself_and_names_why_it_was_not_regenerated() -> None:
    """These prose corrections could not be made by re-running the harness.

    The input bundle lived on an ephemeral volume the host reboot removed. A hand-edited report
    is only trustworthy if it says so and if the edit is bounded, so the bound is stated as a
    numeric-invariance check and the count is asserted against the file.
    """
    path = EVIDENCE / "holes-and-outliers.json"
    report = json.loads(path.read_text())
    note = report["a_test_i_built_and_discarded"][
        "an_asymmetry_review_found_in_this_discarded_test"
    ]["why_this_report_was_edited_rather_than_regenerated"]

    assert "could not be re-run" in note, note

    leaves = sorted(scalar_leaves(report))
    assert f"all {len(leaves)} scalar non-string leaves" in note, note
    assert sum(1 for _, value in leaves if isinstance(value, bool)) == 1, leaves
    assert "49 numbers and one flag" in note, note

    # The count alone is invariant to exactly the change it is supposed to detect: alter
    # any one of the fifty values and it is still fifty, so the sentence still matches and
    # the test still passes. A reviewer caught that, in the one check whose job is to
    # bound a hand-edit. The digest compares values; a count never can.
    projection = hashlib.sha256(json.dumps(leaves, sort_keys=True).encode()).hexdigest()
    assert projection == PROSE_ONLY_EDIT_PROJECTION, projection
    assert PROSE_ONLY_EDIT_PROJECTION in note, note


def test_the_published_construction_spells_out_how_a_path_is_built() -> None:
    """A reviewer reproduced everything except the path spelling, and got a different digest.

    Working only from the construction as first written -- sorted `(path, value)` over every
    non-string leaf, `json.dumps`, `sort_keys=True` -- they computed `262f0cad...`. The
    projection, the sort and the dump were all right; they had written dot-separated paths
    with no leading separator. The digest is only checkable from its description once the
    description says which of those two spellings it means.

    This asserts the description against the construction the test above actually uses,
    rather than merely asserting that some sentence about paths exists.
    """
    report = json.loads((EVIDENCE / "holes-and-outliers.json").read_text())
    note = report["a_test_i_built_and_discarded"][
        "an_asymmetry_review_found_in_this_discarded_test"
    ]["why_this_report_was_edited_rather_than_regenerated"]

    assert "262f0cad" in note, (
        "the reviewer's value is not recorded as the near miss it was"
    )

    leaves = sorted(scalar_leaves(report))
    first_path = leaves[0][0]
    assert first_path in note, (first_path, "the note names a different first leaf")
    assert first_path.startswith("/"), first_path
    assert '"/" + key' in note, note
    assert '"[i]"' in note, note


def test_a_differently_spelled_path_really_does_change_the_digest() -> None:
    """So the clause above is load-bearing rather than decorative."""
    report = json.loads((EVIDENCE / "holes-and-outliers.json").read_text())

    def dotted(node: object, path: str = "") -> list[tuple[str, object]]:
        if isinstance(node, dict):
            return [
                leaf
                for key, value in node.items()
                for leaf in dotted(value, f"{path}.{key}" if path else key)
            ]
        if isinstance(node, list):
            return [
                leaf
                for index, value in enumerate(node)
                for leaf in dotted(value, f"{path}.{index}")
            ]
        return [] if isinstance(node, str) else [(path, node)]

    ours = sorted(scalar_leaves(report))
    theirs = sorted(dotted(report))

    assert len(ours) == len(theirs), (len(ours), len(theirs))
    assert [value for _, value in ours] == [value for _, value in theirs]

    mine = hashlib.sha256(json.dumps(ours, sort_keys=True).encode()).hexdigest()
    alternative = hashlib.sha256(
        json.dumps(theirs, sort_keys=True).encode()
    ).hexdigest()
    assert mine == PROSE_ONLY_EDIT_PROJECTION
    assert alternative != mine, (
        "path spelling would not matter and the clause is unnecessary"
    )
