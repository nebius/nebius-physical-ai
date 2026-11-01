"""Hole and outlier figures, the controls behind them, and the test that was discarded.

These read committed evidence; the meshes are retained privately and the harness needs
the real bundle. What is testable here is that each figure is reported with the control
that makes it readable, that the outer patch edge is not counted as a hole, and that
the trade the hole count exposes stays in the record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
REPORT = EVIDENCE / "holes-and-outliers.json"
RECORD = EVIDENCE / "capability-record-holes-and-outliers.json"


def _report() -> dict:
    return json.loads(REPORT.read_text())


def _record() -> dict:
    return json.loads(RECORD.read_text())


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


def test_the_report_does_not_overclaim_on_either_figure() -> None:
    """Holes are not called defects and outliers are not called noise.

    Neither is decidable without ground truth, which nothing in this chain carries.
    """
    caveat = _report()["what_this_does_not_claim"]
    assert "ground truth" in caveat, caveat
    assert "not automatically a defect" in caveat, caveat
    assert "real thin structure" in caveat, caveat


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
