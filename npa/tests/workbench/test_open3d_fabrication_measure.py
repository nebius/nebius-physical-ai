"""The ground truth these measurements were scored against is lenient, and by how much.

`runner.py` records the leniency and leaves it open: fabrication was checked as unsigned
distance to the whole reference surface, so surface that corresponds to no part of the
reference can still be near some part of it and score as faithful. The note names two possible
replacements, a signed distance or a per-region correspondence, and says neither was
implemented.

`fabrication_measure_leniency.py` reproduces it and measures three candidates. These tests pin
what it found, from the committed report, because the harness needs Open3D and the sample
geometry and CI has neither.

The numbers here are the last four significant figures of a Poisson reconstruction, which is
intrinsically nondeterministic -- so they are pinned with tolerances, and the conclusions,
which are stable, are pinned exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
HARNESS = EVIDENCE / "harness"
REPORT = json.loads((EVIDENCE / "fabrication-measure-leniency.json").read_text())

#: Where a conclusion actually turns on this measure's value.
FALSE_POSITIVE_THRESHOLD = 0.05


def test_the_leniency_reproduces_at_all() -> None:
    """The first thing acceptance required, and it is not a given.

    If the current measure had caught the bridge, there would be nothing here to fix and the
    right answer would have been to say so.
    """
    assert REPORT["the_leniency"]["reproduces"] is True

    bridged = REPORT["bridged"]
    assert bridged["unsigned_distance_only"] == 0.0, bridged
    assert bridged["with_normal_correspondence"] > 0.0, bridged


def test_the_current_measure_reports_exactly_zero_where_surface_was_fabricated() -> (
    None
):
    """Not merely understated. Zero, on a mesh where Poisson demonstrably closed a gap."""
    assert REPORT["bridged"]["unsigned_distance_only"] == 0.0
    assert REPORT["geometry"]["gap_in_voxels"] == pytest.approx(1.0, abs=0.01)


def test_signed_distance_does_not_help_which_is_the_point_of_measuring_it() -> None:
    """The runner named signed distance first. On this failure mode it changes nothing.

    Not because nothing lies inside the reference -- in the comb cases some surface does. It
    changes nothing because everything inside is *also* more than a voxel away, so the inside
    set is a subset of the far set the unsigned measure already had. The obvious fix is
    redundant here rather than inapplicable, which is a stronger statement and not what I
    assumed before measuring it.
    """
    cases = [
        REPORT["faithful"],
        REPORT["bridged"],
        *REPORT["how_large_it_gets"].values(),
    ]

    for case in cases:
        assert case["signed_distance"] == case["unsigned_distance_only"], case
        assert case["area_inside_the_reference"] <= case["unsigned_distance_only"], case

    assert any(case["area_inside_the_reference"] > 0.0 for case in cases), (
        "if nothing is ever inside the reference, this is proving something weaker than it says"
    )


def test_normal_correspondence_catches_it_at_every_size() -> None:
    for name, row in sorted(REPORT["how_large_it_gets"].items()):
        assert row["with_normal_correspondence"] > row["unsigned_distance_only"], (
            name,
            row,
        )


def test_the_stricter_measure_does_not_simply_flag_good_surface_too() -> None:
    """A measure that catches the bridge by also condemning correct surface is no better.

    The control is a single box reconstructed the same way, where nothing is fabricated.
    Poisson rounds its sharp edges, which is exactly where a normal-based test would be
    expected to misfire, so this control is the one that matters.
    """
    faithful = REPORT["faithful"]

    assert faithful["unsigned_distance_only"] == 0.0
    assert faithful["with_normal_correspondence"] < 0.001, faithful
    assert (
        REPORT["bridged"]["with_normal_correspondence"]
        > faithful["with_normal_correspondence"] * 20
    ), (REPORT["bridged"], faithful)


def test_the_hidden_fabrication_grows_with_the_number_of_closed_gaps() -> None:
    rows = sorted(REPORT["how_large_it_gets"].values(), key=lambda r: r["gaps"])
    strict = [row["with_normal_correspondence"] for row in rows]

    assert strict == sorted(strict), strict
    assert max(strict) > 0.2, strict


def test_there_is_geometry_where_the_two_measures_reach_different_verdicts() -> None:
    """The finding that makes this more than a difference in decimal places.

    At these gap counts the current measure reads below the threshold that decides a false
    positive while the stricter one reads above it, so the two disagree about the conclusion
    and not just the number.
    """
    threat = REPORT["does_it_threaten_any_accepted_conclusion"]

    assert threat["the_threshold_that_matters"] == FALSE_POSITIVE_THRESHOLD
    assert threat["gap_counts_where_the_verdict_differs"] == [9, 11, 13]

    for row in REPORT["how_large_it_gets"].values():
        if row["gaps"] not in threat["gap_counts_where_the_verdict_differs"]:
            continue
        assert row["unsigned_distance_only"] < FALSE_POSITIVE_THRESHOLD, row
        assert row["with_normal_correspondence"] > FALSE_POSITIVE_THRESHOLD, row


def test_the_band_is_recomputed_from_the_rows_rather_than_asserted() -> None:
    """The published band has to follow from the published readings.

    A summary that states its own conclusion is the shape I have had to repair twice in this
    lane, so this derives it independently.
    """
    threat = REPORT["does_it_threaten_any_accepted_conclusion"]
    derived = sorted(
        row["gaps"]
        for row in REPORT["how_large_it_gets"].values()
        if (row["unsigned_distance_only"] < FALSE_POSITIVE_THRESHOLD)
        != (row["with_normal_correspondence"] < FALSE_POSITIVE_THRESHOLD)
    )

    assert derived == threat["gap_counts_where_the_verdict_differs"], derived
    assert threat["largest_hidden_share"] == pytest.approx(
        max(
            r["with_normal_correspondence"]
            for r in REPORT["how_large_it_gets"].values()
        )
    )


def test_the_record_says_which_way_the_bias_runs_and_does_not_just_repeat_the_note() -> (
    None
):
    """The runner says the bias is in the tool's favour. For this conclusion it is not.

    Understated fabrication lowers the measured floor, which flatters the tool. The same
    understatement satisfies `false_positive`'s `fabricated < 0.05` clause on rows where real
    fabrication exists, which blames the tool for readings that were right. Both directions
    belong in the record, because the note states only one.
    """
    stated = REPORT["does_it_threaten_any_accepted_conclusion"][
        "which_way_the_bias_runs_there"
    ]

    assert "cried wolf" in stated, stated
    assert "the other way" in stated, stated
    assert "floor" in stated, stated


def test_the_construction_that_failed_is_still_in_the_record() -> None:
    """The first attempt did not exercise the leniency, and deleting that would be dishonest.

    A torus with a disc over its hole was caught perfectly by the current measure, because the
    middle of such a disc is nowhere near the reference. That is a failed construction, not
    evidence the leniency is absent, and the distinction is the whole reason the second
    construction was built.
    """
    source = (HARNESS / "fabrication_measure_leniency.py").read_text()

    assert "torus" in source, "the abandoned first construction has been written out"
    assert "0.2576" in source, source[:0] or "the number it actually produced is gone"
    assert "fails to exercise an effect" in source


def test_the_report_names_the_seed_and_the_cutoff_it_used() -> None:
    assert REPORT["sampling_seed"] == 20261101
    assert REPORT["normal_agreement_cutoff_degrees"] == 60.0
    assert REPORT["faithful"]["median_normal_disagreement_degrees"] < 5.0, (
        "a permissive cutoff is only permissive if good surface sits far below it"
    )
