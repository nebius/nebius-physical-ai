"""Criterion 5 closed with "no change" needs guarding more carefully than a change would.

"Nothing moved" is the easiest conclusion in the world to produce by accident: a measure that
silently reads the wrong field, a row set that quietly excludes the interesting case, or a
threshold comparison against a value nobody recomputed all produce it. So these tests
recompute the verdict from the published readings rather than reading the verdict field, and
they check the two things that would make a true "no change" misleading -- that the rows which
*could* have moved were the ones measured, and that the stricter measure was actually stricter
on them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
RESCORE = EVIDENCE / "band-validity-rescore.json"
SWEEP = EVIDENCE / "band-validity-sweep.json"
RECORD = EVIDENCE / "capability-record-band-validity-rescore.json"

FALSE_POSITIVE_THRESHOLD = 0.05


@pytest.fixture(scope="module")
def rescore() -> dict:
    return json.loads(RESCORE.read_text())


@pytest.fixture(scope="module")
def sweep_rows() -> list[dict]:
    return json.loads(SWEEP.read_text())["rows"]


def test_the_committed_evidence_was_not_overwritten(sweep_rows: list[dict]) -> None:
    """The whole point of the scope review set. These are the values as accepted."""
    fabricated = [row["fabricated_area_fraction"] for row in sweep_rows]
    assert len(fabricated) == 8
    assert fabricated[0] == 0.0
    assert fabricated[1] == pytest.approx(0.0027378386368625777, abs=1e-18)
    assert sum(value >= FALSE_POSITIVE_THRESHOLD for value in fabricated) == 6


def test_exactly_the_rows_below_the_threshold_were_measured(
    rescore: dict, sweep_rows: list[dict]
) -> None:
    """A "no change" result is worthless if the rows that could change were skipped.

    Derived from the committed values rather than from the harness's own row list, so the two
    have to agree.
    """
    could_move = {
        index
        for index, row in enumerate(sweep_rows)
        if row["fabricated_area_fraction"] < FALSE_POSITIVE_THRESHOLD
    }
    measured = {row["row"] for row in rescore["measured_rows"]}
    assert measured == could_move == {0, 1}


def test_the_row_review_called_immune_was_measured_anyway(rescore: dict) -> None:
    """Exact zero is where the unsigned measure is most lenient, not least.

    #1006's two-box bridge read 0.000000 unsigned and 0.004602 under normal correspondence.
    A row at exactly zero therefore has to be measured rather than reasoned away, and this
    holds the record to recording that the reasoning did not change the outcome.
    """
    row = next(r for r in rescore["measured_rows"] if r["row"] == 0)
    assert row["committed_unsigned"] == 0.0
    assert row["largest_stricter_reading"] > 0.0, (
        "the stricter measure lifts this row off zero; if it no longer does, the reason for "
        "measuring it has changed and the record should say so"
    )
    assert row["largest_stricter_reading"] < FALSE_POSITIVE_THRESHOLD


def test_the_stricter_measure_is_never_below_the_unsigned_one(rescore: dict) -> None:
    """The monotonicity the six-row argument rests on, checked against the readings.

    `far | misoriented` contains `far`, so this cannot fail unless the implementation stopped
    computing a superset -- which is exactly the regression worth catching, because it would
    silently invalidate the argument that six rows need no measurement.
    """
    for row in rescore["measured_rows"]:
        assert all(delta >= 0.0 for delta in row["delta_within_each_run"]), row
        fixed = row["one_cloud_reconstructed_three_times"]
        for strict, unsigned in zip(fixed["stricter"], fixed["unsigned"]):
            assert strict >= unsigned, (row["row"], strict, unsigned)


def test_the_stricter_measure_actually_did_something(rescore: dict) -> None:
    """ "No verdict moves" from a measure that changed nothing at all is not the same finding.

    If the stricter reading equalled the unsigned one everywhere, the result would be that the
    instrument is inert on this geometry, not that the conclusion survives it.
    """
    lifted = [
        row
        for row in rescore["measured_rows"]
        if max(row["delta_within_each_run"]) > 0.0
    ]
    assert len(lifted) == len(rescore["measured_rows"]), (
        "the measure was inert somewhere"
    )


def test_the_verdict_recomputes_from_the_readings(rescore: dict) -> None:
    """Recomputed rather than read, because a report asserting its own verdict proves nothing."""
    crossings = [
        row["row"]
        for row in rescore["measured_rows"]
        if row["largest_stricter_reading"] >= FALSE_POSITIVE_THRESHOLD
    ]
    assert crossings == [], crossings
    assert rescore["verdict"]["any_verdict_moves"] is False
    for row in rescore["measured_rows"]:
        expected = row["largest_stricter_reading"] >= FALSE_POSITIVE_THRESHOLD
        assert row["crosses_the_threshold"] is expected, row["row"]


def test_the_two_sources_of_variation_are_reported_separately(rescore: dict) -> None:
    """Because conflating them is the mistake this harness was corrected for.

    The three-fresh-draws spread bounds sampling and reconstruction together; the one-cloud
    spread isolates reconstruction. On row 1 the first is far larger, which is the finding.
    """
    row = next(r for r in rescore["measured_rows"] if r["row"] == 1)
    combined = row["unsigned_this_run"]["spread"]
    reconstruction_only = row["one_cloud_reconstructed_three_times"]["unsigned_spread"]
    assert reconstruction_only < combined, (reconstruction_only, combined)
    assert combined / max(reconstruction_only, 1e-12) > 10, (
        "sampling no longer dominates reconstruction here, so the attribution in the record "
        "is stale"
    )
    note = rescore["what_the_spreads_mean"]
    assert "advances the global RNG" in note["a_correction_to_my_own_first_write_up"]
    assert "2.17e-3" in note["relation_to_the_bound_published_in_1004"]


def test_the_record_does_not_present_a_safe_row_as_a_catch() -> None:
    """The row I added to review's scope lifted, and lifted nowhere near the threshold.

    Saying so plainly is the difference between reporting a measurement and claiming a save.
    """
    record = json.loads(RECORD.read_text())
    section = record["which_rows_could_move"]
    assert "did_not_change_the_outcome" in " ".join(section)
    assert "safe anyway" in section["the_correction_did_not_change_the_outcome"]


def test_the_wider_spread_seen_across_runs_is_disclosed() -> None:
    """Three repeats understate row 1's instability, and the record has to say so.

    An earlier run drew a sample set giving 1.183e-05 where this run's three draws gave
    2.75e-03 to 3.01e-03. Presenting the three-run spread as the bound would be the kind of
    understatement that looks like rigour.
    """
    record = json.loads(RECORD.read_text())
    caveat = record["an_honest_caveat_on_how_much_data_bounds_this"]
    assert "1.183e-05" in caveat["what_happened"]
    assert "should not take the three-run spread as the bound" in caveat["so"]


def test_the_intermittent_crash_is_recorded_rather_than_hidden() -> None:
    """One run died on a core-dumping signal and the identical rerun passed.

    A harness that segfaults intermittently is a fact about reproducing this evidence, so it
    belongs in the record even though the published run is clean.
    """
    record = json.loads(RECORD.read_text())
    assert "core" in record["one_run_dumped_core"].lower()
    assert "rerun" in record["one_run_dumped_core"]


def test_the_separate_open_question_is_not_claimed_as_settled() -> None:
    """The weakened-reading question from #1004 is a different question and still open."""
    record = json.loads(RECORD.read_text())
    note = record["what_this_does_not_settle"]
    assert "weakened" in note
    assert "review" in note and "root" in note
