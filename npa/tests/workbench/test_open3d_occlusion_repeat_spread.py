"""Guards for the measured run-to-run spread, and for how such a figure may be stated.

This record exists because a reviewer falsified a number in `audit-view-occlusion.json`.
That record declared the visible fractions "agree to 2e-5". The figure came from two runs,
and the reviewer's own two runs exceeded it before their numbers were compared against the
record at all.

The interesting test here is the last one. The specific wrong figure is easy to correct;
what stops the next one is requiring a declared reproducibility figure to carry the number
of observations behind it, and to match what those observations actually show. I made this
same n=2 mistake twice in one day -- once here, shipped, and once on the Poisson
reproducibility record, caught before publishing. A guard beats remembering.

Reads records only. No Open3D, so this runs wherever the suite runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs" / "evidence" / "open3d"
SPREAD = EVIDENCE / "occlusion-repeat-spread.json"
AUDIT = EVIDENCE / "audit-view-occlusion.json"
HARNESS = EVIDENCE / "harness" / "occlusion_repeat_spread.py"

#: The figure the audit record used to declare, from two runs.
WITHDRAWN_FIGURE = 2e-5

#: The gate the occlusion verdict rests on.
GATE = 0.5


@pytest.fixture
def spread() -> dict:
    return json.loads(SPREAD.read_text())


@pytest.fixture
def audit() -> dict:
    return json.loads(AUDIT.read_text())


def test_the_replaced_figure_is_named_and_exceeded(spread) -> None:
    """The record has to say what it replaces, and show that it was exceeded.

    A correction that quietly swaps one constant for another leaves no trace that the
    first one was wrong, and the next reader has no reason to distrust the new one.
    """
    observed = spread["observed_deviation"]
    assert observed["the_figure_this_replaces"] == WITHDRAWN_FIGURE
    assert observed["largest_either_way"] > WITHDRAWN_FIGURE, (
        "the replacement figure does not exceed the one it replaces, which would mean "
        "there was nothing to correct"
    )
    assert observed["ratio_to_the_replaced_figure"] == pytest.approx(
        observed["largest_either_way"] / WITHDRAWN_FIGURE, rel=1e-9
    )


def test_the_published_deviation_is_the_largest_one_recorded(spread) -> None:
    """Derived, not asserted: the headline must be the maximum over the stored rows.

    This is the mechanism that makes the figure unable to drift from its observations.
    The rows are in the same file, so a reader can do this by hand.
    """
    observed = spread["observed_deviation"]
    between, from_committed = [], []
    for scene in spread["scenes"].values():
        for row in scene["fractions"].values():
            between.append(row["largest_deviation_between_runs"])
            from_committed.append(row["largest_deviation_from_committed"])
            assert row["span"] == pytest.approx(row["max"] - row["min"], abs=1e-15), row
            assert len(row["values"]) == spread["repeats"]

    assert observed["largest_between_runs"] == pytest.approx(max(between), abs=1e-15)
    assert observed["largest_from_the_committed_record"] == pytest.approx(
        max(from_committed), abs=1e-15
    )
    assert observed["largest_either_way"] == pytest.approx(
        max(max(between), max(from_committed)), abs=1e-15
    )


def test_the_within_runs_deviation_alone_exceeds_the_old_figure(spread) -> None:
    """The reviewer's actual argument, pinned.

    Their point was not that their environment disagreed with mine. It was that their own
    two runs exceeded the published figure before my record was involved, which is what
    makes it a defect in the figure rather than a difference between machines.
    """
    assert spread["observed_deviation"]["largest_between_runs"] > WITHDRAWN_FIGURE


def test_it_is_not_presented_as_a_bound(spread) -> None:
    """A sample maximum is not a bound, and this record must not read like one."""
    assert spread["repeats"] >= 6
    assert "not a bound" in spread["observed_deviation"]["how_to_read_it"]
    assert any(
        "a seventh run may exceed it" in item
        for item in spread["what_this_does_not_claim"]
    ), spread["what_this_does_not_claim"]


def test_the_moving_sample_count_is_explained_rather_than_absorbed(spread) -> None:
    """A reviewer's measurement was right and their inference was not; both are recorded.

    They measured the barycentric sample count moving between runs and read it as the
    sampling seed failing to bind. `samples` is exactly `6 * removed_triangles` in every
    run and in the committed record, so it moves because the triangle count moves. The
    relation is the evidence, so it is asserted here rather than described.
    """
    note = spread["the_sample_count_moves_and_it_is_not_the_seed"]
    assert note["holds_in_every_record_checked"] is True
    for name, scene in spread["scenes"].items():
        assert (
            scene["samples_equals_samples_per_triangle_times_removed_triangles"] is True
        ), name
        samples = scene["counts"]["samples"]
        triangles = scene["counts"]["removed_triangles"]
        # The counts do move -- if they did not, there would be nothing to explain.
        assert samples["span"] > 0, name
        assert samples["span"] == 6 * triangles["span"], name


def test_the_controls_are_still_what_locates_the_variation(spread) -> None:
    """Carried-forward claims are the ones that rot, so this one was re-measured."""
    controls = spread["the_controls_still_do_not_move"]
    assert controls["numeric_fields_checked"] > 0
    assert controls["all_identical_across_runs_and_equal_to_committed"] is True
    assert controls["fields_that_moved"] == []


def test_the_verdict_still_clears_the_gate_by_orders_of_magnitude(spread) -> None:
    """If a reproducibility finding ever did threaten the verdict, that is a new PR."""
    unaffected = spread["does_this_change_the_verdict"]
    assert unaffected["gate"] == GATE
    observed = unaffected["observed_deviation"]
    assert observed == pytest.approx(
        spread["observed_deviation"]["largest_either_way"], abs=1e-15
    )
    # The smallest measured margin is the armadillo at 0.818 against the 0.5 gate.
    assert (0.818 - GATE) / observed > 1000.0


def test_the_audit_record_no_longer_declares_the_falsified_figure(audit) -> None:
    """The guard that outlives this correction.

    A declared reproducibility figure has to carry the number of observations behind it,
    and has to be the figure those observations show. The old text said "agree to 2e-5"
    with no run count, which is how a two-run sample minimum survived as a bound until
    someone ran it a third time.

    Scoped to this claim rather than written as a rule about prose: it checks that the
    withdrawn figure is gone, that a run count is present, and that the number quoted is
    the one the spread record measured.
    """
    note = None

    def walk(node: object) -> None:
        nonlocal note
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "and_still_not_bit_exact":
                    note = value
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(audit)
    assert note is not None, "the reproducibility note is gone from the audit record"

    record = json.loads(SPREAD.read_text())
    measured = record["observed_deviation"]
    assert f"{measured['largest_either_way']:.1e}" in note, note
    assert f"across {record['repeats']} runs" in note, note
    assert "agree to 2e-5" not in note, note
    assert "occlusion-repeat-spread.json" in note, note


def test_the_harness_and_record_are_committed_together(spread) -> None:
    assert HARNESS.exists()
    assert spread["environment"]["open3d"].startswith("0.20")
    assert "NPA_OCCLUSION_OUT" in spread["how_the_repeats_were_isolated"]
