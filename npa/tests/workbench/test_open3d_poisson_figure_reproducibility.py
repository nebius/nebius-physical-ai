"""Guards for the Poisson reproducibility finding.

The finding is a negative one: figures this lane published to sixteen significant digits
do not survive a second run of the harness that wrote them. These tests pin the finding,
the cause, and the two remedies that fail -- and one of them pins a mistake I made while
measuring it, because that mistake is the reason the record's wording is derived from the
numbers instead of written beside them.

Reads the record only. No Open3D, so this runs wherever the suite runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "docs" / "evidence" / "open3d" / "poisson-figure-reproducibility.json"
HARNESS = (
    REPO
    / "docs"
    / "evidence"
    / "open3d"
    / "harness"
    / "poisson_figure_reproducibility.py"
)

#: The threshold the audited record's conclusion rests on.
MINIMUM_VISIBLE_AREA = 0.5


@pytest.fixture
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_finding_is_that_no_condition_reproduces(record) -> None:
    """The negative result, stated for every condition rather than in general."""
    conditions = record["cause"]["conditions"]
    assert set(conditions) == {"default", "seeded_before_each_call", "single_threaded"}
    for name, row in conditions.items():
        assert row["repeats"] >= 4, f"{name} has too few repeats to say anything"
        assert row["surface_area"]["identical"] is False, (
            f"{name} reproduced exactly, which would retract the finding -- if that is "
            f"real the record needs rewriting, not this test relaxing"
        )
        assert row["surface_area"]["relative_spread"] > 0.0


def test_the_cause_is_positions_and_not_a_different_mesh(record) -> None:
    """Stable topology is what makes "same mesh, moved vertices" a measurement.

    Without it the finding would be consistent with Poisson producing a genuinely
    different reconstruction each time, which would be a far larger claim.
    """
    for name, row in record["cause"]["conditions"].items():
        assert row["topology_is_stable"] is True, name
        assert row["vertices"] > 0 and row["triangles"] > 0, name


def test_seeding_is_not_offered_as_a_remedy(record) -> None:
    """The seeded condition has to be no better, and the record has to say so.

    There is a real sampling seed in `audit_view_occlusion.py`, and it does bind what it
    claims to bind. The trap is extending that to Poisson, whose variation is not drawn
    from Open3D's generator at all.
    """
    conditions = record["cause"]["conditions"]
    assert conditions["seeded_before_each_call"]["seeded_before_each_call"] is True
    assert conditions["seeded_before_each_call"]["surface_area"]["identical"] is False
    assert "Does not help" in record["what_the_remedies_do"]["seeding"]


def test_the_single_threading_verdict_agrees_with_its_own_numbers(record) -> None:
    """This test exists because of a mistake, and the mistake is worth keeping visible.

    A first probe used two repeats and put single-threading two orders of magnitude ahead
    of the default. At four repeats it is within a small factor, and the spreads
    themselves move between runs. "Two orders of magnitude" was about to go into the
    record as a finding.

    So the verdict is now derived from the measured spreads rather than written next to
    them, and this checks the derivation still holds: the factor quoted in the prose has
    to be the factor the two conditions actually show.
    """
    conditions = record["cause"]["conditions"]
    default = conditions["default"]["surface_area"]["relative_spread"]
    single = conditions["single_threaded"]["surface_area"]["relative_spread"]
    verdict = record["what_the_remedies_do"]["single_threading"]

    assert "Does not fix it" in verdict
    assert f"{default:.2e}" in verdict, verdict
    assert f"{single:.2e}" in verdict, verdict
    assert f"factor of {default / single:.1f}" in verdict, verdict


def test_the_conclusion_is_not_in_danger_and_the_record_shows_the_arithmetic(
    record,
) -> None:
    """A reproducibility finding that did change a conclusion would be a different PR.

    The point of publishing the margin beside the drift is that a reader can see which
    kind of finding this is without taking my word for it.
    """
    unaffected = record["does_this_change_any_conclusion"]
    margin = unaffected["margin_against_the_threshold"]
    observed = unaffected["observed_difference"]

    published = record["the_published_figures"]["values"][
        "visible_with_only_its_own_self_occlusion"
    ]
    assert margin == pytest.approx(published - MINIMUM_VISIBLE_AREA, abs=1e-12)
    assert observed > 0.0
    assert unaffected["ratio"] == pytest.approx(margin / observed, rel=1e-9)
    # Three orders is the weakest form of "the drift cannot reach the threshold".
    assert unaffected["ratio"] > 1000.0, unaffected


def test_the_published_figures_are_never_pinned_by_exact_equality() -> None:
    """The guard that outlives the finding.

    A figure that does not reproduce must not be asserted with `==` anywhere in the
    suite. Such a test would pass on the machine that wrote the record and fail on the
    next one, and because it would look like a tightening rather than a mistake, the
    likely repair is someone regenerating the record to match -- which buries the finding
    this PR exists to record.

    Scoped to the two literals this record is about, so it stays a check on a known
    non-reproducible quantity rather than a style rule about floats.
    """
    literals = ("0.7848810005123376", "0.3521503113541809")
    offenders: list[str] = []
    for path in (REPO / "npa" / "tests").rglob("test_*.py"):
        text = path.read_text()
        for literal in literals:
            if literal not in text:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if literal in line and "==" in line and "approx" not in line:
                    offenders.append(
                        f"{path.relative_to(REPO)}:{number}: {line.strip()}"
                    )
    assert not offenders, (
        "a non-reproducible figure is pinned by exact equality:\n"
        + "\n".join(offenders)
    )


def test_the_harness_and_record_are_committed_together(record) -> None:
    """A record whose harness is absent cannot be re-measured by a reviewer."""
    assert HARNESS.exists()
    assert record["environment"]["open3d"].startswith("0.20")
    assert record["environment"]["input_points"] > 0
    assert record["environment"]["poisson_depth"] >= 8
