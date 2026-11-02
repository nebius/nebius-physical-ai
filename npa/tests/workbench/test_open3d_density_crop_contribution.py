"""`low_density_vertices_removed` described work the next stage was going to do.

`run_reconstruct` applies two filters in sequence: the density crop removes vertices
in the lowest quantile of Poisson's sample density, and the support crop then removes
vertices farther than `support_distance_factor` voxels from any observed point. The
report published the density crop's removal count, which reads as surface that filter
took out of the delivered mesh.

Measured on the upstream demo scan at the shipped defaults, in
`docs/evidence/open3d/density-crop-contribution.json`:

| configuration                       | result                                     |
| ----------------------------------- | ------------------------------------------ |
| defaults, with the density crop     | 518 vertices removed                       |
| of those, within the support distance | **0**                                    |
| delivered mesh without the crop     | identical, to the triangle and to 1e-6 m2  |
| control, support cropping disabled  | ~1025 triangles and ~6.72 m2 removed       |
| pushed to the 0.20 quantile         | removes supported surface                  |

The count was never wrong. Reading it as this filter's effect on the delivered
surface was, and nothing in the report distinguished the two. The run now publishes
how much of the removal was the density crop's own.

The control carries the weight here: with support cropping off the density crop is the
only filter and it visibly changes the mesh, so "it changed nothing" at the defaults
is overlap with the next filter rather than a measurement that cannot see a crop.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from npa.workbench.open3d.artifacts import (
    Open3dError,
    validate_density_crop_contribution,
)

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "evidence"
    / "open3d"
    / "density-crop-contribution.json"
)


def _record() -> dict:
    return json.loads(EVIDENCE.read_text())


def _report(**overrides: Any) -> dict:
    contribution = {
        "vertices_removed": 518,
        "removed_only_by_this_crop": 0,
        "support_crop_was_disabled": False,
        "support_distance": 0.05,
    }
    contribution.update(overrides.pop("contribution", {}))
    report = {
        "low_density_vertices_removed": 518,
        "density_crop_contribution": contribution,
    }
    report.update(overrides)
    return report


def test_a_report_that_states_its_contribution_validates() -> None:
    validate_density_crop_contribution(_report())


def test_a_report_predating_the_contribution_block_validates() -> None:
    validate_density_crop_contribution({"low_density_vertices_removed": 518})


@pytest.mark.parametrize("nothing", [{}, None], ids=["empty mapping", "null"])
def test_a_contribution_published_as_nothing_is_rejected(nothing) -> None:
    """Absent predates the block. Both of these published the measurement as nothing.

    `None` used to pass while `{}` was refused -- the less alarming shape guarded and
    the one the sentinel's own comment names let through. That collapsed the two
    facts the sentinel exists to separate, so a report could say it had measured the
    contribution and supply nothing.
    """

    with pytest.raises(Open3dError, match="non-empty mapping"):
        validate_density_crop_contribution(
            {"low_density_vertices_removed": 1, "density_crop_contribution": nothing}
        )


def test_a_report_without_the_block_is_the_only_shape_that_skips_the_check() -> None:
    """The sentinel's distinction, asserted rather than described."""

    validate_density_crop_contribution({"low_density_vertices_removed": 1})


@pytest.mark.parametrize(
    ("contribution", "expected"),
    [
        pytest.param(
            {"removed_only_by_this_crop": 600},
            "more exclusive removals than removals",
            id="exclusive_exceeds_the_total",
        ),
        pytest.param(
            {"vertices_removed": 400},
            "different number of removals",
            id="disagrees_with_the_published_count",
        ),
        pytest.param(
            {"support_crop_was_disabled": True},
            "every density removal is this crop's own",
            id="support_off_but_credited_with_none",
        ),
        pytest.param(
            {"support_crop_was_disabled": "no"},
            "whether support cropping was on",
            id="does_not_say_whether_support_ran",
        ),
        pytest.param(
            {"vertices_removed": -1},
            "must be a count",
            id="a_negative_removal_count",
        ),
        pytest.param(
            {"removed_only_by_this_crop": True},
            "must be a count",
            id="a_boolean_exclusive_count",
        ),
        pytest.param(
            {"removed_only_by_this_crop": None},
            "must be a count",
            id="a_measurement_that_was_not_taken",
        ),
    ],
)
def test_a_contribution_that_contradicts_its_report_is_rejected(
    contribution: dict, expected: str
) -> None:
    with pytest.raises(Open3dError, match=expected):
        validate_density_crop_contribution(_report(contribution=contribution))


def test_the_support_disabled_case_is_the_one_that_must_claim_everything() -> None:
    """With nothing to overlap with, every density removal is this crop's own.

    A report claiming a partial share with support cropping off is describing a
    pipeline that did not run, which is worth its own test because it is the only
    guard that depends on the two filters' relationship rather than on arithmetic.
    """

    validate_density_crop_contribution(
        _report(
            contribution={
                "removed_only_by_this_crop": 518,
                "support_crop_was_disabled": True,
            }
        )
    )
    with pytest.raises(Open3dError, match="every density removal"):
        validate_density_crop_contribution(
            _report(
                contribution={
                    "removed_only_by_this_crop": 517,
                    "support_crop_was_disabled": True,
                }
            )
        )


def test_the_harness_measured_the_defaults_the_product_actually_ships() -> None:
    """The record is worthless if it describes a configuration nobody runs.

    The harness restates the quantile and support factor as constants, so they are
    checked against the schema rather than trusted.
    """

    from npa.workbench.open3d.schemas import ReconstructRequest

    record = _record()
    shipped = record["at_the_shipped_defaults"]
    fields = ReconstructRequest.model_fields
    assert shipped["density_quantile"] == fields["density_quantile"].default, shipped
    assert (
        shipped["support_distance_factor"] == fields["support_distance_factor"].default
    ), shipped


def test_the_density_crop_removed_nothing_the_support_crop_would_have_kept() -> None:
    shipped = _record()["at_the_shipped_defaults"]
    assert shipped["density_removed_vertices"] > 0, (
        "the crop removed nothing at all, so this record is not measuring overlap"
    )
    assert shipped["of_those_within_the_support_distance"] == 0, shipped
    assert shipped["delivered_mesh_is_identical"] is True, shipped
    with_crop = shipped["with_the_density_crop"]
    without = shipped["without_the_density_crop"]
    assert with_crop["triangles"] == without["triangles"], (with_crop, without)
    assert with_crop["surface_area"] == without["surface_area"], (with_crop, without)


def test_the_control_proves_the_measurement_can_see_a_crop() -> None:
    """Without it, "the density crop changed nothing" has a second explanation."""

    control = _record()["control_support_cropping_disabled"]
    assert control["support_distance_factor"] == 0.0
    assert control["the_density_crop_changes_the_mesh_here"] is True, control
    assert control["triangles_removed"] > 0, control
    assert control["surface_area_removed"] > 0, control


def test_the_grid_records_where_the_crop_starts_doing_harm() -> None:
    """At a high quantile it removes surface the scan does support.

    Which is why the finding is not "raise the quantile until it contributes".
    """

    rows = _record()["grid"]
    assert len(rows) >= 9, rows
    # At the shipped support distance, not at any support distance. Two rows clear
    # 0.20 -- the shipped factor at about -785 triangles and a wider one near -8980 --
    # and an `any` over both let either carry the argument alone. The finding is about
    # the configuration the product ships, so the cost has to be shown there: the
    # shipped row could have gone to zero with this test still green. The sign is what
    # is asserted, not the count: the count moves by about 3 between runs and sits 783
    # from zero, which `how_stable_are_these_figures` in the record measures.
    from npa.workbench.open3d.schemas import ReconstructRequest

    shipped_support = ReconstructRequest.model_fields["support_distance_factor"].default
    supported = [
        row
        for row in rows
        if row["support_distance_factor"] == shipped_support
        and row["density_quantile"] >= 0.20
    ]
    assert supported, rows
    assert all(row["triangles_versus_no_density_crop"] < 0 for row in supported), (
        "the aggressive quantile no longer removes supported surface at the shipped "
        f"support distance, so the grid no longer shows the cost it exists to show: "
        f"{supported}"
    )
    at_defaults = [
        row
        for row in rows
        if row["support_distance_factor"] > 0.0 and row["density_quantile"] < 0.05
    ]
    assert at_defaults and all(
        row["triangles_versus_no_density_crop"] == 0 for row in at_defaults
    ), at_defaults


def test_the_record_shows_the_crop_is_selective_not_arbitrary() -> None:
    """The finding is about overlap, not about a filter that trims at random.

    Recorded because the headline invites the opposite reading: the removals are far
    from any observed point by a wide margin, which is exactly why they duplicate the
    support crop so completely.
    """

    selectivity = _record()["is_the_density_crop_selective"]
    assert selectivity["spearman_density_against_distance_to_cloud"] < -0.5, selectivity
    for row in selectivity["by_quantile"]:
        assert row["median_distance_removed"] > row["median_distance_kept"], row


def test_the_record_declines_the_claims_it_cannot_support() -> None:
    limits = _record()["what_this_does_not_claim"]
    assert set(limits) == {
        "not_that_the_density_crop_is_wrong",
        "not_that_it_should_be_removed",
        "not_a_general_result_about_every_scan",
    }
    assert (
        "removes extrapolated surface" in limits["not_that_the_density_crop_is_wrong"]
    )
    assert "separate decision" in limits["not_that_it_should_be_removed"]


def test_the_record_says_which_of_its_figures_move_between_runs() -> None:
    """A reader regenerating this must be able to tell noise from a regression.

    An independent lane reproduced every structural result here and found 31 figures
    differing in the fourth significant digit -- including between its own two runs,
    so this is the Poisson solve being unseeded rather than anything about a machine.
    The record measures its own spread instead of conceding it in prose.
    """

    stability = _record()["how_stable_are_these_figures"]
    assert stability["repeats"] >= 4, stability["repeats"]

    moved = stability["moved_between_repeats"]
    assert moved, "a stability block that found nothing moving measured nothing"
    for field, observed in moved.items():
        assert len(observed["values"]) == stability["repeats"], field
        assert observed["spread"] > 0, (
            f"{field} is listed as moving with a spread of zero, so either the repeats "
            "are not independent solves or this field does not belong here"
        )
        assert observed["relative_spread"] < 0.01, (field, observed)


def test_every_figure_the_argument_rests_on_is_the_same_in_every_repeat() -> None:
    """The partition is the point, not the disclosure.

    "Some numbers here move" would be true and useless. What a reader needs is which
    ones, and the four that carry this record's conclusion are not among them.
    """

    stability = _record()["how_stable_are_these_figures"]
    constant = stability["identical_in_every_repeat"]
    record = _record()
    shipped = record["at_the_shipped_defaults"]

    assert constant["density_removed_vertices"] == shipped["density_removed_vertices"]
    assert (
        constant["of_those_within_the_support_distance"]
        == shipped["of_those_within_the_support_distance"]
    )
    assert constant["delivered_mesh_is_identical"] is True
    assert constant["the_density_crop_changes_the_mesh_here"] is True
    # The control's verdict and the shipped verdict are opposite booleans, and both hold
    # in every repeat. That pair is the whole finding: the crop works, and at the
    # defaults its removals are the next filter's work.
    assert (
        record["control_support_cropping_disabled"][
            "the_density_crop_changes_the_mesh_here"
        ]
        is True
    )


def test_the_threshold_a_test_asserts_is_nowhere_near_the_noise() -> None:
    """The reason the moving figures are safe to publish.

    One grid figure has an assertion applied to it, and the honest question is not
    whether it wobbles but whether it could wobble across the line. It sits hundreds
    of units away from zero with a spread of a few, measured over the same repeats.
    """

    margin = _record()["how_stable_are_these_figures"]["the_margin_the_assertions_have"]
    assert margin["asserted"] == "that it is negative"
    assert all(one < 0 for one in margin["values"]), margin
    assert margin["nearest_any_repeat_came_to_zero"] > 20 * margin["spread"], margin
    assert margin["margin_in_units_of_the_observed_spread"] > 20, margin


def test_the_record_binds_the_harness_that_produced_it() -> None:
    provenance = _record()["provenance"]
    assert "this evidence directory" in provenance["path_is_relative_to"]
    harness = EVIDENCE.parent / provenance["harness"]
    assert harness.exists(), harness
    assert (
        hashlib.sha256(harness.read_bytes()).hexdigest() == provenance["harness_sha256"]
    ), "the harness changed without the record being regenerated"


def test_the_readme_quotes_the_records_own_figures() -> None:
    """The body-versus-record class, caught twice by a reader on this PR alone.

    Once was three figures disagreeing with the record; once was a spread written
    as 3 where the record's field is 2, because there are three distinct values and
    the spread is their range. Both were found by a person reading carefully, which
    is the wrong mechanism for a number that a regeneration can move.

    Only the figures the stability paragraph rests on, quoted exactly. A general
    prose parser would be guessing at which digits in the file are claims.
    """

    readme = (EVIDENCE.parent / "README.md").read_text()
    margin = _record()["how_stable_are_these_figures"]["the_margin_the_assertions_have"]
    paragraph = next(
        block
        for block in readme.split("\n\n")
        if "figure a test applies a threshold to" in block
    )
    nearest = margin["nearest_any_repeat_came_to_zero"]
    assert f"sits {nearest} from that threshold" in paragraph, paragraph
    assert f"spread of {margin['spread']}" in paragraph, (
        f"the README's spread disagrees with the record's {margin['spread']}: "
        f"{paragraph}"
    )
    assert f"{min(abs(one) for one in margin['values'])} through " in paragraph, (
        "the paragraph explains the spread with a range that is not the record's"
    )
