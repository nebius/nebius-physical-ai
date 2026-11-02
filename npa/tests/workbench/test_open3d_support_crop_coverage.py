"""The support crop's cost to coverage, and the record that measures it.

The capability record lists a bit-identical coverage result among its supported claims,
phrased as a property of the crop rule. It is not one: the crop tests vertices and
removes triangles, coverage measures sample-to-surface distance, and surface within one
voxel of a sample can be removed either because the factor is below 1.0 or because one
distant vertex took its triangle with it. The second does not close at any factor.

These tests cover the arithmetic that now reports that cost, and hold the evidence
record to its own rows. They do not need Open3D: the query is separated from the
definitions in `runner`, because `open3d` is not installed on any CI shard -- it appears
in `npa/pyproject.toml` only as a container image tag, is in no optional-dependency
extra, and is not in `npa/ci/requirements.txt`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from npa.workbench.open3d.artifacts import Open3dError, validate_support
from npa.workbench.open3d.runner import _coverage_cost, _coverage_from_distances

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "docs" / "evidence" / "open3d" / "support-crop-coverage-guarantee.json"
HARNESS = (
    REPO
    / "docs"
    / "evidence"
    / "open3d"
    / "harness"
    / "support_crop_coverage_guarantee.py"
)
VOXEL = 1.0


def record() -> dict:
    return json.loads(RECORD.read_text())


def rows(mechanism: str) -> list[dict]:
    return record()["mechanisms"][mechanism]["factors"]


def test_a_sample_that_loses_its_only_nearby_surface_is_counted() -> None:
    before = np.array([0.1, 0.9, 5.0])
    after = np.array([0.1, 8.0, 5.0])
    cost = _coverage_cost(before, after, VOXEL)
    assert cost["samples_covered_before_crop"] == 2
    assert cost["samples_abandoned_by_crop"] == 1
    assert cost["coverage_is_bit_identical"] is False
    assert cost["worst_abandoned_sample_voxels_from_surface"] == pytest.approx(8.0)


def test_a_crop_that_costs_coverage_nothing_says_so() -> None:
    before = np.array([0.1, 0.9, 5.0])
    after = np.array([0.1, 0.95, 9.0])
    cost = _coverage_cost(before, after, VOXEL)
    assert cost["samples_abandoned_by_crop"] == 0
    assert cost["coverage_is_bit_identical"] is True
    assert cost["worst_abandoned_sample_voxels_from_surface"] is None


def test_a_sample_that_was_never_covered_is_not_counted_as_a_loss() -> None:
    """Moving an already-uncovered sample further out costs coverage nothing.

    Without this the count would conflate "the crop abandoned a sample it was
    explaining" with "the crop moved surface the sample was never near", and the
    second is what cropping unsupported area is supposed to do.
    """

    cost = _coverage_cost(np.array([4.0]), np.array([40.0]), VOXEL)
    assert cost["samples_covered_before_crop"] == 0
    assert cost["samples_abandoned_by_crop"] == 0
    assert cost["coverage_is_bit_identical"] is True


def test_the_count_is_a_count_and_not_the_drop_in_the_fractions() -> None:
    """Deliberately an input a real crop cannot produce, to pin which one it reports.

    Removing surface cannot bring a sample closer to it, so on real data the number of
    abandoned samples equals the drop in coverage, and `validate_support` now requires
    that. The two are written differently anyway -- a count of samples that lost their
    surface, not a subtraction of two fractions -- so if the premise is ever violated
    the published number is the one about samples. Here two samples swap distances: the
    fractions read as unchanged while one sample has lost its surface.
    """

    before = np.array([0.5, 4.0])
    after = np.array([4.0, 0.5])
    cost = _coverage_cost(before, after, VOXEL)
    assert cost["coverage_is_bit_identical"] is True
    assert cost["samples_abandoned_by_crop"] == 1


def test_the_voxel_is_the_unit_the_distances_are_reported_in() -> None:
    before = np.array([0.4])
    after = np.array([1.6])
    assert _coverage_cost(before, after, 1.0)[
        "worst_abandoned_sample_voxels_from_surface"
    ] == pytest.approx(1.6)
    # Same geometry, voxel twice as large: nothing is abandoned, because 1.6 is now
    # inside one voxel.
    assert _coverage_cost(before, after, 2.0)["samples_abandoned_by_crop"] == 0


def test_the_worst_abandoned_sample_is_the_worst_one_in_voxels() -> None:
    """Two abandoned samples at different distances, measured in a voxel that is not 1.

    With one abandoned sample the worst and the least are the same number, and with a
    voxel of 1.0 the conversion is a no-op, so neither alone tests anything. Both
    mistakes survived until this case existed.
    """

    cost = _coverage_cost(np.array([0.5, 0.5]), np.array([3.0, 10.0]), 2.0)
    assert cost["samples_abandoned_by_crop"] == 2
    assert cost["worst_abandoned_sample_voxels_from_surface"] == pytest.approx(5.0)


def test_the_cost_count_uses_the_same_boundary_as_the_fractions_beside_it() -> None:
    """One voxel out is not covered, in the count as well as in the fractions.

    These two are published next to each other. If the count treated a sample exactly
    one voxel out as covered while `fraction_within_voxel` did not, a run could report
    an abandoned sample that the fraction says was never there.
    """

    assert _coverage_cost(np.array([1.0]), np.array([5.0]), VOXEL) == {
        "samples": 1,
        "samples_covered_before_crop": 0,
        "samples_abandoned_by_crop": 0,
        "coverage_is_bit_identical": True,
        "worst_abandoned_sample_voxels_from_surface": None,
    }
    # And a sample that ends exactly one voxel out has been abandoned, by the same rule.
    landed_on_the_boundary = _coverage_cost(np.array([0.5]), np.array([1.0]), VOXEL)
    assert landed_on_the_boundary["samples_abandoned_by_crop"] == 1


def test_the_coverage_definitions_are_a_strict_boundary() -> None:
    """A sample exactly one voxel out is not covered; the definition uses `<`."""

    assert (
        _coverage_from_distances(np.array([1.0]), VOXEL)["fraction_within_voxel"] == 0
    )
    assert (
        _coverage_from_distances(np.array([0.999999]), VOXEL)["fraction_within_voxel"]
        == 1
    )


@pytest.mark.parametrize(
    "mechanism",
    ["factor_below_one_opens_a_band", "a_triangle_dies_for_one_vertex"],
)
def test_each_published_row_agrees_with_the_cost_arithmetic(mechanism: str) -> None:
    """Every recorded row must be consistent with the definitions shipped in runner.

    The harness measures with its own copy of the coverage question. This checks the
    two cannot drift apart on the part that is pure arithmetic: a row reporting
    abandoned samples must not also report that coverage was bit-identical.
    """

    for row in rows(mechanism):
        fell = row["coverage_within_voxel_before"] != row["coverage_within_voxel_after"]
        assert row["coverage_is_bit_identical"] is not fell
        if row["samples_abandoned"]:
            assert fell, f"factor {row['support_distance_factor']} abandoned a sample "
            assert row["worst_abandoned_voxels_from_surface"] is not None
            assert row["worst_abandoned_voxels_from_surface"] >= 1.0
        else:
            assert row["worst_abandoned_voxels_from_surface"] is None


def test_the_band_closes_at_the_schema_default_and_not_below() -> None:
    mech = record()["mechanisms"]["factor_below_one_opens_a_band"]
    assert mech["factors_where_coverage_fell"], (
        "the band mechanism must demonstrate at least one factor where coverage fell, "
        "or it demonstrates nothing"
    )
    assert max(mech["factors_where_coverage_fell"]) < 1.0
    assert 1.0 in mech["factors_where_nothing_was_removed"]


def test_raising_the_factor_is_not_recorded_as_a_remedy_for_the_asymmetry() -> None:
    """The claim that no factor fixes the asymmetry must rest on an empty set.

    A factor above every vertex's own distance removes nothing, and coverage then
    survives because there was no crop. That row is in the record deliberately, and
    this test is what stops it being read as the remedy: the set of factors that both
    cropped and kept coverage has to be empty.
    """

    mech = record()["mechanisms"]["a_triangle_dies_for_one_vertex"]
    assert mech["factors_where_coverage_held_after_a_real_crop"] == []
    assert 1.0 in mech["factors_where_coverage_fell"], (
        "the asymmetry must be shown at the schema default, which is the factor a "
        "consumer gets without asking"
    )
    assert [f for f in mech["factors_where_coverage_fell"] if f > 1.0], (
        "and at least one factor above it, or 'raising the factor does not help' is "
        "untested"
    )
    for factor in mech["factors_where_nothing_was_removed"]:
        row = next(r for r in mech["factors"] if r["support_distance_factor"] == factor)
        assert row["vertices_removed"] == 0
        assert row["triangles_before"] == row["triangles_after"]


def test_the_vacuous_hold_is_disclosed_rather_than_counted() -> None:
    """A hold with nothing removed must be described as such in the prose, not listed
    beside the real ones."""

    mech = record()["mechanisms"]["a_triangle_dies_for_one_vertex"]
    reading = mech["reading"]
    assert "no crop" in reading
    assert "removed zero vertices" in reading
    assert str(mech["factors_where_nothing_was_removed"]) in reading


def test_the_record_does_not_dispute_the_measurement_it_reinterprets() -> None:
    """This lane did not re-measure the recorded dataset and must not imply it did."""

    examined = record()["the_claim_examined"]
    assert examined["the_run_used_factor"] == 0.87
    assert examined["the_schema_default_is"] == 1.0
    assert "re-measures" in examined["what_is_not_disputed"]
    claims = record()["what_this_does_not_claim"]
    assert any("not re-measured" in c or "not reachable" in c for c in claims)
    assert any("constructed" in c for c in claims)


def test_the_examined_claim_is_quoted_from_the_record_it_cites() -> None:
    """The quoted text must be what that file says, or this is arguing with a paraphrase."""

    cited = record()["the_claim_examined"]
    source = REPO / cited["source"]
    field = json.loads(source.read_text())["claim_scope"]["supported_claims"][1]
    assert cited["text"] == field


def _report(**overrides) -> dict:
    """A reconstruction report that `validate_support` accepts, before mutation."""

    bands = {
        "unsupported_area_fraction": 0.1,
        "unsupported_area_beyond_1_5_voxels": 0.05,
        "unsupported_area_beyond_3_voxels": 0.01,
        "max_vertex_distance_to_sample": 0.9,
    }
    report = {
        "support_distance_factor": 1.0,
        "support_before_crop": {**bands, "unsupported_area_fraction": 0.2},
        "support": dict(bands),
        "coverage_before_crop": {
            "samples": 100,
            "fraction_within_voxel": 0.95,
            "sample_to_surface_rmse": 0.01,
        },
        "coverage": {
            "samples": 100,
            "fraction_within_voxel": 0.93,
            "sample_to_surface_rmse": 0.02,
        },
        "coverage_cost_of_crop": {
            "samples": 100,
            "samples_covered_before_crop": 95,
            "samples_abandoned_by_crop": 2,
            "coverage_is_bit_identical": False,
            "worst_abandoned_sample_voxels_from_surface": 3.4,
        },
    }
    report.update(overrides)
    return report


def test_a_consistent_cost_block_is_accepted() -> None:
    validate_support(_report(), voxel_size=1.0)


def test_a_report_written_before_the_cost_block_existed_still_validates() -> None:
    """The new pair is checked when present. Requiring it would reject every report
    already in this repository's evidence, which were not wrong when they were written.
    """

    older = _report()
    del older["coverage_before_crop"]
    del older["coverage_cost_of_crop"]
    validate_support(older, voxel_size=1.0)


@pytest.mark.parametrize(
    ("what", "mutate"),
    [
        (
            "coverage rose across the crop",
            lambda r: r["coverage"].__setitem__("fraction_within_voxel", 0.97),
        ),
        (
            "the abandonment count does not account for the drop",
            lambda r: r["coverage_cost_of_crop"].__setitem__(
                "samples_abandoned_by_crop", 1
            ),
        ),
        (
            "the covered-before count disagrees with the fraction",
            lambda r: r["coverage_cost_of_crop"].__setitem__(
                "samples_covered_before_crop", 90
            ),
        ),
        (
            "bit-identical is claimed while coverage moved",
            lambda r: r["coverage_cost_of_crop"].__setitem__(
                "coverage_is_bit_identical", True
            ),
        ),
        (
            "the sample counts do not match",
            lambda r: r["coverage_before_crop"].__setitem__("samples", 99),
        ),
        (
            "an abandoned sample is reported as nearer than one voxel",
            lambda r: r["coverage_cost_of_crop"].__setitem__(
                "worst_abandoned_sample_voxels_from_surface", 0.4
            ),
        ),
        (
            "a worst abandoned sample is reported when none was abandoned",
            lambda r: (
                r["coverage_cost_of_crop"].update(
                    samples_abandoned_by_crop=0, coverage_is_bit_identical=True
                )
                or r["coverage"].__setitem__("fraction_within_voxel", 0.95)
            ),
        ),
        (
            "a disabled crop abandoned a sample",
            lambda r: r.__setitem__("support_distance_factor", 0.0),
        ),
        (
            "the cost block is absent while its partner is present",
            lambda r: r.pop("coverage_cost_of_crop"),
        ),
        (
            "an abandonment count is a boolean",
            lambda r: r["coverage_cost_of_crop"].__setitem__(
                "samples_abandoned_by_crop", True
            ),
        ),
    ],
)
def test_an_inconsistent_cost_block_is_rejected(what: str, mutate) -> None:
    report = _report()
    mutate(report)
    with pytest.raises(Open3dError):
        validate_support(report, voxel_size=1.0)
    assert what


def test_the_real_data_control_reports_a_measured_zero() -> None:
    """The constructed failures must be published beside a real run that costs nothing.

    Without this the record reads as though cropping routinely abandons samples. It
    does not, on the one real reconstruction measured here, at any factor. The control
    is what keeps the finding the size it actually is.
    """

    control = record()["on_real_reconstruction_data_it_costs_nothing"]
    assert control["every_report_was_accepted_by_the_validator"] is True
    assert control["factors"], "the control must actually run the product path"
    assert all(r["samples_abandoned_by_crop"] == 0 for r in control["factors"])
    assert all(r["coverage_is_bit_identical"] for r in control["factors"])
    # And it has to include a factor that removed something, or it shows only that
    # doing nothing costs nothing.
    assert max(r["unsupported_vertices_removed"] for r in control["factors"]) > 0
    assert any(
        r["support_distance_factor"] == 1.0 and r["unsupported_vertices_removed"] >= 0
        for r in control["factors"]
    )


def test_the_record_does_not_generalise_the_control_into_a_guarantee_either() -> None:
    """A zero on one input is not a bound, which is the mistake this lane already made
    once on a different figure."""

    claims = record()["what_this_does_not_claim"]
    assert any("unmeasured" in c for c in claims)


def test_the_harness_states_the_two_mechanisms_it_separates() -> None:
    text = HARNESS.read_text()
    assert "remove_vertices_by_mask" in text
    assert "factor_below_one_opens_a_band" in text
    assert "a_triangle_dies_for_one_vertex" in text
