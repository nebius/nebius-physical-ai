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

import ast
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from npa.workbench.open3d.artifacts import Open3dError, validate_support
from npa.workbench.open3d.runner import _coverage_cost, _coverage_from_distances

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "npa" / "src" / "npa" / "workbench" / "open3d"
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
    assert examined["the_run_used_factor"] == 1.0
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
    "hollow",
    [
        {"coverage_cost_of_crop": None},
        {"coverage_before_crop": None},
        {"coverage_cost_of_crop": None, "coverage_before_crop": None},
        {"coverage_cost_of_crop": {}},
    ],
    ids=["cost_none", "before_none", "both_none", "cost_empty"],
)
def test_publishing_the_keys_with_nothing_in_them_is_not_the_same_as_not_having_them(
    hollow: dict,
) -> None:
    """Absent is exempt; present and empty is an error. They are different reports.

    Keyed on `None` instead of absence, the exemption above was a switch a writer
    could throw: emitting both keys as `None` left them present, satisfied the guard
    that reads the keys, and skipped every check here. A reviewer threw it with two
    one-word edits and the suite stayed green. `both_none` is that case; the others
    are the halves of it.
    """

    report = _report()
    report.update(hollow)
    with pytest.raises(Open3dError, match="missing coverage"):
        validate_support(report, voxel_size=1.0)


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


#: The functions in `runner.py` whose job is to produce a coverage block. A value
#: emitted by calling one of these is a measurement; add to this set when a new one is
#: written, which is a visible cost and the point of listing them.
MEASUREMENT_HELPERS = frozenset({"_coverage_from_distances", "_coverage_cost"})


def _is_a_measurement(node: ast.expr) -> bool:
    """Does this emitted value come from measuring something?

    Phrased as a whitelist after three rounds of the opposite. Rejecting hollow shapes
    one at a time does not converge: a check against `ast.Constant` missed `{}`, adding
    `Dict`/`List`/`Tuple`/`Set` missed `dict()`, and the next shape would have been
    found the same way. A reviewer named the pattern and the stable form -- accept what
    is intended rather than enumerate what is not -- and this is that.

    Deliberately permissive about *how*: a plain name is accepted, because binding the
    block to a local before putting it in the report is an ordinary thing to do and the
    validator checks the value either way. A qualified call such as
    `helpers._coverage_cost(...)` is accepted too. What it will not accept is a value
    that cannot have measured anything.

    The boundary, mapped by a reviewer looking for honest code this rejects: a
    defensive copy spelled `{**_coverage_cost(...)}` or `dict(_coverage_cost(...))`,
    a walrus, and a ternary. All four are a one-line fix -- bind the block to a name --
    and widening the rule to admit them is how the enumeration this replaced kept
    acquiring holes.
    """

    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return name in MEASUREMENT_HELPERS
    return False


def test_the_reconstruction_still_publishes_the_pair_the_validator_reads() -> None:
    """The join between the writer and the validator, checked where Open3D is absent.

    `validate_support` reads the coverage-cost pair only when a report carries it, so
    that records written before it existed still validate. A reviewer pointed out what
    that leaves open: remove the emission from `run_reconstruct` and nothing goes red,
    because the validator then skips the invariant by design and no test here feeds it a
    report the runner actually produced.

    The functional smoke now requires the pair on a real report, which is the stronger
    check -- it runs the pipeline. But that eval needs Open3D, so it runs in the
    container and on no CI shard. This reads the writer's own returned keys out of the
    source instead, which is weaker about the values and runs everywhere.
    """

    tree = ast.parse((SRC / "runner.py").read_text())
    writer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_reconstruct"
    )
    emitted = {
        key.value: value
        for node in ast.walk(writer)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    for block in ("coverage_before_crop", "coverage_cost_of_crop", "coverage"):
        assert block in emitted, (
            f"run_reconstruct no longer emits {block}, and validate_support skips the "
            "coverage-cost invariant on a report that lacks the pair, so the whole "
            "check would be switched off without a failing test"
        )
        # And the key must carry a measurement, because present-but-hollow is the same
        # evasion one word smaller. The validator rejects that at runtime; this is the
        # cheaper notice, failing in CI rather than in a pod.
        assert _is_a_measurement(emitted[block]), (
            f"run_reconstruct emits {block} as something that cannot have measured "
            f"anything. Emit it by calling one of {sorted(MEASUREMENT_HELPERS)}, or "
            "bind the block to a name first; if a new measurement helper was added, "
            "add it to MEASUREMENT_HELPERS"
        )


def test_the_functional_smoke_requires_the_pair_on_a_real_report() -> None:
    """And that requirement must stay in the eval that runs the real pipeline.

    Stated as its own test because the static check above cannot see values. If this
    assertion is deleted, the only thing proving a real report satisfies the invariant
    is gone, and nothing on a CI shard can notice.
    """

    smoke = (SRC / ".." / ".." / "smoke" / "test_open3d_functional.py").resolve()
    text = smoke.read_text()
    assert "coverage_cost_of_crop" in text
    assert "coverage_before_crop" in text
    assert "validate_support" in text


def test_public_report_allowlist_covers_every_new_measurement_leaf() -> None:
    """The fail-closed public report generator must accept the additive report block."""

    module = runpy.run_path(
        str(REPO / "docs/evidence/open3d/harness/derive_public_run_report.py")
    )
    allowed = module["ALLOWED"]["surface/result.json"]
    expected = {
        "coverage_before_crop.fraction_within_two_voxels",
        "coverage_before_crop.fraction_within_voxel",
        "coverage_before_crop.sample_to_surface_p95",
        "coverage_before_crop.sample_to_surface_rmse",
        "coverage_before_crop.samples",
        "coverage_cost_of_crop.coverage_is_bit_identical",
        "coverage_cost_of_crop.samples",
        "coverage_cost_of_crop.samples_abandoned_by_crop",
        "coverage_cost_of_crop.samples_covered_before_crop",
        "coverage_cost_of_crop.worst_abandoned_sample_voxels_from_surface",
    }
    assert expected <= set(allowed)
    assert (
        allowed["coverage_cost_of_crop.worst_abandoned_sample_voxels_from_surface"]
        == module["FLOAT_OR_NULL"]
    )


def test_the_real_data_observation_is_scoped_to_its_record() -> None:
    """One retained real-data result is an observation, not a crop guarantee."""

    observation = record()["recorded_run_observation"]
    assert observation["support_distance_factor"] == 1.0
    assert (
        observation["coverage_within_voxel_before"]
        == observation["coverage_within_voxel_after"]
    )
    assert "not a guarantee" in observation["reading"]
    assert observation["source"].endswith("capability-record-support-crop.json")


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
