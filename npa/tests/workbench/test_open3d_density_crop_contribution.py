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
import pathlib
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
        "support_distance_factor": 1.0,
        "voxel_size": 0.05,
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


def test_a_flag_that_contradicts_the_factor_it_is_derived_from_is_rejected() -> None:
    """The direction the counts cannot catch.

    `support_crop_was_disabled: false` beside `support_distance_factor: 0.0` credits
    the density crop with 0 of 518 removals -- which, with no support crop to overlap
    with, is impossible by the same construction the True branch enforces. The
    arithmetic checks all pass: 0 <= 518, the totals agree, the disabled branch does
    not fire because the flag says enabled. Only the configuration contradicts it.
    """

    with pytest.raises(Open3dError, match="enabled while the report ran at a support"):
        validate_density_crop_contribution(
            _report(support_distance_factor=0.0),
        )


def test_a_report_predating_the_factor_still_validates() -> None:
    """Compatibility: the tie is skipped when the field it ties to is not numeric.

    Reports written before `voxel_size` was published carry a factor and no voxel, and
    the oldest carry neither. Neither shape is a contradiction, and refusing them
    would reject records that are internally consistent.
    """

    old = _report()
    old.pop("support_distance_factor", None)
    validate_density_crop_contribution(old)
    without_voxel = _report()
    without_voxel.pop("voxel_size", None)
    validate_density_crop_contribution(without_voxel)


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
        pytest.param(
            {"support_crop_was_disabled": True, "removed_only_by_this_crop": 518},
            "disabled while the report ran at a support distance factor of 1.0",
            id="claims_support_was_off_while_the_report_ran_it",
        ),
        pytest.param(
            {"support_distance": 0.5},
            "voxel size and factor give",
            id="measured_against_a_distance_the_report_does_not_name",
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

    Both reports here declare the factor that makes the flag true. They did not until
    the flag was tied to it, which meant this test's own fixture described a run with
    support cropping both off (the flag) and on (the factor) -- the contradiction the
    tie now refuses.
    """

    validate_density_crop_contribution(
        _report(
            support_distance_factor=0.0,
            contribution={
                "removed_only_by_this_crop": 518,
                "support_crop_was_disabled": True,
                "support_distance": None,
            },
        )
    )
    with pytest.raises(Open3dError, match="every density removal"):
        validate_density_crop_contribution(
            _report(
                support_distance_factor=0.0,
                contribution={
                    "removed_only_by_this_crop": 517,
                    "support_crop_was_disabled": True,
                    "support_distance": None,
                },
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
    # 0.20 -- the shipped factor near -780 triangles and a wider one near -9000 --
    # and an `any` over both let either carry the argument alone. The finding is about
    # the configuration the product ships, so the cost has to be shown there: the
    # shipped row could have gone to zero with this test still green. The sign is what
    # is asserted, not the count: the count's observed values span 2 and sit some 783
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


def _reconstruct_report(**request_fields: Any) -> dict:
    """The block `run_reconstruct` publishes, built by the shipped helper.

    The helper, not a hand-written dict: F1 was the runner emitting `null` where the
    validator requires a mapping, so a fixture that assembles the block itself cannot
    see the defect. This calls `_density_crop_contribution` with the arguments the
    `density_quantile == 0` path passes it -- an empty removal set -- and assembles the
    rest of the report the way the producer does.
    """

    import numpy as np

    from npa.workbench.open3d.runner import (
        ReconstructRequest,
        _density_crop_contribution,
    )

    o3d = pytest.importorskip("open3d")
    fields = {
        "input_path": "s3://bucket/registered.json",
        "output_path": "s3://bucket/out",
        "run_id": "r",
        **request_fields,
    }
    request = ReconstructRequest(**fields)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.random.default_rng(0).random((256, 3)))
    voxel = 0.05
    return {
        "low_density_vertices_removed": 0,
        "support_distance_factor": request.support_distance_factor,
        "voxel_size": voxel,
        "density_crop_contribution": _density_crop_contribution(
            o3d, cloud, [], voxel, request
        ),
    }


@pytest.mark.parametrize(
    "support_distance_factor",
    [1.0, 0.0],
    ids=["support crop on", "both crops off"],
)
def test_a_run_with_the_density_crop_off_publishes_a_valid_report(
    support_distance_factor: float,
) -> None:
    """`--density-quantile 0` is schema-legal, documented, and used to fail the stage.

    The field was nullable and the validator requires a mapping, so a run that chose
    to keep the raw Poisson surface published `null` and was then refused by its own
    runtime -- after the solve, both mesh writes and both support passes, and before
    anything durable was written. The message named the artifact, so an operator was
    pointed at an artifact bug rather than at their flag.

    Both factors, because the two filters disable independently and the disabled
    branch of the validator is reached only by the second.
    """

    report = _reconstruct_report(
        density_quantile=0.0, support_distance_factor=support_distance_factor
    )
    contribution = report["density_crop_contribution"]
    assert contribution["vertices_removed"] == 0
    assert contribution["removed_only_by_this_crop"] == 0
    assert contribution["support_crop_was_disabled"] is (support_distance_factor <= 0.0)
    validate_density_crop_contribution(report)


def test_the_block_is_the_same_shape_whether_or_not_the_crop_ran() -> None:
    """Omitting the key also passes the validator, and would not be a fix.

    `null` and absent and a mapping are three shapes for one field, and the absent
    case is the compatibility sentinel for reports written before the block existed.
    Reusing it for "this run did not crop" would make a current run indistinguishable
    from an old one, to every consumer and to the validator.
    """

    cropped = _reconstruct_report(density_quantile=0.02)
    uncropped = _reconstruct_report(density_quantile=0.0)
    assert set(cropped["density_crop_contribution"]) == set(
        uncropped["density_crop_contribution"]
    )


def _public_report_deriver():
    """The publication table, loaded from the harness directory it lives in."""

    import importlib.util

    path = EVIDENCE.parent / "harness" / "derive_public_run_report.py"
    spec = importlib.util.spec_from_file_location("derive_public_run_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_every_field_the_runner_publishes_is_classified_for_publication() -> None:
    """The table denies by default, so a new producer field aborts the next report.

    `derive_public_run_report.py` raises SystemExit on a leaf it cannot classify.
    That fails closed, which is right, and it still fails: the block was added to the
    report without being added to the table, so the next derivation of a public
    report would have stopped on it. Read out of the producer by AST rather than
    listed here, so adding a key to the runner and not to the table fails here
    instead of at publication time.
    """

    import ast

    from npa.workbench.open3d import runner as runner_module

    source = ast.parse(pathlib.Path(runner_module.__file__).read_text())
    emitted: set[str] = set()
    for node in ast.walk(source):
        if (
            not isinstance(node, ast.FunctionDef)
            or node.name != "_density_crop_contribution"
        ):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Dict):
                continue
            for key in inner.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    emitted.add(key.value)
    assert emitted, "could not read the contribution keys out of the producer"

    table = _public_report_deriver().ALLOWED["surface/result.json"]
    missing = sorted(
        key for key in emitted if f"density_crop_contribution.{key}" not in table
    )
    assert not missing, (
        "the runner publishes these and the publication table would abort on them: "
        f"{missing}"
    )

    # And the voxel size, read out of the report the runner returns. Requiring it in
    # the table alone left the other side free: dropping `"voxel_size": voxel` from
    # the runner costs nothing in the table (which permits rather than requires) and
    # silently makes the support-distance check unreachable, because that check runs
    # only when the report carries a numeric voxel size. A guard that can be disabled
    # by deleting the field it reads is not a guard.
    returned: set[str] = set()
    reconstruct = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "run_reconstruct"
    )
    for node in ast.walk(reconstruct):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    returned.add(key.value)
    assert "support_distance_factor" in returned, "read the wrong return statement"
    assert "voxel_size" in returned, (
        "run_reconstruct no longer publishes the voxel size, so the support distance "
        "it reports cannot be checked against the factor it claims to have run at"
    )
    assert "voxel_size" in table, "the voxel size the support distance is derived from"


def test_the_record_reports_constancy_rather_than_filtering_for_it() -> None:
    """A block named "identical in every repeat" used to be built by dropping.

    The five structural facts were filtered to the ones that held, so a figure that
    moved left the block with nothing saying it had. Four of the five are asserted by
    name elsewhere in this file, so their disappearance would raise KeyError;
    `vertices_before_any_crop` was asserted nowhere while being published as a fixed
    figure at the top of the record.
    """

    stability = _record()["how_stable_are_these_figures"]
    by_repeat = stability["structural_results_by_repeat"]
    assert set(by_repeat) == {
        "vertices_before_any_crop",
        "density_removed_vertices",
        "of_those_within_the_support_distance",
        "delivered_mesh_is_identical",
        "the_density_crop_changes_the_mesh_here",
    }
    for field, entry in by_repeat.items():
        assert isinstance(entry["identical"], bool), field
        if entry["identical"]:
            assert entry["values"] is None, field
        else:
            assert len(set(map(repr, entry["values"]))) > 1, field
    assert set(stability["identical_in_every_repeat"]) == {
        field for field, entry in by_repeat.items() if entry["identical"]
    }
    assert stability["structural_results_that_moved"] == sorted(
        field for field, entry in by_repeat.items() if not entry["identical"]
    )


def test_the_records_limits_cite_only_what_it_measured() -> None:
    """The block whose job is to bound the claim was over-claiming.

    It read "one real scan and the pipeline's synthetic fragments, both of which show
    the same complete overlap at the defaults". No artifact in the evidence directory
    measures this quantity on the synthetic fragments, so the corroboration was not
    there to cite -- and a reader checking how far the result generalises was given a
    second scene that had not been tested.
    """

    record = _record()
    limits = record["what_this_does_not_claim"]["not_a_general_result_about_every_scan"]
    assert "synthetic fragments as showing" in limits, (
        "the correction should say what the sentence used to claim, not quietly drop it"
    )
    assert "one real scan" in limits
    assert record["input"], "the one scan the record does measure"


def test_the_runner_publishes_the_block_whether_or_not_the_crop_ran() -> None:
    """The publish decision, which the tests above cannot reach.

    Those build the block by calling the shipped helper, so they prove the helper is
    honest about a crop that did not run -- and would pass unchanged if
    `run_reconstruct` went back to initialising the field to `None` and filling it in
    only inside `if request.density_quantile > 0.0`, which was F1. The defect was the
    publish decision, not the helper, and `run_reconstruct` needs a registered
    manifest, two writable mesh paths and a Poisson solve to execute, so the decision
    is read out of its source instead.

    What this does not do: check that the call is given the right arguments, or that
    the block it builds is correct. Those are the tests above.
    """

    import ast

    from npa.workbench.open3d import runner as runner_module

    tree = ast.parse(pathlib.Path(runner_module.__file__).read_text())
    reconstruct = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_reconstruct"
    )

    def calls_the_helper(node: ast.AST) -> bool:
        value = getattr(node, "value", None)
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_density_crop_contribution"
        )

    conditional: set[int] = set()
    for node in ast.walk(reconstruct):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
            for inner in ast.walk(node):
                if inner is not node and calls_the_helper(inner):
                    conditional.add(id(inner))

    assignments = [
        node
        for node in ast.walk(reconstruct)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and calls_the_helper(node)
    ]
    assert assignments, "run_reconstruct does not build a contribution block at all"
    assert any(id(node) not in conditional for node in assignments), (
        "every call to _density_crop_contribution in run_reconstruct sits inside a "
        "branch, so a run that does not crop publishes whatever the field was "
        "initialised to -- which the validator refuses when that is None"
    )

    initialisers = [
        node
        for node in ast.walk(reconstruct)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "density_contribution"
    ]
    for node in initialisers:
        assert not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ), "density_contribution is initialised to None, which the validator refuses"
