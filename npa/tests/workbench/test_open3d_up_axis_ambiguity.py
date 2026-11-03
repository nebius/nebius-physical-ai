"""The camera's up axis, and whether the scan or the run decided it.

`_up_axis` infers an up vector from the largest planar segment and the view cameras
consume it, so a wrong axis rolls the whole scene. `segment_plane` is RANSAC: on a
scan holding two planes of comparable point count, successive fits land on either of
them, and every field the function published to make the inference visible -- the
source string, the inlier fraction, the tilt -- took overlapping values on both
branches. One run's record could not show that the next run would roll the view by
ninety degrees.

These tests run without Open3D. They cover the aggregation and the validator, which
is where the decision lives; the measurements behind the figures are in
`docs/evidence/open3d/up-axis-ambiguity.json` and the real-geometry path is asserted
in the functional smoke.
"""

from __future__ import annotations

import re
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from npa.workbench.open3d.artifacts import Open3dError, validate_up_axis_inference

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
RECORD = EVIDENCE / "up-axis-ambiguity.json"


def _record() -> dict[str, Any]:
    return json.loads(RECORD.read_text())


def _counterexample() -> dict[str, Any]:
    from npa.workbench.open3d.runner import UP_AXIS_AGREEMENT_COUNTEREXAMPLE

    return dict(UP_AXIS_AGREEMENT_COUNTEREXAMPLE)


def agreeing_report(**overrides: Any) -> dict[str, Any]:
    """A report whose repeats agreed, so it may name a measured axis."""

    report = {
        "source": "largest-plane-normal-snapped-to-axis",
        "up": [0.0, 0.0, -1.0],
        "measured_plane_normal": [0.02, 0.01, -0.999],
        "angle_to_snapped_axis_degrees": 1.3,
        "plane_inliers": 68000,
        "plane_inlier_fraction": 0.34,
        "agreement": {
            "repeats": 9,
            "attempted_fits": 9,
            "distinct_axes": 1,
            "candidates": [{"up": [0.0, 0.0, -1.0], "chosen_by_repeats": 9}],
            "runner_up_plane": {
                "measured": True,
                "inliers": 45000,
                "inliers_relative_to_the_winner": 0.66,
                "up_it_would_imply": [0.0, -1.0, 0.0],
                "implies_a_different_axis": True,
            },
            "agreement_is_not_meaningfulness": {
                "measured_counterexample": _counterexample(),
                "so": "Agreement proves only that one plane dominates.",
            },
        },
    }
    report.update(overrides)
    return {"up_axis_inference": report}


def ambiguous_report(**overrides: Any) -> dict[str, Any]:
    """A report whose repeats disagreed, so it must publish the declared default."""

    report = {
        "source": "default-ambiguous-plane",
        "up": [0.0, 0.0, 1.0],
        "plane_inliers": 8200,
        "plane_inlier_fraction": 0.506,
        "agreement": {
            "repeats": 9,
            "attempted_fits": 9,
            "distinct_axes": 2,
            "candidates": [
                {"up": [0.0, 0.0, 1.0], "chosen_by_repeats": 7},
                {"up": [1.0, 0.0, 0.0], "chosen_by_repeats": 2},
            ],
            "runner_up_plane": {
                "measured": True,
                "inliers": 8000,
                "inliers_relative_to_the_winner": 0.97,
                "up_it_would_imply": [1.0, 0.0, 0.0],
                "implies_a_different_axis": True,
            },
            "agreement_is_not_meaningfulness": {
                "measured_counterexample": _counterexample(),
                "so": "Disagreement proves the scan does not determine an axis.",
            },
        },
    }
    report.update(overrides)
    return {"up_axis_inference": report}


class _SelectableCloud:
    """Enough of a point cloud for `_runner_up_plane` to run unmocked."""

    def __init__(self, count: int = 400) -> None:
        self.points = [0] * count

    def select_by_index(self, indices: list[int], invert: bool = False):
        kept = len(self.points) - len(indices) if invert else len(indices)
        return _SelectableCloud(max(kept, 0))


def _reading(axis: list[float], weak: bool = False) -> dict[str, Any]:
    return {
        "weak": weak,
        "up": axis,
        "measured_plane_normal": axis,
        "angle_to_snapped_axis_degrees": 0.4,
        "plane_inliers": 8100,
        "plane_inlier_fraction": 0.5,
        "inlier_indices": [0, 1, 2],
    }


def _drive_up_axis(
    monkeypatch: pytest.MonkeyPatch,
    axes: list[list[float]],
    weak: list[bool] | None = None,
    runner_up: dict[str, Any] | None = None,
) -> dict:
    """Run the real `_up_axis` over a scripted sequence of plane fits.

    The validator tests below check reports that already exist. This drives the code
    that *writes* them, with the Open3D calls replaced by a script, so the decision
    logic is covered where CI runs rather than only in the functional smoke.

    `weak` scripts the weak flag per fit. Passing `runner_up` leaves the real
    `_runner_up_plane` in place and scripts the remainder's fit instead, so a caller
    that needs the fields that function actually publishes gets them from it rather
    than from a hand-written stand-in.
    """

    from npa.workbench.open3d import runner

    flags = weak if weak is not None else [False] * len(axes)
    readings = [_reading(axis, flag) for axis, flag in zip(axes, flags, strict=True)]
    if runner_up is not None:
        readings.append(runner_up)
    supply = iter(readings)
    monkeypatch.setattr(
        runner, "_plane_up_reading", lambda *_args, **_kwargs: next(supply)
    )
    cloud: Any = object()
    if runner_up is None:
        monkeypatch.setattr(
            runner,
            "_runner_up_plane",
            lambda *_args, **_kwargs: {
                "measured": True,
                "inliers_relative_to_the_winner": 0.9,
            },
        )
    else:
        cloud = _SelectableCloud()
    return runner._up_axis(object(), cloud, 0.05, repeats=len(axes) - 1)


def test_disagreeing_fits_publish_the_declared_default_not_the_winning_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the fallback, driven deterministically.

    Both candidate axes here are something other than the declared default, because
    the obvious construction -- a floor and a wall -- has the majority branch equal
    to the default, which makes publishing the default and publishing the winner
    indistinguishable. A mutation that returned the winner survived the functional
    smoke for exactly that reason until this case existed.
    """

    x_axis, y_axis = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    report = _drive_up_axis(monkeypatch, [y_axis, y_axis, y_axis, x_axis])

    assert report["source"] == "default-ambiguous-plane", report
    assert report["up"] == [0.0, 0.0, 1.0], (
        "the repeats disagreed and the report published a measured branch as the "
        f"camera's up vector: {report['up']}"
    )
    candidates = {
        tuple(one["up"]): one["chosen_by_repeats"]
        for one in report["agreement"]["candidates"]
    }
    assert candidates == {tuple(y_axis): 3, tuple(x_axis): 1}, candidates
    validate_up_axis_inference({"up_axis_inference": report})


def test_agreeing_fits_publish_the_axis_they_agreed_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other direction: unanimity must still produce a named axis."""

    axis = [0.0, 0.0, -1.0]
    report = _drive_up_axis(monkeypatch, [axis] * 5)

    assert report["source"] == "largest-plane-normal-snapped-to-axis", report
    assert report["up"] == axis, report
    assert report["agreement"]["distinct_axes"] == 1, report["agreement"]
    validate_up_axis_inference({"up_axis_inference": report})


def test_a_single_fit_cannot_establish_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With repeats turned off there is nothing to disagree, and that is not evidence.

    Recorded rather than guarded: one fit still publishes an axis, because the
    caller asked for no repeats. The figure that matters is `repeats`, which a
    reader can see is 1.
    """

    report = _drive_up_axis(monkeypatch, [[0.0, 0.0, 1.0]])
    assert report["agreement"]["repeats"] == 1, report["agreement"]
    assert report["agreement"]["distinct_axes"] == 1, report["agreement"]


def test_a_report_whose_repeats_agreed_may_name_an_axis() -> None:
    validate_up_axis_inference(agreeing_report())


def test_a_report_whose_repeats_disagreed_may_publish_the_default() -> None:
    validate_up_axis_inference(ambiguous_report())


def test_a_record_predating_the_inference_still_validates() -> None:
    """Absent, not hollow: an older recording report simply has no such block."""

    validate_up_axis_inference({"sha256": "abc", "bytes": 10})


def test_a_report_that_publishes_the_block_as_nothing_is_rejected() -> None:
    """The distinction that keeps the exemption from being a switch."""

    with pytest.raises(Open3dError, match="not a report"):
        validate_up_axis_inference({"up_axis_inference": None})


def test_naming_an_axis_while_the_repeats_disagreed_is_rejected() -> None:
    """The defect this guard exists for: a coin flip published as a measurement."""

    payload = agreeing_report()
    payload["up_axis_inference"]["agreement"]["distinct_axes"] = 2
    payload["up_axis_inference"]["agreement"]["candidates"] = [
        {"up": [0.0, 0.0, -1.0], "chosen_by_repeats": 6},
        {"up": [1.0, 0.0, 0.0], "chosen_by_repeats": 3},
    ]
    with pytest.raises(Open3dError, match="names a measured axis"):
        validate_up_axis_inference(payload)


def test_claiming_ambiguity_while_the_repeats_agreed_is_rejected() -> None:
    """The other direction, so the fallback cannot be used to duck a real answer."""

    payload = ambiguous_report()
    payload["up_axis_inference"]["agreement"]["distinct_axes"] = 1
    payload["up_axis_inference"]["agreement"]["candidates"] = [
        {"up": [0.0, 0.0, 1.0], "chosen_by_repeats": 9}
    ]
    with pytest.raises(Open3dError, match="all agreed"):
        validate_up_axis_inference(payload)


@pytest.mark.parametrize(
    "source", ["default", "default-weak-plane", "default-ambiguous-plane"]
)
def test_a_source_meaning_no_axis_must_publish_the_declared_default(
    source: str,
) -> None:
    """Every "we could not measure it" source has to publish the same vector.

    Otherwise the honest-sounding source string can carry a measured branch anyway,
    which is the defect wearing a different label.
    """

    payload = ambiguous_report(source=source, up=[1.0, 0.0, 0.0])
    if source != "default-ambiguous-plane":
        payload["up_axis_inference"].pop("agreement")
    with pytest.raises(Open3dError, match="declared default"):
        validate_up_axis_inference(payload)


def test_claiming_ambiguity_without_saying_what_disagreed_is_rejected() -> None:
    payload = ambiguous_report()
    payload["up_axis_inference"].pop("agreement")
    with pytest.raises(Open3dError, match="does not say what they disagreed"):
        validate_up_axis_inference(payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda block: block.update({"distinct_axes": 0}), "count the axes"),
        (lambda block: block.update({"repeats": 0}), "count its repeats"),
        (lambda block: block.update({"distinct_axes": 3, "repeats": 2}), "3 axes in 2"),
        (lambda block: block.update({"candidates": []}), "counts 1 axes"),
        (
            lambda block: block["candidates"][0].update({"chosen_by_repeats": 2}),
            "do not add up",
        ),
        (
            lambda block: block.pop("agreement_is_not_meaningfulness"),
            "does not publish its limitation",
        ),
        (
            lambda block: block["agreement_is_not_meaningfulness"].update(
                {"measured_counterexample": "the armadillo, agreeing on a tilted plane"}
            ),
            "not backed by a measured counterexample",
        ),
        (
            lambda block: block["agreement_is_not_meaningfulness"][
                "measured_counterexample"
            ].update({"repeats_that_agreed": 38}),
            "does not show agreement failing",
        ),
        (
            lambda block: block["agreement_is_not_meaningfulness"][
                "measured_counterexample"
            ].update({"plane_tilt_to_the_snapped_axis_degrees": 4.0}),
            "degrees off the axis",
        ),
        (
            lambda block: block["agreement_is_not_meaningfulness"][
                "measured_counterexample"
            ].update({"plane_tilt_to_the_snapped_axis_degrees": None}),
            "finite numbers",
        ),
        (
            lambda block: block["agreement_is_not_meaningfulness"][
                "measured_counterexample"
            ].pop("of_repeats"),
            "does not publish",
        ),
    ],
)
def test_the_agreement_report_is_rejected_when_it_does_not_hold_together(
    mutate, match: str
) -> None:
    payload = agreeing_report()
    mutate(payload["up_axis_inference"]["agreement"])
    with pytest.raises(Open3dError, match=match):
        validate_up_axis_inference(payload)


def test_every_rejection_says_something_different() -> None:
    """A guard whose messages collapse cannot tell a reader which claim broke."""

    messages = set()
    builders = [
        (agreeing_report, lambda block: block.update({"distinct_axes": 2})),
        (ambiguous_report, lambda block: block.update({"distinct_axes": 1})),
        (agreeing_report, lambda block: block.update({"repeats": 0})),
        (agreeing_report, lambda block: block.update({"candidates": []})),
        (agreeing_report, lambda block: block.pop("agreement_is_not_meaningfulness")),
    ]
    for build, mutate in builders:
        payload = build()
        mutate(payload["up_axis_inference"]["agreement"])
        with pytest.raises(Open3dError) as caught:
            validate_up_axis_inference(payload)
        messages.add(str(caught.value))
    assert len(messages) == len(builders), messages


def _clearance_the_record_justifies() -> float:
    """The *most* margin the published claim can carry, not the least that would do.

    Taking the smallest sufficient level was the first attempt and it left the rule
    defenceless in the direction that matters. Relaxing the tolerated crossing rate from
    5% to 30% dropped the derived level from 0.30 to 0.10 with all 53 assertions still
    green, because a shipped bound comfortably inside a strict rule is also inside a lax
    one. No assertion about the artifact can catch a loosened gate.

    Maximising instead inverts that. The level is the largest fraction the sweep measured
    that a floor above the record's own degree line could still carry, so loosening an
    input cannot loosen the gate: relaxing the tolerance leaves the level untouched, and
    weakening the degree line *raises* it and fails the bound.

    What the tolerance still does is reject the whole arrangement -- if the most margin
    this claim can carry is not enough to hold the crossing rate down, that is a result
    about the record and not a number to adjust.
    """
    rule = _record()["the_margin_rule_the_bounds_must_satisfy"]
    case = next(
        one for one in _record()["cases"] if one["cloud"] == "armadillo_sampled"
    )
    low, high = case["single_fit_distribution"]["branches"][0][
        "tilt_to_snapped_axis_degrees_range"
    ]
    affordable = (low - rule["floor_must_stay_above_degrees"]) / (high - low)

    offered = sorted(float(one) for one in rule["crossing_rate_by_clearance"])
    within_reach = [one for one in offered if one <= affordable]
    assert within_reach, (
        f"every level the sweep measured asks more than the {affordable:.3f} of the "
        "range a floor above the record's degree line can carry"
    )
    return within_reach[-1]


def test_the_published_counterexample_reconstructs_from_the_record() -> None:
    """Every figure in the constant must be findable in the retained evidence.

    A reviewer went looking for a figure in an earlier record of mine and could not
    find it, because it came from a run the harness was not retaining. The tilt and
    fraction here are ranges over the record's own 40 fits, and the headline tilt is
    the *smallest* of them, so the claim holds at the weakest observation.
    """

    case = next(
        one for one in _record()["cases"] if one["cloud"] == "armadillo_sampled"
    )
    branches = case["single_fit_distribution"]["branches"]
    published = _counterexample()

    assert len(branches) == 1, "the counterexample needs the repeats to have agreed"
    assert branches[0]["chosen_by_fits"] == published["repeats_that_agreed"]
    assert case["single_fit_distribution"]["fits"] == published["of_repeats"]

    # Bounds, checked by direction *and* by margin. Equality against the measured
    # extremes went stale first, so this became a direction -- and that fixed the
    # wrong half: a bound on the right side of one draw by a hair is still pinned to
    # that draw. The tilt floor was 14.0 against observed minima of 14.11, 14.32,
    # 13.74 and 13.41, so it held twice and then did not.
    #
    # The margin each bound needs is measured, not picked. Each branch publishes the
    # range its 40 fits covered, and a bound has to clear the nearest end of that
    # range by a fraction of its width. A flat percentage was the first attempt and it
    # was the wrong rule: it asks as much of the inlier fraction, whose maximum has
    # moved 0.25% across four runs, as of the tilt, whose minimum has moved 3%.
    #
    # The *level* was then wrong too, which is a separate mistake and a worse one. A
    # floor on margin has to be set where a bound that merely satisfies it is still
    # safe, and a second lane's sweep over 24 regenerations put numbers on that: a bound
    # sitting exactly at a tenth is later crossed 26.3% of the time, at 0.2 it is 9.5%,
    # at 0.3 it is 2.9%, and the k=0 control returns the 50% symmetry demands. A tenth
    # permitted bounds that fail a quarter of the time. The floor shipped above survived
    # 24 of 24 only because it happens to sit at 0.149 rather than at the limit the rule
    # allowed -- the rule was not what made it safe.
    #
    # One k for both quantities, not two. At a tenth the crossing rates were 26.3% for
    # the tilt and 24.5% for the inlier: scaling by each quantity's own range already
    # equalises the risk, so a second level would be unjustified.
    #
    # The level is read out of the record rather than written here, and that is the
    # point. A test cannot defend its own threshold: with 0.30 as a literal, relaxing it
    # back to a tenth left all 53 assertions passing, because the shipped bounds clear
    # the weaker rule too. Changing it now means editing the sweep in the evidence, and
    # the assertions below are against that.
    clearance = _clearance_the_record_justifies()

    def _clears(bound: float, nearest: float, spread: float, what: str) -> None:
        assert abs(nearest - bound) >= clearance * spread, (
            f"the published {what} {bound} sits {abs(nearest - bound):.4g} from the "
            f"{nearest} this run measured, inside a tenth of the {spread:.4g} the "
            "record's own fits span. A bound that close is pinned to this draw: the "
            "tilt floor was 14.0 against a measured 14.11 by exactly this margin, "
            "and two regenerations later the evidence had crossed it"
        )

    low, high = branches[0]["tilt_to_snapped_axis_degrees_range"]
    floor = published["plane_tilt_to_the_snapped_axis_degrees"]
    assert published["this_tilt_is_a_floor_the_measurement_cleared"] is True
    assert low >= floor, (
        f"the record's smallest tilt {low} fell below the {floor} the code publishes "
        "as a floor, so the published claim is no longer covered by its evidence"
    )
    _clears(floor, low, high - low, "tilt floor")

    fraction_low, fraction_high = branches[0]["plane_inlier_fraction_range"]
    ceiling = published["plane_inlier_fraction_was_at_most"]
    assert fraction_high <= ceiling, (
        f"the record's largest inlier fraction {fraction_high} exceeded the "
        f"{ceiling} the code publishes"
    )
    _clears(ceiling, fraction_high, fraction_high - fraction_low, "inlier ceiling")

    # Both bounds must still be worth stating. A floor of zero degrees and a ceiling
    # of every point in the cloud would clear every assertion above, and would say
    # nothing: the claim is that this plane is tilted well off the axis it implies
    # and is supported by a small minority of the points.
    #
    # The line comes from the record for the same reason the fraction does: as a literal
    # here it could be weakened to zero with the whole suite still green.
    worth_stating = _record()["the_margin_rule_the_bounds_must_satisfy"][
        "floor_must_stay_above_degrees"
    ]
    assert published["plane_tilt_to_the_snapped_axis_degrees"] >= worth_stating, (
        published
    )
    # And it is published *at* that line, not above it, which is worth asserting
    # because of what it costs. The margin rule's level is the largest its sweep
    # measured that a floor above this line can afford, so the room between the two is
    # the distance between two sweep steps -- a fifth of a degree here. Four floors
    # were picked inside that room (14.0, 12.0, 10.5, 10.2) and each one satisfied the
    # rule against the fits it was written against and failed against the next. At the
    # line itself the margin comparison below is an identity, and the gate that still
    # bites is the affordability check inside `_clearance_the_record_justifies`: it
    # raises when the measured minimum comes within the sweep's level of this line.
    # That trade is stated here so a later reader sees it rather than finding a
    # comparison that cannot fail.
    assert published["plane_tilt_to_the_snapped_axis_degrees"] == worth_stating, (
        published
    )
    assert ceiling <= 0.25, published


def test_the_record_and_the_code_publish_the_same_counterexample() -> None:
    """Close the loop the other way, so the two cannot drift apart silently.

    The test above checks the constant against the record's measurements. This one
    checks the record's own copy of the constant against the constant, which is the
    drift that actually happened while writing this: the figures were corrected in
    the code and the record went on quoting the old ones.
    """

    def plain(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: plain(one) for key, one in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(one) for one in value]
        return value

    assert plain(_record()["counterexample_the_code_publishes"]) == plain(
        _counterexample()
    ), "the record's copy of the counterexample is not what the code publishes"


_WEAK_READING = {"weak": True, "plane_inliers": 90, "plane_inlier_fraction": 0.09}


def _drive_readings(monkeypatch, readings: list[dict | None], repeats: int = 8) -> dict:
    """Script the fits themselves, including weak and absent ones.

    `_drive_up_axis` scripts only the axis each usable fit implies, which cannot
    express "this fit found no plane worth using". That is the seam the vacuous-
    agreement defect lived behind.
    """

    from npa.workbench.open3d import runner

    iterator = iter(readings)
    monkeypatch.setattr(runner, "_plane_up_reading", lambda *_a, **_k: next(iterator))
    monkeypatch.setattr(
        runner, "_runner_up_plane", lambda *_a, **_k: {"measured": False}
    )
    return runner._up_axis(None, None, 0.05, repeats=repeats)


def _strong(axis: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> dict:
    return {
        "weak": False,
        "up": list(axis),
        "measured_plane_normal": list(axis),
        "angle_to_snapped_axis_degrees": 0.05,
        "plane_inliers": 5000,
        "plane_inlier_fraction": 0.5,
        "inlier_indices": list(range(10)),
    }


def test_one_usable_fit_in_nine_is_not_a_measured_axis(monkeypatch) -> None:
    """A majority of the attempts must be usable, not merely the first one.

    Weakness of the *first* reading disqualified the scan; the other eight were
    filtered out silently, so their weakness only shrank the sample. A scan that fit
    weakly eight times in nine and strongly on the draw that happened to be first
    was published as `largest-plane-normal-snapped-to-axis` with `repeats: 1`, and
    the record guard accepted it, because "the repeats agreed" is vacuously true of
    one repeat. That is the publication this whole change exists to prevent.

    It is reachable from the committed evidence, not hypothetical: the Armadillo
    case records inlier fractions of 0.122 to 0.134 against a weak threshold of 0.1,
    so ordinary RANSAC variance puts some of its fits either side of the line.
    """

    inference = _drive_readings(monkeypatch, [_strong()] + [dict(_WEAK_READING)] * 8)

    assert inference["source"] == "default-weak-plane", inference
    assert inference["up"] == [0.0, 0.0, 1.0], inference
    assert inference["usable_fits"] == 1, inference
    assert inference["attempted_fits"] == 9, inference
    validate_up_axis_inference({"up_axis_inference": inference})


@pytest.mark.parametrize(
    ("usable", "expect_measured"),
    [
        pytest.param(1, False, id="one_of_nine"),
        pytest.param(4, False, id="four_of_nine_is_not_a_majority"),
        pytest.param(5, True, id="five_of_nine_is"),
        pytest.param(9, True, id="all_nine"),
    ],
)
def test_the_boundary_of_a_usable_majority_is_where_it_says(
    monkeypatch, usable: int, expect_measured: bool
) -> None:
    """Four of nine is not a majority and five is, so the guard is off by nothing."""

    readings = [_strong()] * usable + [dict(_WEAK_READING)] * (9 - usable)
    inference = _drive_readings(monkeypatch, readings)
    measured = inference["source"] == "largest-plane-normal-snapped-to-axis"
    assert measured is expect_measured, inference
    validate_up_axis_inference({"up_axis_inference": inference})


def test_a_record_naming_an_axis_from_a_minority_of_fits_is_rejected() -> None:
    """The validator must refuse it too, not only the producer.

    The runner no longer emits this shape, so without a guard here the record
    contract would depend on the producer staying correct.
    """

    inference = {
        "source": "largest-plane-normal-snapped-to-axis",
        "up": [1.0, 0.0, 0.0],
        "agreement": {
            "repeats": 1,
            "attempted_fits": 9,
            "distinct_axes": 1,
            "candidates": [{"up": [1.0, 0.0, 0.0], "chosen_by_repeats": 1}],
            "agreement_is_not_meaningfulness": {
                "measured_counterexample": _counterexample(),
                "so": "agreement proves only that one plane dominates",
            },
        },
    }
    with pytest.raises(Open3dError, match="not a majority"):
        validate_up_axis_inference({"up_axis_inference": inference})


def test_a_report_omitting_its_attempt_count_cannot_skip_the_majority_check() -> None:
    """The bypass is omission, and it is the same payload the check above refuses.

    When `attempted_fits` was optional this record validated: the minority rule sat
    behind a presence test, so a producer that never emitted the field was never
    asked the question. Every other field in the block is required, and this one
    carries the only check that stands between one usable fit and the word
    "measured", so optional was the wrong shape for it.
    """

    inference = {
        "source": "largest-plane-normal-snapped-to-axis",
        "up": [1.0, 0.0, 0.0],
        "agreement": {
            "repeats": 1,
            "distinct_axes": 1,
            "candidates": [{"up": [1.0, 0.0, 0.0], "chosen_by_repeats": 1}],
            "agreement_is_not_meaningfulness": {
                "measured_counterexample": _counterexample(),
                "so": "agreement proves only that one plane dominates",
            },
        },
    }
    with pytest.raises(Open3dError, match="how many fits were attempted"):
        validate_up_axis_inference({"up_axis_inference": inference})


def test_both_branches_of_the_runner_state_their_attempt_count(monkeypatch) -> None:
    """Required means the producer must supply it, so bind the two together.

    Reading the source would show the field twice and say nothing about whether the
    branch that returns reaches it. These drive the real function: a scan whose fits
    agree and one whose fits split, which are the two returns that carry an
    `agreement` block, and the validator is then asked to accept each.
    """

    for readings, expect in (
        ([_strong((0.0, 0.0, 1.0))] * 9, 9),
        ([_strong((0.0, 0.0, 1.0))] * 5 + [_strong((1.0, 0.0, 0.0))] * 4, 9),
    ):
        report = _drive_readings(monkeypatch, readings)
        agreement = report["agreement"]
        assert agreement["attempted_fits"] == expect, report
        validate_up_axis_inference({"up_axis_inference": report})


def test_an_agreement_report_claiming_more_repeats_than_attempts_is_rejected() -> None:
    inference = {
        "source": "largest-plane-normal-snapped-to-axis",
        "up": [0.0, 0.0, 1.0],
        "agreement": {
            "repeats": 9,
            "attempted_fits": 3,
            "distinct_axes": 1,
            "candidates": [{"up": [0.0, 0.0, 1.0], "chosen_by_repeats": 9}],
            "agreement_is_not_meaningfulness": {
                "measured_counterexample": _counterexample(),
                "so": "agreement proves only that one plane dominates",
            },
        },
    }
    with pytest.raises(Open3dError, match="out of"):
        validate_up_axis_inference({"up_axis_inference": inference})


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("largest-plane-normal", id="a_truncation_of_the_real_string"),
        pytest.param("plane-normal-v2", id="a_plausible_successor"),
        pytest.param("Largest-Plane-Normal-Snapped-To-Axis", id="different_case"),
        pytest.param("default-ambiguous", id="a_truncated_default"),
    ],
)
def test_an_unrecognised_source_is_refused_rather_than_waved_through(
    source: str,
) -> None:
    """Both semantic checks are keyed to literal strings, so an unknown one met none.

    A record could name an axis, report five different ones across eight repeats,
    and be accepted, provided its source string was not one the validator knew. The
    dangerous shape is not a wild string: it is a near miss like
    `largest-plane-normal`, which turns both checks off silently rather than failing.
    """

    inference = {
        "source": source,
        "up": [1.0, 0.0, 0.0],
        "agreement": {
            "repeats": 8,
            "attempted_fits": 8,
            "distinct_axes": 5,
            "candidates": [
                {"up": [1.0, 0.0, 0.0], "chosen_by_repeats": 4},
                {"up": [0.0, 1.0, 0.0], "chosen_by_repeats": 1},
                {"up": [0.0, 0.0, 1.0], "chosen_by_repeats": 1},
                {"up": [-1.0, 0.0, 0.0], "chosen_by_repeats": 1},
                {"up": [0.0, -1.0, 0.0], "chosen_by_repeats": 1},
            ],
            "agreement_is_not_meaningfulness": {
                "measured_counterexample": _counterexample(),
                "so": "agreement proves only that one plane dominates",
            },
        },
    }
    with pytest.raises(Open3dError, match="unrecognised source"):
        validate_up_axis_inference({"up_axis_inference": inference})


def test_every_source_the_runner_can_emit_is_one_the_validator_knows() -> None:
    """The closed set must not lock out the producer.

    Rejecting unknown sources is only safe if the two lists agree, so this reads the
    sources out of the runner's own branches rather than restating them.
    """

    import re

    from npa.workbench.open3d import artifacts, runner

    source = Path(runner.__file__).read_text()
    emitted = set(re.findall(r'"source": "([a-z-]+)"', source))
    assert emitted, "no source strings found in the runner; has the shape changed?"
    assert emitted <= artifacts._KNOWN_UP_SOURCES, (
        f"the runner can emit {sorted(emitted - artifacts._KNOWN_UP_SOURCES)}, which "
        "the validator would now refuse"
    )


def test_the_record_keeps_a_negative_control_and_more_than_one_of_each_kind() -> None:
    """One ambiguous cloud and one determined cloud would prove very little."""

    clouds = {one["cloud"]: one for one in _record()["cases"]}
    assert "real_demo_scan" in clouds, "no real-data negative control"
    ambiguous = [
        name
        for name, one in clouds.items()
        if one["single_fit_distribution"]["distinct_axes"] > 1
    ]
    determined = [
        name
        for name, one in clouds.items()
        if one["single_fit_distribution"]["distinct_axes"] == 1
    ]
    assert len(ambiguous) >= 1 and len(determined) >= 3, clouds.keys()
    for name in determined:
        answers = clouds[name]["what_the_pipeline_publishes"]["answers"]
        named = [
            one
            for one in answers
            if one["source"] == "largest-plane-normal-snapped-to-axis"
        ]
        assert named, f"{name} has an answer and the pipeline refused to give one"


def test_the_detection_rate_the_code_states_is_the_one_the_record_measured() -> None:
    """The figures live in a constant so this test can exist.

    They were in a comment first, and the comment went stale twice while this was
    being written: the rate moves by a few counts between runs, so every
    regeneration of the record invalidated a sentence nothing was checking. Carried
    as data, the drift is a failure instead of a quiet inaccuracy.
    """

    from npa.workbench.open3d.runner import (
        UP_AXIS_DETECTION_FLOOR_OF_THIRTY,
        UP_AXIS_DETECTION_RISES_WITH_FITS,
        UP_AXIS_REPEATS,
    )

    rates = sorted(
        _record()["detection_by_repeat_count"]["rates"], key=lambda one: one["fits"]
    )
    assert len(rates) >= 3, rates
    for rate in rates:
        assert rate["of_runs"] == 30, rate
    # The floor belongs to the shipped count and to nothing else. Its own comment in
    # runner.py derives it from five sweeps of the shipped count, and applying it to
    # the four-repeat row asserted something the constant never claimed: that row came
    # back at 19 of 30 on a regeneration, correctly, because fewer fits detect less
    # often. That is the claim RISES_WITH_FITS makes about the lower rows.
    shipped = [one for one in rates if one["is_the_shipped_default"]]
    assert len(shipped) == 1, rates
    detected_at_default = shipped[0]["ambiguity_detected_in_runs"]
    assert detected_at_default >= UP_AXIS_DETECTION_FLOOR_OF_THIRTY, (
        "the retained sweep fell below the floor the code publishes at the shipped "
        f"count; the figure in runner.py no longer describes the evidence: {shipped[0]}"
    )
    # The floor has to be worth clearing: a floor of zero would always pass, and a
    # floor at the observed figure would go stale on the next sweep, which is what
    # happened to the exact counts this replaced.
    assert 0 < UP_AXIS_DETECTION_FLOOR_OF_THIRTY < detected_at_default

    # Monotonicity is asserted where it can be measured. The thirty-run sweep cannot
    # order two configurations a few runs apart -- a regeneration returned 27 at five
    # fits against 26 at nine -- so requiring it there pinned the structural claim to
    # a measurement without the power to support it.
    rise = _record()["detection_by_repeat_count"][
        "does_detection_rise_with_the_fit_count"
    ]
    assert UP_AXIS_DETECTION_RISES_WITH_FITS is rise["every_step_rose"], rise
    assert rise["each_process_is_a_fresh_interpreter"] is True, rise
    assert rise["runs_per_configuration"] >= 300, rise
    assert rise["detection_rate"] == sorted(rise["detection_rate"]), rise

    # The sigma floor is applied to the step the shipped count rests on, and not to
    # whichever step happens to be smallest. Those were the same assertion while both
    # steps looked firm; they are not. An independent lane ran this harness and
    # returned 2.91 sigma on the upper step where this run had 4.15, so a floor of 3.0
    # over the minimum was a gate that a reproduction of my own measurement failed.
    # The lower step has come back above five sigma in every run, including theirs.
    shipped_step = rise["the_step_the_shipped_count_rests_on"]
    assert shipped_step is not None, rise
    assert shipped_step["to_fits"] == UP_AXIS_REPEATS + 1, shipped_step
    assert shipped_step["in_sigmas"] > 3.0, shipped_step
    assert not shipped_step["an_endpoint_detected_every_run"], (
        "the step the default rests on now ends at a perfect rate, where the "
        f"two-proportion sigma is optimistic, so this floor is flattered: {shipped_step}"
    )
    # The upper step is ordered, not bounded. Putting a floor under it would be
    # re-pinning a claim to a measurement that has not reproduced at that strength.
    for pair in rise["consecutive_pairs"]:
        assert pair["detection_rose_by"] > 0.0, pair
    for pair in rise["consecutive_pairs"]:
        rebuilt = pair["detection_rose_by"] / pair["one_sigma_of_the_difference"]
        assert pair["in_sigmas"] == pytest.approx(rebuilt, rel=0.02), pair
    assert any(
        one["this_pair_includes_the_shipped_count"] for one in rise["consecutive_pairs"]
    ), rise


def test_a_second_lane_measured_these_rates_on_its_own_machine() -> None:
    """Two sweeps sharing nothing but the harness bytes agree on the rates.

    One 900-run sweep measures one machine. An independent lane ran these harness
    bytes on its own host, in its own virtualenv, against a fresh download of the
    demo meshes, with no seed anywhere in the code, and landed within a couple of
    sigma per configuration. That is a stronger statement about the rates than any
    single run, and it is the reason this record can put numbers on the default
    rather than an ordering.
    """

    reproduction = _record()["detection_by_repeat_count"][
        "does_detection_rise_with_the_fit_count"
    ]["an_independent_lane_ran_this_too"]

    assert reproduction["runs_each"] >= 300, reproduction
    assert len(reproduction["detections_they_reported"]) == len(
        reproduction["per_configuration"]
    )
    # The citation is bound to the harness constant, in both directions. Recomputing
    # the sigmas from the record's own numbers cannot catch a misquoted citation: the
    # arithmetic stays self-consistent whatever the counts say. Parsed rather than
    # imported so this does not need Open3D installed.
    cited = next(
        ast.literal_eval(node.value)
        for node in ast.parse(
            (EVIDENCE / "harness" / "up_axis_ambiguity.py").read_text()
        ).body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", None) == "INDEPENDENT_RUN_DETECTIONS"
    )
    assert list(reproduction["detections_they_reported"]) == list(cited), (
        "the record's citation of the independent run has drifted from the figures "
        f"the harness computes against: {reproduction['detections_they_reported']} "
        f"in the record, {list(cited)} in the harness"
    )
    # Recomputed from the two counts rather than read back, so a published sigma that
    # does not follow from the proportions beside it fails here.
    for row in reproduction["per_configuration"]:
        mine, theirs = row["this_run"], row["the_independent_run"]
        pooled = (mine + theirs) / 2.0
        error = (2.0 * pooled * (1.0 - pooled) / reproduction["runs_each"]) ** 0.5
        if error == 0:
            assert row["difference_in_sigmas"] is None, row
            continue
        assert row["difference_in_sigmas"] == pytest.approx(
            (theirs - mine) / error, rel=0.02
        ), row

    assert reproduction["widest_disagreement_in_sigmas"] < 3.0, (
        "the independent run no longer agrees with this one, which is a finding about "
        f"the measurement rather than a detail of the record: {reproduction}"
    )
    where = reproduction["where_the_two_runs_are_softest"]
    assert "seventeen" in where, where
    assert "rests on the lower step" in where, where


def test_every_string_the_tests_demand_comes_from_the_harness() -> None:
    """A disclosure only a hand-edit carries is dropped by the next regeneration.

    `how_this_sample_was_sized` was the reviewer's condition for publishing the
    pooled z, and it sat in the record while no producer emitted it. The test that
    demands it passed against a file nothing reproduces, so regenerating removed the
    disclosure and failed the test -- which is how this was found. Checking the
    harness source for the keys this file asserts on is cheap and catches the shape
    rather than the instance.
    """

    # The harness and the product both legitimately produce keys here: the harness
    # writes its own measurements, and it quotes the runner's `up_axis_inference`
    # output verbatim rather than restating it. Either is a producer. A key that is in
    # neither is a hand-edit.
    import npa.workbench.open3d

    producers = "".join(
        path.read_text()
        for path in [EVIDENCE / "harness" / "up_axis_ambiguity.py"]
        + sorted((Path(npa.workbench.open3d.__file__).parent).glob("*.py"))
    )
    asserted = set(
        re.findall(r'\[\s*"([a-z][a-z0-9_]{6,})"\s*\]', Path(__file__).read_text())
    )
    published = json.dumps(_record())
    missing = sorted(
        key
        for key in asserted
        if f'"{key}"' not in producers and f'"{key}"' in published
    )
    assert not missing, (
        "these keys are asserted by this file and present in the record, but nothing "
        f"produces them, so a regeneration drops them: {missing}"
    )


def test_the_soft_step_is_named_rather_than_floored() -> None:
    """The record must not present two steps of different firmness as one result.

    `smallest_step_in_sigmas` was gated at 3.0, and an independent reproduction came
    back at 2.91 on the upper step. The claim that survives both runs is the ordering
    everywhere and a strong bound only on the step the default rests on.
    """

    rise = _record()["detection_by_repeat_count"][
        "does_detection_rise_with_the_fit_count"
    ]
    caveat = rise["not_every_step_is_equally_firm"]
    assert "below three" in caveat, caveat
    assert "flatters" in caveat or "optimistic" in caveat, caveat
    assert "Nothing shipped depends on that upper step" in caveat, caveat

    # Checked in both directions. Only the first was asserted at first, and it has no
    # teeth in a record where no rate reached 1.0: the loop ran zero times and a flag
    # hardcoded to False passed. The converse is the one that matters, because the
    # flag exists to warn that a perfect proportion contributes nothing to the pooled
    # standard error and so flatters the sigma published beside it.
    rate_of = dict(zip(rise["fits_per_configuration"], rise["detection_rate"]))
    for pair in rise["consecutive_pairs"]:
        endpoints = (rate_of[pair["from_fits"]], rate_of[pair["to_fits"]])
        assert pair["an_endpoint_detected_every_run"] == (1.0 in endpoints), (
            f"the flag disagrees with the rates it summarises: {endpoints} against "
            f"{pair['an_endpoint_detected_every_run']}"
        )


def test_the_record_publishes_a_detection_rate_rather_than_a_guarantee() -> None:
    """The check misses sometimes, and the record has to say so with numbers.

    Asserted against the 300-run block, not the thirty-run sweep. At a true rate near
    0.92 a sweep of thirty returns a clean 30 of 30 about once in twelve, and one
    did: requiring the sweep to show a miss made the published limitation depend on
    the measurement being unlucky. 276 of 300 is the same claim with the power to
    hold it, and it is below perfect by many times its own noise.
    """

    detection = _record()["detection_by_repeat_count"]
    rates = detection["rates"]
    assert len(rates) >= 3, rates
    default = [one for one in rates if one["is_the_shipped_default"]]
    assert len(default) == 1, "the sweep does not mark the shipped default"

    rise = detection["does_detection_rise_with_the_fit_count"]
    shipped = rise["fits_per_configuration"].index(default[0]["fits"])
    detected = rise["detections"][shipped]
    of_runs = rise["runs_per_configuration"]
    assert detected < of_runs, (
        "the retained record shows perfect detection at the shipped count over "
        f"{of_runs} runs, which would make the published limitation unsupported by "
        "its own evidence"
    )
    # Below perfect by more than noise, so the claim does not rest on a single miss.
    # One sigma of the count at this rate is sqrt(n p (1-p)).
    rate = detected / of_runs
    noise = (of_runs * rate * (1.0 - rate)) ** 0.5
    assert of_runs - detected > 3.0 * noise, (detected, of_runs, noise)
    # This block deliberately does not require the thirty-run counts to be ordered.
    # They are not, reliably: one regeneration returned 27 detections at five fits
    # against 26 at nine. The scatter is this record's subject, and the ordering is
    # measured at 300 runs per configuration in
    # `does_detection_rise_with_the_fit_count`.
    assert all(
        0 <= one["ambiguity_detected_in_runs"] <= one["of_runs"] for one in rates
    )


def test_the_runner_up_separation_is_an_ordering_with_a_narrow_margin() -> None:
    """The ordering survives a regeneration. The width of it does not.

    Exact ranges were quoted here first and drifted twice, so this became the
    ordering plus bounds "loose enough to mean something": above 0.95 and below 0.90.
    The next regeneration returned a determined cloud at 0.904 and broke the second
    bound, which was the useful answer -- the margin between the groups is about 0.05
    and the determined endpoint alone has moved 0.866, 0.878, 0.904, so a bound of
    that kind was never loose enough and the separation is not wide.

    What is asserted here is the ordering, which has held every time, and that the
    margin is published rather than implied. Reinstating an absolute bound a third
    time would be pinning a figure this record exists to say does not reproduce.
    """

    ambiguous, determined = [], []
    for case in _record()["cases"]:
        band = case["what_the_pipeline_publishes"]["runner_up_relative_to_winner_range"]
        if case["single_fit_distribution"]["distinct_axes"] > 1:
            ambiguous.append(band)
        else:
            determined.append(band)
    assert len(ambiguous) >= 2 and len(determined) >= 2, (
        len(ambiguous),
        len(determined),
    )

    lowest_ambiguous = min(band[0] for band in ambiguous)
    highest_determined = max(band[1] for band in determined)
    assert lowest_ambiguous > highest_determined, (
        "the ambiguous and determined runner-up ratios now overlap, so the separation "
        f"the code describes no longer holds: {lowest_ambiguous} vs {highest_determined}"
    )
    # How wide, not just which way round. A margin this narrow is the reason the
    # source must not describe the two groups as well separated, and the runtime
    # string a consumer reads is where that correction has to land.
    margin = lowest_ambiguous - highest_determined
    assert 0.0 < margin < 0.2, (
        f"the margin is {margin}; if the groups have genuinely separated by more than "
        "the endpoints move, say so in the source before asserting it here"
    )
    from npa.workbench.open3d import runner

    source = Path(runner.__file__).read_text()
    _, _, note = source.partition('"no_threshold_is_applied_to_this": (')
    note, _, _ = note.partition("),")
    assert "as little as" in note, (
        "the note a consumer reads still describes the two groups without saying how "
        f"narrow the gap is: {note}"
    )
    assert "no overlap" not in note, (
        "the earlier claim of a clean separation is back in the note a consumer reads"
    )


def test_the_chance_agreement_model_is_tested_at_power_not_asserted() -> None:
    """The record asserted this model, then withdrew it, and now measures it.

    The withdrawal was justified by a per-fit minority rate that moved between
    processes. It does not: the probe that found the movement ran its repeats in one
    interpreter while being named for separate ones, and at forty fits per estimate
    it could not have separated movement from counting noise in either direction.
    The replacement spawns twelve interpreters at four hundred fits each, and the
    assertions below are about its structure and its power rather than its numbers.
    """

    detection = _record()["detection_by_repeat_count"]
    high = detection["independence_of_the_fits"][
        "does_the_minority_rate_vary_between_processes"
    ]
    assert high["each_process_is_a_fresh_interpreter"] is True, high
    assert high["processes"] >= 12, high
    assert high["fits_attempted_per_process"] >= 400, high

    # The homogeneity statistic, recomputed from the per-process counts the record
    # publishes. Asserting the published value against itself would pass on a record
    # that printed a constant.
    pooled = high["pooled_per_fit_minority"]
    rebuilt = sum(
        (minority - pooled * usable) ** 2 / (pooled * usable * (1.0 - pooled))
        for minority, usable in (
            (round(rate * usable), usable)
            for rate, usable in zip(
                high["per_fit_minority_per_process"],
                high["usable_fits_per_process"],
                strict=True,
            )
        )
    )
    assert high["chi_square"] == pytest.approx(rebuilt, rel=0.02), high
    assert high["degrees_of_freedom"] == high["processes"] - 1, high
    # One fixed rate across processes. The earlier probe's 0.2-to-0.375 spread would
    # be about eight sigma here, so this is a measurement with the power to disagree.
    assert high["homogeneity_rejected"] is False, high
    assert high["chi_square_per_degree_of_freedom"] < 3.0, high
    assert high["one_sigma_of_counting_noise_at_that_rate"] < 0.03, high

    model = high["and_does_it_predict_the_miss_rate"]
    assert model["checks_run"] >= 3000, model
    assert model["fits_per_check"] == 9, model
    # Whether this model survives is a result, not a property of the code, and it has
    # already changed sign between runs: the gap was within noise when this block was
    # written and is -3.73 in the recorded run. So assert the comparison is reported
    # honestly rather than that it came out favourably.
    gap = (
        model["misses_observed"]
        - model["misses_predicted_by_each_process_own_minority_rate"]
    ) / model["one_sigma_of_the_prediction"]
    assert model["gap_in_sigmas"] == pytest.approx(gap, abs=0.02), model

    # The gap's direction across runs, derived from the list rather than described.
    # One run cannot separate a tenth-sized bias from noise, so a record that only
    # ever showed the latest gap would keep inviting the reading that this block
    # records getting wrong.
    pooled = model["where_the_sign_of_this_gap_stands"]
    gaps = pooled["gaps_in_sigmas"]
    assert len(gaps) >= 6, gaps
    assert pooled["this_runs_gap_is_the_last_of_them"] is True, pooled
    assert gaps[-1] == pytest.approx(model["gap_in_sigmas"], abs=0.01), (gaps, model)
    assert pooled["mean_gap_in_sigmas"] == pytest.approx(
        sum(gaps) / len(gaps), abs=0.01
    ), pooled
    assert pooled["pooled_z"] == pytest.approx(
        pooled["mean_gap_in_sigmas"] * len(gaps) ** 0.5, rel=0.02
    ), pooled
    # The sample size depended on the data in it: the block was run again because
    # the first gaps looked like a bias. That weakens the z, in the direction of
    # overstating it, and it has to sit in the same object as the number rather than
    # in a paragraph a reader of the record may never see.
    assert "adaptively" in pooled["how_this_sample_was_sized"], pooled

    # The rejection line is published next to the statistic it judges, and both the
    # per-run flag and the pooled flag follow from it. The record previously carried
    # the line as a bare 3.0 in the code and "a couple of sigma" in the prose beside a
    # z of -3.01: the prose was written once and the number kept moving.
    line = pooled["rejection_line_in_sigmas"]
    assert line == pytest.approx(3.0), pooled
    assert pooled["runs_pooled"] == len(gaps), pooled
    assert pooled["pooled_model_rejected"] is (abs(pooled["pooled_z"]) > line), pooled
    assert model["model_rejected"] is (abs(model["gap_in_sigmas"]) > line), (
        model,
        line,
    )
    # And the sentence agrees with the flag, in whichever direction the flag points.
    assert f"{len(gaps)} runs" in pooled["so"], pooled
    rejecting = "at or past the three-sigma line" in pooled["so"]
    assert rejecting is pooled["pooled_model_rejected"], pooled

    # Rejected here means fewer misses than independent fits would give, so the check
    # fires more often than the model predicts. The record has to say which direction,
    # because the other one would mean the shipped repeat count is optimistic.
    if pooled["pooled_model_rejected"]:
        assert pooled["pooled_z"] < 0, pooled
        assert (
            model["misses_observed"]
            < model["misses_predicted_by_each_process_own_minority_rate"]
        ), model
        assert "fewer misses observed than predicted" in pooled["negative_means"]

    # Admissible, not confirmed, and the record has to keep saying which.
    assert "not" in high["what_this_still_does_not_establish"]


def test_the_superseded_probe_is_relabelled_rather_than_quietly_dropped() -> None:
    """Its spread is why the high-power probe exists, so a reader should see it.

    Two things were wrong with it: the name claimed separate processes it never
    spawned, and forty fits could not decide the question. The record keeps it under
    a name that matches what it did.
    """

    independence = _record()["detection_by_repeat_count"]["independence_of_the_fits"]
    assert "across_processes" not in independence, (
        "the probe that never spawned a process must not be published under that name"
    )
    within = independence["repeats_within_one_process"]
    low, high_rate = within["per_fit_minority_range"]
    assert low < high_rate, "a single minority value is not a range"
    assert within["spread_in_sigmas"] == pytest.approx(
        (high_rate - low) / within["one_sigma_of_counting_noise_at_that_rate"], rel=0.02
    ), within
    assert "one process" in within["so"], within["so"]


def test_the_runner_up_measurement_is_published_without_a_threshold() -> None:
    """It is evidence for a reader. Five clouds cannot place a safety line."""

    block = agreeing_report()["up_axis_inference"]["agreement"]["runner_up_plane"]
    assert "inliers_relative_to_the_winner" in block
    clouds = {one["cloud"]: one for one in _record()["cases"]}
    ambiguous = clouds["ambiguous_room"]["what_the_pipeline_publishes"][
        "runner_up_relative_to_winner_range"
    ]
    determined = [
        clouds[name]["what_the_pipeline_publishes"][
            "runner_up_relative_to_winner_range"
        ]
        for name in ("floor_four_to_one", "floor_ten_to_one", "real_demo_scan")
    ]
    assert all(one is not None for one in [ambiguous, *determined])
    assert min(ambiguous) > max(high for _, high in determined), (
        "the ambiguous cloud's runner-up no longer sits above every determined "
        "cloud's, so the record's separation claim needs re-measuring"
    )
    assert (
        "no_threshold_is_applied_to_this" in str(_record()["what_this_does_not_claim"])
        or "gates nothing" in _record()["what_this_does_not_claim"]
    ), "the record should say the runner-up number gates nothing"


def test_the_record_binds_the_harness_that_produced_it() -> None:
    """A record nobody can re-run is the one that most needs its producer named.

    The fit distributions here are not reproducible -- that is the finding -- so a
    reader cannot check this file by regenerating it and comparing numbers. The usual
    argument for a harness hash is therefore inverted and stronger: the only thing left
    to pin is which code produced it, and without the hash the harness could be edited
    and this file left stale with nothing anywhere to notice.

    The pattern is copied from density-crop-contribution.json rather than invented.
    """
    provenance = _record()["provenance"]
    assert "this evidence directory" in provenance["path_is_relative_to"]

    harness = EVIDENCE / provenance["harness"]
    assert harness.exists(), harness
    assert (
        hashlib.sha256(harness.read_bytes()).hexdigest() == provenance["harness_sha256"]
    ), "the harness changed without the record being regenerated"


def test_the_margin_rule_is_set_where_a_bound_that_merely_satisfies_it_is_safe() -> (
    None
):
    """The level, which is a different question from the scale and was wrong longer.

    A floor on margin is a statement about the worst bound it permits, not about the
    bound that happens to be shipped. At a tenth of the range the worst permitted bound
    is crossed 26.3% of the time by a later regeneration, measured by a second lane over
    24 of them; the shipped floor survived only because it carried half again the margin
    the rule asked for.

    So this asserts the two things that made the level what it is, both of which a
    future edit could quietly break: that the published bound carries the margin the
    record's own sweep justifies, and that it still clears the degrees keeping this a
    claim about a badly tilted plane. Those squeeze from opposite sides, and at this
    record's spread they leave no room above 0.359 -- which is why the 0.5 where the
    crossing rate first reaches zero is not available here, and why the residual is
    2.9% rather than nothing.

    That tension is asserted rather than applied as a silent cap. If a later
    regeneration widens the spread enough that both could be satisfied at once, this
    fails and asks for the level to be revisited, instead of relaxing itself.
    """
    rule = _record()["the_margin_rule_the_bounds_must_satisfy"]
    case = next(
        one for one in _record()["cases"] if one["cloud"] == "armadillo_sampled"
    )
    branch = case["single_fit_distribution"]["branches"][0]
    low, high = branch["tilt_to_snapped_axis_degrees_range"]
    spread = high - low
    floor = _counterexample()["plane_tilt_to_the_snapped_axis_degrees"]
    required = _clearance_the_record_justifies()

    carried = (low - floor) / spread
    assert carried >= required, (
        f"the published floor {floor} carries {carried:.3f} of the {spread:.3f} range "
        f"the record's fits span, below the {required} its sweep justifies"
    )

    rates = rule["crossing_rate_by_clearance"]
    residual = rates[f"{required:.2f}"]
    assert residual <= rule["tolerated_crossing_rate"], (
        f"the most margin this claim can carry is {required}, and a bound there is "
        f"still crossed {residual:.1%} of the time -- past the "
        f"{rule['tolerated_crossing_rate']:.1%} this record says it will tolerate. "
        "That is a result about the evidence, not a tolerance to raise"
    )
    assert rates[f"{0.0:.2f}"] > 0.4, (
        "the sweep's zero-margin control should come out near the 50% symmetry demands; "
        f"{rates[f'{0.0:.2f}']} means the sweep measured something else"
    )


def _public_report_deriver() -> Any:
    """The public-run selector, loaded from the evidence harness it lives in."""

    import importlib.util

    path = EVIDENCE / "harness" / "derive_public_run_report.py"
    spec = importlib.util.spec_from_file_location("_derive_public_run_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leaves(node: Any, where: str = "") -> list[tuple[str, Any]]:
    """Every leaf the selector would walk, in the path notation its table uses."""

    if isinstance(node, dict):
        found: list[tuple[str, Any]] = []
        for key, value in node.items():
            found.extend(_leaves(value, f"{where}.{key}" if where else key))
        return found
    if isinstance(node, list):
        found = []
        for one in node:
            found.extend(_leaves(one, f"{where}[]"))
        return found
    return [(where, node)]


@pytest.mark.parametrize(
    ("expected_source", "axes", "weak", "runner_up"),
    [
        (
            "largest-plane-normal-snapped-to-axis",
            [[0.0, 0.0, 1.0]] * 9,
            None,
            _reading([0.0, -1.0, 0.0]),
        ),
        (
            "default-ambiguous-plane",
            [[0.0, 0.0, 1.0]] * 5 + [[1.0, 0.0, 0.0]] * 4,
            None,
            _reading([1.0, 0.0, 0.0], weak=True),
        ),
        (
            "default-weak-plane",
            [[0.0, 0.0, 1.0]] * 9,
            [False] + [True] * 8,
            None,
        ),
    ],
)
def test_every_field_the_producer_publishes_is_classified_for_a_public_run(
    monkeypatch: pytest.MonkeyPatch,
    expected_source: str,
    axes: list[list[float]],
    weak: list[bool] | None,
    runner_up: dict[str, Any] | None,
) -> None:
    """A field the producer publishes that the public-run selector has not classified
    stops the next public run.

    The selector denies by default, so a new `up_axis_inference` leaf raises rather
    than leaking: the safe direction, and still a break. Nothing bound the producer to
    the selector's table, and the whole agreement block had landed without an entry in
    it. These three cases are the producer's three published shapes, driven through the
    real `_up_axis` -- and, where the runner-up fields are reached, the real
    `_runner_up_plane`, so the names come from the function instead of from a fixture
    that could repeat the same mistake.
    """

    report = _drive_up_axis(monkeypatch, axes, weak=weak, runner_up=runner_up)
    assert report["source"] == expected_source

    derive = _public_report_deriver()
    allowed = derive.ALLOWED["reports/rrd-manifest.json"]
    leaves = _leaves({"up_axis_inference": report})
    assert leaves
    for path, value in leaves:
        # SystemExit for a leaf nobody classified, or one whose value is not the kind
        # its name was allowed for.
        derive._classify(path, value, allowed)


def test_the_weak_plane_branch_publishes_its_sample_outside_the_agreement_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its counts are on the inference object, because in that branch nothing agreed.

    Allowing them as `agreement.usable_fits` would have classified nothing: the
    producer never emits that path, and the selector would still have failed closed on
    the real one. The test above would pass either way, so the location is pinned here.
    """

    report = _drive_up_axis(
        monkeypatch, [[0.0, 0.0, 1.0]] * 9, weak=[False] + [True] * 8
    )
    assert report["source"] == "default-weak-plane"
    assert "agreement" not in report

    emitted = {path for path, _ in _leaves({"up_axis_inference": report})}
    assert "up_axis_inference.usable_fits" in emitted
    assert "up_axis_inference.attempted_fits" in emitted

    allowed = _public_report_deriver().ALLOWED["reports/rrd-manifest.json"]
    assert "up_axis_inference.agreement.usable_fits" not in allowed


def test_a_boolean_is_not_accepted_where_a_count_belongs() -> None:
    """`isinstance(True, int)` is True, and two of the three counts relied on that.

    `attempted_fits` already rejected it. So a record could carry `distinct_axes: true`
    or `repeats: true`, be read as a count of one everywhere downstream, and validate --
    while the same value in the field beside them did not.
    """

    for field in ("distinct_axes", "repeats"):
        report = agreeing_report()
        report["up_axis_inference"]["agreement"][field] = True
        with pytest.raises(Open3dError, match="positive|count|whole"):
            validate_up_axis_inference(report)


def test_one_fit_cannot_constitute_agreement() -> None:
    """A majority of the attempts is a ratio, and a ratio is satisfied by 1 of 1.

    `_up_axis(..., repeats=0)` produces exactly that report: one fit, no agreement
    possible, published as a measured axis. It passed every guard here. The product
    call site uses the shipped default, so nothing live produced it, which is not what
    this validator is for.
    """

    report = agreeing_report()
    agreement = report["up_axis_inference"]["agreement"]
    agreement["repeats"] = 1
    agreement["attempted_fits"] = 1
    agreement["candidates"] = [{"up": [0.0, 0.0, -1.0], "chosen_by_repeats": 1}]
    with pytest.raises(Open3dError, match="one fit cannot agree"):
        validate_up_axis_inference(report)

    # Two is the floor, not more: two repeats are the fewest that can agree, and
    # anything above that would be this validator setting the producer's fit count.
    agreement["repeats"] = 2
    agreement["attempted_fits"] = 2
    agreement["candidates"] = [{"up": [0.0, 0.0, -1.0], "chosen_by_repeats": 2}]
    validate_up_axis_inference(report)


def test_the_producer_can_actually_emit_the_report_that_guard_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reachability, so the guard above is not defending against a hand-written shape."""

    report = _drive_up_axis(monkeypatch, [[0.0, 0.0, 1.0]])
    assert report["source"] == "largest-plane-normal-snapped-to-axis"
    assert report["agreement"]["attempted_fits"] == 1
    with pytest.raises(Open3dError, match="one fit cannot agree"):
        validate_up_axis_inference({"up_axis_inference": report})


def test_the_published_axis_has_to_be_the_one_the_repeats_agreed_on() -> None:
    """The counts were all constrained; which axis got published was not.

    So a report naming a measured up of (0,0,1), whose single agreeing candidate is
    (1,0,0), satisfied every check. The vector the view cameras consume could differ
    from the axis the agreement block attests to, with the record still valid.
    """

    report = agreeing_report()
    report["up_axis_inference"]["agreement"]["candidates"] = [
        {"up": [1.0, 0.0, 0.0], "chosen_by_repeats": 9}
    ]
    with pytest.raises(Open3dError, match="agreed on"):
        validate_up_axis_inference(report)

    # Written as floats by one producer and ints by another is not a disagreement.
    report["up_axis_inference"]["up"] = [0, 0, -1]
    report["up_axis_inference"]["agreement"]["candidates"] = [
        {"up": [0.0, 0.0, -1.0], "chosen_by_repeats": 9}
    ]
    validate_up_axis_inference(report)


def test_the_ambiguous_branch_is_still_bound_to_the_declared_default() -> None:
    """The binding above is for the measured branch; this one already had its own.

    Asserting it here keeps the new check from being read as covering both, and would
    catch the measured binding being applied to the wrong branch.
    """

    report = ambiguous_report()
    assert report["up_axis_inference"]["agreement"]["distinct_axes"] == 2
    validate_up_axis_inference(report)

    report["up_axis_inference"]["up"] = [1.0, 0.0, 0.0]
    with pytest.raises(Open3dError):
        validate_up_axis_inference(report)


def test_a_candidate_that_is_not_a_report_fails_the_machine_readable_contract() -> None:
    """It raised `AttributeError` from the `.get` below, which the CLI does not catch.

    The CLI maps `(Open3dError, ValueError)` to an exit code and a message. Everything
    else is a traceback, so this one malformed shape broke the contract the rest of the
    validator keeps.
    """

    report = agreeing_report()
    report["up_axis_inference"]["agreement"]["candidates"] = ["(0,0,-1)"]
    with pytest.raises(Open3dError, match="not a report"):
        validate_up_axis_inference(report)
