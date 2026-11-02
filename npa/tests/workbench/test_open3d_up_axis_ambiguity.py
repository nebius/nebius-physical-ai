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


def _drive_up_axis(monkeypatch: pytest.MonkeyPatch, axes: list[list[float]]) -> dict:
    """Run the real `_up_axis` over a scripted sequence of plane fits.

    The validator tests below check reports that already exist. This drives the code
    that *writes* them, with the Open3D calls replaced by a script, so the decision
    logic is covered where CI runs rather than only in the functional smoke.
    """

    from npa.workbench.open3d import runner

    readings = [
        {
            "weak": False,
            "up": axis,
            "measured_plane_normal": axis,
            "angle_to_snapped_axis_degrees": 0.4,
            "plane_inliers": 8100,
            "plane_inlier_fraction": 0.5,
            "inlier_indices": [0, 1, 2],
        }
        for axis in axes
    ]
    supply = iter(readings)
    monkeypatch.setattr(
        runner, "_plane_up_reading", lambda *_args, **_kwargs: next(supply)
    )
    monkeypatch.setattr(
        runner,
        "_runner_up_plane",
        lambda *_args, **_kwargs: {
            "measured": True,
            "inliers_relative_to_the_winner": 0.9,
        },
    )
    return runner._up_axis(object(), object(), 0.05, repeats=len(axes) - 1)


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

    # Bounds, checked by direction. Equality against the measured extremes is what
    # went stale here: these fits are not reproducible, so the record's minimum moves
    # on every regeneration and a constant quoting it is wrong by the next run.
    low, _high = branches[0]["tilt_to_snapped_axis_degrees_range"]
    assert published["this_tilt_is_a_floor_the_measurement_cleared"] is True
    assert low >= published["plane_tilt_to_the_snapped_axis_degrees"], (
        f"the record's smallest tilt {low} fell below the "
        f"{published['plane_tilt_to_the_snapped_axis_degrees']} the code publishes as "
        "a floor, so the published claim is no longer covered by its evidence"
    )
    _low, fraction_high = branches[0]["plane_inlier_fraction_range"]
    assert fraction_high <= published["plane_inlier_fraction_was_at_most"], (
        f"the record's largest inlier fraction {fraction_high} exceeded the "
        f"{published['plane_inlier_fraction_was_at_most']} the code publishes"
    )
    # The bound must still be worth stating: a floor of zero would always pass.
    assert published["plane_tilt_to_the_snapped_axis_degrees"] >= 10.0, published


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
    detected = [one["ambiguity_detected_in_runs"] for one in rates]
    assert UP_AXIS_DETECTION_RISES_WITH_FITS is True
    assert detected == sorted(detected), (
        "detection no longer rises with the fit count, which is the structural claim "
        f"the repeat default rests on: {detected}"
    )
    # The floor has to be worth clearing: a floor of zero would always pass, and a
    # floor at the observed figure would go stale on the next sweep, which is what
    # happened to the exact counts this replaced.
    assert 0 < UP_AXIS_DETECTION_FLOOR_OF_THIRTY < detected_at_default


def test_the_record_publishes_a_detection_rate_rather_than_a_guarantee() -> None:
    """The check misses sometimes, and the record has to say so with numbers."""

    rates = _record()["detection_by_repeat_count"]["rates"]
    assert len(rates) >= 3, rates
    default = [one for one in rates if one["is_the_shipped_default"]]
    assert len(default) == 1, "the sweep does not mark the shipped default"
    assert default[0]["ambiguity_detected_in_runs"] < default[0]["of_runs"], (
        "the retained record shows perfect detection at the default, which would make "
        "the published limitation unsupported by its own evidence"
    )
    by_count = sorted((one["fits"], one["ambiguity_detected_in_runs"]) for one in rates)
    assert [count for _, count in by_count] == sorted(count for _, count in by_count), (
        f"detection should not fall as fits increase: {by_count}"
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
    assert abs(model["gap_in_sigmas"]) <= 3.0, model
    assert model["model_rejected"] is False, model
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
