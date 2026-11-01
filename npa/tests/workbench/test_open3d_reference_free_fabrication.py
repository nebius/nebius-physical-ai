"""A negative result needs guarding as much as a positive one, and for a different reason.

The finding is that no reference-free reading tried here separates invented surface from
sparse correct coverage, so the runner's withdrawal of that claim stands. The risk to a
negative result is not that someone contradicts it; it is that someone adds a measure like
these without re-establishing the gates it failed, because the record said "rejected" and the
numbers that rejected it were never checkable.

So these tests recompute the verdict from the published rows. The report's own conclusion
fields are never the thing asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
REPORT = EVIDENCE / "reference-free-fabrication.json"
HARNESS = EVIDENCE / "harness" / "reference_free_fabrication.py"

CANDIDATE_A = "misoriented_far_share_of_unsupported_area"
CANDIDATE_B = "orientation_unsupported_far_share_of_unsupported_area"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def clean(report: dict) -> list[dict]:
    """Cases with nothing invented anywhere."""
    rows = report["cases"] + report["sparse_density_sweep_zero_invented_everywhere"]
    assert all(row["invented_area_fraction"] == 0.0 for row in rows)
    return rows


@pytest.fixture(scope="module")
def bridged(report: dict) -> list[dict]:
    """Cases with real invented surface, established against a withheld reference."""
    rows = report["bridged_cases_real_invented_surface"]
    assert all(row["invented_area_fraction"] > 0.0 for row in rows)
    return rows


def test_the_first_gate_is_the_products_own_counterexample(report: dict) -> None:
    """If the published 0.4169 does not reproduce, nothing measured afterwards means anything.

    The claim allowed 0.02. Recomputed here from the reported readings rather than read out of
    the report's own difference field, which would be the report checking itself.
    """
    gate = report["gate_1_the_products_counterexample_reproduces"]
    recorded = gate["what_the_product_records"]
    measured = gate["what_this_run_measures"]
    assert set(recorded) == set(measured)
    worst = max(abs(recorded[key] - measured[key]) for key in recorded)
    assert worst < 0.02, (worst, recorded, measured)


def test_the_construction_check_is_validated_and_not_merely_asserted(
    report: dict,
) -> None:
    """A reference with internal faces makes "invented" meaningless, so this is load-bearing.

    It reads 0.2203 on the construction that had the defect and 0.0 on the two that do not,
    which is what makes it an instrument rather than a claim.
    """
    validated = report["construction_validity_instrument"]["validated_against"]
    assert validated["three_boxes_added_together_my_first_slot"] > 0.2
    assert validated["extruded_u_profile_used_here"] == 0.0
    assert validated["plain_box_control"] == 0.0
    assert validated["open3d_boolean_difference_output"] > 0.0, (
        "the boolean output's internal faces are why the U is hand-built; without this the reason is lost"
    )


def test_every_reference_actually_measured_is_free_of_internal_faces(
    bridged: list[dict],
) -> None:
    """The check above is validated; this is it applied to the geometries that were used."""
    for row in bridged:
        assert row["internal_face_sample_share_of_the_reference"] == 0.0, row["case"]


@pytest.mark.parametrize("field", [CANDIDATE_A, CANDIDATE_B])
def test_the_invented_and_clean_readings_overlap(
    field: str, clean: list[dict], bridged: list[dict]
) -> None:
    """The core of the negative result, recomputed.

    If a later change made either candidate separate the two sets, this test fails and the
    record it guards is out of date -- which is the right outcome, because the record would
    then be stating a rejection that no longer holds.
    """
    highest_clean = max(row[field] for row in clean)
    lowest_invented = min(row[field] for row in bridged)
    assert lowest_invented <= highest_clean, (
        field,
        lowest_invented,
        highest_clean,
        "the candidate now separates the sets; re-run the frozen acceptance",
    )


@pytest.mark.parametrize("field", [CANDIDATE_A, CANDIDATE_B])
def test_both_candidates_read_zero_on_the_case_where_a_roof_is_parallel_to_a_floor(
    field: str, bridged: list[dict]
) -> None:
    """The decisive case, and the one most like real capture.

    A bridge over a gap in a surface is usually roughly parallel to that surface. Both
    candidates read exactly nothing on it.
    """
    slots = [row for row in bridged if "narrow slot" in row["case"]]
    assert len(slots) == 2, [row["case"] for row in bridged]
    for row in slots:
        assert row["invented_area_fraction"] > 0.05, row
        assert row[field] == 0.0, (row["case"], field, row[field])


def test_the_readings_invert_rather_than_merely_blur(
    clean: list[dict], bridged: list[dict]
) -> None:
    """Weak separation and reversed order are different findings, and this is the reversed one.

    Stated as a comparison between the product's own counterexample, where nothing is
    invented, and the least-detected invented case. For both candidates the clean case reads
    strictly higher.
    """
    control = next(
        row for row in clean if row["case"].startswith("dense on five faces")
    )
    for field in (CANDIDATE_A, CANDIDATE_B):
        least = min(row[field] for row in bridged)
        assert least < control[field], (field, least, control[field])


@pytest.mark.parametrize("field", [CANDIDATE_A, CANDIDATE_B])
def test_both_candidates_climb_as_correct_coverage_thins(
    field: str, report: dict
) -> None:
    """Criterion 4: a measure that rises with sparsity is a density detector.

    That is the error the product removed once already, so it is the criterion whose failure
    matters most. Monotone is not required -- only that thinning pushes the reading into the
    range used to flag fabrication.
    """
    sweep = report["sparse_density_sweep_zero_invented_everywhere"]
    readings = [row[field] for row in sweep]
    assert len(readings) >= 4, readings
    assert readings[-1] > 0.05, (field, readings)
    assert readings[-1] > readings[0], (field, readings)


def test_the_recorded_verdict_matches_a_verdict_recomputed_from_the_rows(
    report: dict, clean: list[dict], bridged: list[dict]
) -> None:
    """The report states a pass count per candidate. This recomputes both and compares.

    Written because a report asserting its own validity is the failure mode this lane has
    corrected several times.
    """
    control = next(
        row for row in clean if row["case"].startswith("dense on five faces")
    )
    floor = next(row for row in clean if row["case"].startswith("even coverage"))
    sweep = report["sparse_density_sweep_zero_invented_everywhere"]

    expected = {}
    for name, field in (("candidate_a", CANDIDATE_A), ("candidate_b", CANDIDATE_B)):
        met = [
            control[field] < 0.05,
            min(row[field] for row in bridged) > 0.30,
            min(row[field] for row in bridged) > max(row[field] for row in clean),
            all(row[field] < 0.05 for row in sweep),
            floor[field] < 0.01,
        ]
        expected[name] = sum(met)

    recorded = report["verdict_against_the_frozen_acceptance"]["criteria_met"]
    assert recorded["out_of"] == 5
    assert recorded["candidate_a"] == expected["candidate_a"] == 1
    assert recorded["candidate_b"] == expected["candidate_b"] == 2


def test_each_per_criterion_flag_matches_the_reading_beside_it(
    report: dict, clean: list[dict], bridged: list[dict]
) -> None:
    """The verdict is stated twice, and the test above only checked the sum.

    A reviewer set candidate A's criterion 1 `met` to true -- making the record claim that a
    reading of 0.081198 satisfies `< 0.05` -- and all fifteen tests passed, because the
    summary count was recomputed and the per-criterion booleans were not. The flags were
    correct; the exposure was that nothing held them to the readings.

    So this recomputes all ten from the rows. Two representations of one verdict cannot be
    allowed to separate, and the cheap fix is to assert both rather than to delete one: the
    per-criterion detail is the useful part of the record for a reader, and the summary is the
    useful part for a scanner.
    """
    verdict = report["verdict_against_the_frozen_acceptance"]
    control = next(
        row for row in clean if row["case"].startswith("dense on five faces")
    )
    floor = next(row for row in clean if row["case"].startswith("even coverage"))
    sweep = report["sparse_density_sweep_zero_invented_everywhere"]

    blocks = {
        "candidate_a_misoriented_against_nearest_samples": CANDIDATE_A,
        "candidate_b_no_orientation_consistent_sample_within_12_voxels": CANDIDATE_B,
    }
    assert set(blocks) <= set(verdict), sorted(verdict)

    for block, field in blocks.items():
        criteria = verdict[block]
        recomputed = {
            "1_products_counterexample_below_0_05": control[field] < 0.05,
            "2_every_invented_case_above_0_30": min(row[field] for row in bridged)
            > 0.30,
            "3_no_overlap_between_invented_and_clean": (
                min(row[field] for row in bridged) > max(row[field] for row in clean)
            ),
            "4_sparser_coverage_stays_below_0_05": all(
                row[field] < 0.05 for row in sweep
            ),
            "5_false_alarm_floor_below_0_01": floor[field] < 0.01,
        }
        assert set(criteria) == set(recomputed), (block, sorted(criteria))
        for name, expected_flag in recomputed.items():
            assert criteria[name]["met"] is expected_flag, (block, name, criteria[name])

        assert (
            sum(entry["met"] for entry in criteria.values())
            == verdict["criteria_met"][block.split("_")[0] + "_" + block.split("_")[1]]
        ), (block, "the per-criterion flags and the summary count disagree")


def test_the_readings_each_criterion_quotes_are_the_readings_it_was_judged_on(
    report: dict, clean: list[dict]
) -> None:
    """Each criterion block also quotes the number it judged, which is a third copy.

    Holding the quoted value to the row it came from closes the same class of drift one level
    down: a flag can agree with a quoted reading while the quoted reading has drifted from the
    measurement.
    """
    verdict = report["verdict_against_the_frozen_acceptance"]
    control = next(
        row for row in clean if row["case"].startswith("dense on five faces")
    )
    floor = next(row for row in clean if row["case"].startswith("even coverage"))

    for block, field in (
        ("candidate_a_misoriented_against_nearest_samples", CANDIDATE_A),
        ("candidate_b_no_orientation_consistent_sample_within_12_voxels", CANDIDATE_B),
    ):
        criteria = verdict[block]
        assert (
            criteria["1_products_counterexample_below_0_05"]["reading"]
            == control[field]
        )
        assert criteria["5_false_alarm_floor_below_0_01"]["reading"] == floor[field]


def test_the_two_constructions_that_failed_to_exercise_the_effect_are_still_recorded() -> (
    None
):
    """A construction that fails to exercise an effect is not evidence the effect is absent.

    A sphere with a cap of samples removed invents 0.0004 of its area, and a thin torus
    invents nothing; neither tests a fabrication detector. Keeping them is what stops the next
    person choosing them, and this holds the harness to saying so.
    """
    source = HARNESS.read_text()
    assert "0.0004" in source, "the sphere's invented fraction is no longer recorded"
    assert "thin torus" in source, (
        "the torus that invents nothing is no longer recorded"
    )
    assert "1.5 voxels" in source, (
        "the width at which the slot is finally roofed is not stated"
    )


def test_the_broken_slot_construction_is_recorded_with_what_it_cost() -> None:
    """It produced a plausible 0.1041 and a conclusion that was entirely an artefact.

    The number alone did not reveal it; checking the mechanism did. That is the reusable part,
    so the harness has to keep saying it rather than quietly shipping the corrected geometry.
    """
    source = HARNESS.read_text()
    assert "0.1041" in source, (
        "the fraction the broken construction produced is not recorded"
    )
    assert "internal" in source.lower()
    assert "2" in source and "percent inside the slot" in source


def test_the_record_does_not_claim_this_overturns_the_reference_based_measure(
    report: dict,
) -> None:
    """#1006 found normal correspondence works *against a reference*. This does not undo that.

    Two readings of the same idea reaching opposite conclusions is exactly the shape that gets
    miscited later, so the record has to name the difference.
    """
    note = report["what_this_means_for_the_product"][
        "where_normal_correspondence_does_work"
    ]
    assert "1006" in note
    assert "reference" in note


def test_the_visual_allowance_is_claimed_with_a_reason(report: dict) -> None:
    """Nonvisual work may mark this not_applicable, but it has to say why."""
    evaluation = report["visual_evaluation"]
    assert evaluation["status"] == "not_applicable"
    assert len(evaluation["reason"]) > 80
