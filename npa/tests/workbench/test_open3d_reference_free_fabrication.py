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

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
REPORT = EVIDENCE / "reference-free-fabrication.json"
CAPABILITY = EVIDENCE / "capability-record-reference-free-fabrication.json"
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


def test_historical_control_literals_are_not_presented_as_recomputed(
    report: dict,
) -> None:
    instrument = report["construction_validity_instrument"]
    retained = instrument["retained_literal_controls_not_recomputed_by_this_harness"]
    assert retained["three_boxes_added_together_my_first_slot"] > 0.2
    assert retained["extruded_u_profile_used_here"] == 0.0
    assert "retained literals" in report["report_authorship"]["control_limit"]


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
def test_both_candidates_read_zero_on_the_unlocalized_slot_rows(
    field: str, bridged: list[dict]
) -> None:
    """The inversion is retained without inventing an unsupported roof/floor mechanism."""
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


def recomputed_criteria(
    report: dict, clean: list[dict], bridged: list[dict]
) -> dict[str, dict[str, bool]]:
    """The five frozen criteria, recomputed from the published rows, for each candidate.

    One definition deliberately. This verdict is now stated in four places -- a summary count,
    per-criterion flags, the reading each criterion quotes, and a sentence in the capability
    record -- and a fourth copy of the arithmetic that checks them would reproduce, inside the
    guard, exactly the defect the guard exists to catch.
    """
    control = next(
        row for row in clean if row["case"].startswith("dense on five faces")
    )
    floor = next(row for row in clean if row["case"].startswith("even coverage"))
    sweep = report["sparse_density_sweep_zero_invented_everywhere"]

    return {
        name: {
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
        for name, field in (("candidate_a", CANDIDATE_A), ("candidate_b", CANDIDATE_B))
    }


def test_the_recorded_verdict_matches_a_verdict_recomputed_from_the_rows(
    report: dict, clean: list[dict], bridged: list[dict]
) -> None:
    """The report states a pass count per candidate. This recomputes both and compares.

    Written because a report asserting its own validity is the failure mode this lane has
    corrected several times.
    """
    expected = {
        name: sum(flags.values())
        for name, flags in recomputed_criteria(report, clean, bridged).items()
    }

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
    blocks = {
        "candidate_a_misoriented_against_nearest_samples": "candidate_a",
        "candidate_b_no_orientation_consistent_sample_within_12_voxels": "candidate_b",
    }
    assert set(blocks) <= set(verdict), sorted(verdict)

    all_recomputed = recomputed_criteria(report, clean, bridged)
    for block, candidate in blocks.items():
        criteria = verdict[block]
        recomputed = all_recomputed[candidate]
        assert set(criteria) == set(recomputed), (block, sorted(criteria))
        for name, expected_flag in recomputed.items():
            assert criteria[name]["met"] is expected_flag, (block, name, criteria[name])

        assert (
            sum(entry["met"] for entry in criteria.values())
            == verdict["criteria_met"][candidate]
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
    assert "slotted_box(0.40, 0.03)" in source
    assert "cannot call that area a roof" in source


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


def test_hand_written_report_sections_are_disclosed(report: dict) -> None:
    authorship = report["report_authorship"]
    hand_written = set(
        authorship["hand_written_annotations_not_emitted_by_the_harness"]
    )
    assert {
        "verdict_against_the_frozen_acceptance",
        "the_decisive_case",
        "why_both_candidates_fail_in_one_sentence_each",
        "what_this_means_for_the_product",
        "visual_evaluation",
    } == hand_written
    assert "does not reproduce" in authorship["reproduction_limit"]


def test_harness_requires_a_scratch_output_and_discloses_raw_scope() -> None:
    source = HARNESS.read_text()
    assert 'os.environ.get("OPEN3D_OUT_DIR")' in source
    assert "OPEN3D_OUT_DIR is required" in source
    assert '"report_scope"' in source
    assert '"docs/evidence/open3d"' not in source


@pytest.fixture
def capability() -> dict:
    return json.loads(CAPABILITY.read_text())


def prose_leaves(node: object, path: str = "") -> list[tuple[str, str]]:
    """Every string leaf, with the path that reaches it."""
    if isinstance(node, dict):
        return [
            leaf
            for key, value in node.items()
            for leaf in prose_leaves(value, f"{path}/{key}")
        ]
    if isinstance(node, list):
        return [
            leaf
            for index, value in enumerate(node)
            for leaf in prose_leaves(value, f"{path}[{index}]")
        ]
    return [(path, node)] if isinstance(node, str) else []


def numbers_in(node: object) -> set[float]:
    """Every number anywhere in a record."""
    if isinstance(node, dict):
        return {value for item in node.values() for value in numbers_in(item)}
    if isinstance(node, list):
        return {value for item in node for value in numbers_in(item)}
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return set()
    return {float(node)}


#: Figures the capability record quotes that the report does not hold, each from a measurement
#: that was not carried into it: the discarded construction's invented fraction, the sphere
#: that failed to exercise the effect, and a diagnostic neighbourhood radius. Pinning the set
#: rather than counting it means a *bound* figure that stops resolving shows up here by name.
QUOTED_BUT_NOT_IN_THE_REPORT = {"0.1041", "0.0004", "0.0697"}


def test_the_sentence_a_human_reads_first_agrees_with_the_recomputed_counts(
    report: dict, clean: list[dict], bridged: list[dict], capability: dict
) -> None:
    """The verdict exists a fourth time, in prose, in the record a reader opens first.

    Review found it after the three guarded copies were closed. They replaced

        "Candidate A meets 1 of 5 criteria; candidate B meets 2 of 5. Both rejected."

    with "4 of 5 ... 5 of 5. Both accepted." and all 137 tests across the three Open3D test
    files passed. I reproduced that before fixing it. A rejection that can turn into an
    acceptance in the summary artifact, silently, is worse than drift inside the report,
    because this is the sentence most readers will see instead of the rows.

    So the numerals are parsed out and compared against the counts recomputed from the rows,
    and the verdict word is held to them as well.
    """
    sentence = capability["acceptance_was_frozen_first"]["result"]
    stated = re.findall(r"candidate ([ab]) meets (\d+) of (\d+)", sentence, re.I)
    assert len(stated) == 2, sentence

    recomputed = recomputed_criteria(report, clean, bridged)
    for letter, met, out_of in stated:
        candidate = f"candidate_{letter.lower()}"
        flags = recomputed[candidate]
        assert int(out_of) == len(flags), (sentence, candidate)
        assert int(met) == sum(flags.values()), (sentence, candidate, flags)

    accepted = [name for name, flags in recomputed.items() if all(flags.values())]
    if accepted:
        assert "accepted" in sentence.lower(), (sentence, accepted)
    else:
        assert "rejected" in sentence.lower(), sentence
        assert "accepted" not in sentence.lower(), sentence


def test_every_figure_the_capability_record_quotes_is_still_a_figure_the_report_holds(
    report: dict, capability: dict
) -> None:
    """The verdict sentence was one instance of a class: prose quoting the report's numbers.

    The capability record paraphrases the report throughout -- gate 1's margin, the decisive
    case's readings, the climb as coverage thins -- so regenerating the report would leave
    every one of those sentences quoting a measurement that no longer exists, and nothing
    would say so.

    This pins the exception set rather than a count, which is the stricter of the two: three
    quoted figures come from measurements the report does not carry, and if a figure that
    *is* backed by the report ever stops resolving, it joins that set by name and this fails.
    """
    held = numbers_in(report)
    unresolved = set()
    resolved = 0
    for _, text in prose_leaves(capability):
        for literal in set(re.findall(r"(?<![\w.])0\.\d{4,}(?![\w.])", text)):
            decimals = len(literal.split(".")[1])
            value = float(literal)
            if any(round(number, decimals) == value for number in held):
                resolved += 1
            else:
                unresolved.add(literal)

    assert unresolved == QUOTED_BUT_NOT_IN_THE_REPORT, {
        "no_longer_backed_by_the_report": sorted(
            unresolved - QUOTED_BUT_NOT_IN_THE_REPORT
        ),
        "newly_backed_so_remove_from_the_exception_set": sorted(
            QUOTED_BUT_NOT_IN_THE_REPORT - unresolved
        ),
    }
    assert resolved >= 15, resolved


def test_the_capability_record_binds_current_harness_and_report_bytes(
    capability: dict,
) -> None:
    bound = capability["provenance"]["harness_that_produced_this"]
    expected = {item["path"]: item["sha256"] for item in bound}
    assert set(expected) == {
        "harness/reference_free_fabrication.py",
        "reference-free-fabrication.json",
    }
    for relative, digest in expected.items():
        path = EVIDENCE / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def collected_guard_count(source: str) -> int:
    """How many test cases this file yields, counting a parametrized test once per case.

    The distinction is not pedantic and I got it wrong first. Three tests here are
    parametrized over the two candidates, so at the head review examined this file held 14
    functions and ran 17 cases. The record's count is 17, meaning cases. My first version of
    the test below counted `^def test_` -- 17 after I added three functions -- and it passed
    against a record that meant something else entirely. A guard that agrees for the wrong
    reason is worse than no guard, because it looks like evidence.
    """
    total = 0
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        cases = 1
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and getattr(decorator.func, "attr", "") == "parametrize"
                and len(decorator.args) >= 2
                and isinstance(decorator.args[1], (ast.List, ast.Tuple))
            ):
                cases *= len(decorator.args[1].elts)
        total += cases
    return total


def test_the_record_counts_the_guards_it_actually_has(capability: dict) -> None:
    """The record states how many regression guards exist, which is a number that rots.

    It said 17 while the file ran 17, and the only reason it was right is that nobody had
    added one since. Counting them makes the claim true by construction.
    """
    guards = capability["regression_guards"]
    actual = collected_guard_count(Path(__file__).read_text())
    assert guards["count"] == actual, (guards["count"], actual)
    assert guards["file"] == f"npa/tests/workbench/{Path(__file__).name}"
    assert guards["what_count_means"] == "collected test cases, not test functions"
