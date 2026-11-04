"""Bind the figures in this PR's capability records to what measured them.

Review audited this branch by corrupting one evidence record at a time with the
full Open3D suite as the oracle, and the result fell along a clean line: every
*measurement report* the new tests read is held, and every *capability record*
is loose. One absorbed 88 corrupted fields and another 325 with no test
objecting. The measurements are the load-bearing part, but a capability record
is what a reader reaches for first, and a figure in one could invert silently.

This file closes that for the four capability records this PR adds. It does not
sweep the other forty, which review explicitly did not ask for.

Three kinds of binding, because the figures are three kinds of thing:

* A figure that *quotes* a measurement must resolve to a value in the report
  that produced it. The exceptions are pinned by name rather than counted, so a
  quoted figure that stops resolving cannot hide inside an allowance.
* A figure that is *derived* is checked against its derivation.
* A figure that *counts something in the repository* is checked against the
  thing it counts -- the test count against what pytest collects, the inventory
  against the partition that fills it, the changed-harness count against the
  list the tests parametrize over. These are the ones that rot quietly, and one
  of them already had: the harness-usage record claimed 12 regression tests
  where pytest collects 17.
"""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[3] / "docs/evidence/open3d"
TESTS = Path(__file__).resolve().parent

#: Which report measured each capability record's figures. Two of the four
#: records name no JSON file at all and a third names two, so this mapping is
#: mine rather than derived -- and `test_the_pairing_agrees_with_the_record`
#: checks it against the one record that does state its source.
QUOTED_FROM = {
    "capability-record-fabrication-measure.json": "fabrication-measure-leniency.json",
    "capability-record-band-validity-rescore.json": "band-validity-rescore.json",
}

# Exact record-to-report bindings. Matching against an unordered bag of numbers lets two
# fields be swapped while every value still appears somewhere in the report.
QUOTE_BINDINGS = {
    "capability-record-fabrication-measure.json": {
        "/the_leniency_reproduces/what_the_current_measure_says": "/bridged/unsigned_distance_only",
        "/the_leniency_reproduces/what_normal_correspondence_says": "/bridged/with_normal_correspondence",
        "/normal_correspondence_is_the_one_that_works/false_alarm_on_the_control": "/faithful/with_normal_correspondence",
        "/the_vertex_area_diagnostic_crosses_the_threshold/the_threshold": "/does_it_threaten_any_accepted_conclusion/the_threshold_that_matters",
        "/the_vertex_area_diagnostic_crosses_the_threshold/gap_counts_where_the_two_measures_disagree[0]": "/does_it_threaten_any_accepted_conclusion/gap_counts_where_the_verdict_differs[0]",
        "/the_vertex_area_diagnostic_crosses_the_threshold/gap_counts_where_the_two_measures_disagree[1]": "/does_it_threaten_any_accepted_conclusion/gap_counts_where_the_verdict_differs[1]",
        "/the_vertex_area_diagnostic_crosses_the_threshold/gap_counts_where_the_two_measures_disagree[2]": "/does_it_threaten_any_accepted_conclusion/gap_counts_where_the_verdict_differs[2]",
    },
    "capability-record-band-validity-rescore.json": {
        "/the_measurement/row_0/samples": "/measured_rows[0]/samples",
        "/the_measurement/row_0/committed_unsigned": "/measured_rows[0]/committed_unsigned",
        "/the_measurement/row_0/stricter": "/measured_rows[0]/largest_stricter_reading",
        "/the_measurement/row_1/samples": "/measured_rows[1]/samples",
        "/the_measurement/row_1/committed_unsigned": "/measured_rows[1]/committed_unsigned",
        "/the_measurement/row_1/largest_stricter_seen": "/measured_rows[1]/largest_stricter_reading",
        "/two_sources_of_variation_separated/the_separation/row_1_three_fresh_draws_unsigned_spread": "/measured_rows[1]/unsigned_this_run/spread",
        "/two_sources_of_variation_separated/the_separation/row_1_one_cloud_reconstructed_three_times_unsigned_spread": "/measured_rows[1]/one_cloud_reconstructed_three_times/unsigned_spread",
        "/provenance/sampling_seed": "/sampling_seed",
    },
}

#: Figures that are not quotations, excluded from the resolution check and
#: carried by a named test instead. Pinned by path rather than counted, and
#: required to exist, so the exclusion cannot quietly widen or outlive its
#: field. `/tests/count` is here because it counts the suite, not the
#: measurement -- and note that it *does* resolve against the report today, by
#: coincidence: 11 is also a reading. Letting the resolution check claim it
#: would be a guard agreeing for the wrong reason.
NOT_QUOTATIONS = {
    "capability-record-fabrication-measure.json": {"/tests/count"},
    "capability-record-band-validity-rescore.json": {
        "/the_measurement/row_0/absolute_gap_to_the_threshold",
    },
}

#: Files whose collected-case count is published by a record in this PR, plus
#: this file, which is where the counter itself is exercised.
COUNTED_FILES = (
    "test_open3d_reading_contract.py",
    "test_open3d_fabrication_measure.py",
    "test_open3d_harness_usage.py",
    "test_open3d_band_validity_rescore.py",
    "test_open3d_capability_record_binding.py",
)

#: Every claim of the form "this record names a test file and a count".
TEST_COUNT_CLAIMS = (
    ("capability-record-decayed-readings.json", "tests"),
    ("capability-record-fabrication-measure.json", "tests"),
    ("capability-record-harness-usage.json", "regression_tests"),
)


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text())


def numbers(value: object, path: str = "") -> list[tuple[str, float]]:
    """Every numeric leaf with its path. Booleans are not numbers here."""
    if isinstance(value, bool):
        return []
    if isinstance(value, dict):
        out: list[tuple[str, float]] = []
        for key, child in value.items():
            out.extend(numbers(child, f"{path}/{key}"))
        return out
    if isinstance(value, list):
        out = []
        for index, child in enumerate(value):
            out.extend(numbers(child, f"{path}[{index}]"))
        return out
    if isinstance(value, (int, float)):
        return [(path, float(value))]
    return []


#: Wrappers that cannot change how many items a sequence has, so a parametrize
#: argument wrapped in them has the length of what it wraps.
LENGTH_PRESERVING = {"sorted", "list", "tuple", "reversed"}
LENGTH_PRESERVING_METHODS = {"items", "keys", "values"}


def parametrize_width(argument: ast.expr, module: ast.Module, source: Path) -> int:
    """How many cases one parametrize argument expands to.

    Raises rather than guessing. Counting `def test_` is the wrong instrument
    and I have been caught by it before; the subtler version of the same trap is
    an expander that meets a form it does not understand, quietly scores the
    function as one case, and returns a number that looks right. The first
    version of this function did exactly that -- it handled only literal
    sequences, met `sorted(NEEDS_ARGUMENTS.items())`, and reported 11 where
    pytest collects 17. A wrong number returned confidently is worse than an
    error, so unknown forms are an error now.
    """
    if isinstance(argument, (ast.List, ast.Tuple)):
        return len(argument.elts)
    if isinstance(argument, ast.Name):
        return module_sequence_length(source, argument.id)
    if isinstance(argument, ast.Attribute) or (
        isinstance(argument, ast.Call) and isinstance(argument.func, ast.Attribute)
    ):
        call = argument if isinstance(argument, ast.Call) else None
        attribute = call.func if call else argument
        assert isinstance(attribute, ast.Attribute)
        if attribute.attr in LENGTH_PRESERVING_METHODS:
            return parametrize_width(attribute.value, module, source)
    if (
        isinstance(argument, ast.Call)
        and isinstance(argument.func, ast.Name)
        and argument.func.id in LENGTH_PRESERVING
        and len(argument.args) == 1
    ):
        return parametrize_width(argument.args[0], module, source)
    raise AssertionError(
        f"cannot determine the width of {ast.dump(argument)[:80]} in {source.name}"
    )


def collected_cases(path: Path) -> int:
    """How many cases pytest would collect from a file, expanding parametrize."""
    module = ast.parse(path.read_text())
    total = 0
    for node in ast.walk(module):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        cases = 1
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and getattr(decorator.func, "attr", "") == "parametrize"
                and len(decorator.args) >= 2
            ):
                cases *= parametrize_width(decorator.args[1], module, path)
        total += cases
    return total


def module_sequence_length(path: Path, name: str) -> int:
    """Length of a module-level list/tuple/dict literal, read without importing."""
    for node in ast.parse(path.read_text()).body:
        target = None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        if target != name:
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            return len(node.value.elts)
        if isinstance(node.value, ast.Dict):
            return len(node.value.keys)
    raise AssertionError(f"{name} not found in {path.name}")


def value_at(document: object, pointer: str) -> object:
    """Resolve the small JSON-pointer-plus-list-index form used above."""
    value = document
    for component in pointer.strip("/").split("/") if pointer != "/" else []:
        match = re.fullmatch(r"([^[]+)(?:\[(\d+)\])?", component)
        assert match, pointer
        assert isinstance(value, dict), (pointer, component)
        value = value[match.group(1)]
        if match.group(2) is not None:
            assert isinstance(value, list), (pointer, component)
            value = value[int(match.group(2))]
    return value


@pytest.mark.parametrize("record_name", sorted(QUOTED_FROM))
def test_every_figure_a_capability_record_quotes_is_bound_to_its_exact_report_field(
    record_name,
):
    """A value moved to a different semantic field must not satisfy the binding."""
    record = load(record_name)
    report = load(QUOTED_FROM[record_name])
    excluded = NOT_QUOTATIONS[record_name]
    present = {path for path, _ in numbers(record)}
    assert excluded <= present, excluded - present
    quoted = present - excluded
    bindings = QUOTE_BINDINGS[record_name]
    assert quoted == set(bindings), (quoted - set(bindings), set(bindings) - quoted)
    for record_path, report_path in bindings.items():
        assert math.isclose(
            float(value_at(record, record_path)),
            float(value_at(report, report_path)),
            rel_tol=1e-6,
            abs_tol=1e-9,
        ), (record_path, report_path)


def test_the_pairing_agrees_with_the_record_that_states_its_source():
    """One of the four names its report. The mapping must not contradict it."""
    record_name = "capability-record-band-validity-rescore.json"
    text = (EVIDENCE / record_name).read_text()
    assert QUOTED_FROM[record_name] in text


def test_the_gap_to_the_threshold_is_the_derivation_it_claims():
    """The one figure that is arithmetic rather than a quotation.

    `absolute_gap_to_the_threshold` does not appear in the report because it is
    the threshold minus the stricter reading, rounded. Checking it against the
    report would have failed; excluding it without checking it at all would
    leave the only derived figure in the record unguarded.
    """
    record = load("capability-record-band-validity-rescore.json")
    measurement = record["the_measurement"]["row_0"]
    threshold = 0.05
    gap = measurement["absolute_gap_to_the_threshold"]
    stricter = measurement.get("largest_stricter_seen", measurement["stricter"])
    assert math.isclose(gap, round(threshold - stricter, 5), abs_tol=5e-6), (
        gap,
        stricter,
    )


@pytest.mark.parametrize("row", ("row_0", "row_1"))
def test_the_threshold_crossing_flag_follows_from_the_row(row):
    """A boolean verdict beside the number it is a verdict about."""
    record = load("capability-record-band-validity-rescore.json")
    measurement = record["the_measurement"][row]
    readings = [
        measurement[key]
        for key in ("committed_unsigned", "stricter", "largest_stricter_seen")
        if key in measurement
    ]
    assert measurement["crosses_0_05"] is any(value >= 0.05 for value in readings)


def test_the_case_counter_agrees_with_the_collector_it_stands_in_for():
    """Validate the instrument against pytest, on every file it is used on.

    The counter exists so that the record's claim can be checked without
    spawning a collector for each assertion, which means it is a model of
    pytest's behaviour and could be wrong. It already was: before name
    resolution it reported 11 where pytest collects 17. So it is compared
    against the real collector here, once, across all four files whose counts
    this PR publishes plus this file itself.
    """
    for name in sorted(COUNTED_FILES):
        path = TESTS / name
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(path),
                "--collect-only",
                "-q",
                "--no-header",
                "-p",
                "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=path.parents[2],
        )
        found = re.search(r"(\d+) tests? collected", result.stdout)
        assert found, (name, result.stdout[-400:])
        assert collected_cases(path) == int(found.group(1)), name


@pytest.mark.parametrize(("record_name", "block"), TEST_COUNT_CLAIMS)
def test_the_claimed_test_count_is_the_count_pytest_collects(record_name, block):
    """The count a record publishes must be the suite that exists.

    This is the check that found the live defect: the harness-usage record
    claimed 12 while pytest collects 17 -- a number that matched neither the
    function count nor the case count, so it was stale rather than ambiguous.
    """
    claim = load(record_name)[block]
    path = TESTS / Path(claim["file"]).name
    assert path.exists(), claim["file"]
    assert claim["count"] == collected_cases(path)


@pytest.mark.parametrize(("record_name", "block"), TEST_COUNT_CLAIMS)
def test_each_count_says_which_thing_it_counts(record_name, block):
    """A bare number cannot distinguish functions from cases; the record must.

    Without this, a future reader reconciling 17 against 11 visible `def test_`
    lines has no way to tell which the record meant, and a guard comparing
    against the wrong one would agree for the wrong reason.
    """
    claim = load(record_name)[block]
    assert "collected test cases" in claim["what_count_means"]


def test_the_harness_baseline_partition_covers_the_whole_inventory():
    """Four disjoint outcomes over one inventory: they must add up.

    The record's headline -- six harnesses reported a missing input as an
    internal error -- is only meaningful if the six are six out of a stated
    whole. If the partition did not sum, any of the four could drift without
    contradicting the others.
    """
    record = load("capability-record-harness-usage.json")
    baseline = record["the_measured_baseline"]
    parts = (
        "ran_cleanly",
        "reported_usage_correctly",
        "blocked_on_open3d",
        "failed_with_an_internal_exception",
    )
    assert (
        sum(baseline[part] for part in parts)
        == (record["provenance"]["harnesses_inventoried"])
    )


def test_the_enumerated_failures_are_as_many_as_the_count_claims():
    record = load("capability-record-harness-usage.json")
    baseline = record["the_measured_baseline"]
    assert (
        len(baseline["the_six_and_their_messages"])
        == (baseline["failed_with_an_internal_exception"])
    )


def test_the_changed_harness_count_is_bound_to_the_enumerated_changed_files():
    """Do not compare two unrelated seven-item sets and call the count bound."""
    record = load("capability-record-harness-usage.json")
    provenance = record["provenance"]
    changed = provenance["changed_harnesses"]
    assert provenance["harnesses_changed"] == len(changed) == len(set(changed))
    assert all((EVIDENCE / "harness" / name).is_file() for name in changed)


def test_the_messages_the_record_quotes_are_the_failures_it_describes():
    """Each quoted failure names the exception class the prose blames.

    The record's argument is that a missing argument surfaced as an internal
    error. That argument is carried entirely by these six strings, which nothing
    else in the suite reads.
    """
    record = load("capability-record-harness-usage.json")
    messages = record["the_measured_baseline"]["the_six_and_their_messages"]
    classes = {"IndexError", "KeyError", "ValueError", "TypeError", "AttributeError"}
    for harness, message in sorted(messages.items()):
        assert harness.endswith(".py"), harness
        assert any(message.startswith(name + ":") for name in classes), (
            harness,
            message,
        )
