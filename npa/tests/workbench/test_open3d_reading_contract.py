"""The Open3D evidence harnesses consume a reading the runner produces. Hold them together.

Commit `a828e58fd3` renamed that reading from `"extrapolated shell"` to
`"far from any observation"`, on the same branch of the same condition, because the old name
claimed the reading could detect fabrication and it cannot. The runner's own tests pinned the
new value, so the product was covered. Four evidence harnesses still compared against the old
literal, and nothing covered them at all.

The effect was silent and one-directional. `floor-across-geometry-and-sampling` recorded 27 of
32 readings true when its evidence was committed and 0 afterwards; `band-validity-sweep` went
from 6 of 8 to 0. Both still reported their accepted negative conclusions -- no false positive,
no over-call -- because a detector that cannot fire finds nothing to report. That is worse than
a missing measurement, which at least looks missing.

So these tests do not check the readings themselves, which needs Open3D and the sample meshes.
They check that the string the harnesses look for is still a string the runner can emit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "docs" / "evidence" / "open3d" / "harness"
RUNNER = REPO / "npa" / "src" / "npa" / "workbench" / "open3d" / "runner.py"

#: The name the product withdrew. It must not come back as something a harness tests for.
WITHDRAWN_READING = "extrapolated shell"

#: The field whose value the harnesses branch on.
READING_FIELD = "removed_surface_reads_as"


def runner_readings() -> set[str]:
    """Every value the runner can assign to that field."""
    return set(re.findall(r'^\s+reads_as = "([^"]+)"', RUNNER.read_text(), re.M))


def test_the_runner_still_assigns_the_field_this_contract_is_about() -> None:
    """If this fails, the rest of the file is checking a field that no longer exists."""
    source = RUNNER.read_text()

    assert f'"{READING_FIELD}": reads_as' in source, (
        f"{READING_FIELD} is no longer populated from `reads_as`; this contract needs rewriting"
    )
    assert len(runner_readings()) >= 2, runner_readings()


def test_the_constant_the_harnesses_share_is_a_reading_the_runner_emits() -> None:
    far_reading = re.search(
        r'^FAR_READING = "([^"]+)"', (HARNESS / "harness_io.py").read_text(), re.M
    )
    assert far_reading, "harness_io.FAR_READING is gone"

    assert far_reading.group(1) in runner_readings(), (
        far_reading.group(1),
        runner_readings(),
    )


def test_no_harness_compares_the_reading_against_a_bare_literal() -> None:
    """A literal duplicated across four files is what let this drift unnoticed.

    The constant is checked against the runner by the test above, so routing every
    comparison through it is what connects the harnesses to their producer.
    """
    offenders = []
    for path in sorted(HARNESS.glob("*.py")):
        source = path.read_text()
        for number, line in enumerate(source.splitlines(), start=1):
            if READING_FIELD not in line and "reading[" not in line:
                continue
            window = "\n".join(source.splitlines()[number - 1 : number + 2])
            if re.search(r'==\s*\n?\s*"[^"]+"', window):
                offenders.append(f"{path.name}:{number}")

    assert offenders == [], offenders


def test_the_withdrawn_reading_is_not_what_any_harness_tests_for() -> None:
    """Named separately from the general rule, because this is the instance that happened."""
    offenders = [
        f"{path.name}:{number}"
        for path in sorted(HARNESS.glob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if WITHDRAWN_READING in line and "==" in line
    ]

    assert offenders == [], offenders


def test_the_withdrawn_name_is_not_a_key_in_any_report_a_harness_writes() -> None:
    """A key named for the old reading reasserts the claim the product withdrew.

    The runner stopped saying "extrapolated shell" because it cannot tell an invented
    surface from sparse coverage of a real one. A harness that writes
    `reads_as_extrapolated_shell` into its report puts that claim back, whatever it
    compared against to get there.
    """
    offenders = [
        f"{path.name}:{number}"
        for path in sorted(HARNESS.glob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if "reads_as_extrapolated_shell" in line or "zero_error_reads_as_shell" in line
    ]

    assert offenders == [], offenders


@pytest.mark.parametrize(
    "harness",
    [
        "band_validity_sweep.py",
        "band_false_negative_sweep.py",
        "floor_across_geometry_and_sampling.py",
        "poisson_procedure_and_overcall_audit.py",
    ],
)
def test_each_restored_harness_imports_the_shared_constant(harness: str) -> None:
    source = (HARNESS / harness).read_text()

    assert "FAR_READING" in source, harness
    assert re.search(r"^from harness_io import .*FAR_READING", source, re.M), harness


def test_the_committed_evidence_records_the_detector_firing() -> None:
    """The reason we know this decayed rather than never having worked.

    These two files were produced while the reading was live. They are the oracle that
    made the diagnosis possible, and if they are ever regenerated under the broken
    comparison, that evidence is gone -- so the counts are pinned here.
    """
    import json

    expected = {
        "floor-across-geometry-and-sampling.json": 27,
        "band-validity-sweep.json": 6,
    }
    for name, firings in expected.items():
        report = json.loads((HARNESS.parent / name).read_text())
        found = []

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "reads_as_extrapolated_shell" and isinstance(value, bool):
                        found.append(value)
                    else:
                        walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(report)
        assert sum(found) == firings, (name, sum(found), firings)
