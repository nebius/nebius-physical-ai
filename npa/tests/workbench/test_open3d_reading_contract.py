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

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs" / "evidence" / "open3d"
HARNESS = EVIDENCE / "harness"
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


#: Files whose detector column this recomputes, with the firing count pinned elsewhere here.
RECOMPUTABLE = (
    "band-validity-sweep.json",
    "floor-across-geometry-and-sampling.json",
    "band-false-negative-sweep.json",
)


def recorded_rows(report: dict) -> list[dict]:
    """Every row in a report that carries the withdrawn field, at any nesting depth.

    The three reports nest it differently -- flat under `rows` in one, under `cases/<name>/rows`
    in another -- so this searches rather than indexes.
    """
    found: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if f"reads_as_{WITHDRAWN_READING.replace(' ', '_')}" in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(report)
    return found


def runner_threshold() -> float:
    """The share the runner compares against, read from its source.

    Imported, this would resolve to whichever copy of the package the interpreter finds first,
    which on a machine with several worktrees need not be the one under test. That is not
    hypothetical: it is how I first ran the mutation for this test, and moving the threshold in
    this tree left the test green because the value came from somewhere else entirely.
    """
    match = re.search(r"^FAR_BAND_AREA_SHARE = ([0-9.]+)$", RUNNER.read_text(), re.M)
    assert match, (
        "the runner no longer defines the share this contract recomputes against"
    )
    return float(match.group(1))


def test_the_committed_detector_column_recomputes_from_the_committed_numbers() -> None:
    """The one claim in this PR a reviewer could not check, made checkable without Open3D.

    I claimed that restoring the comparison reproduces the committed detector sequences, and
    review could verify the premise, the scope, the counts and the guard by execution but not
    that: the harnesses import Open3D at module scope and no environment on this host has it.

    It does not need Open3D. The runner's reading is a function of two numbers that each row
    already records -- `unsupported_area_beyond_3_voxels / unsupported_area_fraction` against
    `FAR_BAND_AREA_SHARE` -- and the rename touched neither. So the column can be recomputed
    from the committed bytes and compared against the committed column, as a sequence rather
    than a count, by anyone with Python.

    That is stronger than publishing my own run's sequences would have been. It does not ask
    anyone to trust that I ran the harness; it shows the column could not have depended on the
    renamed string in the first place.

    The threshold is read out of the runner's source rather than written as 0.1 here. Pinning
    the number would recreate, in the test that closes this defect, the duplicated literal that
    caused it -- and reading the source rather than importing keeps this file runnable without
    the package, which is how every other test here already reads the runner.
    """
    threshold = runner_threshold()

    field = f"reads_as_{WITHDRAWN_READING.replace(' ', '_')}"
    checked = 0
    for name in RECOMPUTABLE:
        rows = recorded_rows(json.loads((EVIDENCE / name).read_text()))
        assert rows, f"{name} carries no rows with {field}"

        recomputed, recorded = [], []
        for row in rows:
            unsupported = float(row.get("unsupported_area_fraction") or 0.0)
            far = float(row.get("unsupported_area_beyond_3_voxels") or 0.0)
            share = (far / unsupported) if unsupported > 0 else 0.0
            recomputed.append(share >= threshold)
            recorded.append(bool(row[field]))

        assert recomputed == recorded, (
            name,
            [i for i, (a, b) in enumerate(zip(recomputed, recorded)) if a != b],
        )
        checked += len(rows)

    assert checked == 45, checked


def test_the_share_each_row_publishes_is_the_ratio_it_claims_to_be() -> None:
    """Otherwise the recomputation above could agree while reading a different quantity."""
    published = 0
    for name in RECOMPUTABLE:
        for row in recorded_rows(json.loads((EVIDENCE / name).read_text())):
            if "share_beyond_3_voxels" not in row:
                continue
            unsupported = float(row.get("unsupported_area_fraction") or 0.0)
            far = float(row.get("unsupported_area_beyond_3_voxels") or 0.0)
            expected = (far / unsupported) if unsupported > 0 else 0.0
            assert abs(float(row["share_beyond_3_voxels"]) - expected) < 1e-9, (
                name,
                row,
            )
            published += 1
    assert published > 0, "no row publishes the share, so nothing was cross-checked"
