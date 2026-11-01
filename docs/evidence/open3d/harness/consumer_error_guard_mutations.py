"""Corrupt the error-structure record and confirm the guards notice.

A guard suite that passes against the committed record has shown only that the
record is self-consistent today. It has not shown that the guards would object
to a record that was not. Each mutation below is a specific way this evidence
could become wrong -- the threshold quietly moved past the one awkward result,
an object-scale miss hidden behind a stale summary, the labeller's control
weakened until it stops being a control -- and each is applied to the real file,
tested, and reverted.

Mutation 1 is the one worth reading. The candidate's largest phantom component
is exactly the frozen threshold of 8, so moving the threshold to 9 would turn
"one structured component" into "none" and make the record read better. That is
the edit the frozen gate exists to prevent, and it must fail loudly.

    python consumer_error_guard_mutations.py
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RECORD = REPO / "docs/evidence/open3d/consumer-error-structure.json"
TESTS = "npa/tests/workbench/test_open3d_consumer_error_structure.py"
PYTHON = REPO / "npa/.venv/bin/python"


def move_the_threshold_past_the_awkward_result(data: dict) -> None:
    """The candidate's largest component is exactly 8. Nine makes it vanish."""
    data["structured_threshold_cells"] = 9
    candidate = data["variants"]["candidate"]["phantom_obstacle_structure"]
    candidate["structured_components"] = 0
    candidate["cells_in_structured_components"] = 0
    candidate["is_structured"] = False


def drop_the_concession_from_the_prose(data: dict) -> None:
    key = "the_threshold_landed_exactly_on_the_boundary_and_was_not_moved"
    data["the_verdict"][key] = (
        "The threshold of 8 cells was frozen in the claim before any component "
        "size was known, and the candidate's residual error is a 1 cell thick "
        "rind of 8 cells."
    )


def call_the_rind_a_lump(data: dict) -> None:
    """Claim a thickness the extent does not support."""
    block = data["variants"]["candidate"]["phantom_obstacle_structure"]
    block["largest_component_min_thickness_cells"] = 2


def shrink_the_box_below_the_component(data: dict) -> None:
    """337 cells cannot fit in a 2x2x2 box."""
    block = data["variants"]["baseline"]["phantom_obstacle_structure"]
    block["largest_component_extent_cells"] = [2, 2, 2]
    block["largest_component_min_thickness_cells"] = 2


def inflate_the_structured_total(data: dict) -> None:
    block = data["variants"]["candidate"]["phantom_obstacle_structure"]
    block["cells_in_structured_components"] = 40


def break_the_binding_to_the_consumer_record(data: dict) -> None:
    data["variants"]["candidate"]["phantom_obstacle_cells"] = 120
    data["variants"]["candidate"]["phantom_obstacle_structure"]["cells_total"] = 120


def hide_an_object_scale_miss(data: dict) -> None:
    """Add a 40-cell missed component and leave the summary claiming none."""
    block = data["variants"]["candidate"]["missed_structure_structure"]
    block["component_sizes"] = [40, *block["component_sizes"]]


def claim_a_miss_in_a_hole_is_still_a_boundary_cell(data: dict) -> None:
    data["variants"]["candidate"]["missed_distance_in_radii"]["max"] = 7.4
    data["variants"]["candidate"]["missed_distance_in_radii"]["beyond_five_radii"] = 3


def weaken_the_labeller_control(data: dict) -> None:
    """Keep the passing half, drop the half that makes it a control."""
    checks = data["labeller_validation"]["checks"]
    checks["corner_case_would_merge_under_26_connectivity"] = False
    data["labeller_validation"]["labeller_is_valid"] = False


def quote_a_figure_from_nowhere(data: dict) -> None:
    data["the_finding"]["what_the_residual_error_is_instead"] += (
        " Only 3 of the 79 components exceed 6 cells."
    )


def accept_a_failed_reproduction(data: dict) -> None:
    data["reproduces_published_counts"] = False
    data["mismatches_against_published"] = ["candidate.occupied_cells: 1948 -> 1950"]


MUTATIONS: dict[str, Callable[[dict], None]] = {
    "1 move the threshold past the one awkward result": (
        move_the_threshold_past_the_awkward_result
    ),
    "2 drop the concession from the verdict prose": drop_the_concession_from_the_prose,
    "3 call the one-cell rind two cells thick": call_the_rind_a_lump,
    "4 shrink the bounding box below the component": shrink_the_box_below_the_component,
    "5 inflate the structured-cell total": inflate_the_structured_total,
    "6 break the binding to downstream-consumer.json": (
        break_the_binding_to_the_consumer_record
    ),
    "7 hide an object-scale missed component": hide_an_object_scale_miss,
    "8 call a miss five radii out a boundary cell": (
        claim_a_miss_in_a_hole_is_still_a_boundary_cell
    ),
    "9 weaken the labeller control to half a control": weaken_the_labeller_control,
    "10 quote a figure nothing measures": quote_a_figure_from_nowhere,
    "11 publish a failed reproduction gate": accept_a_failed_reproduction,
}


def run_tests() -> tuple[int, int]:
    """Return (failed, passed) counts from a pytest run of the guard file."""
    result = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            TESTS,
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO / "npa",
        capture_output=True,
        text=True,
        check=False,
    )
    summary = "\n".join(
        line
        for line in result.stdout.splitlines()
        if " passed" in line or " failed" in line
    )
    failed = re.search(r"(\d+) failed", summary)
    passed = re.search(r"(\d+) passed", summary)
    return (
        int(failed.group(1)) if failed else 0,
        int(passed.group(1)) if passed else 0,
    )


def restore() -> None:
    subprocess.run(
        ["git", "checkout", "--", str(RECORD.relative_to(REPO))],
        cwd=REPO,
        check=True,
    )


def main() -> int:
    original = RECORD.read_text()

    failed, passed = run_tests()
    print(f"baseline: {passed} passed, {failed} failed")
    if failed:
        print("the committed record does not pass its own guards; stopping")
        return 1

    caught = 0
    for name, mutate in MUTATIONS.items():
        data = json.loads(original)
        mutate(data)
        RECORD.write_text(json.dumps(data, indent=2, sort_keys=True))
        failed, passed = run_tests()
        restore()
        verdict = "caught" if failed else "MISSED"
        caught += bool(failed)
        print(f"  {verdict:7} by {failed} test(s): {name}")

    assert RECORD.read_text() == original, "record not restored"
    dirty = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    print(
        f"\n{caught}/{len(MUTATIONS)} caught; tracked files left modified: {len(dirty)}"
    )
    return 0 if caught == len(MUTATIONS) and not dirty else 1


if __name__ == "__main__":
    raise SystemExit(main())
