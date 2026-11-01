"""Does the evidence-record guard actually fail when a record is wrong?

The guard in `npa/tests/workbench/test_open3d_evidence_record_guards.py` exists because a
reviewer showed that corrupting eight Open3D evidence records left all 120 executing tests
passing. A guard written in answer to that has to be shown failing, or it is the same kind of
artifact as the records it protects: asserted, not demonstrated.

This applies seven mutations to the committed tree, runs the guard after each, restores, and
reports which tests objected. Run it from the repository root:

    npa/.venv/bin/python docs/evidence/open3d/harness/evidence_guard_mutations.py

Mutation 2 is the one that matters. The others are gross damage, or bookkeeping that several
checks notice at once; mutation 2 is a believable re-measurement -- one fraction moved 4 percent, still
finite, still inside [0, 1] -- and exactly one test in the file notices.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EVIDENCE = REPO / "docs" / "evidence" / "open3d"
GUARD = REPO / "npa" / "tests" / "workbench" / "test_open3d_evidence_record_guards.py"
PYTHON = REPO / "npa" / ".venv" / "bin" / "python"


def failing_tests() -> list[str]:
    """Names of the guard's tests that fail right now."""
    done = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            str(GUARD),
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    return sorted(
        line.split("::")[-1].split()[0]
        for line in done.stdout.splitlines()
        if line.startswith("FAILED")
    )


def negate_everything(node: object) -> object:
    """The reviewer's corruption: every boolean flipped, every number negated."""
    if isinstance(node, dict):
        return {key: negate_everything(value) for key, value in node.items()}
    if isinstance(node, list):
        return [negate_everything(value) for value in node]
    if isinstance(node, bool):
        return not node
    if isinstance(node, (int, float)):
        return -node
    return node


def nudge_one_fraction(node: object, target: float, factor: float = 1.04) -> object:
    """Move a single cited figure by a few percent and leave everything else alone."""
    if isinstance(node, dict):
        return {
            key: nudge_one_fraction(value, target, factor)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [nudge_one_fraction(value, target, factor) for value in node]
    if (
        isinstance(node, float)
        and not isinstance(node, bool)
        and round(node, 5) == target
    ):
        return round(node * factor, 8)
    return node


def rewrite_json(path: Path, transform) -> None:
    path.write_text(
        json.dumps(transform(json.loads(path.read_text())), indent=2) + "\n"
    )


def main() -> int:
    baseline = failing_tests()
    if baseline:
        print(f"the guard is already failing before any mutation: {baseline}")
        return 1

    cases: list[tuple[str, Path, object]] = [
        (
            "1. an unpinned record corrupted the way the reviewer corrupted them",
            EVIDENCE / "band-profile-across-scenes.json",
            negate_everything,
        ),
        (
            "2. one cited fraction re-measured by 4 percent, still a valid fraction",
            EVIDENCE / "poisson-procedure-and-overcall-audit.json",
            lambda data: nudge_one_fraction(data, 0.00911),
        ),
        (
            "3. a capability record emptied",
            EVIDENCE / "capability-record-distance-bands.json",
            lambda _: {},
        ),
        (
            "4. a count field made negative",
            EVIDENCE / "band-validity-sweep.json",
            lambda data: json.loads(
                json.dumps(data).replace('"samples": 300000', '"samples": -300000')
            ),
        ),
        (
            "5a. a downstream-consumer record corrupted (unguarded until the vocabulary widened)",
            EVIDENCE / "downstream-consumer.json",
            negate_everything,
        ),
        (
            "5b. the viewer capability record's fit ratios corrupted (reached only by path)",
            EVIDENCE / "capability-record-viewer-ui-capture.json",
            negate_everything,
        ),
        (
            "5. the duplicated filename made to disagree with its twin",
            EVIDENCE / "viewer-ui" / "ordinary-window-audit.json",
            # This record is a list at the top level, not an object, which is worth knowing
            # before writing a transform that assumes every record is a mapping.
            lambda data: (
                [*data, {"_divergence": "this copy now says something else"}]
                if isinstance(data, list)
                else {**data, "_divergence": "this copy now says something else"}
            ),
        ),
    ]

    coverage_record = EVIDENCE / "evidence-record-guards.json"
    cases.append(
        (
            "6. the published coverage figures edited without re-measuring",
            coverage_record,
            lambda data: {
                **data,
                "coverage": {
                    **data["coverage"],
                    "records_with_structural_checks_only": 4,
                },
            },
        )
    )

    results = []
    for label, path, transform in cases:
        original = path.read_text()
        try:
            rewrite_json(path, transform)
            caught = failing_tests()
        finally:
            path.write_text(original)
        results.append({"mutation": label, "tests_that_objected": caught})
        verdict = "CAUGHT" if caught else "MISSED"
        print(f"{verdict:>7}  {label}")
        for name in caught:
            print(f"         {name}")

    # Mutation 7 adds a file rather than editing one, so it does not fit the loop above. It is
    # the dilution case: a record that nothing checks must not be able to join the catalogue
    # while the coverage figures carry on describing the old one.
    newcomer = EVIDENCE / "unguarded-newcomer.json"
    assert not newcomer.exists(), newcomer
    try:
        newcomer.write_text(
            '{"a_note": "plausible prose, no fractions, no counts, no digests"}\n'
        )
        caught = failing_tests()
    finally:
        newcomer.unlink()
    label = "7. a record nothing checks added to the catalogue"
    results.append({"mutation": label, "tests_that_objected": caught})
    print(f"{'CAUGHT' if caught else 'MISSED':>7}  {label}")
    for name in caught:
        print(f"         {name}")

    # Only tracked modifications count. An earlier version of this counted anything in the
    # porcelain output ending in `.json`, which reported this harness's own new, untracked
    # records as a failure to restore.
    left_modified = subprocess.run(
        ["git", "diff", "--name-only", "--", "docs/evidence/open3d"],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    ).stdout.split()
    print(
        f"\ncommitted evidence files left modified: {len(left_modified)} {left_modified}"
    )

    missed = [case["mutation"] for case in results if not case["tests_that_objected"]]
    if missed:
        print(f"\nmutations the guard does not catch: {missed}")
    print(json.dumps({"baseline_failures": baseline, "mutations": results}, indent=2))
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
