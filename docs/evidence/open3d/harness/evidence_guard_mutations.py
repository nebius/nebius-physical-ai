"""Does the evidence-record guard actually fail when a record is wrong?

The guard in `npa/tests/workbench/test_open3d_evidence_record_guards.py` exists because a
reviewer showed that corrupting eight Open3D evidence records left all 120 executing tests
passing. A guard written in answer to that has to be shown failing, or it is the same kind of
artifact as the records it protects: asserted, not demonstrated.

This applies representative mutations to the committed tree, runs the guard after each,
restores, and reports which tests objected. Run it from the repository root:

    npa/.venv/bin/python docs/evidence/open3d/harness/evidence_guard_mutations.py

The guard deliberately does not claim to detect plausible in-range remeasurement; the separate
believable-edit sweep measures that limitation.
"""

from __future__ import annotations

import json
import re
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
    summary = "\n".join(
        line
        for line in done.stdout.splitlines()
        if " passed" in line or " failed" in line
    )
    executed = sum(
        int(value) for value in re.findall(r"(\d+) (?:passed|failed)", summary)
    )
    if done.returncode not in {0, 1} or executed == 0:
        raise RuntimeError(
            "guard pytest did not complete a test run: "
            f"exit={done.returncode}\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
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


def rewrite_json(path: Path, transform) -> None:
    path.write_text(
        json.dumps(transform(json.loads(path.read_text())), indent=2) + "\n"
    )


def run_mutation(label: str, path: Path, transform) -> list[str]:
    """Apply one effective mutation, run the guard, and restore the exact bytes."""
    original = path.read_text()
    try:
        rewrite_json(path, transform)
        if path.read_text() == original:
            raise RuntimeError(f"mutation did not change {path}: {label}")
        return failing_tests()
    finally:
        path.write_text(original)
        if path.read_text() != original:
            raise RuntimeError(f"mutation was not restored for {path}: {label}")


def mutation_cases() -> list[tuple[str, Path, object]]:
    """Return the corruption cases this harness promises to detect."""
    return [
        (
            "1. an unpinned record corrupted the way the reviewer corrupted them",
            EVIDENCE / "band-profile-across-scenes.json",
            negate_everything,
        ),
        (
            "2. a capability record emptied",
            EVIDENCE / "capability-record-distance-bands.json",
            lambda _: {},
        ),
        (
            "3. a count field made negative",
            EVIDENCE / "band-validity-sweep.json",
            lambda data: json.loads(
                json.dumps(data).replace('"samples": 300000', '"samples": -300000')
            ),
        ),
        (
            "4. a downstream-consumer record corrupted",
            EVIDENCE / "downstream-consumer.json",
            negate_everything,
        ),
        (
            "5. the viewer capability record's fit ratios corrupted",
            EVIDENCE / "capability-record-viewer-ui-capture.json",
            negate_everything,
        ),
        (
            "6. the duplicated filename made to disagree with its twin",
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


def modified_mutation_paths(paths: list[Path]) -> list[str]:
    """Return task paths left dirty after restoration."""
    return subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--",
            *(str(path.relative_to(REPO)) for path in paths),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    ).stdout.split()


def main() -> int:
    baseline = failing_tests()
    if baseline:
        print(f"the guard is already failing before any mutation: {baseline}")
        return 1

    results = []
    mutated_paths = []
    for label, path, transform in mutation_cases():
        caught = run_mutation(label, path, transform)
        mutated_paths.append(path)
        results.append({"mutation": label, "tests_that_objected": caught})
        verdict = "CAUGHT" if caught else "MISSED"
        print(f"{verdict:>7}  {label}")
        for name in caught:
            print(f"         {name}")

    left_modified = modified_mutation_paths(mutated_paths)
    print(
        f"\ncommitted evidence files left modified: {len(left_modified)} {left_modified}"
    )

    missed = [case["mutation"] for case in results if not case["tests_that_objected"]]
    if missed:
        print(f"\nmutations the guard does not catch: {missed}")
    print(json.dumps({"baseline_failures": baseline, "mutations": results}, indent=2))
    return 1 if missed or left_modified else 0


if __name__ == "__main__":
    sys.exit(main())
