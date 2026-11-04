"""Repeat review's audit against the capability records this PR adds.

Review corrupted one evidence record at a time on this branch, with the full
Open3D suite as the oracle, and found a clean line: every measurement report the
new tests read was held, and every capability record was loose. One absorbed 88
corrupted fields and another 325 with nothing objecting.

This reruns that audit on the four capability records this PR adds, so the
before and after are produced the same way rather than compared across methods.
Corruption is deliberately indiscriminate -- negate every number, flip every
boolean, in place -- because a guard that only catches a hand-picked edit is a
guard that was written against the edit.

    python capability_record_binding_mutations.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EVIDENCE = REPO / "docs/evidence/open3d"
TESTS = REPO / "npa/tests/workbench"
PYTHON = REPO / "npa/.venv/bin/python"

RECORDS = (
    "capability-record-fabrication-measure.json",
    "capability-record-band-validity-rescore.json",
    "capability-record-harness-usage.json",
    "capability-record-decayed-readings.json",
)


def corrupt(value: object) -> tuple[object, int]:
    """Negate every number and flip every boolean, counting what changed."""
    if isinstance(value, bool):
        return (not value), 1
    if isinstance(value, (int, float)):
        return -value, 1
    if isinstance(value, dict):
        changed = 0
        out = {}
        for key, child in value.items():
            out[key], count = corrupt(child)
            changed += count
        return out, changed
    if isinstance(value, list):
        changed = 0
        items = []
        for child in value:
            new, count = corrupt(child)
            items.append(new)
            changed += count
        return items, changed
    return value, 0


def run_suite() -> tuple[int, int]:
    """The whole Open3D suite as the oracle, as review used it."""
    result = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            *[str(p) for p in sorted(TESTS.glob("test_open3d*.py"))],
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO,
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


def main() -> int:
    failed, passed = run_suite()
    print(f"baseline: {passed} passed, {failed} failed")
    if failed or not passed:
        print("the suite does not pass before mutation; stopping")
        return 1

    results = []
    for name in RECORDS:
        path = EVIDENCE / name
        original = path.read_text()
        data, changed = corrupt(json.loads(original))
        try:
            path.write_text(json.dumps(data, indent=2) + "\n")
            failed, _ = run_suite()
        finally:
            path.write_text(original)
        assert path.read_text() == original, f"{name} not restored"
        results.append((name, changed, failed))
        verdict = f"caught, {failed} failures" if failed else "UNNOTICED"
        print(f"  {name:48} {changed:4} fields corrupted -> {verdict}")

    missed = [name for name, _, failed in results if not failed]
    print(f"\n{len(results) - len(missed)}/{len(results)} records now held")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
