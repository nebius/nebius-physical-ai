"""Which records resist a believable edit, and which only resist a corruption?

Acceptance criterion 2 for this guard was that a *believable* edit -- one cited
fraction re-measured by a few percent, still finite, still in range -- fails a
test. That was demonstrated on one record. Review then quantified it across the
catalogue and found that most records do not resist it, which is not a
contradiction: a range invariant cannot detect an in-range value, by
construction. The useful statement is which mechanism carries the resistance.

This sweep measures that, in two modes, because the distinction matters more
than the headline.

**naive** perturbs every number by 3 percent and keeps proportions inside
[0, 1]. It is the comparable measurement. But it flatters the guard: a count of
42 becomes 43.26, and the count invariant rejects that for not being an integer,
which is detection of a type error rather than of a plausible re-measurement.
Nobody publishes 43.26 harnesses.

**type_respecting** is the edit someone would actually make by accident or by
tuning: a proportion moves 3 percent and stays in [0, 1], a count moves to a
neighbouring *integer*, a signed delta stays in [-1, 1]. Every value remains
something the record could legitimately have contained. This is the mode that
isolates what range invariants cannot see, and the records that still object are
objecting because a figure is pinned to a value outside themselves.

    python believable_edit_sweep.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[4]
GUARDS = REPO / "npa/tests/workbench/test_open3d_evidence_record_guards.py"
PYTHON = REPO / "npa/.venv/bin/python"
OUT = Path("/tmp/believable-edit-sweep.json")

FACTOR = 1.03


def load_guards() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guards", GUARDS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def perturb(value: object, kind_of: object, path: str, mode: str) -> object:
    """Move one value by about 3 percent, honouring its class in type mode."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    kind = kind_of(path)
    moved = value * FACTOR

    if mode == "naive":
        if kind == "proportion" and not 0.0 <= moved <= 1.0:
            moved = value * (2 - FACTOR)
        return moved

    if kind == "count" or (isinstance(value, int) and kind != "proportion"):
        # A count must stay a count, or the integrality rule catches a type
        # error rather than a plausible edit.
        step = max(1, round(abs(value) * (FACTOR - 1)))
        return int(value) + step
    if kind == "proportion":
        if not 0.0 <= moved <= 1.0:
            moved = value * (2 - FACTOR)
        return min(1.0, max(0.0, moved))
    if kind == "delta":
        return max(-1.0, min(1.0, moved))
    if kind == "fit_ratio":
        return abs(moved) or 1.0
    return abs(moved)


def walk(node: object, kind_of, mode: str, path: str = "") -> object:
    if isinstance(node, dict):
        return {k: walk(v, kind_of, mode, f"{path}/{k}") for k, v in node.items()}
    if isinstance(node, list):
        return [walk(v, kind_of, mode, f"{path}[{i}]") for i, v in enumerate(node)]
    return perturb(node, kind_of, path, mode)


def numeric_leaves(node: object) -> int:
    if isinstance(node, bool):
        return 0
    if isinstance(node, dict):
        return sum(numeric_leaves(v) for v in node.values())
    if isinstance(node, list):
        return sum(numeric_leaves(v) for v in node)
    return 1 if isinstance(node, (int, float)) else 0


def run_guards() -> int:
    result = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "pytest",
            str(GUARDS),
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
    return int(failed.group(1)) if failed else 0


def main() -> int:
    guards = load_guards()
    kind_of = guards.invariant_for
    paths = sorted(guards.records(), key=guards.record_key)

    if run_guards():
        print("the catalogue does not pass its guards before mutation; stopping")
        return 1

    report: dict[str, object] = {
        "question": (
            "A range invariant cannot detect an in-range value. So which "
            "records resist a believable re-measurement, and by what mechanism?"
        ),
        "perturbation": f"every number multiplied by {FACTOR}, booleans untouched",
        "modes": {},
    }

    for mode in ("naive", "type_respecting"):
        caught: list[str] = []
        unnoticed: list[str] = []
        skipped: list[str] = []
        for path in paths:
            key = guards.record_key(path)
            original = path.read_text()
            data = json.loads(original)
            if not numeric_leaves(data):
                skipped.append(key)
                continue
            edited = walk(data, kind_of, mode)
            if edited == data:
                skipped.append(key)
                continue
            path.write_text(json.dumps(edited, indent=2, sort_keys=True))
            failures = run_guards()
            path.write_text(original)
            assert path.read_text() == original, f"{key} not restored"
            (caught if failures else unnoticed).append(key)
            print(f"  [{mode:15}] {'caught ' if failures else 'MISSED '} {key}")

        report["modes"][mode] = {
            "records_with_numeric_content": len(caught) + len(unnoticed),
            "caught": len(caught),
            "unnoticed": len(unnoticed),
            "records_without_numeric_content": len(skipped),
            "caught_records": sorted(caught),
            "unnoticed_records": sorted(unnoticed),
        }
        print(
            f"=== {mode}: {len(caught)} caught, {len(unnoticed)} unnoticed, "
            f"{len(skipped)} without numbers"
        )

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
