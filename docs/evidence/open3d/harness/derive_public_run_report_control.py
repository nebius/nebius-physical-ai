"""Controls for the public-report allowlist: what it must refuse, and what it must preserve.

The first version of the generator claimed every unclassified field was refused, and an
independent review disproved it with one line: the allowlist held subtree patterns like
`mesh.*` and `coverage.*`, so a field nobody had ever classified -- including a
string-valued one -- passed straight through under a name that looked measured. The
counterexample is reproduced here as the first control rather than described, because a
claim about what a filter refuses is worth exactly as much as the case that tries it.

Four controls:

- An unknown nested key under a previously wildcarded parent must stop the generator.
- A known key holding the wrong kind of value must stop it too. Naming a leaf is not
  enough on its own: `mesh.sha256` would still have admitted an arbitrary string.
- Every leaf the generator does emit must equal the byte the run wrote.
- The committed report must be exactly what the current generator produces.

Run from the repository root, against the run's retained artifacts:
    python3 docs/evidence/open3d/harness/derive_public_run_report_control.py \\
        --artifacts <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from derive_public_run_report import ARTIFACTS, build

EVIDENCE = Path(__file__).resolve().parent.parent
REPORT = EVIDENCE / "replay-b18d-public-report.json"

#: The exact shape of the independent review's counterexample: an unclassified key nested
#: under `mesh`, which the old `mesh.*` pattern carried through untouched.
UNKNOWN_FIELD = ("surface/result.json", "mesh", "synthetic_unclassified_private_field")
#: A leaf that *is* named, holding something its declared kind does not allow.
WRONG_KIND = ("surface/result.json", "mesh", "sha256")
SYNTHETIC_VALUE = "synthetic-not-a-real-value"


def _leaves(node: object, path: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _leaves(value, f"{path}[{index}]")
    else:
        yield path, node


def _refuses(artifacts: Path, artifact: str, parent: str, key: str) -> str:
    """Plant one field in a copy of the run's artifacts and report how the generator answers."""

    with tempfile.TemporaryDirectory() as workspace:
        planted = Path(workspace) / "artifacts"
        shutil.copytree(artifacts, planted)
        target = planted / artifact
        data = json.loads(target.read_text())
        data[parent][key] = SYNTHETIC_VALUE
        target.write_text(json.dumps(data))
        try:
            build(planted)
        except SystemExit as refusal:
            return f"refused: {refusal}"
    return "carried it through"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args(argv)

    unknown = _refuses(args.artifacts, *UNKNOWN_FIELD)
    wrong_kind = _refuses(args.artifacts, *WRONG_KIND)

    report = build(args.artifacts)
    drifted = []
    emitted = 0
    for artifact, derived in report["derived_fields"].items():
        original = dict(_leaves(json.loads((args.artifacts / artifact).read_text())))
        for path, value in _leaves(derived):
            emitted += 1
            if path not in original or original[path] != value:
                drifted.append(f"{artifact}:{path}")

    committed = REPORT.read_bytes()
    regenerated = (json.dumps(report, indent=2, sort_keys=False) + "\n").encode()

    result = {
        "what_this_checks": (
            "that the allowlist refuses what it says it refuses, and that what it keeps "
            "is unchanged from the bytes the run wrote"
        ),
        "generator": "harness/derive_public_run_report.py",
        "controls": {
            "unknown_nested_key": {
                "planted": f"{UNKNOWN_FIELD[0]} {UNKNOWN_FIELD[1]}.{UNKNOWN_FIELD[2]}",
                "why": "the independent review's counterexample; the old mesh.* pattern kept it",
                "outcome": unknown,
            },
            "named_leaf_wrong_kind": {
                "planted": f"{WRONG_KIND[0]} {WRONG_KIND[1]}.{WRONG_KIND[2]} as an arbitrary string",
                "why": "naming a leaf does not constrain what arrives in it",
                "outcome": wrong_kind,
            },
            "fidelity": {
                "leaves_emitted": emitted,
                "leaves_that_differ_from_the_original": len(drifted),
                "examples": drifted[:5],
            },
            "committed_report_is_current": {
                "committed_sha256": hashlib.sha256(committed).hexdigest(),
                "regenerated_sha256": hashlib.sha256(regenerated).hexdigest(),
                "identical": committed == regenerated,
            },
        },
        "artifacts_read": list(ARTIFACTS),
    }
    print(json.dumps(result, indent=2))

    failures = []
    if not unknown.startswith("refused"):
        failures.append("an unknown nested key was not refused")
    if not wrong_kind.startswith("refused"):
        failures.append("a named leaf holding the wrong kind of value was not refused")
    if drifted:
        failures.append(f"{len(drifted)} emitted leaf/leaves differ from the original")
    if committed != regenerated:
        failures.append("the committed report is not what this generator produces")
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
