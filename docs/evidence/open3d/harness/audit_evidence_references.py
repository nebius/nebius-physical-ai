"""Check that this evidence set's records point at things that exist and hash as claimed.

Written after a review lane found, in sequence, a record citing a generator that was never
committed, a record citing a frame that had been moved, and a recording whose provenance fields
were invented. Each was a different shape of the same problem: a record that reads as though it
is backed by something checkable, and is not.

The first version of this guard was itself an instance of that problem. It announced that "every
checkable hash matches its file" while doing two much weaker things: it only looked at hashes
whose key name was one of three it knew about, and it asked whether a hash matched *some* file in
the directory rather than *the file the record named*. A null byte appended to a cited recording
left it green, and two records with their hashes swapped would both have passed. The same review
lane had caught membership-where-binding-was-intended once before in this program, in image
identity, which is why it is worth naming here rather than quietly fixing.

So the check is now a binding one. A hash is verified when the record says which file it belongs
to -- a `<name>` and `<name>_sha256` pair in the same object, or a `sha256` beside a `file`,
`path`, `frame`, `script` or `recording` -- and verification means hashing that named file. Hashes
that name no file cannot be bound, and the summary says how many there were instead of counting
them as checked.

Run from the repository root:
    python3 docs/evidence/open3d/harness/audit_evidence_references.py

Exits non-zero on any finding, so it can gate.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Paths inside a record that refer to files *produced by a workflow run* rather than to
#: evidence committed here. Naming them descriptively is correct, so they are not findings.
RUN_ARTIFACT_NAMES = frozenset(
    {
        "manifest.json",
        "result.json",
        "pose_graph.json",
        "status.json",
        "validation.json",
        "rrd-manifest.json",
        "cluster.json",
        "calibration-attempts.json",
    }
)

#: Keys whose value is a path that a sibling bare `sha256` belongs to.
PATH_KEYS = ("file", "path", "frame", "script", "recording", "harness", "generator")

REFERENCE = re.compile(r"\b((?:harness|viewer-ui)/[\w./-]+)\b")
SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
#: A JSON key that is itself a filename, which several records use to key a hash by its file.
LOOKS_LIKE_A_FILE = re.compile(r"\A[\w./-]+\.[A-Za-z0-9]{1,6}\Z")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def resolve(value: str) -> Path | None:
    """Find the file a record names, whether it gave a path here or a bare filename."""

    candidate = ROOT / value
    if candidate.is_file():
        return candidate
    matches = [p for p in ROOT.rglob(value) if p.is_file()] if "/" not in value else []
    return matches[0] if len(matches) == 1 else None


def bound_pairs(
    node: object, where: str = ""
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Split every hash in a record into those bound to a named file and those not.

    Returns (bound, unbound) where each bound entry is (json path, named file, claimed hash).
    """

    bound: list[tuple[str, str, str]] = []
    unbound: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{where}.{key}" if where else key
            if isinstance(value, str) and SHA256.match(value):
                # `<name>_sha256` binds to a sibling `<name>`; a bare `sha256` binds to
                # whichever path key the same object carries.
                named = None
                if key.endswith("_sha256"):
                    named = node.get(key[: -len("_sha256")])
                elif key == "sha256":
                    named = next(
                        (node[k] for k in PATH_KEYS if isinstance(node.get(k), str)),
                        None,
                    )
                elif LOOKS_LIKE_A_FILE.match(key):
                    # Several records key a hash by the filename itself, as in
                    # `artifact_hashes: {"multiway/fused.ply": "<sha>"}`. The key is the binding.
                    named = key
                if isinstance(named, str) and named:
                    bound.append((here, named, value))
                else:
                    unbound.append(here)
            else:
                sub_bound, sub_unbound = bound_pairs(value, here)
                bound += sub_bound
                unbound += sub_unbound
    elif isinstance(node, list):
        for index, value in enumerate(node):
            sub_bound, sub_unbound = bound_pairs(value, f"{where}[{index}]")
            bound += sub_bound
            unbound += sub_unbound
    return bound, unbound


def main() -> int:
    findings: list[str] = []
    verified = 0
    unresolvable: list[str] = []
    unbound_total: list[str] = []

    for record in sorted(ROOT.glob("*.json")):
        text = record.read_text()
        for reference in sorted(set(REFERENCE.findall(text))):
            if not (ROOT / reference).exists():
                findings.append(f"{record.name}: cites missing {reference}")

        bound, unbound = bound_pairs(json.loads(text))
        unbound_total += [f"{record.name}:{w}" for w in unbound]
        for where, named, claimed in bound:
            if Path(named).name in RUN_ARTIFACT_NAMES:
                # A hash of a run's own artifact, which is not committed here to compare against.
                unresolvable.append(f"{record.name}:{where} -> {named} (run artifact)")
                continue
            target = resolve(named)
            if target is None:
                unresolvable.append(
                    f"{record.name}:{where} -> {named} (not found here)"
                )
                continue
            actual = digest(target)
            if actual != claimed:
                findings.append(
                    f"{record.name}: {where} claims {claimed[:12]}.. for "
                    f"{target.relative_to(ROOT)} which is actually {actual[:12]}.."
                )
            else:
                verified += 1

    if findings:
        print(f"{len(findings)} finding(s):")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    print(
        f"Every cited harness and view exists. {verified} hash(es) were bound to a named file "
        f"here and match its bytes."
    )
    # Said out loud rather than folded into the word "checkable", which is how the first version
    # of this guard overstated what it had done. Counts by default, names on request, because a
    # gate that prints seventy lines on success stops being read.
    verbose = "--verbose" in sys.argv
    held_elsewhere = "name a file this directory does not hold, mostly a run's own artifacts recorded by hash"
    for label, items in (
        (held_elsewhere, unresolvable),
        ("name no file and so cannot be bound to anything", unbound_total),
    ):
        if items:
            count = len(items)
            print(
                f"{count} hash(es) {label}."
                + ("" if verbose else " Pass --verbose to list.")
            )
            if verbose:
                for item in items:
                    print(f"  . {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
