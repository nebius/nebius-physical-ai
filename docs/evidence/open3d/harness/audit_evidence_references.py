"""Check that this evidence set's records point at things that exist and hash as claimed.

Written after a review lane found, in sequence, a record citing a generator that was never
committed, a record citing a frame that had been moved, and a recording whose provenance fields
were invented. Each was a different shape of the same problem: a record that reads as though it
is backed by something checkable, and is not.

Two kinds of check:

  * every `harness/...` or `viewer-ui/...` path a record mentions must exist here, so a reader
    can actually run or open it;
  * every SHA256 a record claims for a file in this directory must match that file's bytes.

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

REFERENCE = re.compile(r"\b((?:harness|viewer-ui)/[\w./-]+)\b")
SHA256 = re.compile(r"\b[0-9a-f]{64}\b")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def claimed_hashes(node: object, path: str = "") -> list[tuple[str, str]]:
    """Pull (json path, sha256) pairs out of a record, wherever they are nested."""

    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found += claimed_hashes(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found += claimed_hashes(value, f"{path}[{index}]")
    elif isinstance(node, str) and SHA256.fullmatch(node):
        found.append((path, node))
    return found


def main() -> int:
    findings: list[str] = []
    # One digest per file, since several records cite the same artifacts.
    by_digest: dict[str, Path] = {}
    for candidate in ROOT.rglob("*"):
        if candidate.is_file() and candidate.suffix != ".json":
            by_digest.setdefault(digest(candidate), candidate)

    for record in sorted(ROOT.glob("*.json")):
        text = record.read_text()
        for reference in sorted(set(REFERENCE.findall(text))):
            if not (ROOT / reference).exists():
                findings.append(f"{record.name}: cites missing {reference}")

        data = json.loads(text)
        for where, sha in claimed_hashes(data):
            # A claimed hash is only checkable if the record also says which file it belongs
            # to; those are the ones sitting beside a `file`/`frame`/`script` key, which the
            # digest index resolves by content.
            if sha not in by_digest and any(
                token in where.lower()
                for token in ("frame_sha256", "script_sha256", "harness_sha256")
            ):
                findings.append(
                    f"{record.name}: {where} claims {sha[:12]}.. which matches no file here"
                )

    for name in sorted(RUN_ARTIFACT_NAMES):
        if (ROOT / name).exists():
            findings.append(
                f"{name} is committed here but is treated as a run-artifact name by this audit; "
                "rename it or narrow RUN_ARTIFACT_NAMES"
            )

    if findings:
        print(f"{len(findings)} finding(s):")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print("Every cited harness and view exists, and every checkable hash matches its file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
