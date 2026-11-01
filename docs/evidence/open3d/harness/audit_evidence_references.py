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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#: Where `ROOT` sits inside the repository, for naming a path to `git show`.
PREFIX = "docs/evidence/open3d"

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


def resolve(value: str, run_dir: str | None) -> Path | None:
    """Find the file a record names, whether it gave a path here or a bare filename.

    A record that commits its run's artifacts says where with `artifacts_committed_under`, and
    only then do its within-run paths like `surface/result.json` resolve there. Matching such a
    path against the whole tree looks helpful and is not: two runs have a `prepared/manifest.json`
    with the same relative path and different bytes, so a free-floating suffix match bound one
    run's recorded hashes to another run's files and reported five findings against records that
    were correct.
    """

    candidate = ROOT / value
    if candidate.is_file():
        return candidate
    if run_dir:
        within = ROOT / run_dir / value
        if within.is_file():
            return within
    if "/" in value or Path(value).name in RUN_ARTIFACT_NAMES:
        # A bare `status.json` belongs to whichever run its record describes. Searching the tree
        # for that name found a *different* run's file and called three correct records wrong.
        # Run-artifact names resolve only through a declared run directory.
        return None
    matches = [p for p in ROOT.rglob(value) if p.is_file()]
    return matches[0] if len(matches) == 1 else None


def bound_pairs(
    node: object, where: str = ""
) -> tuple[list[tuple[str, str, str, str | None]], list[str]]:
    """Split every hash in a record into those bound to a named file and those not.

    Returns (bound, unbound) where each bound entry is
    (json path, named file, claimed hash, commit the bytes are claimed to be at).

    That fourth element is the difference between two claims this check used to treat
    as one. `capture.frame_sha256` says *this committed PNG has these bytes*, and the
    working tree is the right place to look. `generator.script_sha256` says *the
    artifact was produced by a script with these bytes*, which is a statement about
    history, and the working tree is only the right place to look until someone
    improves the generator.

    Conflating them pushes in a bad direction. A record here attests to a viewer
    capture whose input clouds were never committed and which cannot be regenerated
    from a checkout; editing its generator made this check report the true provenance
    hash as a mismatch, and the way to clear the report is to overwrite a correct
    historical claim with bytes that post-date the artifact. That record's own
    `supersedes` block documents an earlier capture that invented provenance to
    satisfy required fields. A gate that makes doing it again the path of least
    resistance has the wrong rule, not the wrong data.

    So a record may say where the bytes live, with `<name>_bytes_committed_at` beside
    `<name>_sha256`, or `bytes_committed_at` beside a bare `sha256`. That is a
    redirection, never an exemption: the hash is still verified, against the file at
    that commit.
    """

    bound: list[tuple[str, str, str, str | None]] = []
    unbound: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{where}.{key}" if where else key
            if isinstance(value, str) and SHA256.match(value):
                # `<name>_sha256` binds to a sibling `<name>`; a bare `sha256` binds to
                # whichever path key the same object carries.
                named = None
                at_commit = None
                if key.endswith("_sha256"):
                    stem = key[: -len("_sha256")]
                    named = node.get(stem)
                    at_commit = node.get(f"{stem}_bytes_committed_at")
                elif key == "sha256":
                    named = next(
                        (node[k] for k in PATH_KEYS if isinstance(node.get(k), str)),
                        None,
                    )
                    at_commit = node.get("bytes_committed_at")
                elif LOOKS_LIKE_A_FILE.match(key):
                    # Several records key a hash by the filename itself, as in
                    # `artifact_hashes: {"multiway/fused.ply": "<sha>"}`. The key is the binding.
                    named = key
                if isinstance(named, str) and named:
                    bound.append(
                        (here, named, value, at_commit if isinstance(at_commit, str) else None)
                    )
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


def digest_at_commit(commit: str, path: str) -> str | None:
    """Hash a file as it was at a commit, or None when this checkout lacks that commit.

    A missing commit is reported rather than treated as a mismatch. A shallow clone
    is a property of the checkout, and a check that turns one into a finding about
    the evidence is describing the wrong thing.
    """
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if present.returncode != 0:
        return None
    shown = subprocess.run(
        ["git", "show", f"{commit}:{PREFIX}/{path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if shown.returncode != 0:
        return ""
    return hashlib.sha256(shown.stdout).hexdigest()


def main() -> int:
    findings: list[str] = []
    verified = 0
    verified_at_a_commit = 0
    unresolvable: list[str] = []
    #: Kept apart from `unresolvable`, whose label says "mostly a run's own artifacts".
    #: A commit this checkout does not have is a different reason, and filing it under
    #: that label would describe it wrongly to the one reader who needs to act on it.
    commit_missing: list[str] = []
    unbound_total: list[str] = []

    for record in sorted(ROOT.glob("*.json")):
        text = record.read_text()
        for reference in sorted(set(REFERENCE.findall(text))):
            if not (ROOT / reference).exists():
                findings.append(f"{record.name}: cites missing {reference}")

        data = json.loads(text)
        run_dir = data.get("artifacts_committed_under") if isinstance(data, dict) else None
        bound, unbound = bound_pairs(data)
        unbound_total += [f"{record.name}:{w}" for w in unbound]
        for where, named, claimed, at_commit in bound:
            if at_commit:
                # The claim is about bytes at a commit, so that is the tree to read.
                actual = digest_at_commit(at_commit, named)
                if actual is None:
                    commit_missing.append(
                        f"{record.name}:{where} -> {named} at {at_commit[:12]}"
                    )
                    continue
                if actual != claimed:
                    findings.append(
                        f"{record.name}: {where} claims {claimed[:12]}.. for {named} at "
                        f"{at_commit[:12]} which is actually "
                        f"{(actual or 'absent there')[:12]}.."
                    )
                else:
                    verified += 1
                    verified_at_a_commit += 1
                continue
            # Resolve before exempting: a run's artifacts are often not committed here, but when
            # a run directory *is* committed its hashes are checkable like anything else, and the
            # exemption used to skip them before anything looked.
            target = resolve(named, run_dir)
            if target is None and Path(named).name in RUN_ARTIFACT_NAMES:
                unresolvable.append(f"{record.name}:{where} -> {named} (run artifact)")
                continue
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
        f"and match its bytes."
    )
    if verified_at_a_commit:
        # Named separately because it is a different claim: not "this file has these
        # bytes" but "the thing this record describes was produced by these bytes",
        # checked against the commit the record says they are at.
        one = verified_at_a_commit == 1
        print(
            f"{verified_at_a_commit} of those "
            f"{'is a provenance claim' if one else 'are provenance claims'}, verified "
            f"against the commit the record names rather than the working tree."
        )
    # Said out loud rather than folded into the word "checkable", which is how the first version
    # of this guard overstated what it had done. Counts by default, names on request, because a
    # gate that prints seventy lines on success stops being read.
    verbose = "--verbose" in sys.argv
    held_elsewhere = "name a file this directory does not hold, mostly a run's own artifacts recorded by hash"
    commit_missing_label = (
        "claim bytes at a commit this checkout does not have, so they could not be "
        "checked either way"
    )
    for label, items in (
        (held_elsewhere, unresolvable),
        (commit_missing_label, commit_missing),
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
