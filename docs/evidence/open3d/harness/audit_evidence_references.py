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

Some of those are still checkable without a file: an equality claim a record makes side by side,
or an input two records both hash. Those are checked. What is left over is genuinely beyond this
directory's reach -- a run's artifacts, a module mounted from a dirty tree -- and has to say so
in a `hashes_that_name_no_file_here` note, because a count nobody has to justify is a count that
drifts. Declaring never substitutes for either check above.

Run from the repository root:
    python3 docs/evidence/open3d/harness/audit_evidence_references.py

Exits non-zero on any finding, so it can gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("NPA_EVIDENCE_ROOT", Path(__file__).resolve().parent.parent))

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

#: Key-name transforms that turn one half of an equality claim into the other half. A record that
#: writes `runner_py_at_commit` beside `runner_py_inside_image` is asserting the two are the same
#: bytes -- that assertion is the entire content of `byte_verified_not_label_trusted` -- and the
#: check needs no file, because both values are already sitting in the same object.
#:
#: This closes a gap between two different predicates the guard used to conflate. It reported 37
#: hashes as "naming no file and so cannot be bound to anything" and therefore checked none of
#: them, but "cannot be bound to a file" and "cannot be verified" are not the same thing. Twelve
#: of those 37 were equality claims, so a stale or swapped value in any of them passed in silence,
#: including the pair whose only purpose is to be an equality check.
EQUALITY_TRANSFORMS = (
    (re.compile(r"\A(.+)_at_commit\Z"), r"\1_inside_image"),
    (re.compile(r"\Acommitted_(.+)\Z"), r"regenerated_\1"),
)

#: Objects whose keys name a run input and whose values are that input's digest. Two records that
#: both hash the same input have to agree about it. Deliberately a short explicit list rather than
#: a rule over key names: digests called `sha256` or `original_sha256` appear all over this set
#: with legitimately different values, so a general "same leaf name must match" rule would invent
#: findings. These two containers key their children by the input's own name.
INPUT_HASH_CONTAINERS = ("input_hashes", "inputs.fragments")

#: How a record declares that a hash deliberately names no file committed here and is not an
#: equality claim either -- the bytes of a run's own artifact, or of a source file pinned at a
#: commit the working tree has moved past. The key may sit beside the hash or at any ancestor
#: and covers everything beneath it.
#:
#: Declaring never overrides binding or an equality claim: those are checked whatever a note
#: says, so this cannot be used to excuse a stale digest. It only takes a hash out of the
#: "nothing could check this" count, and only by saying what the bytes are.
#:
#: The twelve equality claims above came out of that count by being *checked*, which is the
#: better outcome and the one to reach for first. What remains cannot be checked here at all,
#: and was being reported as a number on a passing run: a review lane asked for it to be
#: reduced or justified, one record was justified in prose that nothing read, and the rest
#: stayed as they were for a month inside a count that never moved. A category nothing
#: enforces drifts, so an undeclared one is now a finding.
DECLARATION_KEY = "hashes_that_name_no_file_here"

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
    node: object, where: str = "", declared: bool = False
) -> tuple[list[tuple[str, str, str, bool, str | None]], list[tuple[str, bool]]]:
    """Split every hash in a record into those bound to a named file and those not.

    Returns (bound, unbound). A bound entry is (json path, named file, claimed hash,
    whether a `DECLARATION_KEY` note covers it, commit holding historical bytes); an
    unbound entry is (json path, whether a note covers it).

    Both sides carry the note because naming a file is not the same as the file being
    here. A hash whose named file does not resolve is exactly as uncheckable as one
    that named nothing, and until the note was threaded through this side too, adding
    a path key that resolves to nothing moved a hash out of the gated bucket and into
    the ungated one -- same hash, same unverifiability, no justification demanded.
    """

    bound: list[tuple[str, str, str, bool, str | None]] = []
    unbound: list[tuple[str, bool]] = []
    if isinstance(node, dict):
        note = node.get(DECLARATION_KEY)
        declared = declared or bool(isinstance(note, str) and note.strip())
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
                    historical = at_commit if isinstance(at_commit, str) else None
                    bound.append((here, named, value, declared, historical))
                else:
                    unbound.append((here, declared))
            else:
                sub_bound, sub_unbound = bound_pairs(value, here, declared)
                bound += sub_bound
                unbound += sub_unbound
    elif isinstance(node, list):
        for index, value in enumerate(node):
            sub_bound, sub_unbound = bound_pairs(value, f"{where}[{index}]", declared)
            bound += sub_bound
            unbound += sub_unbound
    return bound, unbound


def digest_at_commit(commit: str, path: str) -> str | None:
    """Hash a named evidence file at a declared commit; None means unavailable history."""

    try:
        present = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if present.returncode != 0:
        return None
    prefix = subprocess.run(
        ["git", "rev-parse", "--show-prefix"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    shown = subprocess.run(
        ["git", "show", f"{commit}:{prefix}{path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if shown.returncode != 0:
        return ""
    return hashlib.sha256(shown.stdout).hexdigest()


def equality_claims(
    node: object, where: str = ""
) -> tuple[list[str], list[str], set[str]]:
    """Check the hash pairs a record places side by side as an assertion that they match.

    Returns (findings, verified descriptions, json paths covered). The covered set lets the
    summary stop counting these as unverifiable, which is what they were being called.
    """

    findings: list[str] = []
    verified: list[str] = []
    covered: set[str] = set()

    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{where}.{key}" if where else key
            if isinstance(value, str) and SHA256.match(value):
                for pattern, replacement in EQUALITY_TRANSFORMS:
                    if not pattern.match(key):
                        continue
                    partner = pattern.sub(replacement, key)
                    other = node.get(partner)
                    if not (isinstance(other, str) and SHA256.match(other)):
                        continue
                    partner_path = f"{where}.{partner}" if where else partner
                    covered |= {here, partner_path}
                    if value != other:
                        findings.append(
                            f"{here} is {value[:12]}.. and {partner} is {other[:12]}.., but "
                            "putting them side by side asserts they are the same bytes"
                        )
                    else:
                        verified.append(f"{here} == {partner}")
                    # A record that also states its own conclusion has to agree with its values.
                    for flag in ("identical", "equal", "matches"):
                        if isinstance(node.get(flag), bool) and node[flag] != (
                            value == other
                        ):
                            findings.append(
                                f"{where}.{flag} says {node[flag]} while {key} and {partner} "
                                f"{'match' if value == other else 'differ'}"
                            )
            else:
                sub = equality_claims(value, here)
                findings += sub[0]
                verified += sub[1]
                covered |= sub[2]
    elif isinstance(node, list):
        for index, value in enumerate(node):
            sub = equality_claims(value, f"{where}[{index}]")
            findings += sub[0]
            verified += sub[1]
            covered |= sub[2]
    return findings, verified, covered


def input_hashes(data: object, where: str = "") -> dict[str, tuple[str, str]]:
    """Collect `{input name: (digest, json path)}` from the containers that key inputs by name."""

    found: dict[str, tuple[str, str]] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            here = f"{where}.{key}" if where else key
            if here in INPUT_HASH_CONTAINERS and isinstance(value, dict):
                for name, claimed in value.items():
                    if isinstance(claimed, str) and SHA256.match(claimed):
                        found[name] = (claimed, f"{here}.{name}")
            else:
                found.update(input_hashes(value, here))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            found.update(input_hashes(value, f"{where}[{index}]"))
    return found


def main() -> int:
    findings: list[str] = []
    verified = 0
    equalities = 0
    unresolvable: list[str] = []
    unbound_total: list[str] = []
    undeclared: list[str] = []
    #: input name -> (digest, "record:path"), for the cross-record agreement check.
    inputs_seen: dict[str, list[tuple[str, str, str]]] = {}

    for record in sorted(ROOT.glob("*.json")):
        text = record.read_text()
        for reference in sorted(set(REFERENCE.findall(text))):
            if not (ROOT / reference).exists():
                findings.append(f"{record.name}: cites missing {reference}")

        data = json.loads(text)
        run_dir = (
            data.get("artifacts_committed_under") if isinstance(data, dict) else None
        )

        pair_findings, pair_verified, covered = equality_claims(data)
        findings += [f"{record.name}: {f}" for f in pair_findings]
        # Counted in hashes rather than comparisons: a verified pair checks two of them, which is
        # the number that used to be reported as checkable by nothing.
        equalities += 2 * len(pair_verified)

        for name, (claimed, path) in input_hashes(data).items():
            inputs_seen.setdefault(name, []).append((claimed, record.name, path))

        bound, unbound = bound_pairs(data)
        # An equality claim is verifiable without a file, so it no longer counts as something
        # nothing could check.
        uncovered = [(w, d) for w, d in unbound if w not in covered]
        unbound_total += [f"{record.name}:{w}" for w, _ in uncovered]
        undeclared += [f"{record.name}:{w}" for w, d in uncovered if not d]
        for where, named, claimed, declared, at_commit in bound:
            if at_commit:
                actual = digest_at_commit(at_commit, named)
                if actual is None:
                    # The checkout cannot adjudicate a historical claim it does not contain.
                    # This is distinct from declaring an artifact unavailable in the evidence
                    # directory: the claim remains a binding and is checked when history exists.
                    continue
                if actual != claimed:
                    findings.append(
                        f"{record.name}: {where} claims {claimed[:12]}.. for {named} at "
                        f"{at_commit[:12]} which is actually "
                        f"{(actual or 'absent there')[:12]}.."
                    )
                else:
                    verified += 1
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
                # And it needs the same note as a hash that named nothing at all. A name
                # the directory does not hold is not a binding: nothing reads those bytes,
                # so the claim rests on the record's word either way. Leaving this subset
                # ungated made the gate avoidable by adding a path key -- which is also
                # how an accident presents, when a record cites a committed file that is
                # later moved and is silently reclassified from verified to held
                # elsewhere. The `(run artifact)` subset above stays exempt because its
                # name is itself the description, which is what the note would say.
                if not declared:
                    undeclared.append(f"{record.name}:{where} -> {named}")
                continue
            actual = digest(target)
            if actual != claimed:
                findings.append(
                    f"{record.name}: {where} claims {claimed[:12]}.. for "
                    f"{target.relative_to(ROOT)} which is actually {actual[:12]}.."
                )
            else:
                verified += 1

    # An input hashed by two records is checkable without any file being present: the records have
    # to agree about it. One record naming it alone proves nothing, so it stays uncounted.
    for name, seen in sorted(inputs_seen.items()):
        if len(seen) < 2:
            continue
        first, first_record, first_path = seen[0]
        for claimed, where_record, where_path in seen[1:]:
            if claimed != first:
                findings.append(
                    f"{where_record}: {where_path} hashes input {name} as {claimed[:12]}.. "
                    f"while {first_record}:{first_path} hashes it as {first[:12]}.."
                )
        if all(claimed == first for claimed, _, _ in seen):
            equalities += len(seen)
        checked_here = {f"{r}:{p}" for _, r, p in seen}
        unbound_total = [item for item in unbound_total if item not in checked_here]
        # A hash two records agree about is checked, so it needs no note. Dropping it from
        # both counts and not just the printed one is the difference between a gate and a
        # label: left in `undeclared` it would demand a declaration for something the audit
        # had just verified.
        undeclared = [item for item in undeclared if item not in checked_here]

    findings += [
        f"{one} is a hash this directory cannot check -- it names no file here, or "
        "names one the directory does not hold -- is not an equality claim, and no "
        f"enclosing object declares why; add a {DECLARATION_KEY!r} note saying what "
        "these bytes are"
        for one in undeclared
    ]

    if findings:
        print(f"{len(findings)} finding(s):")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    print(
        f"Every cited harness and view exists. {verified} hash(es) were bound to a named file "
        f"here and match its bytes, and {equalities} hash(es) that name no file were still "
        f"checked, as equality claims the records make side by side."
    )
    # Said out loud rather than folded into the word "checkable", which is how the first version
    # of this guard overstated what it had done. Counts by default, names on request, because a
    # gate that prints seventy lines on success stops being read.
    verbose = "--verbose" in sys.argv
    # Split by which of the two exemptions applies, because they are now enforced
    # differently and one printed number covering both said less than it appeared to.
    # A run-artifact name describes itself; a name this directory simply does not hold
    # has to be declared, exactly like a hash that named nothing.
    run_artifacts = [one for one in unresolvable if one.endswith("(run artifact)")]
    not_held_here = [one for one in unresolvable if one not in set(run_artifacts)]
    for label, items in (
        (
            (
                "name a run's own artifact by a name this directory does not hold, which is "
                "what those names are"
            ),
            run_artifacts,
        ),
        (
            (
                f"name a file this directory does not hold and are declared as such by a "
                f"{DECLARATION_KEY!r} note"
            ),
            not_held_here,
        ),
        (
            f"name no file here and are declared as such by a {DECLARATION_KEY!r} note",
            unbound_total,
        ),
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
