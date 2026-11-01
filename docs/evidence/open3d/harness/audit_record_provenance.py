#!/usr/bin/env python3
"""Check that every Open3D capability record says where it came from, in one shape.

A review lane found provenance recorded in six different ways across these records, which means an
auditor cannot read the set mechanically -- they must learn each file. Worse, two records recorded
none at all, and nothing distinguished that from a record whose provenance simply had not been
reached yet.

The shape required here is deliberately small:

    "provenance": {
      "source_commit": "<bare hex sha, or null with a stated reason>",
      "main_is_an_ancestor_of_this_commit": <bool, or null if unresolvable>
    }

**Why the ancestry flag and not just the commit.** Five of the commits these records name are not
ancestors of `main`, and they are right to name them: they are commits on the retired Open3D lane
branches, where the measurements were actually taken. PR 602 landed as its own commit, so those
originals never became ancestors. Rewriting them to ancestors would falsify the records. The
problem was never the values; it was that a reader could not tell a deliberate non-ancestor from a
stale or mistyped reference. Stating it turns an ambiguity into a fact.

**Why this is a check and not a convention.** The flag is a claim about the repository, so it can
be wrong, and a wrong one is worse than none because it reads as verified. Where the commit
resolves, the claim is compared against `git merge-base --is-ancestor`.

**A commit this clone cannot resolve is still checkable, one way.** Every ancestor of `main` is
present in any clone that has `main`, so a missing object cannot be an ancestor of it. Absence
therefore confirms `false` and refutes `true`. That inference is what lets this guard give the same
answer in CI, which fetches no lane branches, as on a machine that has them -- and it is the only
reason the flag is worth recording, because **six of the commits these records name are on zero
remote refs** and exist nowhere but the clones that made them. A shallow clone breaks the
inference, since its history is truncated by design rather than by the commit's absence, so
shallowness is detected and the comparison is skipped and reported instead of guessed at.

Reachability from a remote ref is reported where it can be, and deliberately not gated: it depends
on how CI fetched the repository, and a guard that fails because of a fetch flag is a guard someone
deletes.

What this does not do: it says nothing about whether a record's measurements are right, and a commit
that resolves and is correctly flagged can still be the wrong commit. This checks that the
provenance a record states is internally consistent and consistent with the repository, which is
strictly less than knowing it is true.

Reads committed json and queries git. No network.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BARE_SHA = re.compile(r"\A[0-9a-f]{7,40}\Z")

# Where "main" is, tried in order. A PR checkout does not always have the remote ref, and a guard
# that fails because of how CI fetched the repository is a guard people delete.
MAIN_CANDIDATES = ("origin/main", "refs/remotes/origin/main", "main")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
    )


def main_ref() -> str | None:
    """The ref this clone knows as main, or None if it knows none of them."""
    for candidate in MAIN_CANDIDATES:
        if (
            _git(
                "rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"
            ).returncode
            == 0
        ):
            return candidate
    return None


def resolves(sha: str) -> bool:
    return _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def is_shallow() -> bool:
    return _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"


def on_a_remote_ref(sha: str) -> bool:
    return bool(
        _git("for-each-ref", "--contains", sha, "refs/remotes/origin/*").stdout.strip()
    )


def is_ancestor(sha: str, main: str) -> bool:
    return _git("merge-base", "--is-ancestor", sha, main).returncode == 0


def check_record(path: Path, main: str | None) -> tuple[list[str], str]:
    """Findings for one record, and a one-word outcome for the summary."""
    findings: list[str] = []
    name = path.name
    record = json.loads(path.read_text(encoding="utf-8"))

    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        findings.append(
            f"{name}: no `provenance` object. A capability record that does not say where it came "
            f"from cannot be audited, and its absence is indistinguishable from an oversight."
        )
        return findings, "missing"

    if "source_commit" not in provenance:
        findings.append(
            f"{name}: `provenance` has no `source_commit` key. Use null with "
            f"`why_no_source_commit` if the record genuinely does not attest to one."
        )
        return findings, "missing"

    sha = provenance["source_commit"]
    if sha is None:
        if not str(provenance.get("why_no_source_commit", "")).strip():
            findings.append(
                f"{name}: `source_commit` is null without `why_no_source_commit`. Declining to "
                f"name a commit is allowed; declining silently is the defect this guard exists for."
            )
            return findings, "declined"
        return findings, "declined"

    if not isinstance(sha, str) or not BARE_SHA.match(sha):
        findings.append(
            f"{name}: `source_commit` is {sha!r}, which is not a bare hex sha. Prose belongs in a "
            f"neighbouring key; a value no tool can parse is a value no tool will check."
        )
        return findings, "unparseable"

    # One record also carries the commit inside a `generator` block that describes the script run.
    # Two keys holding one value is how they drift, so they are required to agree.
    generator = record.get("generator")
    if isinstance(generator, dict) and "repository_commit" in generator:
        echoed = generator["repository_commit"]
        if echoed != sha:
            findings.append(
                f"{name}: `generator.repository_commit` is {echoed!r} but "
                f"`provenance.source_commit` is {sha!r}. One value in two places has drifted, "
                f"and a reader has no way to tell which is current."
            )
            return findings, "self-inconsistent"

    claimed = provenance.get("main_is_an_ancestor_of_this_commit", "absent")
    if claimed == "absent":
        findings.append(
            f"{name}: names {sha[:11]} without `main_is_an_ancestor_of_this_commit`. Five commits "
            f"in this set are deliberately not ancestors, so a reader cannot tell a deliberate "
            f"reference from a stale one unless the record says which it is."
        )
        return findings, "unflagged"

    if main is None:
        return findings, "main-unknown"

    if not resolves(sha):
        if is_shallow():
            return findings, "unresolvable-shallow-clone"
        # Absence is informative here: a clone that has main has every ancestor of main, so a
        # missing object cannot be one. This confirms false and refutes true.
        if claimed is True:
            findings.append(
                f"{name}: claims {sha[:11]} is an ancestor of main, but this clone has main and "
                f"does not have that commit. Every ancestor of main is present in a clone that "
                f"has main, so the claim cannot hold."
            )
            return findings, "wrong"
        if claimed is None:
            findings.append(
                f"{name}: records null ancestry for {sha[:11]}. This clone has main and not that "
                f"commit, which settles it: not an ancestor. Say false."
            )
            return findings, "needlessly-unstated"
        return findings, "verified-non-ancestor-by-absence"

    actual = is_ancestor(sha, main)
    if claimed is not actual:
        findings.append(
            f"{name}: claims `main_is_an_ancestor_of_this_commit` is {claimed!r} for {sha[:11]}, "
            f"but against {main} it is {actual!r}."
        )
        return findings, "wrong"

    return findings, "verified" if actual else "verified-non-ancestor"


def main() -> int:
    records = sorted(ROOT.glob("capability-record*.json"))
    if not records:
        print(f"No capability records under {ROOT}", file=sys.stderr)
        return 1

    main = main_ref()
    findings: list[str] = []
    outcomes: dict[str, list[str]] = {}
    for path in records:
        record_findings, outcome = check_record(path, main)
        findings.extend(record_findings)
        outcomes.setdefault(outcome, []).append(path.name)

    if findings:
        print(f"{len(findings)} provenance finding(s):")
        for finding in findings:
            print(f"  - {finding}")
        return 1

    counts = ", ".join(
        f"{len(names)} {outcome}" for outcome, names in sorted(outcomes.items())
    )
    print(
        f"All {len(records)} capability records state provenance in one shape: {counts}."
    )
    if main is None:
        print(
            "This clone knows no main ref, so stated ancestry was not compared against the "
            "repository. Shape and resolvability were still checked.",
            file=sys.stderr,
        )
    skipped = outcomes.get("unresolvable-shallow-clone", [])
    if skipped:
        # Said loudly on purpose. The counts above would otherwise read as though the ancestry of
        # these records had been checked, when in a shallow clone it cannot be.
        print(
            f"\n{len(skipped)} record(s) name a commit this shallow clone does not have, so their "
            f"stated ancestry was NOT checked. Absence proves nothing here: the history is "
            f"truncated by the fetch rather than by the commit being absent. Re-run in a full "
            f"clone to check these.",
            file=sys.stderr,
        )
        for name in skipped:
            print(f"  unchecked: {name}", file=sys.stderr)
    for name in outcomes.get("declined", []):
        print(
            f"  note: {name} names no commit, with its reason recorded in the record."
        )

    # Not gated, because it depends on how the repository was fetched. Reported, because it is the
    # thing a reader actually needs: a commit on no remote ref cannot be looked up by anyone else.
    unpushed = sorted(
        {
            path.name: json.loads(path.read_text(encoding="utf-8"))["provenance"][
                "source_commit"
            ]
            for path in records
        }.items()
    )
    stranded = [
        (name, sha)
        for name, sha in unpushed
        if isinstance(sha, str) and not on_a_remote_ref(sha)
    ]
    if stranded:
        print(
            f"\n{len(stranded)} record(s) name a commit on no remote ref in this clone, so another "
            f"reader cannot resolve it. Not a failure -- it depends on the fetch -- but it is why "
            f"each of these records also names the commit on main where its work landed:"
        )
        for name, sha in stranded:
            print(f"  - {name}: {sha[:11]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
