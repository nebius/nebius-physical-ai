#!/usr/bin/env python3
"""Test whether the commits named by capability records are reachable from any remote ref.

`audit_record_provenance.py` already reports reachability, but it reports it *from the clone
it happens to be run in*, and it says so. That turned out to matter. The records make the
stronger claim -- "on ZERO remote refs", in ten files -- and the clone that produced it held
only `refs/heads/*` and was shallow. This repository advertises 1271 refs, 955 of them under
`refs/pull/`, so the original check consulted roughly a quarter of the refs a reader could
resolve a commit from. The claim happens to be true. It was not established.

So this harness consults what the remote actually advertises rather than what the local clone
has fetched, and it closes the two holes that made the earlier answer unsound:

1. **Unfetched refs.** `git ls-remote` is the ground truth for what exists remotely. Every
   `refs/pull/*/head` and `refs/tags/*` is fetched into a namespace of its own before any
   reachability question is asked. `refs/pull/*/merge` is deliberately skipped: a merge ref is
   GitHub's computed merge of the base and the PR head, so it contains the head as an ancestor
   and cannot contain a commit the head does not.

2. **Shallow truncation.** A shallow clone makes absence unsound in one direction -- a commit
   below the graft boundary is unreachable in the walk whether or not a ref contains it, so
   "no ref contains it" could be an artefact of the clone. The test is not whether the clone is
   shallow but whether any *named* commit could be concealed by it, which is true only if that
   commit is an ancestor of a graft point. That is checked per commit and reported, and a
   commit that could be concealed is reported as indeterminate rather than as stranded.

Reachability is still not a correctness claim. A commit on no remote ref means no other reader
can look it up; it says nothing about whether the measurements taken there were right.

Writes docs/evidence/open3d/commit-reachability-audit.json.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
REPO = EVIDENCE.parents[2]
OUT = EVIDENCE / "commit-reachability-audit.json"

# Namespaces this harness fetches into. Kept out of refs/remotes/origin/* so a run never
# rewrites what the clone's own tracking refs mean.
AUDIT_NS = "refs/remotes/provenance-audit/"
PR_NS = AUDIT_NS + "pr"
TAG_NS = AUDIT_NS + "tag"

SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def git(*args: str, check: bool = True) -> str:
    done = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
    )
    if check and done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def advertised_refs() -> dict[str, str]:
    """What the remote says exists, which is the only honest denominator here."""
    refs = {}
    for line in git("ls-remote", "origin").splitlines():
        sha, _, name = line.partition("\t")
        if name:
            refs[name.strip()] = sha.strip()
    return refs


def fetch_everything_resolvable(refs: dict[str, str]) -> dict[str, int]:
    pulls = [r for r in refs if r.startswith("refs/pull/") and r.endswith("/head")]
    tags = [r for r in refs if r.startswith("refs/tags/")]
    if pulls:
        git(
            "fetch",
            "--quiet",
            "--no-write-fetch-head",
            "origin",
            f"refs/pull/*/head:{PR_NS}/*",
        )
    if tags:
        git(
            "fetch",
            "--quiet",
            "--no-write-fetch-head",
            "origin",
            f"refs/tags/*:{TAG_NS}/*",
        )
    return {
        "pull_heads_advertised": len(pulls),
        "pull_merges_skipped": len([r for r in refs if r.endswith("/merge")]),
        "tags_advertised": len(tags),
        "fetched_pull_refs": len(
            git("for-each-ref", PR_NS, "--format=%(refname)").splitlines()
        ),
        "fetched_tag_refs": len(
            git("for-each-ref", TAG_NS, "--format=%(refname)").splitlines()
        ),
    }


def graft_points() -> list[str]:
    shallow = Path(git("rev-parse", "--git-common-dir")) / "shallow"
    if not shallow.is_absolute():
        shallow = REPO / shallow
    if not shallow.exists():
        return []
    return [ln.strip() for ln in shallow.read_text().splitlines() if ln.strip()]


def could_be_concealed(sha: str, grafts: list[str]) -> bool:
    """True if truncated history could hide this commit, making absence uninformative.

    Shallowness removes the ancestors of each graft point. A commit is therefore invisible to
    a walk only if it is itself among those ancestors. Testing that directly beats arguing
    from commit dates, which rebases and clock skew make unreliable.
    """
    for graft in grafts:
        done = subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor", sha, graft],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode == 0:
            return True
    return False


def object_type(sha: str) -> str | None:
    done = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-t", sha],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip() if done.returncode == 0 else None


def named_commits() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Every commit-shaped provenance value in the capability records, by record.

    Written first with an allowlist of the three key names I expected, which found seven of
    the eleven sha-shaped values the records actually carry -- the same under-coverage defect
    as checking a quarter of the refs, one level up. Any value in `provenance` that is shaped
    like an object name is now taken as a candidate, and object *type* decides what it is:
    `producing_tree_identity` holds a tree, and asking whether a tree is on a branch is a
    category error rather than a finding.
    """
    found: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}
    for record in sorted(EVIDENCE.glob("capability-record-*.json")):
        prov = json.loads(record.read_text()).get("provenance", {})
        shas = []
        for key, value in prov.items():
            if not (isinstance(value, str) and SHA_RE.match(value)):
                continue
            kind = object_type(value) or "unknown_to_this_clone"
            kinds[value] = kind
            if kind == "commit" or kind == "unknown_to_this_clone":
                shas.append(value)
            else:
                found.setdefault(f"{record.name}#non-commit", []).append(
                    f"{key}={kind}"
                )
        if shas:
            found[record.name] = sorted(set(shas))
    return found, kinds


def classify(sha: str, grafts: list[str]) -> dict[str, object]:
    if (
        git("cat-file", "-e", sha, check=False) == ""
        and subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-e", sha],
            capture_output=True,
            check=False,
        ).returncode
        != 0
    ):
        return {
            "commit": sha,
            "verdict": "absent_from_this_clone",
            "remote_refs_containing": None,
            "note": "The object is not here, so this clone cannot answer. It is not evidence "
            "that the commit does not exist.",
        }
    containing = git(
        "for-each-ref", "--contains", sha, "--format=%(refname)"
    ).splitlines()
    # Only refs that trace back to a configured remote count. This clone carries orphaned
    # tracking refs -- refs/remotes/npabase627, refs/remotes/npapr607 and others left behind by
    # remotes that were removed -- which live in the remotes namespace while being as local as
    # any branch. Counting them inflated one commit's reachability to 684 refs. The direction
    # is safe for the stranded verdict, since a wider net can only find more, but "another
    # reader can resolve this" is a claim about the remote and must be scoped to it.
    live = tuple(f"refs/remotes/{name}/" for name in git("remote").splitlines())
    remote = [r for r in containing if r.startswith(live) or r.startswith(AUDIT_NS)]
    orphaned = [
        r for r in containing if r.startswith("refs/remotes/") and r not in remote
    ]
    concealable = could_be_concealed(sha, grafts)
    if remote:
        verdict = "resolvable_remotely"
    elif concealable:
        verdict = "indeterminate_shallow_clone_could_conceal_it"
    else:
        verdict = "on_zero_remote_refs"
    return {
        "commit": sha,
        "verdict": verdict,
        "remote_refs_containing": len(remote),
        "local_refs_containing": len(containing) - len(remote),
        "shallow_clone_could_conceal_it": concealable,
        "orphaned_tracking_refs_containing": len(orphaned),
        "example_remote_refs": sorted(remote)[:3],
    }


def build() -> dict[str, object]:
    refs = advertised_refs()
    coverage = fetch_everything_resolvable(refs)
    grafts = graft_points()
    by_record, kinds = named_commits()
    distinct = sorted(
        {
            sha
            for key, shas in by_record.items()
            if not key.endswith("#non-commit")
            for sha in shas
        }
    )
    results = [classify(sha, grafts) for sha in distinct]
    tally: dict[str, int] = {}
    for r in results:
        tally[str(r["verdict"])] = tally.get(str(r["verdict"]), 0) + 1
    return {
        "what_this_answers": "For every commit a capability record names, can another reader "
        "resolve it from a ref this remote advertises?",
        "why_it_exists": "The records claimed zero remote refs. The clone that checked held "
        "only refs/heads/* out of 1271 advertised refs, and was shallow. The "
        "claim was true and unestablished, which is its own kind of defect.",
        "ref_coverage": {
            "total_refs_advertised_by_remote": len(refs),
            **coverage,
            "why_merge_refs_are_skipped": "A refs/pull/N/merge is the computed merge of the "
            "base and the PR head, so it has the head as an "
            "ancestor and cannot contain what the head lacks.",
        },
        "shallow_clone": {
            "is_shallow": git("rev-parse", "--is-shallow-repository") == "true",
            "graft_points": len(grafts),
            "why_this_does_not_invalidate_the_result": "Truncation removes the ancestors of a "
            "graft point. A named commit is hidden "
            "only if it is one of them, which is "
            "tested per commit rather than argued "
            "from commit dates.",
            "commits_that_could_be_concealed": sum(
                1 for r in results if r.get("shallow_clone_could_conceal_it")
            ),
        },
        "tally": tally,
        "commits": results,
        "object_kinds": dict(sorted(kinds.items())),
        "records_naming_them": {k: v for k, v in sorted(by_record.items())},
        "what_this_does_not_establish": "That any measurement taken at these commits is "
        "correct. Reachability is about whether a reader can "
        "look the commit up, nothing more.",
    }


def main() -> int:
    report = build()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    tally = report["tally"]
    assert isinstance(tally, dict)
    stranded = tally.get("on_zero_remote_refs", 0)
    unknown = tally.get("indeterminate_shallow_clone_could_conceal_it", 0)
    coverage = report["ref_coverage"]
    assert isinstance(coverage, dict)
    print(
        f"Consulted {coverage['total_refs_advertised_by_remote']} advertised refs "
        f"({coverage['fetched_pull_refs']} PR heads, {coverage['fetched_tag_refs']} tags). "
        f"{stranded} commit(s) on zero remote refs, "
        f"{tally.get('resolvable_remotely', 0)} resolvable, {unknown} indeterminate."
    )
    print(f"Wrote {OUT.relative_to(REPO)}")
    # Indeterminate is the only state that makes the report untrustworthy: it means the clone
    # could not tell stranded from reachable, and no record should rest on that.
    return 1 if unknown else 0


if __name__ == "__main__":
    sys.exit(main())
