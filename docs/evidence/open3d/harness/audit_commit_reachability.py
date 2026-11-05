#!/usr/bin/env python3
"""Test advertised-ref containment and direct-SHA fetchability for named commits.

`audit_record_provenance.py` already reports reachability, but it reports it *from the clone
it happens to be run in*, and it says so. That turned out to matter. The records make the
stronger claim -- "on no advertised remote ref" -- and the clone that produced it held
only `refs/heads/*` and was shallow. Most of what this remote advertises is under `refs/pull/`,
so the original check consulted roughly a quarter of the refs a reader could resolve a commit
from. The claim happens to be true. It was not established.

Counts are deliberately absent from this prose. They are in the report, because this remote
gains refs continuously -- three in the three quarters of an hour between a reviewer's two
measurements -- and a number written into a docstring is wrong shortly after it is written. One
of mine was already wrong when it was published.

So this harness consults what the remote actually advertises rather than what the local clone
has fetched, and it closes the two holes that made the earlier answer unsound:

1. **Unfetched refs.** `git ls-remote` is the ground truth for what exists remotely.
   Every advertised ref -- `refs/pull/*/head`, `refs/pull/*/merge` and
   `refs/tags/*` -- is fetched into a namespace of its own before any reachability question is
   asked. Merge refs were skipped at first, on the argument that a merge ref is GitHub's
   computed merge of the base and the PR head and so cannot contain a commit the head lacks. A
   reviewer tested that on every merge ref here and found one where the property holds for a
   different reason: the head had moved since the merge ref was computed. It was harmless, and
   a force-push is the case that would not be, so the argument is gone and the refs are
   measured.

2. **Shallow truncation.** A shallow clone makes absence unsound in one direction -- a commit
   below the graft boundary is unreachable in the walk whether or not a ref contains it, so
   "no ref contains it" could be an artefact of the clone. The test is not whether the clone is
   shallow but whether any *named* commit could be concealed by it, which is true only if that
   commit is an ancestor of a graft point. That is checked per commit and reported, and a
   commit that could be concealed is reported as indeterminate rather than as stranded.

Reachability is still not a correctness claim. No advertised ref means only that no current ref
contains the commit. Direct-SHA fetchability is probed separately, and neither result says whether
the measurements taken there were right.

Writes docs/evidence/open3d/commit-reachability-audit.json.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
REPO = EVIDENCE.parents[2]
OUT = EVIDENCE / "commit-reachability-audit.json"
SOURCE_REPO = REPO
AUDIT_REPO = REPO

# Namespaces this harness fetches into. Kept out of refs/remotes/origin/* so a run never
# rewrites what the clone's own tracking refs mean.
AUDIT_NS = "refs/remotes/provenance-audit/"
PR_NS = AUDIT_NS + "pr"
MERGE_NS = AUDIT_NS + "merge"
TAG_NS = AUDIT_NS + "tag"
BRANCH_NS = AUDIT_NS + "branch"

SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def git(*args: str, check: bool = True) -> str:
    done = subprocess.run(
        ["git", "-C", str(AUDIT_REPO), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def source_git(*args: str, check: bool = True) -> str:
    done = subprocess.run(
        ["git", "-C", str(SOURCE_REPO), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and done.returncode != 0:
        raise RuntimeError(f"source git {' '.join(args)} failed: {done.stderr.strip()}")
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
    """Mirror every resolvable ref locally. Refspecs are forced because PR heads move: a second
    run of this audit otherwise fails on the first non-fast-forward update, which is a stale
    mirror reported as a broken harness."""
    pulls = [r for r in refs if r.startswith("refs/pull/") and r.endswith("/head")]
    merges = [r for r in refs if r.startswith("refs/pull/") and r.endswith("/merge")]
    tags = [r for r in refs if r.startswith("refs/tags/")]
    branches = [r for r in refs if r.startswith("refs/heads/")]
    if branches:
        git(
            "fetch",
            "--quiet",
            "--no-write-fetch-head",
            "origin",
            f"+refs/heads/*:{BRANCH_NS}/*",
        )
    if pulls:
        git(
            "fetch",
            "--quiet",
            "--no-write-fetch-head",
            "origin",
            f"+refs/pull/*/head:{PR_NS}/*",
        )
    if merges:
        git(
            "fetch",
            "--quiet",
            "--no-write-fetch-head",
            "origin",
            f"+refs/pull/*/merge:{MERGE_NS}/*",
        )
    if tags:
        git(
            "fetch",
            "--quiet",
            "--no-write-fetch-head",
            "origin",
            f"+refs/tags/*:{TAG_NS}/*",
        )
    other = len(refs) - len(pulls) - len(merges) - len(tags) - len(branches)
    return {
        "pull_heads_advertised": len(pulls),
        "pull_merges_advertised": len(merges),
        "tags_advertised": len(tags),
        "branches_advertised": len(branches),
        "refs_in_no_other_category": other,
        "fetched_pull_refs": len(
            git("for-each-ref", PR_NS, "--format=%(refname)").splitlines()
        ),
        "fetched_merge_refs": len(
            git("for-each-ref", MERGE_NS, "--format=%(refname)").splitlines()
        ),
        "fetched_tag_refs": len(
            git("for-each-ref", TAG_NS, "--format=%(refname)").splitlines()
        ),
        "fetched_branch_refs": len(
            git("for-each-ref", BRANCH_NS, "--format=%(refname)").splitlines()
        ),
    }


def graft_points() -> list[str]:
    shallow = Path(git("rev-parse", "--git-common-dir")) / "shallow"
    if not shallow.is_absolute():
        shallow = AUDIT_REPO / shallow
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
            ["git", "-C", str(AUDIT_REPO), "merge-base", "--is-ancestor", sha, graft],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode == 0:
            return True
    return False


def object_type(sha: str) -> str | None:
    done = subprocess.run(
        ["git", "-C", str(AUDIT_REPO), "cat-file", "-t", sha],
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
    for record in sorted(EVIDENCE.glob("capability-record*.json")):
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


def evidence_outside_the_record_glob() -> dict[str, object]:
    """How much evidence this audit's glob does not reach, as a number rather than a hope.

    A reviewer measured the whole evidence directory rather than the records this change
    rewrote and found seven JSON files that name no commit a reader can resolve and are not
    `capability-record-*`, so no amount of hardening inside the glob will ever reach them. The
    shape is right and its enforcement stops short of the files that still need it.

    Widening the glob is the real fix and is deliberately not done here: those files carry no
    provenance block at all, so it would mean designing provenance for six artifacts this
    change did not produce, on a base 310 commits behind main where the seventh does not yet
    exist. Reporting the boundary as a count keeps it from being a silent hole in the meantime,
    which is the part that was actually wrong.
    """
    outside = []
    for path in sorted(EVIDENCE.glob("*.json")):
        if path.name.startswith("capability-record"):
            continue
        data = json.loads(path.read_text())
        # Not every evidence file is an object. Some are top-level arrays of readings, which
        # cannot carry a provenance block at all and are counted as unreached rather than
        # crashing the audit that is supposed to be measuring them.
        provenance = data.get("provenance") if isinstance(data, dict) else None
        blob = json.dumps(data)
        outside.append(
            {
                "file": path.name,
                "has_a_provenance_block": isinstance(provenance, dict),
                "carries_digests": bool(re.search(r"[0-9a-f]{64}", blob)),
                "names_a_commit": isinstance(provenance, dict)
                and any(
                    isinstance(v, str) and SHA_RE.match(v) for v in provenance.values()
                ),
            }
        )
    unreached = [f for f in outside if not f["has_a_provenance_block"]]
    carrying = [f for f in unreached if f["carries_digests"]]
    return {
        "evidence_json_outside_this_audits_glob": len(outside),
        "of_those_with_no_provenance_block_at_all": len(unreached),
        "of_those_that_also_carry_digests": len(carrying),
        "why_three_numbers_and_not_one": "A reviewer measured this as seven files. That is the "
        "subset carrying digests and naming no resolvable commit; the broadest count here is "
        "every non-record evidence JSON. Different populations rather than a disagreement, and "
        "the digest-carrying count is the one that matters, because a digest with no declared "
        "origin is the case this change exists to close.",
        "files_carrying_digests_with_no_declared_provenance": carrying,
        "files": unreached,
        "why_this_is_reported_instead_of_fixed": "Widening the glob means designing provenance "
        "for artifacts this change did not produce, on a base well behind main. Reported so the "
        "enforcement edge is a number a reader can see rather than an absence they have to "
        "notice.",
    }


def classify(sha: str, grafts: list[str]) -> dict[str, object]:
    direct_fetch_attempted = False
    if object_type(sha) != "commit":
        direct_fetch_attempted = True
        subprocess.run(
            [
                "git",
                "-C",
                str(AUDIT_REPO),
                "fetch",
                "--quiet",
                "--no-write-fetch-head",
                "origin",
                sha,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    object_available = object_type(sha) == "commit"
    direct_fetch_succeeded = direct_fetch_attempted and object_available
    if not object_available:
        return {
            "commit": sha,
            "verdict": "absent_from_this_clone",
            "remote_refs_containing": None,
            "direct_sha_fetch_attempted": direct_fetch_attempted,
            "direct_sha_fetchable": direct_fetch_succeeded,
            "local_refs_containing": 0,
            "shallow_clone_could_conceal_it": False,
            "orphaned_tracking_refs_containing": 0,
            "example_remote_refs": [],
            "note": "The isolated audit repository could neither find the object through an "
            "advertised ref nor fetch it directly by SHA at measurement time.",
        }
    containing = git(
        "for-each-ref", "--contains", sha, "--format=%(refname)"
    ).splitlines()
    # The audit repository is fresh and private. Every persistent ref in it is
    # created from the current ls-remote result under AUDIT_NS, so stale local or
    # orphaned tracking refs cannot be counted as remote evidence.
    remote = [r for r in containing if r.startswith(AUDIT_NS)]
    if not remote and not direct_fetch_attempted:
        direct_fetch_attempted = True
        fetched = subprocess.run(
            [
                "git",
                "-C",
                str(AUDIT_REPO),
                "fetch",
                "--quiet",
                "--no-write-fetch-head",
                "origin",
                sha,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        direct_fetch_succeeded = (
            fetched.returncode == 0 and object_type(sha) == "commit"
        )
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
        "direct_sha_fetch_attempted": direct_fetch_attempted,
        "direct_sha_fetchable": (
            direct_fetch_succeeded if direct_fetch_attempted else None
        ),
        "local_refs_containing": 0,
        "shallow_clone_could_conceal_it": concealable,
        "orphaned_tracking_refs_containing": 0,
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
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "source_repository_head": source_git("rev-parse", "HEAD^{commit}"),
        "remote_name": "origin",
        "audit_workspace": "fresh private bare repository removed after this run",
        "what_this_answers": "For every commit a capability record names, does a current "
        "advertised origin ref contain it, and if not can origin fetch the object by SHA?",
        "why_it_exists": "The records claimed no advertised remote refs. The clone that checked held "
        "only refs/heads/* out of every ref this remote advertises, and was shallow. The "
        "claim was true and unestablished, which is its own kind of defect.",
        "ref_coverage": {
            "total_refs_advertised_by_remote": len(refs),
            **coverage,
            "why_merge_refs_are_no_longer_skipped": "The first version skipped them, arguing a "
            "merge ref has its PR head as an ancestor and so cannot hold a commit the head "
            "lacks. A reviewer tested it on all of them and found one where the head had moved "
            "since the merge ref was computed, so the property held for a different reason than "
            "the one I gave. It was harmless, and a force-push is the case that would not be. "
            "They are now fetched and tested, so the audit rests on no argument about refs.",
            "counts_here_are_a_moment_not_a_constant": "This remote gains refs continuously. A "
            "count written into prose is stale almost immediately -- one of mine was wrong by "
            "the time it was published -- so the numbers live in this report and the prose "
            "points here.",
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
        "what_this_audits_glob_does_not_reach": evidence_outside_the_record_glob(),
        "what_this_does_not_establish": "That any measurement taken at these commits is "
        "correct. Reachability is about whether a reader can "
        "look the commit up, nothing more.",
    }


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--remote", default="origin")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global SOURCE_REPO, AUDIT_REPO, EVIDENCE, OUT
    args = _arguments(argv)
    SOURCE_REPO = args.repo.resolve()
    EVIDENCE = args.evidence_dir.resolve()
    OUT = args.out.resolve()
    remote_url = source_git("remote", "get-url", args.remote)

    with tempfile.TemporaryDirectory(prefix="open3d-reachability-") as temporary:
        AUDIT_REPO = Path(temporary) / "audit.git"
        subprocess.run(
            ["git", "init", "--quiet", "--bare", str(AUDIT_REPO)], check=True
        )
        subprocess.run(
            ["git", "-C", str(AUDIT_REPO), "remote", "add", "origin", remote_url],
            check=True,
        )
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
        f"{stranded} commit(s) on zero advertised refs with the object available, "
        f"{tally.get('resolvable_remotely', 0)} resolvable, {unknown} indeterminate."
    )
    print(f"Wrote {OUT}")
    # Indeterminate is the only state that makes the report untrustworthy: it means the clone
    # could not tell stranded from reachable, and no record should rest on that.
    return 1 if unknown else 0


if __name__ == "__main__":
    sys.exit(main())
