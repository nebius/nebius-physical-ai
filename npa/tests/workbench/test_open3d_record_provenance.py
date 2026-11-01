"""The provenance guard runs on the committed records, and fails on each thing it claims to catch.

A review lane found that neither existing Open3D evidence harness was run by any test or CI job, so
both could have gone red on the committed set with nothing noticing. A new guard with no test is
the same defect, one file along.

Negative controls work on a copy of the evidence directory, so a test run never mutates the
committed records. The copy keeps the real repository as its git context, because the guard asks
git whether a commit resolves and whether main is an ancestor of it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/evidence/open3d"
GUARD = "harness/audit_record_provenance.py"


def run(evidence: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(evidence / GUARD)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    """A copy of the evidence directory, inside the real repository so git still answers."""
    copy = ROOT / "docs/evidence" / f"_provenance_test_{tmp_path.name}"
    shutil.copytree(EVIDENCE, copy)
    try:
        yield copy
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def edit(evidence: Path, name: str, mutate) -> None:
    path = evidence / name
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data, indent=2) + "\n")


@pytest.fixture
def guard(tmp_path: Path):
    """The guard imported as a module, so the git helpers can be replaced deterministically."""
    spec = importlib.util.spec_from_file_location(
        "audit_record_provenance", EVIDENCE / GUARD
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _record_dir(tmp_path: Path) -> Path:
    global _RECORD_DIR
    _RECORD_DIR = tmp_path
    return tmp_path


def _record_file(provenance: dict) -> Path:
    path = _RECORD_DIR / "capability-record-synthetic.json"
    path.write_text(json.dumps({"provenance": provenance}, indent=2))
    return path


def test_the_committed_records_pass() -> None:
    result = run(EVIDENCE)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "in one shape" in result.stdout


def test_every_committed_record_carries_the_shape() -> None:
    """The guard is what enforces this; this is the readable statement of what it enforces."""
    records = sorted(EVIDENCE.glob("capability-record*.json"))
    assert len(records) >= 10, records
    for path in records:
        provenance = json.loads(path.read_text()).get("provenance")
        assert isinstance(provenance, dict), path.name
        assert "source_commit" in provenance, path.name
        if provenance["source_commit"] is None:
            assert provenance.get("why_no_source_commit"), path.name
        else:
            assert "main_is_an_ancestor_of_this_commit" in provenance, path.name


def test_a_record_with_no_provenance_is_caught(evidence: Path) -> None:
    """The state ten of these records were in before this change."""
    edit(
        evidence, "capability-record-distance-bands.json", lambda d: d.pop("provenance")
    )

    result = run(evidence)
    assert result.returncode == 1, result.stdout
    assert "no `provenance` object" in result.stdout


def test_a_sha_with_prose_beside_it_is_caught(evidence: Path) -> None:
    """The real pre-fix value in capability-record.json, which no scanner could parse."""

    def mangle(data: dict) -> None:
        sha = data["provenance"]["source_commit"]
        data["provenance"]["source_commit"] = f"{sha} (npa worktree HEAD at build)"

    edit(evidence, "capability-record.json", mangle)

    result = run(evidence)
    assert result.returncode == 1, result.stdout
    assert "not a bare hex sha" in result.stdout


def test_a_commit_named_without_the_ancestry_flag_is_caught(evidence: Path) -> None:
    """Asserts which check fired, not just that something did.

    Mutation testing caught this: with the explicit branch disabled, the missing key falls through
    to the comparison and is reported as *claiming* an ancestry of `'absent'`, which no record
    claims. The guard still fails, so a test asserting only the key name or the exit code cannot
    tell the two apart -- and the message a reader would get is wrong about what they did.
    """
    edit(
        evidence,
        "capability-record-standard-runtime.json",
        lambda d: d["provenance"].pop("main_is_an_ancestor_of_this_commit"),
    )

    result = run(evidence)
    assert result.returncode == 1, result.stdout
    assert "without `main_is_an_ancestor_of_this_commit`" in result.stdout
    assert "claims" not in result.stdout, (
        "reported as a false claim rather than a missing key"
    )


def test_an_ancestry_claim_that_contradicts_the_repository_is_caught(
    evidence: Path,
) -> None:
    """A wrong flag is worse than no flag, because it reads as verified."""
    edit(
        evidence,
        "capability-record.json",
        lambda d: d["provenance"].update({"main_is_an_ancestor_of_this_commit": False}),
    )

    result = run(evidence)
    assert result.returncode == 1, result.stdout
    assert "but against" in result.stdout


def test_absence_from_a_full_clone_refutes_an_ancestor_claim(guard) -> None:
    """A clone holding main holds every ancestor of main, so a missing object cannot be one.

    Checked against the function rather than through the subprocess, because this repository is a
    shallow clone and the guard correctly declines to draw the inference there. Testing it through
    git would mean the control passes or vanishes depending on how the host fetched the repository,
    which is the kind of test that looks like coverage and is not.
    """
    guard.is_shallow = lambda: False
    guard.resolves = lambda sha: False

    findings, outcome = guard.check_record(
        _record_file(
            {"source_commit": "0" * 40, "main_is_an_ancestor_of_this_commit": True}
        ),
        "origin/main",
    )
    assert outcome == "wrong", (findings, outcome)
    assert "does not have that commit" in findings[0]


def test_absence_from_a_full_clone_confirms_a_non_ancestor_claim(guard) -> None:
    """The other half of the inference, which must not be rejected.

    Without this control the guard could satisfy its sibling by refusing every unresolvable
    commit, and then fail on nine committed records the moment it ran in a full clone that lacks
    the retired lane branches.
    """
    guard.is_shallow = lambda: False
    guard.resolves = lambda sha: False

    findings, outcome = guard.check_record(
        _record_file(
            {"source_commit": "0" * 40, "main_is_an_ancestor_of_this_commit": False}
        ),
        "origin/main",
    )
    assert findings == []
    assert outcome == "verified-non-ancestor-by-absence"


def test_a_shallow_clone_reports_the_ancestry_it_could_not_check(guard) -> None:
    """Degrading quietly would be worse than failing: the summary would read as verified."""
    guard.is_shallow = lambda: True
    guard.resolves = lambda sha: False

    findings, outcome = guard.check_record(
        _record_file(
            {"source_commit": "0" * 40, "main_is_an_ancestor_of_this_commit": True}
        ),
        "origin/main",
    )
    assert findings == []
    assert outcome == "unresolvable-shallow-clone", outcome


def test_a_duplicated_commit_that_drifts_is_caught(evidence: Path) -> None:
    """One value in two places is how they come apart."""
    edit(
        evidence,
        "capability-record-provenance-bound-preview.json",
        lambda d: d["generator"].update({"repository_commit": "d06e95846"}),
    )

    result = run(evidence)
    assert result.returncode == 1, result.stdout
    assert "has drifted" in result.stdout


def test_declining_to_name_a_commit_needs_a_reason(evidence: Path) -> None:
    def silent(data: dict) -> None:
        data["provenance"]["source_commit"] = None
        data["provenance"].pop("why_no_source_commit", None)

    edit(evidence, "capability-record-distance-bands.json", silent)

    result = run(evidence)
    assert result.returncode == 1, result.stdout
    assert "declining silently" in result.stdout


def test_declining_with_a_reason_is_accepted(evidence: Path) -> None:
    """The benign control. A gate that rejects every null is not measuring the reason."""

    def stated(data: dict) -> None:
        data["provenance"]["source_commit"] = None
        data["provenance"]["why_no_source_commit"] = "No single commit produced this."

    edit(evidence, "capability-record-distance-bands.json", stated)

    result = run(evidence)
    assert result.returncode == 0, result.stdout + result.stderr


RECOVERED_PROBES = {
    "capability-record-viewer-reconciliation.json": (
        "test_a_cloud_only_run_offers_no_empty_surface_tab"
    ),
    "capability-record-support-claim-correction.json": "FAR_BAND_AREA_SHARE",
}


def test_the_recovered_probes_actually_reached_the_tree() -> None:
    """The half of the recovery argument that every clone can check.

    The record's claim is that each recovered commit's contribution reached main, so the probe
    symbol must be in the tree. This needs no history and no unreachable objects, which is why
    it is split out from the trace below.
    """
    for name, probe in RECOVERED_PROBES.items():
        found = subprocess.run(
            ["git", "-C", str(ROOT), "grep", "--quiet", probe, "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert found.returncode == 0, (
            f"{name} names probe {probe}, absent from the tree at HEAD"
        )


def test_the_recovered_commits_are_the_ones_the_records_name() -> None:
    """Three records named no commit; each value was recovered from evidence, not chosen.

    Pinned because a plausible-looking commit is the easiest thing in this change to get wrong,
    and the recovery argument lives in prose that nothing else checks.

    It cannot be pinned everywhere, and that is this change's own finding rather than a weakness
    in the test. These commits are on zero remote refs, so a clone made by anyone else -- CI's
    included -- does not have the objects, and `git log -S` has nothing to search. The first
    version of this test asserted the trace unconditionally and failed in CI for exactly the
    reason the record documents. Skipping where the objects are absent is honest; asserting a
    trace that cannot run is not, and silently passing would be worse than either.
    """
    missing = [
        sha
        for name in RECOVERED_PROBES
        if (
            sha := json.loads((EVIDENCE / name).read_text())["provenance"][
                "source_commit"
            ]
        )
        and subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", sha],
            capture_output=True,
            check=False,
        ).returncode
        != 0
    ]
    if missing:
        pytest.skip(
            "This clone does not hold "
            + ", ".join(sha[:11] for sha in missing)
            + ". They are on zero remote refs, so only the clone that made them can trace "
            "them; see commit-reachability-audit.json. The tree-level half of this claim is "
            "checked by test_the_recovered_probes_actually_reached_the_tree."
        )

    for name, probe in RECOVERED_PROBES.items():
        sha = json.loads((EVIDENCE / name).read_text())["provenance"]["source_commit"]
        found = subprocess.run(
            ["git", "-C", str(ROOT), "log", "-S", probe, "--format=%H", "--all"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        assert sha in found, (
            f"{name} names {sha[:11]}, but no commit introducing {probe} matches it"
        )


def test_the_reachability_audit_consulted_the_refs_the_remote_advertises() -> None:
    """The zero-remote-refs claim must rest on the remote's refs, not the clone's fetch refspec.

    The first version of that claim was asserted from a clone whose only refspec is
    refs/heads/*, against a remote advertising 1274 refs. It was true and unestablished. This
    pins the coverage so a future run cannot quietly narrow back to the branches it happens to
    have.
    """
    audit = json.loads((EVIDENCE / "commit-reachability-audit.json").read_text())
    coverage = audit["ref_coverage"]

    assert coverage["total_refs_advertised_by_remote"] >= 1000
    # PR heads are the bulk of what a branch-only fetch misses, and the reason the first answer
    # was unsound. If they stop being consulted the audit is back to checking a quarter of them.
    assert coverage["fetched_pull_refs"] >= coverage["pull_heads_advertised"] - 1
    assert coverage["fetched_tag_refs"] == coverage["tags_advertised"]
    # Merge refs were skipped on an argument until a reviewer found a ref the argument did not
    # describe. They are fetched and tested now, so the audit rests on no argument about refs.
    assert coverage["fetched_merge_refs"] >= coverage["pull_merges_advertised"] - 1
    # The decomposition has to close, or "the rest are branches" is hiding a category.
    assert (
        coverage["pull_heads_advertised"]
        + coverage["pull_merges_advertised"]
        + coverage["tags_advertised"]
        + coverage["branches_advertised"]
        + coverage["refs_in_no_other_category"]
        == coverage["total_refs_advertised_by_remote"]
    ), coverage


def test_shallow_truncation_is_ruled_out_per_commit_not_waved_at() -> None:
    """Absence of a ref is only evidence if truncation could not have hidden the commit."""
    audit = json.loads((EVIDENCE / "commit-reachability-audit.json").read_text())

    assert audit["shallow_clone"]["commits_that_could_be_concealed"] == 0
    # Indeterminate is the state that would make the report untrustworthy: the clone could not
    # tell stranded from reachable. The harness exits nonzero on it, so a committed report
    # containing one means the report was produced by an older or patched harness.
    assert "indeterminate_shallow_clone_could_conceal_it" not in audit["tally"]


def test_reachability_is_scoped_to_the_remote_not_to_local_remote_shaped_refs() -> None:
    """Orphaned tracking refs live under refs/remotes/ while being as local as a branch.

    Counting them inflated one commit's reachability to 684 refs. For a stranded verdict the
    wider net is harmless, but "another reader can resolve this" is a claim about the remote,
    and this clone has 71 refs that look remote and are not.
    """
    audit = json.loads((EVIDENCE / "commit-reachability-audit.json").read_text())
    stranded = [c for c in audit["commits"] if c["verdict"] == "on_zero_remote_refs"]

    assert len(stranded) == 7, [c["commit"][:12] for c in stranded]
    for commit in stranded:
        assert commit["remote_refs_containing"] == 0, commit
        assert commit["orphaned_tracking_refs_containing"] == 0, commit


def test_every_record_claiming_no_remote_ref_names_a_commit_the_audit_agrees_is_stranded() -> (
    None
):
    """The roll-up count was wrong once; this checks the per-record prose against the data.

    My summary said "nine of the ten named commits", conflating nine records with seven
    distinct commits because three records share one. Each individual claim was right, so a
    test on the records alone would have passed. This one joins them to the audit.
    """
    audit = json.loads((EVIDENCE / "commit-reachability-audit.json").read_text())
    verdicts = {c["commit"]: c["verdict"] for c in audit["commits"]}

    claiming = []
    for record in sorted(EVIDENCE.glob("capability-record-*.json")):
        provenance = json.loads(record.read_text()).get("provenance", {})
        if "this_commit_is_on_no_remote_ref" not in provenance:
            continue
        sha = provenance.get("source_commit") or provenance.get("candidate_commit")
        claiming.append(record.name)
        assert verdicts.get(sha) == "on_zero_remote_refs", (
            record.name,
            sha,
            verdicts.get(sha),
        )

    assert len(claiming) == 9, claiming
    # Nine records, seven commits: the gap is the whole point of the correction.
    named = {
        json.loads((EVIDENCE / name).read_text())["provenance"].get("source_commit")
        or json.loads((EVIDENCE / name).read_text())["provenance"].get(
            "candidate_commit"
        )
        for name in claiming
    }
    assert len(named) == 7, sorted(s[:12] for s in named)


def test_the_audit_reports_the_evidence_its_own_glob_cannot_reach() -> None:
    """An enforcement edge should be a number a reader can see, not an absence they must notice.

    A reviewer measured the whole evidence directory rather than the records this change
    rewrote, and found files that carry digests, name no resolvable commit, and are not
    `capability-record-*`, so no hardening inside the glob will ever reach them. Widening it
    means designing provenance for artifacts this change did not produce; reporting the count
    is what stops it being silent in the meantime.
    """
    audit = json.loads((EVIDENCE / "commit-reachability-audit.json").read_text())
    edge = audit["what_this_audits_glob_does_not_reach"]

    assert edge["evidence_json_outside_this_audits_glob"] > 0
    # The digest-carrying subset is the one that matters: a digest with no declared origin is
    # exactly the case this change exists to close. The reviewer measured seven on main; this
    # base is behind main and uses a slightly different population, and the lists overlap in
    # five. Pinning the exact count would pin that difference rather than the property.
    assert edge["of_those_that_also_carry_digests"] >= 5, edge
    assert (
        len(edge["files_carrying_digests_with_no_declared_provenance"])
        == (edge["of_those_that_also_carry_digests"])
    )
    for entry in edge["files_carrying_digests_with_no_declared_provenance"]:
        assert entry["carries_digests"] is True, entry
        assert entry["has_a_provenance_block"] is False, entry
        assert not entry["file"].startswith("capability-record-"), entry
