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


def test_the_recovered_commits_are_the_ones_the_records_name(evidence: Path) -> None:
    """Three records named no commit; each value was recovered from evidence, not chosen.

    Pinned because a plausible-looking commit is the easiest thing in this change to get wrong,
    and the recovery argument lives in prose that nothing else checks.
    """
    expected = {
        "capability-record-viewer-reconciliation.json": (
            "test_a_cloud_only_run_offers_no_empty_surface_tab"
        ),
        "capability-record-support-claim-correction.json": "FAR_BAND_AREA_SHARE",
    }
    for name, probe in expected.items():
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
