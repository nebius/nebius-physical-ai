"""Run the evidence audit, and prove it can fail.

`docs/evidence/open3d/harness/audit_evidence_references.py` recomputes every digest
an evidence record binds to a named file and exits non-zero on a mismatch. It is a
good check that nothing invoked: no workflow under `.github` named it, so an
unbound frame stayed unbound for a month and a review lane appended a byte to a
cited frame and got a byte-identically green audit.

A test is a better home than a workflow step. `harness-guardrails.yml` runs only in
the merge queue, so a step there would gate merges and show a PR reviewer nothing,
while this runs wherever pytest runs. The audit is stdlib-only and takes 50ms over
7.8 MB, so it costs nothing to run it everywhere.

Running it is not enough on its own. A check that passes proves nothing unless the
same check is shown to fail on the thing it is for, so the tamper cases below drive
the audit against a constructed directory and assert each finding.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

AUDIT = (
    Path(__file__).resolve().parents[3]
    / "docs/evidence/open3d/harness/audit_evidence_references.py"
)


def _audit(root: Path | None = None):
    """Load a fresh copy of the audit, optionally rooted at a constructed tree.

    `ROOT` is module-level and derived from the script's own location, so a tamper
    case needs its own module instance rather than a monkeypatched attribute shared
    with the test that runs against the real tree.
    """

    spec = importlib.util.spec_from_file_location(f"audit_{id(root)}", AUDIT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if root is not None:
        module.ROOT = root
    return module


def _run(module, *argv: str) -> int:
    saved = sys.argv
    sys.argv = ["audit", *argv]
    try:
        return module.main()
    finally:
        sys.argv = saved


def test_the_committed_evidence_passes_its_own_audit(capsys) -> None:
    """The gate itself, on the real tree."""

    assert AUDIT.exists(), AUDIT
    assert _run(_audit()) == 0, capsys.readouterr().out


def test_the_audit_binds_rather_than_counts(capsys) -> None:
    """A passing audit has to have hashed something, or it is a no-op in disguise.

    The first version of this guard announced that every checkable hash matched
    while asking whether a hash matched *some* file in the directory. A count of
    zero bound hashes would read as success.
    """

    _run(_audit())
    out = capsys.readouterr().out
    bound = int(out.split("hash(es) were bound")[0].split()[-1])
    assert bound >= 18, out


@pytest.fixture
def constructed(tmp_path: Path) -> Path:
    """A minimal evidence directory: one record, one frame it binds."""

    frame = tmp_path / "frame.png"
    frame.write_bytes(b"pretend-png-bytes")
    (tmp_path / "record.json").write_text(
        json.dumps(
            {
                "frames_measured": [
                    {
                        "frame": "frame.png",
                        "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    }
                ]
            },
            indent=2,
        )
    )
    return tmp_path


def test_a_replaced_frame_fails(constructed, capsys) -> None:
    """Review's own tamper test, which the audit used to pass byte-identically."""

    assert _run(_audit(constructed)) == 0, capsys.readouterr().out
    (constructed / "frame.png").write_bytes(b"pretend-png-bytes-plus-one")
    assert _run(_audit(constructed)) == 1
    out = capsys.readouterr().out
    assert "frame.png" in out and "which is actually" in out, out


def test_a_frame_cited_by_name_alone_is_a_finding(constructed, capsys) -> None:
    """The shape the nine refit frames were in: named, present, bound to nothing.

    Listing a filename proves the file exists. It does not say these are the bytes
    the measurement read, which is the only claim a reader of a record cares about.
    """

    (constructed / "record.json").write_text(
        json.dumps({"frames_measured": ["frame.png"], "note_sha256": "0" * 64})
    )
    assert _run(_audit(constructed)) == 1
    assert "no enclosing object declares why" in capsys.readouterr().out


def test_an_unbindable_hash_is_accepted_only_once_declared(constructed, capsys) -> None:
    """The category that drifted for a month inside a number that never moved.

    Unbindable hashes are legitimate -- a run's own artifacts, a source file pinned
    at a commit -- and they were reported as a count on a passing run. A review lane
    asked for them to be reduced or justified; one record was justified in prose that
    nothing read and the rest stayed as they were. So the declaration is enforced,
    and it scopes from wherever it sits down through everything beneath it.
    """

    record = constructed / "record.json"
    payload = {"run_artifacts": {"result_sha256": "a" * 64}}
    record.write_text(json.dumps(payload))
    assert _run(_audit(constructed)) == 1, "an undeclared unbindable hash must fail"
    assert "hashes_that_name_no_file_here" in capsys.readouterr().out

    payload["run_artifacts"]["hashes_that_name_no_file_here"] = (
        "of the artifact as the run wrote it, which this directory does not hold"
    )
    record.write_text(json.dumps(payload))
    assert _run(_audit(constructed)) == 0, capsys.readouterr().out


def test_a_declaration_covers_what_is_beneath_it_and_no_more(
    constructed, capsys
) -> None:
    """Scoping, asserted in both directions.

    A declaration at the root would be a blanket exemption; one beside the hashes it
    explains is a statement about those hashes. Both matter, so the sibling branch
    must stay a finding.
    """

    record = constructed / "record.json"
    record.write_text(
        json.dumps(
            {
                "declared": {
                    "hashes_that_name_no_file_here": "run output, not committed here",
                    "nested": {"deeper_sha256": "b" * 64},
                },
                "undeclared": {"other_sha256": "c" * 64},
            }
        )
    )
    assert _run(_audit(constructed)) == 1
    out = capsys.readouterr().out
    assert "undeclared.other_sha256" in out, out
    assert "declared.nested.deeper_sha256" not in out, (
        "a declaration must reach the hashes nested beneath it"
    )


def test_a_declaration_does_not_exempt_a_bound_hash(constructed, capsys) -> None:
    """Declaring never overrides binding.

    A declaration covers everything beneath it, so without this the cheapest way to
    hide a stale digest would be a note at the record root. The hash of a file the
    record names is hashed and compared regardless: binding is a stronger statement
    than declaring and has to win.
    """

    record = constructed / "record.json"
    record.write_text(
        json.dumps(
            {
                "hashes_that_name_no_file_here": "everything here is held elsewhere",
                "frame": "frame.png",
                "sha256": "0" * 64,
            }
        )
    )
    assert _run(_audit(constructed)) == 1
    assert "which is actually" in capsys.readouterr().out


def test_an_empty_declaration_does_not_count(constructed, capsys) -> None:
    """Otherwise the cheapest way past the gate is a key with nothing in it."""

    record = constructed / "record.json"
    for value in ("", "   ", None, 0):
        record.write_text(
            json.dumps(
                {
                    "run": {
                        "hashes_that_name_no_file_here": value,
                        "result_sha256": "d" * 64,
                    }
                }
            )
        )
        assert _run(_audit(constructed)) == 1, value
        capsys.readouterr()


def test_a_cited_harness_that_does_not_exist_is_a_finding(constructed, capsys) -> None:
    """The first defect this audit was written for, kept under test."""

    (constructed / "record.json").write_text(
        json.dumps({"generator": "harness/never_committed.py"})
    )
    assert _run(_audit(constructed)) == 1
    assert "cites missing harness/never_committed.py" in capsys.readouterr().out


def test_the_report_declares_itself_without_prose_among_its_artifact_paths() -> None:
    """A mapping whose keys are data has no room for commentary.

    `derived_fields` is keyed by artifact path and `derive_public_run_report_control`
    iterates those keys as filenames, so a declaration added beside them made the
    control try to open the note as a file. The declaration belongs at the record
    root, where it is also the more accurate claim: every hash in this report is of
    bytes the run wrote, and there is no branch of it where that stops holding.
    """

    report = json.loads((_audit().ROOT / "replay-b18d-public-report.json").read_text())
    assert report["hashes_that_name_no_file_here"].strip()
    paths = report["derived_fields"]
    assert paths
    assert all("/" in name and name.endswith(".json") for name in paths), list(paths)


def test_the_audit_needs_no_dependency_the_test_shards_lack(tmp_path) -> None:
    """Why this lives in the shared suite rather than behind an Open3D skip.

    The audit hashes files and parses JSON. If it imported Open3D, or numpy, it
    would be skipped in the shards that matter and the gate would be back to
    running nowhere.
    """

    shutil.copy(AUDIT, tmp_path / "audit.py")
    source = (tmp_path / "audit.py").read_text()
    imported = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    }
    assert imported <= {"__future__", "hashlib", "json", "re", "sys", "pathlib"}, (
        imported
    )
