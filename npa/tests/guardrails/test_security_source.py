"""Verify security scanner adapters reject incomplete reports and stable regressions."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "security_source.py"
_SPEC = importlib.util.spec_from_file_location("security_source", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
source_scanner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(source_scanner)


def _python_report(root: Path, line: int = 1) -> dict:
    return {
        "errors": [], "metrics": {str(root / "candidate.py"): {}},
        "results": [{
            "filename": str(root / "candidate.py"), "line_number": line,
            "col_offset": 0, "test_id": "B307", "issue_text": "Unsafe evaluation",
        }],
    }


@pytest.mark.parametrize("errors", [[{"reason": "syntax error"}], None])
def test_python_parse_errors_fail_closed(tmp_path: Path, errors: object) -> None:
    """Reject Python parse errors instead of accepting an incomplete scan.

    Args:
        tmp_path: Isolated synthetic source and report directories.
        errors: Invalid scanner parse-error observations.
    Returns:
        None.
    Raises:
        AssertionError: The scanner safety or finding-identity contract regresses.
    """
    report = _python_report(tmp_path)
    report["errors"] = errors
    with pytest.raises(RuntimeError, match="could not parse"):
        source_scanner._python_findings(report, tmp_path, {"candidate.py"})


def test_unscanned_python_files_fail_closed(tmp_path: Path) -> None:
    """Require coverage of every Python source in the snapshot.

    Args:
        tmp_path: Isolated synthetic source and report directories.
    Returns:
        None.
    Raises:
        AssertionError: The scanner safety or finding-identity contract regresses.
    """
    with pytest.raises(RuntimeError, match="every Python file"):
        source_scanner._python_findings(
            _python_report(tmp_path), tmp_path, {"candidate.py", "omitted.py"},
        )


def test_python_identity_survives_line_moves_and_counts_duplicates(tmp_path: Path) -> None:
    """Preserve finding identity across line moves without losing duplicates.

    Args:
        tmp_path: Isolated synthetic source and report directories.
    Returns:
        None.
    Raises:
        AssertionError: The scanner safety or finding-identity contract regresses.
    """
    candidate = tmp_path / "candidate.py"
    candidate.write_text("eval(request)\n", encoding="utf-8")
    original = source_scanner._python_findings(
        _python_report(tmp_path), tmp_path, {"candidate.py"},
    )
    candidate.write_text("# An unrelated header.\neval(request)\neval(request)\n", encoding="utf-8")
    report = _python_report(tmp_path, line=2)
    report["results"].append({**report["results"][0], "line_number": 3})
    changed = source_scanner._python_findings(report, tmp_path, {"candidate.py"})
    assert len(changed) == 2
    assert original[0]["identity"] == changed[0]["identity"] == changed[1]["identity"]
    assert [finding["line"] for finding in changed] == [2, 3]


def test_python_identity_changes_with_vulnerable_expression(tmp_path: Path) -> None:
    """Distinguish changes to the vulnerable source expression.

    Args:
        tmp_path: Isolated synthetic source and report directories.
    Returns:
        None.
    Raises:
        AssertionError: The scanner safety or finding-identity contract regresses.
    """
    candidate = tmp_path / "candidate.py"
    candidate.write_text("eval(request)\n", encoding="utf-8")
    original = source_scanner._python_findings(
        _python_report(tmp_path), tmp_path, {"candidate.py"},
    )
    candidate.write_text("eval(other_request)\n", encoding="utf-8")
    changed = source_scanner._python_findings(
        _python_report(tmp_path), tmp_path, {"candidate.py"},
    )
    assert original[0]["identity"] != changed[0]["identity"]


def test_multiline_call_uses_full_expression_when_issue_marks_keyword(tmp_path: Path) -> None:
    """Attribute a keyword finding to its complete multiline expression.

    Args:
        tmp_path: Isolated synthetic source and report directories.
    Returns:
        None.
    Raises:
        AssertionError: The scanner safety or finding-identity contract regresses.
    """
    candidate = tmp_path / "candidate.py"
    candidate.write_text("result = run(\n    request,\n    shell=True,\n)\n", encoding="utf-8")
    report = _python_report(tmp_path, line=3)
    report["results"][0].update(col_offset=9, line_range=[1, 2, 3, 4])
    findings = source_scanner._python_findings(report, tmp_path, {"candidate.py"})
    assert len(findings) == 1
    assert findings[0]["line"] == 3


@pytest.mark.parametrize("returncode,stdout", [(2, "{}"), (0, "{"), (1, "[]")])
def test_workflow_scanner_failures_do_not_become_clean_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str,
) -> None:
    """Reject failed or malformed workflow scanner responses.

    Args:
        tmp_path: Isolated synthetic source and report directories.
        monkeypatch: Replaces the external workflow scanner process.
        returncode: Observed scanner process exit code.
        stdout: Synthetic scanner JSON output.
    Returns:
        None.
    Raises:
        AssertionError: The scanner safety or finding-identity contract regresses.
    """
    monkeypatch.setattr(
        source_scanner.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, returncode, stdout, "diagnostic"),
    )
    with pytest.raises(RuntimeError):
        source_scanner._run_report(["zizmor"], tmp_path, tmp_path / "zizmor.json")
    assert (tmp_path / "zizmor.stderr").read_text() == "diagnostic"


def _workflow_issue(root: Path, step: int, row: int) -> dict:
    return {
        "ident": "template-injection", "desc": "Attacker-controlled template",
        "locations": [{
            "symbolic": {
                "kind": "Primary",
                "key": {"Local": {"verbatim_path": str(root / "workflow.yml")}},
                "route": {"route": [{"Key": "jobs"}, {"Key": "build"}, {"Index": step}]},
            },
            "concrete": {
                "feature": "echo ${{ github.event.pull_request.title }}",
                "location": {"start_point": {"row": row}},
            },
        }],
    }


def test_workflow_identity_survives_step_and_line_moves(tmp_path: Path) -> None:
    """Preserve workflow finding identity across step and line movement.

    Args:
        tmp_path: Isolated synthetic source and report directories.
    Returns:
        None.
    Raises:
        AssertionError: The scanner safety or finding-identity contract regresses.
    """
    original = source_scanner._workflow_finding(_workflow_issue(tmp_path, 0, 1), tmp_path)
    changed = source_scanner._workflow_finding(_workflow_issue(tmp_path, 2, 5), tmp_path)
    assert original["identity"] == changed["identity"]
    assert original["line"] == 2
    assert changed["line"] == 6
    assert "github.event" not in json.dumps(original)


def test_missing_workflow_location_fails_closed(tmp_path: Path) -> None:
    """Reject workflow findings without an actionable primary location.

    Args:
        tmp_path: Isolated synthetic source and report directories.
    Returns:
        None.
    Raises:
        AssertionError: The scanner safety or finding-identity contract regresses.
    """
    issue = _workflow_issue(tmp_path, 0, 1)
    issue["locations"][0]["symbolic"]["kind"] = "Related"
    with pytest.raises(RuntimeError, match="primary source location"):
        source_scanner._workflow_finding(issue, tmp_path)


def test_snapshot_symlink_cannot_read_external_source(tmp_path: Path) -> None:
    """Reject snapshot symlinks before scanning external source bytes.

    Args:
        tmp_path: Isolated synthetic source and report directories.
    Returns:
        None.
    Raises:
        AssertionError: The scanner safety or finding-identity contract regresses.
    """
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "external.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(ValueError, match="symlinks"):
        source_scanner.scan_source(root, tmp_path / "reports")


def test_report_directory_cannot_pollute_source_snapshot(tmp_path: Path) -> None:
    """Keep scanner reports outside the source snapshot.

    Args:
        tmp_path: Isolated synthetic source and report directories.
    Returns:
        None.
    Raises:
        AssertionError: The scanner safety or finding-identity contract regresses.
    """
    with pytest.raises(ValueError, match="outside"):
        source_scanner.scan_source(tmp_path, tmp_path / "reports")
