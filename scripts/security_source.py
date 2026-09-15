"""Scan repository snapshots for Python and GitHub Actions security regressions."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import tokenize
from pathlib import Path

_SCANNER_VERSIONS = {"bandit": "1.9.4", "zizmor": "1.30.0"}


def _scanner_binary(scanner: str) -> str:
    """Reject missing or unexpected scanner installations before scanning."""
    executable = shutil.which(scanner)
    if executable is None:
        raise RuntimeError(f"Install the pinned {scanner} scanner before running this gate")
    version = subprocess.run(
        [executable, "--version"], check=True, capture_output=True, text=True,
    ).stdout.split()
    if version[:2] != [scanner, _SCANNER_VERSIONS[scanner]]:
        raise RuntimeError(f"Expected {scanner} {_SCANNER_VERSIONS[scanner]}")
    return executable


def _run_report(command: list[str], root: Path, output: Path) -> object:
    """Keep scanner diagnostics private and reject incomplete scanner execution."""
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    output.write_text(completed.stdout, encoding="utf-8")
    output.with_suffix(".stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"{output.stem} failed with exit {completed.returncode}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{output.stem} did not produce a complete JSON report") from error
    if output.stem == "zizmor" and completed.returncode != 0:
        raise RuntimeError("zizmor failed; its findings exit codes were disabled")
    return report


def _relative_path(filename: str, root: Path) -> str:
    """Reject scanner paths outside the immutable candidate snapshot."""
    path = Path(filename)
    if not path.is_absolute():
        path = root / path
    return path.resolve().relative_to(root).as_posix()


def _identity(value: object) -> str:
    """Keep source contents out of published finding identifiers."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _python_node(root: Path, path: str, issue: dict, trees: dict) -> str:
    """Ignore line movement while preserving changes to the vulnerable expression."""
    if path not in trees:
        with tokenize.open(root / path) as source:
            trees[path] = ast.parse(source.read(), filename=path)
    line = issue["line_number"]
    column = issue["col_offset"]
    start_lines = [line, min(issue.get("line_range") or [line])]
    for start_line in start_lines:
        candidates = [
            node for node in ast.walk(trees[path])
            if getattr(node, "lineno", None) == start_line
            and getattr(node, "col_offset", None) == column
        ]
        if candidates:
            return ast.dump(candidates[0], include_attributes=False)
    raise RuntimeError(f"Bandit finding has no matching source node: {path}:{line}")


def _python_findings(report: object, root: Path, expected: set[str]) -> list[dict]:
    """Require a complete scan before normalizing Python findings."""
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise TypeError("Bandit report has an invalid findings schema")
    if report.get("errors") != []:
        raise RuntimeError("Bandit could not parse or scan every Python file; inspect bandit.json")
    metrics = report.get("metrics", {})
    scanned = {_relative_path(path, root) for path in metrics if path != "_totals"}
    if scanned != expected:
        raise RuntimeError("Bandit report did not cover every Python file in the snapshot")
    trees: dict[str, ast.AST] = {}
    findings = []
    for issue in report["results"]:
        path = _relative_path(issue["filename"], root)
        findings.append({
            "scanner": "bandit", "path": path, "rule": issue["test_id"],
            "identity": _identity(_python_node(root, path, issue, trees)),
            "line": issue["line_number"], "message": issue["issue_text"],
        })
    return findings


def _scan_python(root: Path, output: Path) -> list[dict]:
    """Supply trusted configuration instead of accepting candidate suppressions."""
    expected = {path.relative_to(root).as_posix() for path in root.rglob("*.py")}
    if not expected:
        return []
    configuration = output / "bandit.yaml"
    configuration.write_text("{}\n", encoding="utf-8")
    settings = output / "bandit.ini"
    settings.write_text("[bandit]\n", encoding="utf-8")
    command = [
        _scanner_binary("bandit"), "--recursive", str(root), "--exclude", "",
        "--configfile", str(configuration), "--ini", str(settings),
        "--ignore-nosec", "--severity-level", "medium", "--confidence-level", "medium",
        "--format", "json", "--quiet",
    ]
    report = _run_report(command, root, output / "bandit.json")
    return _python_findings(report, root, expected)


def _workflow_location(issue: dict) -> dict:
    """Require the scanner's primary location for an actionable finding."""
    for location in issue["locations"]:
        if location["symbolic"]["kind"] == "Primary":
            return location
    raise RuntimeError("zizmor finding is missing its primary source location")


def _workflow_finding(issue: dict, root: Path) -> dict:
    """Fingerprint workflow expressions independently of step positions."""
    location = _workflow_location(issue)
    symbolic = location["symbolic"]
    concrete = location["concrete"]
    path = _relative_path(symbolic["key"]["Local"]["verbatim_path"], root)
    route = [part for part in symbolic["route"]["route"] if "Key" in part]
    return {
        "scanner": "zizmor", "path": path, "rule": issue["ident"],
        "identity": _identity([route, concrete["feature"].strip()]),
        "line": concrete["location"]["start_point"]["row"] + 1,
        "message": issue["desc"],
    }


def _scan_workflows(root: Path, output: Path) -> list[dict]:
    """Scan workflows and local actions without candidate ignore files or comments."""
    definitions = set(root.rglob("action.yml")) | set(root.rglob("action.yaml"))
    workflows = root / ".github" / "workflows"
    definitions.update(workflows.glob("*.yml"))
    definitions.update(workflows.glob("*.yaml"))
    if not definitions:
        return []
    command = [
        _scanner_binary("zizmor"), "--offline", "--no-config", "--no-ignores",
        "--strict-collection", "--collect", "all", "--min-severity", "medium",
        "--min-confidence", "medium", "--persona", "regular", "--no-exit-codes",
        "--format", "json", "--no-progress", *map(str, sorted(definitions)),
    ]
    report = _run_report(command, root, output / "zizmor.json")
    if not isinstance(report, list):
        raise TypeError("zizmor report has an invalid findings schema")
    return [_workflow_finding(issue, root) for issue in report]


def scan_source(root: Path, output: Path) -> list[dict]:
    """Find Python and GitHub Actions security issues in a repository snapshot.

    Args:
        root: Directory containing only the candidate's regular tracked files.
        output: Private directory outside the snapshot for scanner reports.

    Returns:
        Findings with stable identities and actionable paths, rules, and lines.

    Raises:
        RuntimeError: A scanner is missing, has the wrong version, or fails to scan.
        ValueError: A source or output path escapes the requested scan boundary.
        OSError: Source or report files cannot be read or written.
    """
    root = root.resolve(strict=True)
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Scanner reports must be outside the candidate snapshot")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Security scans require a regular-file snapshot without symlinks")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        return _scan_python(root, output) + _scan_workflows(root, output)
    except (KeyError, TypeError) as error:
        raise RuntimeError("Scanner report is incomplete or has an invalid schema") from error
