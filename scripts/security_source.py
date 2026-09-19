"""Scan repository snapshots for Python and GitHub Actions security regressions."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import tokenize
from pathlib import Path, PurePosixPath

_SCANNER_VERSIONS = {"bandit": "1.9.4", "zizmor": "1.30.0"}
_DECLARATIVE_B108_PATHS = frozenset(
    {
        "npa/tests/e2e/test_byof_onboarding_live_e2e.py",
        "npa/tests/workflows/test_gymnasium_pod_receipt.py",
    }
)
_DECLARATIVE_B108_FUNCTIONS = {
    "npa/tests/e2e/test_byof_onboarding_live_e2e.py": {"_gymnasium_admitted_volumes"},
    "npa/tests/workflows/test_gymnasium_pod_receipt.py": {
        "test_admitted_policy_accepts_only_reviewed_ipc_population",
        "test_admission_population_and_mount_changes_are_refused",
        "test_profile_accepts_closed_admitted_policy",
        "test_profile_refuses_unapproved_ipc_mount_options",
    },
}


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


def _python_issue_node(root: Path, path: str, issue: dict, trees: dict) -> ast.AST:
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
            return candidates[0]
    raise RuntimeError(f"Bandit finding has no matching source node: {path}:{line}")


def _python_node(root: Path, path: str, issue: dict, trees: dict) -> str:
    return ast.dump(
        _python_issue_node(root, path, issue, trees), include_attributes=False
    )


def _declarative_mount_b108(
    *, path: str, issue: dict, node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> bool:
    """Disposition only the exact inert mount metadata shape under trusted paths.

    This is deliberately narrower than a string or file allowlist: the Bandit
    node must be the ``/dev/shm`` value of a ``mountPath`` key, inside one of
    the reviewed fixture helpers, without a call/operation in its ancestry.
    Any other B108 occurrence remains actionable.
    """

    if issue.get("test_id") != "B108" or path not in _DECLARATIVE_B108_PATHS:
        return False
    reviewed_mount_path = PurePosixPath("/", "dev", "shm")
    if (
        not isinstance(node, ast.Constant)
        or not isinstance(node.value, str)
        or node.value != str(reviewed_mount_path)
        or PurePosixPath(node.value) != reviewed_mount_path
    ):
        return False
    parent = parents.get(node)
    if not isinstance(parent, ast.Dict):
        return False
    mount_keys = {
        key.value
        for key in parent.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    matching_values = [
        value
        for key, value in zip(parent.keys, parent.values)
        if isinstance(key, ast.Constant)
        and key.value == "mountPath"
        and value is node
    ]
    if len(matching_values) != 1 or not mount_keys <= {
        "name", "mountPath", "readOnly", "mountPropagation"
    }:
        return False
    current: ast.AST | None = parent
    function_name = None
    while current is not None:
        if isinstance(current, ast.Call):
            return False
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_name = current.name
            break
        current = parents.get(current)
    return function_name in _DECLARATIVE_B108_FUNCTIONS[path]


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
    parents_by_path: dict[str, dict[ast.AST, ast.AST]] = {}
    findings = []
    for issue in report["results"]:
        path = _relative_path(issue["filename"], root)
        node = _python_issue_node(root, path, issue, trees)
        parents = parents_by_path.setdefault(path, {})
        if not parents:
            parents.update(
                {
                    child: parent
                    for parent in ast.walk(trees[path])
                    for child in ast.iter_child_nodes(parent)
                }
            )
        findings.append({
            "scanner": "bandit", "path": path, "rule": issue["test_id"],
            "identity": _identity(ast.dump(node, include_attributes=False)),
            "line": issue["line_number"], "message": issue["issue_text"],
            "policy_disposition": (
                "trusted-declarative-mount-metadata"
                if _declarative_mount_b108(
                    path=path, issue=issue, node=node, parents=parents
                )
                else "actionable"
            ),
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
