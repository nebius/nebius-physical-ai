"""Select early PR tests and recognize conservative prose-only merge candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


_SUBSYSTEMS = (
    ("npa/src/npa/cli/", ("npa/tests/cli",)),
    ("npa/src/npa/agent_backend/", ("npa/tests/cli",)),
    ("npa/src/npa/workbench/", ("npa/tests/workbench", "npa/tests/cli")),
    ("npa/src/npa/clients/", ("npa/tests/clients",)),
    ("npa/src/npa/orchestration/", ("npa/tests/orchestration", "npa/tests/workflows")),
    ("npa/src/npa/workflows/", ("npa/tests/workflows",)),
    ("npa/workflows/", ("npa/tests/workflows",)),
)
_BROWSER_PREFIXES = (
    "npa/src/npa/agent_backend/",
    "npa/src/npa/cli/agent",
    "npa/tests/cli/test_agent",
    "npa/tests/browser/",
    "npa/scripts/run_agent_cypress.sh",
)
_PROSE_EXCLUSIONS = ("docs/cli/", "docs/security/", "docs/architecture/decisions/")


@dataclass(frozen=True)
class _Change:
    """One Git path change, including the modes needed to reject symlinks."""

    path: str
    status: str
    old_mode: str
    new_mode: str


def _git(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True, capture_output=True
    ).stdout


def _changes(repo_root: Path, base: str, head: str) -> list[_Change]:
    for revision in (base, head):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("CI comparison revisions must be full commit SHAs")
        _git(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    records = _git(repo_root, "diff", "--raw", "--no-renames", "-z", base, head, "--")
    fields = records.decode("utf-8").split("\0")[:-1]
    changes = []
    for metadata, path in zip(fields[::2], fields[1::2], strict=True):
        old_mode, new_mode, _, _, status = metadata.removeprefix(":").split()
        changes.append(_Change(path, status, old_mode, new_mode))
    return changes


def _prose_path(path: str) -> bool:
    if path.startswith(_PROSE_EXCLUSIONS) or PurePosixPath(path).name in {
        "AGENTS.md",
        "SKILL.md",
    }:
        return False
    if path in {"README.md", "CONTRIBUTING.md", "npa/README.md"}:
        return True
    if path.startswith("docs/") and path.endswith(".md"):
        return True
    return path.startswith("npa/workflows/") and PurePosixPath(path).name == "README.md"


def _protected_lines(lines: list[str]) -> set[int]:
    protected = set()
    fence = ""
    frontmatter = bool(lines and lines[0] == "---")
    for index, line in enumerate(lines):
        marker = re.search(r"`{3,}|~{3,}", line)
        if frontmatter:
            protected.add(index)
            if index and line in {"---", "..."}:
                frontmatter = False
        if fence or marker:
            protected.add(index)
        if marker:
            token = marker.group()
            if not fence:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = ""
        if line.startswith(("    ", "\t")) or re.search(r"[`<>]|\{[{%]", line):
            protected.add(index)
    return protected


def _prose_edit(before: str, after: str) -> bool:
    if "\0" in before + after or re.search(r"<(script|style)\b", before + after, re.I):
        return False
    old_lines, new_lines = before.splitlines(), after.splitlines()
    old_protected, new_protected = (
        _protected_lines(old_lines),
        _protected_lines(new_lines),
    )
    for operation, old_start, old_end, new_start, new_end in SequenceMatcher(
        a=old_lines, b=new_lines, autojunk=False
    ).get_opcodes():
        if operation == "equal":
            continue
        if old_protected.intersection(range(old_start, old_end)):
            return False
        if new_protected.intersection(range(new_start, new_end)):
            return False
    return True


def _is_prose(repo_root: Path, base: str, head: str, change: _Change) -> bool:
    if (
        change.status != "M"
        or change.old_mode != "100644"
        or change.new_mode != "100644"
    ):
        return False
    if not _prose_path(change.path):
        return False
    try:
        before = _git(repo_root, "show", f"{base}:{change.path}").decode("utf-8")
        after = _git(repo_root, "show", f"{head}:{change.path}").decode("utf-8")
    except UnicodeDecodeError:
        return False
    return _prose_edit(before, after)


def _affected_paths(change: _Change) -> tuple[str, ...] | None:
    if change.status not in {"A", "M"} or change.new_mode != "100644":
        return None
    path = change.path
    if PurePosixPath(path).name in {"conftest.py", "__init__.py"}:
        return None
    if path.endswith(".md"):
        return None
    if path.startswith("npa/tests/browser/"):
        return ()
    if path.startswith("npa/tests/") and PurePosixPath(path).match("test_*.py"):
        return (path,)
    if path.startswith("npa/tests/"):
        return None
    for prefix, tests in _SUBSYSTEMS:
        if path.startswith(prefix):
            return tests
    return None


def _full_scope() -> dict:
    return {"prose_only": False, "full_suite": True, "browser": True, "test_paths": []}


def classify(repo_root: Path, base: str, head: str, event: str) -> dict:
    """Select test work without allowing uncertain changes to skip the suite.

    Args:
        repo_root: Checkout containing both comparison commits and candidate tests.
        base: Full SHA of the trusted target commit.
        head: Full SHA of the candidate actually checked out by CI.
        event: GitHub event name; only PRs and merge groups may reduce work.
    Returns:
        JSON-compatible scope containing prose, suite, browser, and test selections.
    Raises:
        ValueError: A comparison SHA is malformed.
        subprocess.CalledProcessError: Git cannot prove the comparison.
    """
    if event not in {"pull_request", "merge_group"}:
        return _full_scope()
    changes = _changes(repo_root, base, head)
    if not changes:
        return _full_scope()
    relevant = [
        change for change in changes if not _is_prose(repo_root, base, head, change)
    ]
    if not relevant:
        return {
            "prose_only": True,
            "full_suite": False,
            "browser": False,
            "test_paths": [],
        }
    if event == "merge_group":
        return _full_scope()
    return _pull_request_scope(repo_root, relevant)


def _pull_request_scope(repo_root: Path, changes: list[_Change]) -> dict:
    tests: set[str] = set()
    for change in changes:
        selected = _affected_paths(change)
        if selected is None:
            return _full_scope()
        tests.update(selected)
    if not all((repo_root / path).exists() for path in tests):
        return _full_scope()
    browser = any(change.path.startswith(_BROWSER_PREFIXES) for change in changes)
    return {
        "prose_only": False,
        "full_suite": False,
        "browser": browser,
        "test_paths": sorted(tests),
    }


def main() -> int:
    """Write the selected test work as JSON and optional GitHub step outputs.

    Args:
        None; arguments come from the command line.
    Returns:
        Zero after the comparison and output write succeed.
    Raises:
        ValueError: A comparison SHA is malformed.
        subprocess.CalledProcessError: A Git comparison fails.
        OSError: The output file cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    scope = classify(
        arguments.repo_root, arguments.base, arguments.head, arguments.event
    )
    print(json.dumps(scope, sort_keys=True))
    if arguments.github_output:
        with arguments.github_output.open("a", encoding="utf-8") as output:
            for name, value in scope.items():
                output.write(f"{name}={json.dumps(value)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
