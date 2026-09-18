"""Prove test selection retains full coverage outside narrow recognized changes."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ci_test_scope as scope  # noqa: E402


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=repo, text=True).strip()


def _write(repo: Path, path: str, contents: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents)


def _commit(repo: Path) -> str:
    _git(repo, "add", "--all")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "Synthetic CI fixture")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def comparison_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a local repository with prose, source, and matching test directories.

    Args:
        tmp_path: Isolated fixture directory.
    Returns:
        Repository root and initial commit SHA.
    Raises:
        subprocess.CalledProcessError: Fixture Git setup fails.
    """
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "CI fixture")
    _git(tmp_path, "config", "user.email", "ci@example.test")
    for path, contents in {
        "README.md": "# Guide\n\nAn old explanation.\n\n```sh\nnpa --help\n```\n",
        "docs/guide.md": "# Guide\nAn old explanation.\n",
        "npa/workflows/example/README.md": "An old explanation.\n",
        "npa/src/npa/cli/example.py": "VALUE = 1\n",
        "npa/tests/cli/test_example.py": "def test_example():\n    assert True\n",
        "npa/tests/workflows/test_example.py": "def test_example():\n    assert True\n",
        "npa/tests/browser/example.cy.js": "// Browser fixture\n",
    }.items():
        _write(tmp_path, path, contents)
    return tmp_path, _commit(tmp_path)


@pytest.mark.parametrize("event", ["pull_request", "merge_group"])
@pytest.mark.parametrize(
    "path", ["README.md", "docs/guide.md", "npa/workflows/example/README.md"]
)
def test_existing_prose_edits_skip_only_runtime_suites(comparison_repo, event, path):
    """Allow explanatory edits while preserving unchanged fenced examples.

    Args:
        comparison_repo: Local repository and base SHA.
        event: Candidate event to classify.
        path: Existing documentation path.
    Returns:
        None.
    Raises:
        AssertionError: Ordinary prose does not receive the narrow fast path.
    """
    repo, base = comparison_repo
    _write(repo, path, (repo / path).read_text().replace("old", "clearer"))
    assert scope.classify(repo, base, _commit(repo), event) == {
        "prose_only": True,
        "full_suite": False,
        "browser": False,
        "test_paths": [],
    }


@pytest.mark.parametrize(
    "before,after",
    [
        ("```sh\nnpa --help\n```\n", "```sh\nnpa --version\n```\n"),
        ("~~~yaml\nvalue: 1\n~~~\n", "~~~yaml\nvalue: 2\n~~~\n"),
        ("    npa --help\n", "    npa --version\n"),
        ("Use `npa --help`.\n", "Use `npa --version`.\n"),
        ("---\nvalue: 1\n---\n", "---\nvalue: 2\n---\n"),
        ("<script>\nvalue = 1\n</script>\n", "<script>\nvalue = 2\n</script>\n"),
        ("Example\n", "Example\n```sh\nnpa --help\n```\n"),
        ("```sh\nexample\n```\n", "example\n"),
        ("Text\n", "Text\n{{ include('example') }}\n"),
    ],
)
def test_executable_or_ambiguous_markdown_retains_full_checks(before, after):
    """Reject code, metadata, templates, and Markdown structural changes.

    Args:
        before: Original document.
        after: Candidate document.
    Returns:
        None.
    Raises:
        AssertionError: An executable change is classified as prose.
    """
    assert not scope._prose_edit(before, after)


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/test.yml",
        "npa/pyproject.toml",
        "npa/tests/conftest.py",
        "npa/tests/cli/fixtures/example.json",
        "npa/src/npa/cli/__init__.py",
        "skills/atomic/example/SKILL.md",
        "docs/cli/example.md",
        "docs/security/example.md",
        "scripts/example.py",
        "new-guide.md",
        "docs/AGENTS.md",
    ],
)
def test_unknown_shared_and_policy_changes_run_the_full_suite(comparison_repo, path):
    """Unknown paths and shared configuration must not narrow PR validation.

    Args:
        comparison_repo: Local repository and base SHA.
        path: Unmapped or shared candidate file.
    Returns:
        None.
    Raises:
        AssertionError: The fallback drops full tests or browser coverage.
    """
    repo, base = comparison_repo
    _write(repo, path, "changed\n")
    assert (
        scope.classify(repo, base, _commit(repo), "pull_request") == scope._full_scope()
    )


@pytest.mark.parametrize(
    "operation", ["new", "delete", "rename", "symlink", "executable"]
)
def test_document_file_operations_do_not_receive_the_prose_exception(
    comparison_repo, operation
):
    """Require full validation for changes beyond regular-file prose edits.

    Args:
        comparison_repo: Local repository and base SHA.
        operation: Candidate file operation.
    Returns:
        None.
    Raises:
        AssertionError: File modes or path changes receive the prose exception.
    """
    repo, base = comparison_repo
    readme = repo / "README.md"
    if operation == "new":
        _write(repo, "docs/new.md", "New guide\n")
    elif operation == "rename":
        readme.rename(repo / "docs" / "renamed.md")
    elif operation == "executable":
        readme.chmod(0o755)
    else:
        readme.unlink()
        if operation == "symlink":
            readme.symlink_to("docs/guide.md")
    assert (
        scope.classify(repo, base, _commit(repo), "merge_group") == scope._full_scope()
    )


def test_cli_changes_get_subsystem_tests_before_full_queue_validation(comparison_repo):
    """Move subsystem failures before admission without reducing queue coverage.

    Args:
        comparison_repo: Local repository and base SHA.
    Returns:
        None.
    Raises:
        AssertionError: PR tests are missing or queue validation is narrowed.
    """
    repo, base = comparison_repo
    _write(repo, "npa/src/npa/cli/example.py", "VALUE = 2\n")
    head = _commit(repo)
    assert scope.classify(repo, base, head, "pull_request") == {
        "prose_only": False,
        "full_suite": False,
        "browser": False,
        "test_paths": ["npa/tests/cli"],
    }
    assert scope.classify(repo, base, head, "merge_group") == scope._full_scope()


@pytest.mark.parametrize(
    "path", ["npa/src/npa/cli/agent_ui.html", "npa/src/npa/agent_backend/foxglove.py"]
)
def test_agent_changes_require_both_browser_and_python_feedback(comparison_repo, path):
    """Exercise browser coverage before an agent change enters the queue.

    Args:
        comparison_repo: Local repository and base SHA.
        path: Agent implementation file.
    Returns:
        None.
    Raises:
        AssertionError: UI dependencies omit browser or Python tests.
    """
    repo, base = comparison_repo
    _write(repo, path, "changed\n")
    result = scope.classify(repo, base, _commit(repo), "pull_request")
    assert result["browser"]
    assert result["test_paths"] == ["npa/tests/cli"]


def test_multiple_changes_union_tests_and_do_not_hide_code_behind_prose(
    comparison_repo,
):
    """Classify the complete candidate diff, including all queued predecessors.

    Args:
        comparison_repo: Local repository and base SHA.
    Returns:
        None.
    Raises:
        AssertionError: Mixed changes lose required tests.
    """
    repo, base = comparison_repo
    _write(
        repo, "README.md", (repo / "README.md").read_text().replace("old", "clearer")
    )
    _write(repo, "npa/src/npa/cli/example.py", "VALUE = 2\n")
    _write(repo, "npa/workflows/example/workflow.yaml", "steps: []\n")
    result = scope.classify(repo, base, _commit(repo), "merge_group")
    assert not result["prose_only"]
    assert result["full_suite"]


def test_workflow_readme_examples_do_not_take_the_prose_exception(comparison_repo):
    """Treat changed workflow instructions as full validation, including on PRs.

    Args:
        comparison_repo: Local repository and base SHA.
    Returns:
        None.
    Raises:
        AssertionError: Executable examples are reduced to the documentation path.
    """
    repo, base = comparison_repo
    _write(repo, "npa/workflows/example/README.md", "```sh\nnpa --help\n```\n")
    assert (
        scope.classify(repo, base, _commit(repo), "pull_request") == scope._full_scope()
    )


@pytest.mark.parametrize(
    "path,tests,browser",
    [
        ("npa/tests/cli/test_example.py", ["npa/tests/cli/test_example.py"], False),
        ("npa/tests/browser/example.cy.js", [], True),
        ("npa/src/npa/workbench/example.py", [], True),
    ],
)
def test_direct_test_edits_and_missing_subsystem_tests(
    comparison_repo, path, tests, browser
):
    """Run changed tests directly and fall back when a mapped suite is absent.

    Args:
        comparison_repo: Local repository and base SHA.
        path: Changed test or implementation path.
        tests: Expected selected pytest paths.
        browser: Expected browser requirement.
    Returns:
        None.
    Raises:
        AssertionError: Selection misses changed tests or accepts missing coverage.
    """
    repo, base = comparison_repo
    _write(repo, path, "changed\n")
    result = scope.classify(repo, base, _commit(repo), "pull_request")
    assert result["test_paths"] == tests
    assert result["browser"] == browser
    assert result["full_suite"] == path.startswith("npa/src/npa/workbench/")


def test_affected_subsystems_are_combined(comparison_repo):
    """Retain both changed subsystems when selecting early tests.

    Args:
        comparison_repo: Local repository and base SHA.
    Returns:
        None.
    Raises:
        AssertionError: One subsystem overwrites another.
    """
    repo, base = comparison_repo
    _write(repo, "npa/src/npa/cli/example.py", "VALUE = 2\n")
    _write(repo, "npa/workflows/example/workflow.yaml", "steps: []\n")
    result = scope.classify(repo, base, _commit(repo), "pull_request")
    assert result["test_paths"] == ["npa/tests/cli", "npa/tests/workflows"]


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch", "push", "unknown"])
def test_audits_always_keep_full_coverage(comparison_repo, event):
    """Never apply candidate shortcuts to scheduled, manual, or unknown events.

    Args:
        comparison_repo: Local repository and base SHA.
        event: Non-candidate GitHub event.
    Returns:
        None.
    Raises:
        AssertionError: An audit loses full validation.
    """
    repo, base = comparison_repo
    assert scope.classify(repo, base, base, event) == scope._full_scope()


def test_missing_comparison_fails_and_empty_diff_keeps_full_validation(comparison_repo):
    """A missing comparison must never look like a successful prose decision.

    Args:
        comparison_repo: Local repository and base SHA.
    Returns:
        None.
    Raises:
        AssertionError: Invalid comparisons are accepted or empty diffs skip tests.
    """
    repo, base = comparison_repo
    assert scope.classify(repo, base, base, "merge_group") == scope._full_scope()
    with pytest.raises(subprocess.CalledProcessError):
        scope.classify(repo, "0" * 40, base, "merge_group")
    with pytest.raises(ValueError):
        scope.classify(repo, "main", base, "pull_request")


def _run_scope_step(repo: Path, base: str, head: str) -> list[str]:
    workflow_path = Path(__file__).resolve().parents[2] / ".github/workflows/test.yml"
    workflow = yaml.safe_load(workflow_path.read_text())
    command = workflow["jobs"]["scope"]["steps"][-1]["run"]
    output = repo / "scope-output"
    environment = dict(
        os.environ,
        EVENT_NAME="pull_request",
        BASE_SHA=base,
        GITHUB_SHA=head,
        GITHUB_OUTPUT=str(output),
        RUNNER_TEMP=str(repo),
    )
    environment["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    subprocess.run(["bash", "-c", command], cwd=repo, env=environment, check=True)
    return output.read_text().splitlines()


@pytest.mark.parametrize("installed", [False, True])
def test_workflow_uses_the_base_policy_and_bootstraps_with_full_tests(
    comparison_repo, installed
):
    """Execute the actual scope step and reject a candidate-only policy shortcut.

    Args:
        comparison_repo: Local repository and base SHA without an installed policy.
        installed: Whether the base already contains the trusted selector.
    Returns:
        None.
    Raises:
        AssertionError: The candidate policy runs or first-rollout tests are skipped.
    """
    repo, base = comparison_repo
    if installed:
        _write(repo, "npa/scripts/ci_test_scope.py", Path(scope.__file__).read_text())
        base = _commit(repo)
    _write(
        repo,
        "npa/scripts/ci_test_scope.py",
        "raise SystemExit('candidate policy ran')\n",
    )
    head = _commit(repo)
    assert _run_scope_step(repo, base, head) == [
        "prose_only=false",
        "full_suite=true",
        "browser=true",
        "test_paths=[]",
    ]
