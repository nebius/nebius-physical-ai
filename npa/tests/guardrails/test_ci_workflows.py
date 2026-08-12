"""Guard the CI controls that prevent duplicate and superseded PR work."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
AUTOMATIC_PR_WORKFLOWS = (
    "confidentiality-scan.yml",
    "gitleaks.yml",
    "harness-guardrails.yml",
    "image-security-scan.yml",
    "lint.yml",
    "test.yml",
)


def _load_workflow(name: str) -> dict:
    # BaseLoader preserves the key `on`; PyYAML's YAML 1.1 SafeLoader treats it
    # as a boolean, while GitHub Actions follows YAML 1.2 semantics here.
    with (WORKFLOW_DIR / name).open(encoding="utf-8") as workflow_file:
        return yaml.load(workflow_file, Loader=yaml.BaseLoader)


def test_automatic_pr_workflows_cancel_superseded_commits() -> None:
    expected_group = (
        "${{ github.workflow }}-"
        "${{ github.event.pull_request.number || github.run_id }}"
    )
    expected_cancel = "${{ github.event_name == 'pull_request' }}"

    discovered = {
        path.name
        for path in WORKFLOW_DIR.glob("*.y*ml")
        if "pull_request" in _load_workflow(path.name)["on"]
    }
    assert discovered == set(AUTOMATIC_PR_WORKFLOWS)

    for name in AUTOMATIC_PR_WORKFLOWS:
        workflow = _load_workflow(name)
        assert "pull_request" in workflow["on"], name
        assert workflow["concurrency"] == {
            "group": expected_group,
            "cancel-in-progress": expected_cancel,
        }, name


def test_test_and_lint_do_not_duplicate_feature_branch_pushes() -> None:
    for name in ("test.yml", "lint.yml"):
        workflow = _load_workflow(name)
        assert workflow["on"]["push"] == {"branches": ["main"]}, name


def test_pr_test_matrix_uses_one_version_and_main_keeps_compatibility() -> None:
    workflow = _load_workflow("test.yml")
    versions = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]

    assert "github.event_name == 'pull_request'" in versions
    assert "[\"3.12\"]" in versions
    assert "[\"3.10\",\"3.12\",\"3.14\"]" in versions


def test_advisory_mypy_is_manual_only() -> None:
    lint = _load_workflow("lint.yml")
    typecheck = _load_workflow("typecheck.yml")

    assert "mypy" not in lint["jobs"]
    assert typecheck["on"] == {"workflow_dispatch": ""}
    assert set(typecheck["jobs"]) == {"mypy"}
