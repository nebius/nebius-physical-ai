"""Keep unrelated validation jobs independent while retaining PR supersession.

Repository-wide test slots filled their 100-job queues during a dependency
update batch. Validation needs no shared mutable resource lock: let GitHub
schedule independent jobs, and cancel old commits at the workflow boundary.
"""

from pathlib import Path

import pytest
import yaml


WORKFLOW_DIR = Path(__file__).resolve().parents[3] / ".github/workflows"
CANDIDATE = (
    "${{ github.event.pull_request.number || "
    "github.event.merge_group.head_sha || github.ref }}"
)
WORKFLOW_GROUPS = {
    "merge-queue-report.yml": "merge-queue-feedback",
    "security-regression.yml": "pr-gate-" + CANDIDATE,
    "test.yml": "test-" + CANDIDATE,
    "lint.yml": "lint-" + CANDIDATE,
    "harness-guardrails.yml": "guardrails-" + CANDIDATE,
    "image-security-scan.yml": "image-security-${{ github.run_id }}",
    "confidentiality-scan.yml": "confidentiality-${{ github.ref }}",
    "gitleaks.yml": "gitleaks-${{ github.ref }}",
    "typecheck.yml": None,
}

PRIORITY_JOBS = {
    "security-regression.yml": {
        "gitleaks",
        "pr-precheck",
        "scan",
        "security-scanners",
        "security-regression",
    },
    "test.yml": {"coverage"},
    "image-security-scan.yml": {"base-image-plan", "base-image-cve-scan"},
}
TEST_JOBS = {
    "security-regression.yml": {"security-runtime"},
    "test.yml": {"test", "browser-mocked", "pr-smoke"},
    "lint.yml": {"ruff", "docs-drift"},
    "harness-guardrails.yml": {"guardrails"},
    "image-security-scan.yml": {
        "image-policy",
        "base-image-entry",
        "omniverse-payload-scan",
    },
}


def _workflow(name: str) -> dict:
    """Read workflow scalars without interpreting YAML 1.1's boolean `on`."""
    loader = yaml.BaseLoader((WORKFLOW_DIR / name).read_text())
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


@pytest.mark.parametrize("name", WORKFLOW_GROUPS)
def test_validation_jobs_do_not_share_locks_or_cap_parallelism(name: str) -> None:
    """Prevent independent checks from serializing through a bounded queue.

    Args:
        name: Validation workflow, including reusable callers and audit jobs.
    Returns:
        None.
    Raises:
        AssertionError: A job adds a lock or limits matrix parallelism.
    """
    for job_id, job in _workflow(name)["jobs"].items():
        assert "concurrency" not in job, (name, job_id)
        assert "max-parallel" not in job.get("strategy", {}), (name, job_id)


@pytest.mark.parametrize("name", WORKFLOW_GROUPS)
def test_workflow_groups_isolate_candidates_and_reusable_children(name: str) -> None:
    """Keep supersession scoped to a candidate with distinct child groups.

    Args:
        name: Validation workflow whose group must not lock unrelated work.
    Returns:
        None.
    Raises:
        AssertionError: A workflow shares its group or calls an unguarded child.
    """
    workflow = _workflow(name)
    group = workflow.get("concurrency", {}).get("group")
    assert group == WORKFLOW_GROUPS[name]
    for job in workflow["jobs"].values():
        if "uses" not in job:
            continue
        assert job["uses"].startswith("./.github/workflows/")
        child = job["uses"].removeprefix("./.github/workflows/")
        assert child in WORKFLOW_GROUPS
        assert WORKFLOW_GROUPS[child] != group


@pytest.mark.parametrize("name", WORKFLOW_GROUPS)
def test_candidate_runner_roles_keep_bulk_work_out_of_priority_capacity(name):
    """Reserve configured priority labels for short candidate gates only.

    Args:
        name: Workflow with concrete or reusable jobs.
    Returns:
        None.
    Raises:
        AssertionError: Candidate or background work reaches the wrong pool.
    """
    for job_id, job in _workflow(name)["jobs"].items():
        if "runs-on" not in job:
            continue
        role = None
        if name == "security-regression.yml" and job_id in {
            "scan",
            "security-scanners",
        }:
            role = "SECURITY"
        elif job_id in PRIORITY_JOBS.get(name, set()):
            role = "PRIORITY"
        elif job_id in TEST_JOBS.get(name, set()):
            role = "TEST"
        if role is None:
            assert job["runs-on"] == "ubuntu-latest"
            continue
        pool = f"vars.NPA_CI_{role}_RUNNER"
        if role == "SECURITY":
            pool += " || vars.NPA_CI_PRIORITY_RUNNER"
        expected = (
            '${{ contains(fromJSON(\'["pull_request", "merge_group"]\'), github.event_name) '
            f"&& ({pool} || 'ubuntu-latest') || 'ubuntu-latest' }}}}"
        )
        assert job["runs-on"] == expected, (name, job_id)
