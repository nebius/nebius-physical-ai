"""Keep validation demand bounded without evicting another PR's waiting jobs.

One merge candidate waited seventeen minutes for coverage and another twelve
for its three-second final check. Shared runner slots bound competing PR work;
short scope and completion jobs must have their own slot. These checks pin the
finite routing expressions, cover every runner job, and forbid callers holding
child slots.
"""

from pathlib import Path

import pytest
import yaml


WORKFLOW_DIR = Path(__file__).resolve().parents[3] / ".github/workflows"
POOL = (
    "npa-validation-${{ github.event_name == 'pull_request' && 'pr' || "
    "github.event_name == 'merge_group' && 'merge' || 'audit' }}-"
)
CANDIDATE = 'contains(fromJSON(\'["pull_request", "merge_group"]\'), github.event_name)'
SLOTS = {
    "checks": "checks",
    "docs": "${{ github.event_name == 'pull_request' && 'docs' || 'checks' }}",
    "policy": "${{ " + CANDIDATE + " && 'policy' || 'security' }}",
    "runtime": "${{ " + CANDIDATE + " && 'runtime' || 'security' }}",
    "test": "${{ " + CANDIDATE + " && 'test-1' || 'test' }}",
    "browser": (
        "${{ github.event_name == 'pull_request' && 'test-2' || "
        "github.event_name == 'merge_group' && 'test-5' || 'test' }}"
    ),
    "shards": (
        "${{ github.event_name == 'merge_group' && format('test-{0}', matrix.shard) || "
        "github.event_name == 'pull_request' && "
        "(contains(fromJSON('[1, 3, 5]'), matrix.shard) && 'test-1' || 'test-2') || 'test' }}"
    ),
    "completion": "${{ " + CANDIDATE + " && 'completion' || 'checks' }}",
}
RUNNER_SLOTS = {
    "security-regression.yml": {
        "gitleaks": "checks",
        "scan": "checks",
        "security-scanners": "policy",
        "security-runtime": "runtime",
        "security-regression": "completion",
        "ci-timing-report": "report",
    },
    "test.yml": {
        "scope": "completion",
        "browser-mocked": "browser",
        "pr-smoke": "test",
        "test": "shards",
        "coverage": "completion",
    },
    "lint.yml": {"ruff": "checks", "docs-drift": "docs"},
    "harness-guardrails.yml": {"guardrails": "docs"},
    "image-security-scan.yml": {
        "image-policy": "policy",
        "base-image-cve-scan": "runtime",
        "omniverse-payload-scan": "runtime",
    },
    "confidentiality-scan.yml": {"scan": "checks"},
    "gitleaks.yml": {"gitleaks": "checks"},
    "typecheck.yml": {"mypy": "test"},
}


def _workflow(name: str) -> dict:
    """Read workflow scalars without interpreting YAML 1.1's boolean `on`."""
    loader = yaml.BaseLoader((WORKFLOW_DIR / name).read_text())
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


@pytest.mark.parametrize("name", RUNNER_SLOTS)
def test_every_validation_runner_uses_a_shared_retained_queue(name: str) -> None:
    """Reject unbounded jobs, candidate-specific slots, and pending eviction.

    Args:
        name: Validation workflow whose runner jobs share the finite pools.
    Returns:
        None.
    Raises:
        AssertionError: A runner job bypasses its pool or cancels another job.
    """
    jobs = _workflow(name)["jobs"]
    runners = {job_id: job for job_id, job in jobs.items() if "runs-on" in job}
    assert runners.keys() == RUNNER_SLOTS[name].keys()
    for job_id, job in runners.items():
        slot = RUNNER_SLOTS[name][job_id]
        group = (
            "npa-validation-audit-checks" if slot == "report" else POOL + SLOTS[slot]
        )
        assert job["concurrency"] == {
            "group": group,
            "queue": "max",
            "cancel-in-progress": "false",
        }, (name, job_id)


def test_reusable_callers_never_hold_a_child_runner_slot() -> None:
    """Discover candidate dependencies and reject nested-lock deadlocks.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: A caller holds a runner slot or calls an uncovered gate.
    """
    for name in RUNNER_SLOTS:
        workflow = _workflow(name)
        assert "npa-validation-" not in workflow.get("concurrency", {}).get("group", "")
        for job in workflow["jobs"].values():
            if "uses" in job:
                assert "concurrency" not in job, (name, job["uses"])
                assert job["uses"].startswith("./.github/workflows/")
                assert job["uses"].removeprefix("./.github/workflows/") in RUNNER_SLOTS


def test_merge_completion_slot_excludes_long_and_optional_work() -> None:
    """Reserve candidate coordination for scope, coverage, and the required result.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Long work or optional reporting can delay completion.
    """
    completion = {
        (name, job_id)
        for name in RUNNER_SLOTS
        for job_id, job in _workflow(name)["jobs"].items()
        if "completion" in job.get("concurrency", {}).get("group", "")
    }
    assert completion == {
        ("test.yml", "scope"),
        ("test.yml", "coverage"),
        ("security-regression.yml", "security-regression"),
    }
    report = _workflow("security-regression.yml")["jobs"]["ci-timing-report"]
    assert report["concurrency"]["group"] == "npa-validation-audit-checks"
