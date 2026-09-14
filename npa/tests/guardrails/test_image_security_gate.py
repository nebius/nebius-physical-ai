"""Keep image scans mandatory without duplicate runs or changed alert identity."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def _workflow(name):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
    # PyYAML's YAML 1.1 loader parses GitHub's unquoted `on` key as a boolean.
    workflow["on"] = workflow.pop(True)
    return workflow


def test_required_security_workflow_calls_image_scans_on_every_candidate():
    """Cover all PR paths, merge queues and main through the required check.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Image scans can be filtered, skipped or made advisory.
    """
    workflow = _workflow("security-regression.yml")
    assert workflow["on"] == {"pull_request": None, "merge_group": None, "push": {"branches": ["main"]}}
    calls = [name for name, job in workflow["jobs"].items()
             if job.get("uses") == "./.github/workflows/image-security-scan.yml"]
    assert calls == ["image-security"]
    image_job = workflow["jobs"]["image-security"]
    assert image_job == {
        "uses": "./.github/workflows/image-security-scan.yml",
        "permissions": {"contents": "read", "packages": "read", "security-events": "write"},
    }
    required = workflow["jobs"]["security-regression"]
    assert "image-security" in required["needs"]
    commands = "\n".join(step.get("run", "") for step in required["steps"])
    assert "npa/tests/guardrails/test_image_security_gate.py" in commands


def test_reusable_scans_keep_scheduled_coverage_and_distinct_concurrency():
    """Avoid duplicate PR runs and cancellation of the calling required workflow.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Reusable scans lose their trigger or share the caller's group.
    """
    workflow = _workflow("image-security-scan.yml")
    assert set(workflow["on"]) == {"workflow_call", "schedule", "workflow_dispatch"}
    assert workflow["on"]["schedule"]
    assert workflow["concurrency"] == {
        "group": "image-security-${{ github.event.pull_request.number || github.run_id }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }
    for name, job in workflow["jobs"].items():
        if name != "omniverse-payload-scan":
            assert "if" not in job and "continue-on-error" not in job


def test_sarif_uploads_preserve_existing_alert_configuration():
    """Preserve alert identity and avoid requiring SARIF upload permissions on PRs.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: The existing scan configuration changes when called through the required gate.
    """
    jobs = _workflow("image-security-scan.yml")["jobs"]
    for name, category in [("base-image-cve-scan", "trivy-image-${{ matrix.name }}"),
                           ("dockerfile-static-scan", "trivy-config")]:
        steps = jobs[name]["steps"]
        upload = next(step for step in steps if "upload-sarif" in step.get("uses", ""))
        assert upload["if"] == "always() && github.event_name != 'pull_request'"
        assert upload["with"]["category"] == category
        assert upload["env"]["CODEQL_ACTION_ANALYSIS_KEY"] == f".github/workflows/image-security-scan.yml:{name}"

        assert "matrix" not in upload["with"]
