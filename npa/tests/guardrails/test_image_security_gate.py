"""Keep image scans mandatory and aligned with the OS updates used in builds."""

import json
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
    image_job = workflow["jobs"]["image-security"]
    assert image_job == {
        "uses": "./.github/workflows/image-security-scan.yml",
        "permissions": {"contents": "read", "packages": "read", "security-events": "write"},
    }
    required = workflow["jobs"]["security-regression"]
    assert "image-security" in required["needs"]
    commands = "\n".join(step.get("run", "") for step in required["steps"])
    assert "npa/tests/guardrails/test_image_security_gate.py" in commands
    assert "npa/tests/docker/test_fiftyone_os_updates.py" in commands


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


def test_python_base_scan_uses_the_same_os_updates_as_fiftyone():
    """Prevent a scan of the raw Python parent from drifting from the runtime build.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: The image pin or shared patch step diverges.
    """
    dockerfile = (ROOT / "npa/docker/workbench/fiftyone/Dockerfile").read_text()
    base = dockerfile.split("FROM ", 1)[1].splitlines()[0]
    job = _workflow("image-security-scan.yml")["jobs"]["base-image-cve-scan"]
    entries = [entry for entry in job["strategy"]["matrix"]["include"] if entry["image"] == base]
    assert len(entries) == 1 and entries[0]["upgrade_os_packages"] is True
    script = next(step["run"] for step in job["steps"] if step.get("id") == "scan-target")
    assert "COPY npa/docker/workbench/fiftyone/upgrade_os_packages.sh /tmp/npa-upgrade-os-packages.sh" in script
    assert "RUN sh /tmp/npa-upgrade-os-packages.sh" in script
    assert "docker build --pull --no-cache" in script
    assert "set -euo pipefail" in script
    assert "COPY --chmod=0644 docker/workbench/fiftyone/upgrade_os_packages.sh /opt/npa-upgrade-os-packages.sh" in dockerfile
    assert "RUN sh /opt/npa-upgrade-os-packages.sh" in dockerfile


def test_scans_block_on_findings_and_keep_sarif_categories():
    """Keep the CVE threshold and stable reporting categories without PR upload permissions.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Findings become advisory or disappear from main reporting.
    """
    jobs = _workflow("image-security-scan.yml")["jobs"]
    for name, category in [("base-image-cve-scan", "trivy-image-${{ matrix.name }}"),
                           ("dockerfile-static-scan", "trivy-config")]:
        steps = jobs[name]["steps"]
        gate = next(step for step in steps if step.get("with", {}).get("exit-code") == "1")
        assert "if" not in gate and "continue-on-error" not in gate
        assert "CRITICAL" in gate["with"]["severity"]
        upload = next(step for step in steps if "upload-sarif" in step.get("uses", ""))
        assert upload["if"] == "always() && github.event_name != 'pull_request'"
        assert upload["with"]["category"] == category
        assert upload["env"]["CODEQL_ACTION_ANALYSIS_KEY"] == f".github/workflows/image-security-scan.yml:{name}"


def test_image_alert_identity_excludes_new_build_switches():
    """Retain the original analysis matrix so clean scans update existing alerts.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: A build-only field creates a second alert configuration.
    """
    job = _workflow("image-security-scan.yml")["jobs"]["base-image-cve-scan"]
    upload = next(step for step in job["steps"] if "upload-sarif" in step.get("uses", ""))
    for entry in job["strategy"]["matrix"]["include"]:
        report_matrix = upload["with"]["matrix"]
        expected = {key: entry[key] for key in ("image", "name", "purge_linux_libc_dev")}
        for key, value in expected.items():
            report_matrix = report_matrix.replace("${{ toJSON(matrix." + key + ") }}", json.dumps(value))
        assert json.loads(report_matrix) == expected
