"""Keep every image scan required while isolating each image's disk usage."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import ci_image_security_scope as scope  # noqa: E402


def _workflow(name: str) -> dict:
    workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
    workflow["on"] = workflow.pop(True)
    return workflow


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_image_scope_covers_security_inputs_but_not_unrelated_source() -> None:
    """Select every image policy surface without taxing ordinary source PRs.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Relevant paths are skipped or unrelated paths are selected.
    """

    relevant = [
        ".github/workflows/release.yml",
        "docs/security/image-reproducibility.md",
        "npa/docker/workbench/example/Dockerfile",
        "npa/scripts/image_byte_scan/prepare.py",
        "npa/scripts/scan_base_images.py",
        "npa/tests/docker/test_base_image_scan.py",
        "npa/src/npa/deploy/images.py",
        "scripts/security_install.sh",
        "trivy.yaml",
    ]
    assert all(scope.needs_deep_image_security([path]) for path in relevant)
    assert not scope.needs_deep_image_security(
        ["npa/src/npa/cli/workbench/example.py", "docs/workbench/example.md"]
    )


def test_required_gate_calls_image_security_for_every_candidate() -> None:
    """Require image policy from PRs, merge groups, and main.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Image security becomes detached from the required gate.
    """

    workflow = _workflow("security-regression.yml")
    assert workflow["on"] == {
        "pull_request": None,
        "merge_group": None,
        "push": {"branches": ["main"]},
    }
    image_job = workflow["jobs"]["image-security"]
    assert image_job["needs"] == ["validation-plan", "pr-precheck"]
    assert "needs.validation-plan.result == 'success'" in image_job["if"]
    assert "needs.validation-plan.outputs.mode == 'full'" in image_job["if"]
    assert {key: image_job[key] for key in ("uses", "permissions")} == {
        "uses": "./.github/workflows/image-security-scan.yml",
        "permissions": {
            "contents": "read",
            "packages": "read",
            "security-events": "write",
        },
    }
    required = workflow["jobs"]["security-regression"]
    assert "image-security" in required["needs"]
    assert "security-runtime" in required["needs"]


def test_reusable_scan_preserves_scope_and_required_inventory_aggregation() -> None:
    """Keep irrelevant PRs cheap without skipping the required workflow.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Path scope can leave a required check pending.
    """

    workflow = _workflow("image-security-scan.yml")
    assert set(workflow["on"]) == {"workflow_call", "schedule", "workflow_dispatch"}
    assert workflow["on"]["schedule"]
    assert workflow["concurrency"] == {
        "group": "image-security-${{ github.run_id }}",
        "cancel-in-progress": False,
    }
    jobs = workflow["jobs"]
    assert set(jobs) == {
        "image-policy",
        "base-image-plan",
        "base-image-entry",
        "base-image-cve-scan",
        "omniverse-payload-scan",
    }
    for name in ("image-policy", "base-image-plan"):
        job = jobs[name]
        assert "if" not in job and "strategy" not in job
        scope = _step(job, "Classify image-security scope")
        assert scope["id"] == "scope"
        assert "ci_image_security_scope.py" in scope["run"]
        if name == "image-policy":
            heavy_steps = job["steps"][job["steps"].index(scope) + 1 :]
            assert heavy_steps
            assert all("steps.scope.outputs.deep" in step["if"] for step in heavy_steps)


def test_base_inventory_isolates_each_image_without_dropping_failed_entries() -> None:
    """Use the complete validated inventory and retain every matrix result.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Coverage or failure propagation is weakened.
    """

    jobs = _workflow("image-security-scan.yml")["jobs"]
    plan = _step(jobs["base-image-plan"], "Enumerate every validated base")
    assert "--matrix" in plan["run"] and "base-image-security.json" in plan["run"]
    scan = jobs["base-image-entry"]
    assert scan["needs"] == "base-image-plan"
    assert scan["if"] == "needs.base-image-plan.outputs.deep == 'true'"
    assert scan["strategy"] == {
        "fail-fast": False,
        "matrix": "${{ fromJSON(needs.base-image-plan.outputs.matrix) }}",
    }
    command = _step(scan, "Scan the exact inventory entry")
    assert command["env"]["SCAN_ENTRY"] == "${{ matrix.entry }}"
    assert '--entry-name "$SCAN_ENTRY"' in command["run"]
    assert "base-image-security.json" in command["run"]
    aggregate = jobs["base-image-cve-scan"]
    assert set(aggregate["needs"]) == {"base-image-plan", "base-image-entry"}
    assert aggregate["if"] == "always()"
    assert "continue-on-error" not in scan and "continue-on-error" not in aggregate


def test_sarif_uploads_preserve_existing_alert_categories() -> None:
    """Retain the established config and per-image SARIF identities.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Consolidation silently replaces security alert identity.
    """

    jobs = _workflow("image-security-scan.yml")["jobs"]
    config_upload = _step(jobs["image-policy"], "Upload configuration SARIF")
    assert config_upload["with"]["category"] == "trivy-config"
    assert config_upload["env"]["CODEQL_ACTION_ANALYSIS_KEY"].endswith(
        ":dockerfile-static-scan"
    )

    upload = _step(jobs["base-image-entry"], "Upload base SARIF")
    assert upload["with"]["category"] == "trivy-image-${{ matrix.entry }}"
    assert upload["with"]["sarif_file"].endswith("/trivy-${{ matrix.entry }}.sarif")
    assert upload["env"]["CODEQL_ACTION_ANALYSIS_KEY"].endswith(":base-image-cve-scan")


@pytest.mark.parametrize(
    "plan,deep,scan,accepted",
    [
        ("success", "true", "success", True),
        ("success", "false", "skipped", True),
        ("success", "true", "failure", False),
        ("success", "true", "cancelled", False),
        ("success", "true", "skipped", False),
        ("success", "true", "", False),
        ("failure", "true", "success", False),
        ("cancelled", "false", "skipped", False),
        ("skipped", "false", "skipped", False),
        ("success", "", "success", False),
        ("success", "false", "success", False),
    ],
)
def test_actual_inventory_aggregation_fails_closed(plan, deep, scan, accepted):
    """Execute the workflow's real status guard against incomplete inventories."""
    job = _workflow("image-security-scan.yml")["jobs"]["base-image-cve-scan"]
    command = _step(job, "Require complete scoped inventory")["run"]
    result = subprocess.run(
        ["bash", "-c", command],
        env={**os.environ, "PLAN_RESULT": plan, "DEEP": deep, "SCAN_RESULT": scan},
        capture_output=True,
        check=False,
    )
    assert (result.returncode == 0) is accepted
