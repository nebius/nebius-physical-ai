"""Keep image security fail closed without queue-amplifying scan matrices."""

from pathlib import Path
import sys

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
    assert image_job == {
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


def test_reusable_scan_has_two_automatic_jobs_and_internal_scope() -> None:
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
        "base-image-cve-scan",
        "omniverse-payload-scan",
    }
    for name in ("image-policy", "base-image-cve-scan"):
        job = jobs[name]
        assert "if" not in job and "strategy" not in job
        scope = _step(job, "Classify image-security scope")
        assert scope["id"] == "scope"
        assert "ci_image_security_scope.py" in scope["run"]
        heavy_steps = job["steps"][job["steps"].index(scope) + 1 :]
        assert heavy_steps
        assert all("steps.scope.outputs.deep" in step["if"] for step in heavy_steps)


def test_base_inventory_uses_bounded_parallelism_without_a_matrix() -> None:
    """Prevent seven base images from consuming seven hosted runners.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Base scanning returns to queue-amplifying matrix jobs.
    """

    job = _workflow("image-security-scan.yml")["jobs"]["base-image-cve-scan"]
    assert "strategy" not in job
    scan = _step(job, "Scan all pinned bases with two local workers")
    assert "scan_base_images.py" in scan["run"]
    assert "--workers 2" in scan["run"]
    assert "base-image-security.json" in scan["run"]


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

    uploads = [
        step
        for step in jobs["base-image-cve-scan"]["steps"]
        if "upload-sarif" in step.get("uses", "")
    ]
    assert len(uploads) == 7
    assert len({step["with"]["category"] for step in uploads}) == 7
    assert all(
        step["env"]["CODEQL_ACTION_ANALYSIS_KEY"].endswith(":base-image-cve-scan")
        for step in uploads
    )
