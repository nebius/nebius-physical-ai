"""The two public-mirror workflows must keep sharing one credential path, and the health
check must stay read-only.

Both of the mirror's real failures were credential-shaped, and the fix was to put that logic
in one script instead of inlining it per workflow. Two workflows now depend on it, so these
guard the properties that make the split safe rather than the wording of any one file:

- neither workflow re-inlines credential handling (the drift that makes one of them wrong)
- the health check cannot copy or publish anything, ever
- the health check cannot become permanently red for a known-absent image, which is how a
  scheduled job trains people to ignore it
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"
LOGIN_SCRIPT = "npa/scripts/ci_source_registry_login.sh"
PUBLISH = WORKFLOWS / "publish-public-images.yml"
HEALTH = WORKFLOWS / "public-mirror-health.yml"


def _steps(path: Path) -> list[dict]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [step for job in spec["jobs"].values() for step in job["steps"]]


def _run_scripts(path: Path) -> list[str]:
    return [step["run"] for step in _steps(path) if isinstance(step.get("run"), str)]


@pytest.mark.parametrize("path", [PUBLISH, HEALTH], ids=lambda p: p.name)
def test_both_workflows_exist_and_share_the_login_script(path: Path) -> None:
    assert path.is_file(), path
    assert any(LOGIN_SCRIPT in run for run in _run_scripts(path)), (
        f"{path.name} must authenticate via {LOGIN_SCRIPT}"
    )


def test_the_shared_login_script_is_executable() -> None:
    script = Path(__file__).resolve().parents[3] / LOGIN_SCRIPT
    assert script.is_file()
    # A workflow invokes it as ./npa/scripts/..., which needs the bit set in git.
    assert script.stat().st_mode & 0o111, f"{LOGIN_SCRIPT} must be executable"


def test_scoped_branch_publication_has_dispatch_schema_fallbacks() -> None:
    """GitHub validates dispatch inputs against the default branch, so a release branch
    adding the selector must also support temporary repository variables until merged.
    """
    text = PUBLISH.read_text(encoding="utf-8")
    assert "inputs.source_registry || vars.NPA_PUBLISH_SOURCE_REGISTRY" in text
    assert "inputs.tool || vars.NPA_PUBLISH_TOOL" in text
    assert "(inputs.tool || vars.NPA_PUBLISH_TOOL) == ''" in text


@pytest.mark.parametrize("path", [PUBLISH, HEALTH], ids=lambda p: p.name)
def test_no_workflow_reinlines_the_credential_handling(path: Path) -> None:
    """Two copies of this logic means one of them is wrong, and it is the one nobody ran."""
    for run in _run_scripts(path):
        if LOGIN_SCRIPT in run:
            continue
        assert "iam get-access-token" not in run, (
            f"{path.name} mints a token inline; call {LOGIN_SCRIPT} instead"
        )
        assert "crane auth login" not in run or "ghcr.io" in run, (
            f"{path.name} logs into a source registry inline; call {LOGIN_SCRIPT} instead"
        )


def test_the_health_check_never_copies_anything() -> None:
    """It runs on a schedule with no human watching, against a registry where publication is
    irreversible. Every publish_public invocation in it must be an explicitly read-only mode."""
    read_only = ("--preflight", "--verify-public", "--describe-credential")
    invocations = [run for run in _run_scripts(HEALTH) if "npa.deploy.publish_public" in run]
    assert invocations, "the health check must actually run the publisher's preflight"
    for run in invocations:
        assert any(flag in run for flag in read_only), f"non-read-only publisher call: {run}"
        assert "--dry-run" not in run, "--dry-run skips the preflight; use --preflight"


def test_the_health_check_tolerates_images_that_are_not_built_yet() -> None:
    """Without --skip-missing it would be red for as long as any pin lacks a pushed image,
    and a permanently red scheduled job is one nobody reads."""
    preflight = [
        run for run in _run_scripts(HEALTH) if "--preflight" in run and "publish_public" in run
    ]
    assert preflight, "expected a preflight invocation"
    for run in preflight:
        assert "--skip-missing" in run


def test_the_health_check_is_scheduled_and_read_only_by_permission() -> None:
    spec = yaml.safe_load(HEALTH.read_text(encoding="utf-8"))
    # PyYAML 1.1 resolves a bare `on:` key to the boolean True.
    triggers = spec.get("on") or spec[True]
    assert "schedule" in triggers, "early warning requires a schedule, not just dispatch"
    assert "workflow_dispatch" in triggers, "must also be runnable on demand"
    # No packages: write. It reads the source registry and must not be able to push to GHCR.
    assert spec["permissions"] == {"contents": "read"}


def test_only_the_publish_workflow_can_write_to_ghcr() -> None:
    spec = yaml.safe_load(PUBLISH.read_text(encoding="utf-8"))
    triggers = spec.get("on") or spec[True]
    assert set(triggers) == {"workflow_dispatch"}, (
        "publishing is irreversible; it must never be triggered by a push or a schedule"
    )
    assert spec["permissions"].get("packages") == "write"


def test_visibility_guidance_requires_a_completed_copy_phase() -> None:
    """A pre-copy exception must never produce irreversible visibility instructions."""
    steps = _steps(PUBLISH)
    publish = next(step for step in steps if step.get("name") == "Publish and verify")
    visibility = next(
        step
        for step in steps
        if step.get("name") == "Write the visibility checklist to the job summary"
    )
    pre_copy = next(
        step
        for step in steps
        if step.get("name") == "Write pre-copy failure guidance to the job summary"
    )

    assert publish["id"] == "publish"
    assert "steps.publish.outputs.copy_phase_completed == 'true'" in visibility["if"]
    assert "Images copied, but not yet public" in visibility["run"]
    assert "--tool \"$SELECTED_TOOL\" --mode checklist" in visibility["run"]
    assert "steps.publish.outputs.copy_phase_completed != 'true'" in pre_copy["if"]
    assert "do not change GHCR visibility" in pre_copy["run"]
