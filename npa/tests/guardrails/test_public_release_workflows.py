"""Guard the separate private-candidate and public-release GHCR channels."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
CANDIDATE = WORKFLOWS / "publish-private-candidate-image.yml"
PUBLISH = WORKFLOWS / "publish-public-images.yml"
HEALTH = WORKFLOWS / "public-release-health.yml"


def _spec(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _runs(path: Path) -> str:
    spec = _spec(path)
    return "\n".join(
        str(step.get("run") or "")
        for job in spec["jobs"].values()
        for step in job["steps"]
    )


def test_all_three_channel_workflows_exist() -> None:
    assert CANDIDATE.is_file()
    assert PUBLISH.is_file()
    assert HEALTH.is_file()


def test_candidate_workflow_uses_immutable_private_refs_and_visibility_gate() -> None:
    text = CANDIDATE.read_text(encoding="utf-8")
    spec = _spec(CANDIDATE)
    triggers = spec.get("on") or spec[True]
    assert set(triggers) == {"workflow_dispatch"}
    assert spec["permissions"]["packages"] == "write"
    assert "candidate_image_for_tool" in text
    assert "github.sha" in text
    assert "nebius-physical-ai-private" in text
    assert "visibility" in text and "= private" in text
    assert "anonymously pullable" in text
    assert "NPA_SOURCE_SHA=${{ github.sha }}" in text
    assert "provenance: mode=max" in text
    assert "sbom: true" in text
    assert "{{json .Provenance}}" in text
    assert "{{json .SBOM}}" in text
    assert "trivy-action@v0.36.0" in text


def test_live_payload_scan_can_read_a_private_candidate() -> None:
    text = (WORKFLOWS / "image-security-scan.yml").read_text(encoding="utf-8")
    assert "packages: read" in text
    assert "docker/login-action@v3" in text
    assert "secrets.GITHUB_TOKEN" in text


def test_public_publisher_promotes_candidate_sha_to_separate_target() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    spec = _spec(PUBLISH)
    triggers = spec.get("on") or spec[True]
    assert set(triggers) == {"workflow_dispatch"}
    assert spec["permissions"]["packages"] == "write"
    assert "candidate_sha" in text
    assert "nebius-physical-ai-private" in text
    assert "ghcr.io/${GITHUB_REPOSITORY,,}" in text
    assert "NPA_PRIVATE_GHCR_TOKEN" in text
    assert "iam get-access-token" not in text


def test_public_publisher_can_bootstrap_candidate_from_existing_dispatch_file() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    assert "NPA_BUILD_CANDIDATE_TOOL" in text
    assert "candidate_image_for_tool" in text
    assert "NPA_SOURCE_SHA=${{ inputs.candidate_sha || github.sha }}" in text
    assert "provenance: mode=max" in text
    assert "sbom: true" in text
    assert "{{json .Provenance}}" in text
    assert "{{json .SBOM}}" in text
    assert "trivy-action@v0.36.0" in text
    assert "skypilot-0.12.2-v1" in text
    assert "visibility)\" = private" in text
    assert "NPA_RETIRE_CANDIDATE_REF" in text
    assert "metadata.container.tags" in text
    assert "gh api --method DELETE" in text
    assert "anonymously pullable" in text


def test_public_health_is_anonymous_and_read_only() -> None:
    spec = _spec(HEALTH)
    triggers = spec.get("on") or spec[True]
    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers
    assert spec["permissions"] == {"contents": "read"}
    run = _runs(HEALTH)
    assert "--verify-public" in run
    assert "auth login" not in run
    assert "--preflight" not in run


def test_no_workflow_mentions_nebius_container_registry_auth() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in (CANDIDATE, PUBLISH, HEALTH)
    )
    for forbidden in (
        "NEBIUS_" + "CR_TOKEN",
        "NEBIUS_" + "SA_CREDENTIALS_JSON",
        "iam get-access-token",
        "npa-" + "nebius-registry",
    ):
        assert forbidden not in combined
