"""Guard the single public-development and public-release GHCR model."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
PUBLISH = WORKFLOWS / "publish-public-images.yml"
HEALTH = WORKFLOWS / "public-release-health.yml"
SECURITY_SCAN = WORKFLOWS / "image-security-scan.yml"


def _spec(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _runs(path: Path) -> str:
    spec = _spec(path)
    return "\n".join(
        str(step.get("run") or "")
        for job in spec["jobs"].values()
        for step in job["steps"]
    )


def test_public_only_workflows_exist_without_a_private_candidate_workflow() -> None:
    assert PUBLISH.is_file()
    assert HEALTH.is_file()
    retired = "publish-private-" + "candidate-image.yml"
    assert not (WORKFLOWS / retired).exists()


def test_public_publisher_builds_only_immutable_public_development_refs() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    spec = _spec(PUBLISH)
    triggers = spec.get("on") or spec[True]
    assert set(triggers) == {"workflow_dispatch"}
    assert spec["permissions"]["packages"] == "write"
    assert spec["permissions"]["attestations"] == "write"
    assert "development_image_for_tool" in text
    assert "dev-<sha>" in text
    assert "ghcr.io/nebius/nebius-physical-ai" in text
    assert "nebius-physical-ai-private" not in text
    assert "NPA_PRIVATE" not in text
    for stale_variable in (
        "NPA_PUBLIC_IMAGE_TARGET",
        "NPA_DEVELOPMENT_SHA",
        "NPA_BUILD_DEVELOPMENT_TOOLS",
        "NPA_CLEANUP_DEVELOPMENT_TOOLS",
        "NPA_PUBLISH_TOOL",
    ):
        assert stale_variable not in text


def test_public_development_build_runner_is_dispatch_scoped_and_defaults_hosted() -> None:
    spec = _spec(PUBLISH)
    triggers = spec.get("on") or spec[True]
    inputs = triggers["workflow_dispatch"]["inputs"]

    assert inputs["build_runner_label"] == {
        "description": "Runner label for public development image builds",
        "required": False,
        "default": "ubuntu-latest",
    }
    assert spec["jobs"]["build-development"]["runs-on"] == (
        "${{ inputs.build_runner_label || 'ubuntu-latest' }}"
    )
    for name, job in spec["jobs"].items():
        if name != "build-development":
            assert job["runs-on"] == "ubuntu-latest"


def test_public_channel_workflows_do_not_restore_retired_channel_language() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (PUBLISH, HEALTH, SECURITY_SCAN)
    )
    for retired in (
        "private candidate",
        "private-package",
        "operator candidate",
        "candidate payload",
        "candidate push",
        "nebius-physical-ai-private",
    ):
        assert retired not in combined


def test_prepublication_gates_run_before_the_public_dev_push() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    push = text.index("Push only after every pre-publication gate passes")
    for required in (
        "test_packaging_contract.py",
        "npa.guardrails.confidentiality",
        "gitleaks detect",
        "Prove destination cannot expose unvalidated tagged bytes",
        "scan_image_omniverse_payload.py",
        "scan_image_ltx_payload.py",
        "scan_image_wan_payload.py",
        "scan_image_cosmos3_ray_serve_payload.py",
        "test_ltx_runtime_bootstrap.py",
        "test_cosmos3_ray_serve_image_contract.py",
        "--scanners vuln,secret,license",
        "--format spdx-json",
        "non-root runtime required",
        "cached EULA acceptance",
    ):
        assert required in text
        assert text.index(required) < push
    assert "exact pushed-byte gates and post-push anonymous verification apply" in text
    assert "if matrix and head != sha" in text


def test_public_base_pull_authentication_precedes_local_build() -> None:
    spec = _spec(PUBLISH)
    steps = spec["jobs"]["build-development"]["steps"]
    names = [str(step.get("name") or "") for step in steps]
    auth = names.index("Authenticate immutable public base pulls")
    build = names.index("Build immutable development image locally")
    push = names.index("Push only after every pre-publication gate passes")
    assert steps[auth]["uses"] == "docker/login-action@v3"
    assert auth < build < push


def test_large_image_scan_reclaims_only_disposable_build_cache_and_tar() -> None:
    spec = _spec(PUBLISH)
    step = next(
        item
        for item in spec["jobs"]["build-development"]["steps"]
        if item.get("name")
        == "Enforce runtime, revision, bootstrap, config, and history contracts"
    )
    script = step["run"]
    assert script.index("docker buildx prune --all --force") < script.index(
        'docker save --output "$RUNNER_TEMP/${TOOL}.tar"'
    )
    assert 'rm -f "$RUNNER_TEMP/${TOOL}.tar"' in script


def test_large_image_scan_reclaims_build_cache_and_reuses_large_volume() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    prepare = text.index("Prepare capacity for full-image security scans")
    scan = text.index("Pre-publication vulnerability, secret, and license scan")
    sbom = text.index("Generate pre-publication SBOM")
    push = text.index("Push only after every pre-publication gate passes")
    assert prepare < scan < sbom < push
    assert "docker buildx prune --all --force" in text[prepare:scan]
    assert text[scan:push].count("TRIVY_TEMP_DIR: /mnt/npa-trivy") == 2
    assert text[scan:push].count("--cache-dir /tmp/trivy/cache") == 2
    assert text[scan:push].count("--timeout 2562047h47m16s") == 2
    assert text[scan:push].count("-e TMPDIR=/tmp/trivy") == 2
    trivy = "aquasec/trivy:0.70.0@sha256:be1190afcb28352bfddc4ddeb71470835d16462af68d310f9f4bca710961a41e image"
    assert text[scan:push].count(trivy) == 2
    assert "docker image prune" not in text[scan:push]


def test_base_image_scans_do_not_inherit_trivys_five_minute_timeout() -> None:
    spec = _spec(SECURITY_SCAN)
    steps = spec["jobs"]["base-image-cve-scan"]["steps"]
    scans = [
        step
        for step in steps
        if step.get("uses") == "aquasecurity/trivy-action@v0.36.0"
    ]

    assert len(scans) == 2
    assert all(step["with"]["timeout"] == "2562047h47m16s" for step in scans)


def test_post_push_and_promotion_gates_are_digest_bound() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    for required in (
        "attest-build-provenance@v3",
        "attest-sbom@v3",
        "subject-name: ${{ steps.push.outputs.repository }}",
        "Require both digest-bound attestation results",
        "crane digest",
        'DOCKER_CONFIG="$anonymous_config" crane manifest',
        "--development-sha",
        "--mode preflight",
        "--mode publish",
        "Retained immutable dev tags as release provenance",
    ):
        assert required in text
    visibility = text.index('gh api --method PATCH "$package_api" -f visibility=public')
    anonymous = text.index('DOCKER_CONFIG="$anonymous_config" crane manifest')
    pushed_scan = text.index(
        "scan_image_cosmos3_ray_serve_payload.py", text.index("Verify pushed bytes")
    )
    assert pushed_scan < visibility < anonymous
    prepush = text.index("Prove destination cannot expose unvalidated tagged bytes")
    push = text.index("Push only after every pre-publication gate passes")
    assert prepush < push
    assert "Private destination contains tagged versions" in text[prepush:push]
    assert "tagged_count" in text[prepush:push]
    verify = text[text.index("Verify pushed bytes") :]
    assert "pushed-payload-attempt-${payload_attempt}.log" in verify
    assert "anonymous-manifest-attempt-${anonymous_attempt}.log" in verify
    assert verify.count("TOOMANYREQUESTS|429 Too Many Requests") == 2
    assert verify.count("while true; do") >= 2
    assert "if ! grep -Eq" in verify


def test_post_push_payload_scan_binds_remote_digest_to_local_full_tar() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    post_push = text[text.index("Verify pushed bytes") :]

    assert 'docker pull "$exact"' in post_push
    # Two calls bind the pulled digest to the local image; the third binds the
    # independent cuRobo archive verifier to that same inspected remote image.
    assert post_push.count("docker image inspect --format '{{.Id}}'") == 3
    assert ('test "$(docker image inspect --format \'{{.Id}}\' "$exact")" = \\\n'
            '                "$(docker image inspect --format \'{{.Id}}\' "$IMAGE")"') in post_push
    assert '--expected-image-id "$(docker image inspect --format \'{{.Id}}\' "$exact")"' in post_push
    assert 'docker save --output "$RUNNER_TEMP/${TOOL}-pushed.tar" "$exact"' in post_push
    assert '--tarball "$RUNNER_TEMP/${TOOL}-pushed.tar"' in post_push
    assert 'rm -f "$RUNNER_TEMP/${TOOL}-pushed.tar"' in post_push
    assert 'scan_image_omniverse_payload.py \\\n+            "$exact"' not in post_push


def test_build_and_cleanup_dispatches_cannot_fall_through_to_promotion() -> None:
    promote = _spec(PUBLISH)["jobs"]["promote"]
    condition = str(promote["if"])
    assert "needs.resolve.outputs.build_count == '0'" in condition
    assert "needs.resolve.outputs.cleanup_count == '0'" in condition


def test_failed_development_cleanup_is_exact_and_refuses_shared_digest() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    assert "cleanup-failed-build" in text
    assert "metadata.container.tags" in text
    assert "Refusing cleanup: digest also carries tags" in text
    assert "versions/${version_id}" in text
    assert 'if [ "$(jq length "$versions")" = 1 ]' in text
    assert 'gh api --method DELETE "$package_api"' in text
    assert "Deletion does not revoke downloads" in text
    assert "Requested development tag is already absent" in text


def test_public_health_is_anonymous_and_read_only() -> None:
    spec = _spec(HEALTH)
    triggers = spec.get("on") or spec[True]
    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers
    assert spec["permissions"] == {"contents": "read"}
    run = _runs(HEALTH)
    assert "--verify-accepted-releases" in run
    assert "--verify-public" not in run
    assert "ghcr.io/nebius/nebius-physical-ai" in run
    assert "GITHUB_REPOSITORY" not in run
    assert "auth login" not in run
    assert "--preflight" not in run


def test_additive_release_inputs_are_scoped_to_promotion() -> None:
    spec = _spec(PUBLISH)
    inputs = (spec.get("on") or spec[True])["workflow_dispatch"]["inputs"]
    assert inputs["release_tag"]["default"] == ""
    assert inputs["expected_source_digest"]["default"] == ""
    assert "inputs.release_tag || inputs.development_sha" in spec["concurrency"]["group"]
    assert spec["concurrency"]["cancel-in-progress"] is False
    assert "needs.resolve.result == 'success'" in spec["jobs"]["cleanup-requested"]["if"]
    resolve = next(step for step in spec["jobs"]["resolve"]["steps"] if step.get("name") == "Validate additive release selection without changing defaults")
    assert 'test "$BUILD_COUNT" = 0 && test "$CLEANUP_COUNT" = 0' in resolve["run"]
    assert resolve["env"]["DEVELOPMENT_SHA"] == "${{ inputs.development_sha }}"
    assert "--mode plan" in resolve["run"]
    promote = spec["jobs"]["promote"]
    assert promote["env"]["RELEASE_TAG"] == "${{ inputs.release_tag }}"
    assert promote["env"]["EXPECTED_SOURCE_DIGEST"] == "${{ inputs.expected_source_digest }}"
    for name in ("build-development", "cleanup-requested", "cleanup-failed-build"):
        assert "--release-tag" not in "\n".join(step.get("run", "") for step in spec["jobs"][name]["steps"])


def test_additive_workflow_forwards_exact_selector_and_digest_without_shell_expansion(tmp_path) -> None:
    """Run the checked-in trusted shell adapter against an argv-recording executable."""
    import json
    import os
    import subprocess
    import sys

    executable = tmp_path / "npa/.venv/bin/python"
    executable.parent.mkdir(parents=True)
    recorder = tmp_path / "argv.jsonl"
    executable.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['ARGV_RECORD'], 'a') as output:\n"
        "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    executable.chmod(0o700)
    env = {**os.environ, "TARGET": "ghcr.io/nebius/nebius-physical-ai", "DEVELOPMENT_SHA": "b" * 40, "SELECTED_TOOLS": "detection-training", "RELEASE_TAG": "runtime-recovery-1", "EXPECTED_SOURCE_DIGEST": "sha256:" + "a" * 64, "ARGV_RECORD": str(recorder)}
    steps = _spec(PUBLISH)["jobs"]["promote"]["steps"]
    scripts = [step["run"] for step in steps if step.get("name") in {"Plan and preflight immutable public development digests", "Promote exact validated digests and verify public parity"}]
    # These trusted steps have one harmless GitHub expression in the unselected
    # all-image branch. Render that expression as Actions does before bash parses it.
    for script in scripts:
        script = script.replace("${{ inputs.skip_missing && '--skip-missing' || '' }}", "")
        subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=tmp_path, env={**env, "GITHUB_STEP_SUMMARY": str(tmp_path / "summary")}, check=True, capture_output=True)
    records = [json.loads(line) for line in recorder.read_text().splitlines()]
    assert [record[-1] for record in records] == ["plan", "preflight", "publish"]
    for record in records:
        assert record == [".github/scripts/publish_selected_public_image.py", "--target", env["TARGET"], "--development-sha", env["DEVELOPMENT_SHA"], "--release-tag", env["RELEASE_TAG"], "--expected-source-digest", env["EXPECTED_SOURCE_DIGEST"], "--tool", "detection-training", "--mode", record[-1]]
    resolve = next(step for step in _spec(PUBLISH)["jobs"]["resolve"]["steps"] if step.get("name") == "Validate additive release selection without changing defaults")
    for build_count, cleanup_count in [("1", "0"), ("0", "1"), ("1", "1")]:
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", resolve["run"]], cwd=tmp_path,
            env={**env, "BUILD_COUNT": build_count, "CLEANUP_COUNT": cleanup_count},
            capture_output=True,
        )
        assert result.returncode != 0
    assert len(recorder.read_text().splitlines()) == 3
