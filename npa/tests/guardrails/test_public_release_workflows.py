"""Guard the single public-development and public-release GHCR model."""

from __future__ import annotations

from pathlib import Path

import re

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
PUBLISH = WORKFLOWS / "publish-public-images.yml"
HEALTH = WORKFLOWS / "public-release-health.yml"
SECURITY_SCAN = WORKFLOWS / "image-security-scan.yml"
LIBERO_DOC = ROOT / "docs" / "workbench" / "byof-libero.md"


def _spec(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


_PINNED_ACTION = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+@[0-9a-f]{40}$")


def _assert_pinned(uses: object, action: str) -> None:
    """Third-party actions must be digest-pinned; the `# vN` tag comment is
    stripped by YAML parsing, so only the 40-hex SHA remains to assert on."""
    assert isinstance(uses, str) and _PINNED_ACTION.fullmatch(uses), (
        f"expected digest-pinned {action}, got {uses!r}"
    )
    assert uses.startswith(action + "@")


def _runs(path: Path) -> str:
    spec = _spec(path)
    return "\n".join(
        str(step.get("run") or "")
        for job in spec["jobs"].values()
        for step in job["steps"]
    )


def _verify_pushed_bytes_script() -> str:
    steps = _spec(PUBLISH)["jobs"]["build-development"]["steps"]
    return next(
        step["run"]
        for step in steps
        if step.get("name", "").startswith("Verify pushed bytes")
    )


def _cleanup_requested_script() -> str:
    steps = _spec(PUBLISH)["jobs"]["cleanup-requested"]["steps"]
    return next(
        step["run"]
        for step in steps
        if step.get("name", "").startswith("Delete only the exact")
    )


def _shell_function_call(script: str, name: str) -> str:
    start = script.index(f"{name}() {{")
    call = f"\n{name}\n"
    end = script.index(call, start) + len(call)
    return script[start:end]


def test_libero_namespace_claim_requires_continuous_isolation_evidence() -> None:
    """Sampled inventories must never be presented as run-long isolation proof."""

    text = LIBERO_DOC.read_text(encoding="utf-8")
    start = text.index("Before a future run,\n")
    end = text.index("The execution and payload-proof kubeconfig contexts", start)
    contract = text[start:end]
    for required in (
        "not a run-long isolation proof",
        "admission-enforced exclusive-writer policy",
        "gap-free Kubernetes watch or audit-log interval",
        "Every unexpected\ncreate, update, or delete event fails qualification",
        "must not claim run-long isolation",
    ):
        assert required in contract


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


def test_public_development_build_runner_is_dispatch_scoped_and_defaults_hosted() -> (
    None
):
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


def test_lerobot_development_build_version_is_manifest_scoped() -> None:
    spec = _spec(PUBLISH)
    inputs = (spec.get("on") or spec[True])["workflow_dispatch"]["inputs"]
    text = PUBLISH.read_text(encoding="utf-8")

    assert inputs["lerobot_version"]["default"] == ""
    assert "lerobot_version requires a lerobot development build" in text
    assert 'manifest["supported_versions"]' in text
    assert 'build_args+=(--build-arg "LEROBOT_VERSION=$LEROBOT_VERSION")' in text
    assert '"LEROBOT_VERSION=${{ inputs.lerobot_version }}"' not in text


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
        "scan_image_alpamayo2_payload.py",
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
    assert "refusing before any registry write" in text[:push]
    assert "if matrix and head != sha" in text


def test_publication_uses_the_locked_cryptography_dependency() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    project = (ROOT / "npa" / "pyproject.toml").read_text(encoding="utf-8")

    assert project.count('"cryptography==50.0.0"') == 1
    assert (
        text.count(
            'npa/.venv/bin/pip install -c npa/requirements-lock.txt -e "npa[dev]"'
        )
        == 2
    )
    assert (
        text.count(
            "npa/.venv/bin/python -c 'import cryptography; "
            'assert cryptography.__version__ == "50.0.0"\''
        )
        == 2
    )


def test_public_base_pull_authentication_precedes_local_build() -> None:
    spec = _spec(PUBLISH)
    steps = spec["jobs"]["build-development"]["steps"]
    names = [str(step.get("name") or "") for step in steps]
    crane = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses")
        == "imjasonh/setup-crane@31b88efe9de28ae0ffa220711af4b60be9435f6e"
    )
    auth = names.index("Authenticate immutable public base pulls")
    destination = names.index(
        "Prove destination cannot expose unvalidated tagged bytes"
    )
    build = names.index("Build immutable development image locally")
    push = names.index("Push only after every pre-publication gate passes")
    _assert_pinned(steps[auth]["uses"], "docker/login-action")
    assert steps[auth]["uses"] == (
        "docker/login-action@c94ce9fb468520275223c153574b00df6fe4bcc9"
    )
    assert names.count("Authenticate immutable public base pulls") == 1
    assert "Authenticate immutable LIBERO base pulls" not in names
    assert (
        sum("imjasonh/setup-crane@" in str(step.get("uses") or "") for step in steps)
        == 1
    )
    assert crane < auth < destination < build < push


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
    script = (ROOT / "npa/scripts/scan_base_images.py").read_text()
    assert '"--timeout", "2562047h47m16s"' in script
    job = _spec(SECURITY_SCAN)["jobs"]["base-image-cve-scan"]
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Scan all pinned bases with three local workers"
    )
    assert "scan_base_images.py" in command
    assert "--workers 3" in command


def test_post_push_and_promotion_gates_are_digest_bound() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    for required in (
        "attest-build-provenance@",
        "attest-sbom@",
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
    for action in ("actions/attest-build-provenance", "actions/attest-sbom"):
        assert re.search(rf"{re.escape(action)}@[0-9a-f]{{40}}", text), (
            f"{action} must be digest-pinned"
        )
    visibility = text.index("require_existing_public_visibility")
    anonymous = text.index('DOCKER_CONFIG="$anonymous_config" crane manifest')
    pushed_scan = text.index(
        "scan_image_cosmos3_ray_serve_payload.py", text.index("Verify pushed bytes")
    )
    assert pushed_scan < visibility < anonymous
    alpamayo_scans = [
        index
        for index in range(len(text))
        if text.startswith("scan_image_alpamayo2_payload.py", index)
    ]
    assert len(alpamayo_scans) == 2
    assert alpamayo_scans[0] < text.index(
        "Push only after every pre-publication gate passes"
    )
    assert text.index("Verify pushed bytes") < alpamayo_scans[1] < visibility
    prepush = text.index("Prove destination cannot expose unvalidated tagged bytes")
    push = text.index("Push only after every pre-publication gate passes")
    assert prepush < push
    destination_gate = text[prepush:push]
    assert "refusing before any registry write" in destination_gate
    assert "refuse_first_publication()" in destination_gate
    assert (
        'elif [ "$visibility" = private ]; then\n              if [ "$TOOL" = libero ]; then'
        in destination_gate
    )
    assert (
        "elif grep -q '^HTTP/.* 404 ' \"$response\"; then\n            if [ \"$TOOL\" = libero ]; then"
        in destination_gate
    )
    assert destination_gate.count('echo "NPA_FIRST_PUBLICATION_REQUIRED=1" >> "$GITHUB_ENV"') == 3
    assert "payload-free public-development staging" in destination_gate
    steps = _spec(PUBLISH)["jobs"]["build-development"]["steps"]
    attestations = {
        step["name"]: step
        for step in steps
        if step.get("name")
        in {
            "Attest exact pushed digest provenance",
            "Attest exact pushed digest SBOM",
            "Require both digest-bound attestation results",
        }
    }
    assert set(attestations) == {
        "Attest exact pushed digest provenance",
        "Attest exact pushed digest SBOM",
        "Require both digest-bound attestation results",
    }
    for step in attestations.values():
        assert step["if"] == "matrix.tool != 'ncore'"
    assert (
        attestations["Attest exact pushed digest provenance"]["with"][
            "push-to-registry"
        ]
        == "${{ matrix.tool != 'libero' && env.NPA_FIRST_PUBLICATION_REQUIRED != '1' }}"
    )
    assert (
        attestations["Attest exact pushed digest SBOM"]["with"]["push-to-registry"]
        == "${{ matrix.tool != 'libero' && env.NPA_FIRST_PUBLICATION_REQUIRED != '1' }}"
    )
    result_gate = attestations["Require both digest-bound attestation results"]["run"]
    result_lines = result_gate.splitlines()
    assert result_lines[:2] == [
        'test -s "$PROVENANCE_BUNDLE"',
        'test -s "$SBOM_BUNDLE"',
    ]
    assert not any("&&" in line for line in result_lines[:2])
    assert 'if [ "$TOOL" != libero ]; then' in result_gate
    assert 'test -n "$PROVENANCE_URL" && test -n "$SBOM_URL"' in result_gate
    verify = text[text.index("Verify pushed bytes") :]
    first_publication = verify[
        verify.index('elif [ "$NPA_FIRST_PUBLICATION_REQUIRED" = 1 ]') :
    ]
    visibility = first_publication.index("require_existing_public_visibility")
    assert "length == 1" in first_publication[:visibility]
    assert "all(.[]; .name == $root" in first_publication[:visibility]
    assert "adjacent-package-versions.json" not in first_publication
    assert (
        "no registry-enforced exclusive-writer or atomic compare-and-set"
        in (first_publication[visibility:])
    )
    assert "pushed-payload-attempt-${payload_attempt}.log" in verify
    assert "anonymous-manifest-attempt-${anonymous_attempt}.log" in verify
    assert verify.count("TOOMANYREQUESTS|429 Too Many Requests") == 2
    assert verify.count("while true; do") >= 2
    assert "if ! grep -Eq" in verify


def test_hostile_graph_mutation_before_visibility_gate_cannot_disclose(
    tmp_path: Path,
) -> None:
    import os
    import subprocess
    import sys

    script = _verify_pushed_bytes_script()
    gate = _shell_function_call(script, "require_existing_public_visibility")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    operations = tmp_path / "operations"
    operations.write_text("hostile-graph-mutation\n", encoding="utf-8")
    summary = tmp_path / "summary"
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        "args=sys.argv[1:]\n"
        "with open(os.environ['OPERATIONS'],'a') as stream: stream.write(' '.join(args)+'\\n')\n"
        "query=args[args.index('--jq')+1]\n"
        "print('private' if query == '.visibility' else os.environ['GITHUB_REPOSITORY'])\n",
        encoding="utf-8",
    )
    gh.chmod(0o700)
    completed = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + gate],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "OPERATIONS": str(operations),
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_STEP_SUMMARY": str(summary),
            "package_api": "/orgs/nebius/packages/container/npa-libero",
        },
        text=True,
    )

    assert completed.returncode == 1
    assert (
        "no registry-enforced exclusive-writer or atomic compare-and-set"
        in completed.stdout
    )
    operation_log = operations.read_text(encoding="utf-8")
    assert operation_log.startswith("hostile-graph-mutation\n")
    assert "PATCH" not in operation_log
    assert "no visibility PATCH was attempted" in summary.read_text(encoding="utf-8")


def test_repository_concurrency_never_authorizes_public_visibility() -> None:
    spec = _spec(PUBLISH)
    script = _verify_pushed_bytes_script()
    final_graph = script.index("libero-final-package-versions.json")
    refusal = script.index("Public visibility transition is deferred", final_graph)

    assert spec["concurrency"]["group"] == "public-image-registry-mutation"
    assert "Repository-local workflow concurrency cannot exclude another" in script
    assert "visibility=public" not in script
    assert final_graph < refusal


def test_post_push_payload_scan_binds_remote_digest_to_local_full_tar() -> None:
    text = PUBLISH.read_text(encoding="utf-8")
    post_push = text[text.index("Verify pushed bytes") :]

    assert 'docker pull "$exact"' in post_push
    # Two calls bind the pulled digest to the local image; the third binds the
    # independent cuRobo archive verifier to that same inspected remote image.
    assert post_push.count("docker image inspect --format '{{.Id}}'") == 3
    assert (
        'test "$(docker image inspect --format \'{{.Id}}\' "$exact")" = \\\n'
        '                "$(docker image inspect --format \'{{.Id}}\' "$IMAGE")"'
    ) in post_push
    assert (
        '--expected-image-id "$(docker image inspect --format \'{{.Id}}\' "$exact")"'
        in post_push
    )
    assert (
        'docker save --output "$RUNNER_TEMP/${TOOL}-pushed.tar" "$exact"' in post_push
    )
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
    assert "Refusing non-atomic registry cleanup" in text
    assert "gh api --method DELETE" not in text
    assert "Requested development tag is already absent" in text
    failed_cleanup = _spec(PUBLISH)["jobs"]["cleanup-failed-build"]
    script = next(
        step["run"]
        for step in failed_cleanup["steps"]
        if str(step.get("name") or "").startswith("Remove an exact run-owned")
    )
    assert 'gh api --method DELETE "$package_api"' not in script
    assert 'gh api --method PATCH "$package_api" -f visibility=private' not in script
    assert 'gh api -i "$package_api"' in script
    assert "grep -q '^HTTP/.* 404 '" in script
    assert "Failed-build package absence is unverified" in script
    assert "subject relationship alone is not proof" in script
    assert "NPA_FAILED_PUBLICATION_PRE_GRAPH" in script
    assert 'before_ids="$(jq -c' in script
    assert "Run-created attestation version is not bound" in script
    assert "require_complete_graph" in script
    assert "Unrelated package identity changed after the sealed" in script
    assert "require_complete_libero_graph" in script
    assert 'crane manifest "$repository@$LIBERO_QUALIFIED_OCI_DIGEST"' in script
    assert "$repository@$LIBERO_QUALIFIED_ATTESTATION_MANIFEST_DIGEST" in script
    assert "vnd.docker.reference.digest" in script
    assert "gh api --method DELETE" not in script
    assert script.count("Refusing non-atomic registry cleanup") == 2
    assert "versions and package configuration are retained" in script
    assert "subprocess.check_output" not in script


def test_requested_libero_cleanup_revalidates_but_refuses_nonatomic_delete() -> None:
    script = _cleanup_requested_script()

    assert 'gh api --method DELETE "$package_api"' not in script
    assert "require_complete_requested_libero_graph()" in script
    assert "Requested LIBERO pagination is incomplete or malformed" in script
    assert "Requested LIBERO package identity changed during cleanup" in script
    assert "Requested LIBERO package graph changed during cleanup" in script
    assert "sort_by(.name == $root)" in script
    loop = script.index("while IFS=$'\\t' read -r version_id digest tags; do")
    loop_end = script.index('done < "$cleanup_rows"', loop)
    body = script[loop:loop_end]
    immutable_digest = body.index('crane digest "$repository@$digest"')
    graph_readback = body.index("require_complete_requested_libero_graph")
    identity = body.index('jq -e --arg id "$version_id"')
    refusal = script.index("Refusing non-atomic registry cleanup", loop_end)
    assert immutable_digest < graph_readback < identity
    assert loop_end < refusal < script.index("exit 1", refusal)
    assert "gh api --method DELETE" not in script
    assert "forget_requested_libero_version" not in script


def test_requested_libero_cleanup_refuses_hostile_graph_changes(tmp_path: Path) -> None:
    import json
    import os
    import subprocess

    script = _cleanup_requested_script()
    root_digest = "sha256:" + "1" * 64
    referrer_digests = ["sha256:" + "2" * 64, "sha256:" + "3" * 64]
    tag = "dev-" + "a" * 40
    image = f"ghcr.io/nebius/nebius-physical-ai/npa-libero:{tag}"
    graph = [
        {
            "id": 11,
            "name": root_digest,
            "metadata": {"container": {"tags": [tag]}},
        },
        *[
            {
                "id": 12 + index,
                "name": digest,
                "metadata": {"container": {"tags": []}},
            }
            for index, digest in enumerate(referrer_digests)
        ],
    ]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
log = Path(os.environ["FAKE_GH_LOG"])
with log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(args, separators=(",", ":")) + "\\n")
if args[:3] == ["api", "--paginate", "--slurp"]:
    count_path = Path(os.environ["FAKE_GH_COUNT"])
    count = int(count_path.read_text(encoding="utf-8")) if count_path.exists() else 0
    count_path.write_text(str(count + 1), encoding="utf-8")
    graph = json.loads(os.environ["FAKE_GH_GRAPH"])
    if count:
        mode = os.environ["HOSTILE_MUTATION"]
        if mode == "unrelated":
            graph.append({"id": 99, "name": "sha256:" + "4" * 64,
                          "metadata": {"container": {"tags": []}}})
        elif mode == "retag":
            graph[0]["metadata"]["container"]["tags"].append("latest")
        elif mode == "digest":
            graph[0]["name"] = "sha256:" + "9" * 64
        elif mode == "identity":
            graph[0]["id"] = 101
        elif mode == "pages":
            print(json.dumps([graph, {"partial": True}]))
            raise SystemExit(0)
    print(json.dumps([graph]))
elif args[:1] == ["api"] and "--method" not in args and "-i" not in args:
    print(json.dumps({"visibility": "public",
                      "repository": {"full_name": os.environ["GITHUB_REPOSITORY"]},
                      "name": "nebius-physical-ai/npa-libero"}))
elif args[:3] == ["api", "--method", "DELETE"]:
    print("{}")
else:
    raise SystemExit(f"unexpected gh arguments: {args!r}")
""",
        encoding="utf-8",
    )
    gh.chmod(0o700)
    crane = fake_bin / "crane"
    crane.write_text(
        """#!/usr/bin/env python3
import os
import sys

if sys.argv[1] != "digest":
    raise SystemExit(f"unexpected crane arguments: {sys.argv[1:]!r}")
reference = sys.argv[2]
print(reference.rsplit("@", 1)[1] if "@" in reference else os.environ["ROOT_DIGEST"])
""",
        encoding="utf-8",
    )
    crane.chmod(0o700)

    for mutation in ("unchanged", "unrelated", "retag", "digest", "identity", "pages"):
        run_dir = tmp_path / mutation
        run_dir.mkdir()
        log = run_dir / "gh.jsonl"
        summary = run_dir / "summary.md"
        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "IMAGE": image,
            "TOOL": "libero",
            "LIBERO_QUALIFIED_OCI_DIGEST": root_digest,
            "LIBERO_QUALIFIED_PACKAGE_VERSION_DIGESTS": json.dumps(
                sorted([root_digest, *referrer_digests])
            ),
            "LIBERO_PACKAGE_WRITER_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_STEP_SUMMARY": str(summary),
            "ROOT_DIGEST": root_digest,
            "FAKE_GH_GRAPH": json.dumps(graph),
            "FAKE_GH_LOG": str(log),
            "FAKE_GH_COUNT": str(run_dir / "count"),
            "HOSTILE_MUTATION": mutation,
        }
        completed = subprocess.run(
            ["bash", "-c", script],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0, mutation
        if mutation == "unchanged":
            assert "Refusing non-atomic registry cleanup" in completed.stdout
        calls = [
            json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
        ]
        assert not any("--method" in call and "DELETE" in call for call in calls), (
            mutation,
            completed.stdout,
            completed.stderr,
        )


def test_failed_publication_current_run_delta_accepts_zero_one_or_two_referrers() -> (
    None
):
    import json
    import subprocess

    cleanup = _spec(PUBLISH)["jobs"]["cleanup-failed-build"]
    script = next(
        step["run"]
        for step in cleanup["steps"]
        if str(step.get("name") or "").startswith("Remove an exact run-owned")
    )
    command = script.index("jq -S --argjson before_ids")
    program_start = script.index("'\n", command) + 2
    closing_command = '\' "$current_versions" > "$owned"'
    offset = program_start
    for line in script[program_start:].splitlines(keepends=True):
        if line.strip() == closing_command:
            program_end = offset - 1
            break
        offset += len(line)
    else:
        raise AssertionError("exact closing jq command is absent")
    program = script[program_start:program_end]
    tag = "dev-" + "a" * 40
    prior = {
        "id": 1,
        "name": "sha256:" + "1" * 64,
        "metadata": {"container": {"tags": ["stable"]}},
    }

    for referrer_count in range(3):
        root = {
            "id": 2,
            "name": "sha256:" + "2" * 64,
            "metadata": {"container": {"tags": [tag]}},
        }
        referrers = [
            {
                "id": 3 + index,
                "name": "sha256:" + str(3 + index) * 64,
                "metadata": {"container": {"tags": []}},
            }
            for index in range(referrer_count)
        ]
        completed = subprocess.run(
            [
                "jq",
                "-S",
                "--argjson",
                "before_ids",
                '["1"]',
                "--arg",
                "tag",
                tag,
                program,
            ],
            input=json.dumps([prior, root, *referrers]),
            text=True,
            capture_output=True,
            check=True,
        )
        owned = json.loads(completed.stdout)
        assert owned["root"] == {
            "id": "2",
            "name": root["name"],
            "tags": [tag],
        }
        assert len(owned["referrers"]) == referrer_count


def test_failed_publication_cleanup_seals_failure_cancellation_and_concurrency_state() -> (
    None
):
    spec = _spec(PUBLISH)
    build_steps = spec["jobs"]["build-development"]["steps"]
    names = [str(step.get("name") or "") for step in build_steps]
    snapshot = names.index("Seal adjacent pre-publication package graph")
    upload = names.index("Persist adjacent pre-publication package graph")
    push = names.index("Push only after every pre-publication gate passes")
    provenance = names.index("Attest exact pushed digest provenance")
    assert snapshot < upload < push < provenance
    assert build_steps[upload]["uses"] == (
        "actions/upload-artifact@330a01c490aca151604b8cf639adc76d48f6c5d4"
    )
    assert "github.run_id" in build_steps[upload]["with"]["name"]
    assert "github.run_attempt" in build_steps[upload]["with"]["name"]

    cleanup = spec["jobs"]["cleanup-failed-build"]
    condition = str(cleanup["if"])
    assert 'fromJSON(\'["failure","cancelled"]\')' in condition
    assert spec["concurrency"] == {
        "group": "public-image-registry-mutation",
        "cancel-in-progress": False,
    }
    recover = next(
        step
        for step in cleanup["steps"]
        if step.get("name")
        == "Recover the sealed adjacent pre-publication package graph"
    )
    assert "/actions/runs/${GITHUB_RUN_ID}/artifacts" in recover["run"]
    assert ".total_count == 1" in recover["run"]


def test_failed_publication_cleanup_checks_graph_without_nonatomic_deletion() -> None:
    cleanup = _spec(PUBLISH)["jobs"]["cleanup-failed-build"]
    script = next(
        step["run"]
        for step in cleanup["steps"]
        if str(step.get("name") or "").startswith("Remove an exact run-owned")
    )
    delta = script.index("before_ids=")
    subject = script.index("Run-created attestation version is not bound")
    helper = script.index("require_complete_graph() {")
    refusal = script.index("Refusing non-atomic registry cleanup", helper)
    assert delta < subject < helper < refusal
    assert script.count("require_complete_graph() {") == 1
    referrer_loop = script.index(
        "while IFS=$'\\t' read -r referrer_id referrer_digest; do", helper
    )
    referrer_loop_end = script.index(
        "done < <(jq -r '.referrers[] | [.id,.name] | @tsv' \"$owned\")",
        referrer_loop,
    )
    body = script[referrer_loop:referrer_loop_end]
    assert body.index("require_complete_graph") < body.index(
        'jq -e --arg id "$referrer_id"'
    )
    root_check = script.index("\nrequire_complete_graph\n", referrer_loop_end)
    root_identity = script.index('jq -e --arg id "$root_id"', root_check)
    assert referrer_loop_end < root_check < root_identity < refusal
    assert script.index("exit 1", refusal) > refusal
    assert "gh api --method DELETE" not in script
    assert "forget_owned_version" not in script
    assert "versions and package configuration are retained" in script


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
    assert spec["concurrency"]["group"] == "public-image-registry-mutation"
    assert "inputs." not in spec["concurrency"]["group"]
    assert spec["concurrency"]["cancel-in-progress"] is False
    assert (
        "needs.resolve.result == 'success'" in spec["jobs"]["cleanup-requested"]["if"]
    )
    resolve = next(
        step
        for step in spec["jobs"]["resolve"]["steps"]
        if step.get("name")
        == "Validate additive release selection without changing defaults"
    )
    assert 'test "$BUILD_COUNT" = 0 && test "$CLEANUP_COUNT" = 0' in resolve["run"]
    assert resolve["env"]["DEVELOPMENT_SHA"] == "${{ inputs.development_sha }}"
    assert "--mode plan" in resolve["run"]
    promote = spec["jobs"]["promote"]
    assert promote["env"]["RELEASE_TAG"] == "${{ inputs.release_tag }}"
    assert (
        promote["env"]["EXPECTED_SOURCE_DIGEST"]
        == "${{ inputs.expected_source_digest }}"
    )
    for name in ("build-development", "cleanup-requested", "cleanup-failed-build"):
        assert "--release-tag" not in "\n".join(
            step.get("run", "") for step in spec["jobs"][name]["steps"]
        )


def test_additive_workflow_forwards_exact_selector_and_digest_without_shell_expansion(
    tmp_path,
) -> None:
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
    env = {
        **os.environ,
        "TARGET": "ghcr.io/nebius/nebius-physical-ai",
        "DEVELOPMENT_SHA": "b" * 40,
        "SELECTED_TOOLS": "detection-training",
        "RELEASE_TAG": "runtime-recovery-1",
        "EXPECTED_SOURCE_DIGEST": "sha256:" + "a" * 64,
        "ARGV_RECORD": str(recorder),
    }
    steps = _spec(PUBLISH)["jobs"]["promote"]["steps"]
    scripts = [
        step["run"]
        for step in steps
        if step.get("name")
        in {
            "Plan and preflight immutable public development digests",
            "Promote exact validated digests and verify public parity",
        }
    ]
    # These trusted steps have one harmless GitHub expression in the unselected
    # all-image branch. Render that expression as Actions does before bash parses it.
    for script in scripts:
        script = script.replace(
            "${{ inputs.skip_missing && '--skip-missing' || '' }}", ""
        )
        subprocess.run(
            ["bash", "-euo", "pipefail", "-c", script],
            cwd=tmp_path,
            env={**env, "GITHUB_STEP_SUMMARY": str(tmp_path / "summary")},
            check=True,
            capture_output=True,
        )
    records = [json.loads(line) for line in recorder.read_text().splitlines()]
    assert [record[-1] for record in records] == ["plan", "preflight", "publish"]
    for record in records:
        assert record == [
            ".github/scripts/publish_selected_public_image.py",
            "--target",
            env["TARGET"],
            "--development-sha",
            env["DEVELOPMENT_SHA"],
            "--release-tag",
            env["RELEASE_TAG"],
            "--expected-source-digest",
            env["EXPECTED_SOURCE_DIGEST"],
            "--tool",
            "detection-training",
            "--mode",
            record[-1],
        ]
    resolve = next(
        step
        for step in _spec(PUBLISH)["jobs"]["resolve"]["steps"]
        if step.get("name")
        == "Validate additive release selection without changing defaults"
    )
    for build_count, cleanup_count in [("1", "0"), ("0", "1"), ("1", "1")]:
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", resolve["run"]],
            cwd=tmp_path,
            env={**env, "BUILD_COUNT": build_count, "CLEANUP_COUNT": cleanup_count},
            capture_output=True,
        )
        assert result.returncode != 0
    assert len(recorder.read_text().splitlines()) == 3
