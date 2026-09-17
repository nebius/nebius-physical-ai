"""Static contract for the unbuilt RoboTwin public bootstrap recipe."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from npa.deploy import images


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "npa/docker/workbench/robotwin"
ARCHIVE_DOWNLOAD = "apt-get download ${packages};"
ARCHIVE_INSTALL = (
    "apt-get install -y --no-install-recommends --allow-downgrades "
    '--no-download "${archive_dir}"/*.deb;'
)
LOCK_ONLY_SANITY_GUARD = 'test "$(uniq "${expected_archives}" | wc -l)" = 75;'
ARCHIVE_PREINSTALL_GUARDS = (
    LOCK_ONLY_SANITY_GUARD,
    "! -type f -o ! -name '*.deb'",
    'test "${locked_identity_count}" = 1;',
    'test "$(sha256sum "${archive}" | cut -d \' \' -f1)" = "${archive_sha}";',
    'test "$(stat -c \'%s\' "${archive}")" = "${archive_size}";',
    'cmp "${expected_archives}" "${verified_archives}";',
)


def _assert_archive_install_contract(text: str) -> None:
    download_position = text.index(ARCHIVE_DOWNLOAD)
    install_position = text.index(ARCHIVE_INSTALL)
    assert download_position < install_position
    for guard in ARCHIVE_PREINSTALL_GUARDS:
        assert text.count(guard) == 1
        guard_position = text.index(guard)
        assert guard_position < install_position
        if guard != LOCK_ONLY_SANITY_GUARD:
            assert download_position < guard_position


def _move_guard_after_install(text: str, guard: str) -> str:
    without_guard = text.replace(guard, "", 1)
    boundary = without_guard.index(ARCHIVE_INSTALL) + len(ARCHIVE_INSTALL)
    return without_guard[:boundary] + guard + without_guard[boundary:]


def test_candidate_is_registered_public_but_unbuilt_and_quarantined() -> None:
    contract = yaml.safe_load(
        (ROOT / "npa/docker/workbench/packaging-contract.yaml").read_text()
    )["images"]["robotwin"]
    assert contract["tier"] == "job"
    assert contract["redistribution"] == "public"
    assert contract["skypilot_bootstrap_contract"] == "skypilot-0.12.2-v1"
    assert images.CONTAINER_IMAGE_NAMES["robotwin"] == "npa-robotwin"
    assert images.SUPPORTED_TOOL_VERSIONS["robotwin"].endswith("-unbuilt")
    assert "robotwin" in images.UNVALIDATED_PUBLICATION_TOOLS
    assert "robotwin" not in images.publicly_publishable_tools()


def test_dockerfile_is_nonroot_zero_payload_and_immutably_resolved() -> None:
    text = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "FROM ubuntu:22.04@sha256:"
        "281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986"
    ) in text
    assert "https://snapshot.ubuntu.com/ubuntu/20260912T000000Z/" in text
    assert "apt-get install -y --no-install-recommends --allow-downgrades" in text
    assert 'org.nebius.npa.payload="zero-vendor-payload"' in text
    assert 'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in text
    assert "USER ubuntu" in text
    assert "robotwin-runtime assert-refusal" in text
    assert "--mount=type=secret" not in text
    for forbidden in (
        "FROM nvidia/",
        "FROM nvcr.io/",
        "pip install",
        "git clone",
        "hf_hub_download",
        "COPY --from",
    ):
        assert forbidden not in text


def test_apt_archives_are_verified_one_to_one_before_local_install() -> None:
    text = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")

    _assert_archive_install_contract(text)


@pytest.mark.parametrize("guard", ARCHIVE_PREINSTALL_GUARDS)
def test_apt_archive_contract_rejects_missing_or_late_guards(guard: str) -> None:
    text = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")

    with pytest.raises((AssertionError, ValueError)):
        _assert_archive_install_contract(text.replace(guard, "", 1))
    with pytest.raises(AssertionError):
        _assert_archive_install_contract(_move_guard_after_install(text, guard))


def test_build_script_passes_the_dockerfile_source_sha_argument() -> None:
    build = (IMAGE_ROOT / "build.sh").read_text(encoding="utf-8")
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG NPA_SOURCE_SHA" in dockerfile
    assert (
        "printf '%s\\n' \"${NPA_SOURCE_SHA}\" | grep -Eq '^[0-9a-f]{40}$'"
    ) in dockerfile
    assert '--build-arg "NPA_SOURCE_SHA=$SOURCE_SHA"' in build
    assert '--build-arg "SOURCE_SHA=$SOURCE_SHA"' not in build
    assert 'git -C "$REPO_ROOT" archive "$SOURCE_SHA"' in build
    assert build.index("native-content policy is unresolved") < build.index(
        "docker buildx build"
    )


def test_neutral_locks_are_complete_while_runtime_delivery_is_disabled() -> None:
    lock = json.loads((IMAGE_ROOT / "runtime-lock.json").read_text())
    assert lock["status"] == "bootstrap-complete-runtime-disabled"
    assert lock["bootstrap"]["status"] == "complete"
    assert lock["bootstrap"]["payload_class"] == "zero-vendor-payload"
    assert lock["bootstrap"]["apt"]["binary_package_count"] == 75
    assert lock["bootstrap"]["apt"]["source_package_count"] == 57
    assert lock["bootstrap"]["python_runtime"]["application_artifact_count"] == 0
    assert lock["runtime_delivery"]["status"].startswith("disabled-")
    assert lock["runtime_delivery"]["asset_output_classification_status"] == (
        "complete-no-signature-hold"
    )
    assert lock["runtime_delivery"]["network_side_effects_permitted"] is False
    assert lock["runtime_artifacts"] == []
    assert lock["weights"] == []
    assert lock["access"]["status"] == "not-probed"
    assert lock["access"]["timing"] == "before-provisioning"
    assert lock["access"]["credential_phase"] == "runtime-only-secret-value"
    assert lock["access"]["credential_persistence"] is False
    assert lock["access"]["anonymous_artifacts"] == (
        "exact-revision-payload-byte-probe"
    )
    assert lock["access"]["gated_artifacts"] == (
        "customer-vendor-side-entitlement-and-exact-revision-payload-byte-probe"
    )
    assert lock["access"]["customer_authorization"] == {
        "schema_version": "npa.byof.robotwin.authenticated-customer-authorization.v1",
        "control": "authenticated-customer-control-plane-consume-once",
        "bindings": [
            "verified-issuer",
            "customer-scope-id",
            "run-id",
            "runtime-lock-sha256",
            "issuance",
            "expiry",
            "exact-terms",
            "intended-activity",
            "assertion-id",
            "replay-resistant-nonce",
        ],
        "unsigned_local_file_authoritative": False,
        "manager_context_authoritative": False,
        "repository_authenticator_implementation": False,
        "manager_or_npa_acceptance": False,
    }
    assert lock["cache"]["status"] == "disabled-until-runtime-delivery-approved"
    assert lock["cache"]["tier"] == "node-local-ephemeral"
    assert lock["cache"]["owner_access"] == "single-customer-single-workload"
    assert lock["cache"]["contains_credentials"] is False
    assert {asset["provider_access"] for asset in lock["assets"]} == {"public-ungated"}
    assert {asset["license"] for asset in lock["assets"]} == {"MIT"}
    assert lock["outputs"]["generated_output_restriction"] == (
        "none-found-in-inspected-authoritative-terms"
    )
    assert "aggregate" not in lock["reason"].lower()
    assert (
        "customer authorization assertion or receipt"
        in " ".join(lock["bootstrap"]["forbidden_payloads"]).lower()
    )
    apt_lines = (IMAGE_ROOT / "apt-packages.lock").read_text().splitlines()
    assert "INCOMPLETE" not in "\n".join(apt_lines)
    assert sum(line.startswith("binary\t") for line in apt_lines) == 75
    assert sum(line.startswith("source\t") for line in apt_lines) == 57
    for line in (line.split("\t") for line in apt_lines if line.startswith("binary\t")):
        assert len(line) in {12, 13}
        if len(line) == 13:
            assert line[12] == "gitleaks:allow=public-ubuntu-copyright-sha256"
        assert len(line[4]) == 64
        assert len(line[10]) == 64
    requirements_path = IMAGE_ROOT / "runtime-requirements.lock"
    requirements = requirements_path.read_text()
    assert "status=complete-empty" in requirements
    assert "artifact-count=0" in requirements
    assert "https://" not in requirements
    requirements_sha256 = hashlib.sha256(requirements_path.read_bytes()).hexdigest()
    assert (
        lock["bootstrap"]["python_runtime"]["requirements_lock_sha256"]
        == requirements_sha256
    )
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    requirements_guard = next(
        line
        for line in dockerfile.splitlines()
        if "runtime-requirements.lock" in line and "sha256sum" in line
    )
    assert f'= "{requirements_sha256}";' in requirements_guard


def test_publication_workflow_refuses_robotwin_before_build_selection() -> None:
    workflow = (ROOT / ".github/workflows/publish-public-images.yml").read_text()
    guard = 'if tool == "robotwin":'
    matrix_append = "matrix.append({"
    assert workflow.index(guard) < workflow.index(matrix_append)
    assert "native-content policy and built-byte evidence are incomplete" in workflow


def test_build_refuses_unresolved_native_policy_before_docker(tmp_path: Path) -> None:
    marker = tmp_path / "docker-invoked"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        f"#!/bin/sh\nprintf invoked >{marker}\nexit 99\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o700)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    completed = subprocess.run(
        [
            str(IMAGE_ROOT / "build.sh"),
            "--source-sha",
            head,
            "--image",
            f"local.invalid/npa-robotwin:dev-{head}",
        ],
        cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "native-content policy is unresolved" in completed.stderr
    assert not marker.exists()


def test_public_native_policy_is_intentionally_unusable_until_byte_review() -> None:
    policy = json.loads(
        (
            ROOT / "npa/scripts/image_byte_scan/public_policies/robotwin-bootstrap.json"
        ).read_text()
    )
    assert policy["schema_version"] == "npa.image-native-content-policy.v1"
    assert policy["entries"] == []
    assert set(policy["detector_identity"].values()) == {"UNRESOLVED-PHASE-A"}
