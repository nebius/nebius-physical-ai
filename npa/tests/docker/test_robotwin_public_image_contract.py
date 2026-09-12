"""Static Phase A contract for the unbuilt RoboTwin public bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from npa.deploy import images


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "npa/docker/workbench/robotwin"


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


def test_dockerfile_is_nonroot_zero_payload_and_accidentally_unbuildable() -> None:
    text = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM ubuntu:22.04@sha256:UNRESOLVED-PHASE-A" in text
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
        "apt-get",
        "COPY --from",
    ):
        assert forbidden not in text


def test_build_script_passes_the_dockerfile_source_sha_argument() -> None:
    build = (IMAGE_ROOT / "build.sh").read_text(encoding="utf-8")
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert 'ARG NPA_SOURCE_SHA' in dockerfile
    assert '--build-arg "NPA_SOURCE_SHA=$SOURCE_SHA"' in build
    assert '--build-arg "SOURCE_SHA=$SOURCE_SHA"' not in build


def test_phase_a_locks_are_explicitly_incomplete_and_fetch_nothing() -> None:
    lock = json.loads((IMAGE_ROOT / "runtime-lock.json").read_text())
    assert lock["status"] == "incomplete"
    assert lock["runtime_artifacts"] == []
    assert lock["weights"] == []
    assert lock["cache"]["status"] == "disabled-until-lock-complete"
    assert "INCOMPLETE" in (IMAGE_ROOT / "apt-packages.lock").read_text()
    requirements = (IMAGE_ROOT / "runtime-requirements.lock").read_text()
    assert "INCOMPLETE" in requirements
    assert "https://" not in requirements


def test_publication_workflow_refuses_robotwin_before_build_selection() -> None:
    workflow = (ROOT / ".github/workflows/publish-public-images.yml").read_text()
    guard = 'if tool == "robotwin":'
    matrix_append = "matrix.append({"
    assert workflow.index(guard) < workflow.index(matrix_append)
    assert "Phase A immutable base/apt/runtime locks are incomplete" in workflow


def test_public_native_policy_is_intentionally_unusable_until_byte_review() -> None:
    policy = json.loads(
        (
            ROOT
            / "npa/scripts/image_byte_scan/public_policies/robotwin-bootstrap.json"
        ).read_text()
    )
    assert policy["schema_version"] == "npa.image-native-content-policy.v1"
    assert policy["entries"] == []
    assert set(policy["detector_identity"].values()) == {"UNRESOLVED-PHASE-A"}
