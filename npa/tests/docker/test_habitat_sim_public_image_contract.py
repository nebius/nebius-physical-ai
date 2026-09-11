"""Lock the unbuilt Habitat-Sim image and publication quarantine contract."""

from __future__ import annotations

from pathlib import Path
import re

from npa.deploy import images


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "npa/docker/workbench/habitat-sim"
DOCKERFILE = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")


def test_candidate_is_public_eligible_but_unbuilt_and_unpublishable() -> None:
    assert images.CONTAINER_IMAGE_NAMES["habitat-sim"] == "npa-habitat-sim"
    assert images.SUPPORTED_TOOL_VERSIONS["habitat-sim"].endswith("-unbuilt")
    assert images.is_publicly_redistributable("habitat-sim")
    assert "habitat-sim" in images.UNVALIDATED_PUBLICATION_TOOLS
    assert "habitat-sim" not in images.publicly_publishable_tools()


def test_dedicated_image_pins_base_snapshot_and_ca_bootstrap() -> None:
    base = "ubuntu:22.04@sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986"
    assert f"ARG BASE_IMAGE={base}" in DOCKERFILE
    assert "ARG APT_SNAPSHOT=20260903T121500Z" in DOCKERFILE
    assert "ADD --checksum=sha256:${CA_DEB_SHA256}" in DOCKERFILE
    assert (
        "http://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/pool/main/c/ca-certificates/"
        in DOCKERFILE
    )
    assert "URIs: https://snapshot.ubuntu.com/ubuntu/" in DOCKERFILE
    assert "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg" in DOCKERFILE
    assert DOCKERFILE.index("dpkg-deb -x /tmp/ca-certificates.deb") < DOCKERFILE.index(
        "URIs: https://snapshot.ubuntu.com/ubuntu/"
    )
    assert not re.search(r"URIs:\s+http://", DOCKERFILE)


def test_snapshot_pair_refuses_incompatible_perl_family() -> None:
    assert DOCKERFILE.count("candidate_perl_base=") == 2
    assert DOCKERFILE.count("candidate_perl=") == 2
    assert DOCKERFILE.count('test "$installed_perl" = "$candidate_perl_base"') == 2
    assert DOCKERFILE.count('test "$installed_perl" = "$candidate_perl"') == 2
    assert DOCKERFILE.count("5.34.0-3ubuntu1.8") == 2


def test_build_frontend_uses_supported_release_environment() -> None:
    assert "SKBUILD_CMAKE_BUILD_TYPE=Release" in DOCKERFILE
    assert "--config-settings" not in DOCKERFILE
    for setting in (
        "HABITAT_BUILD_GUI_VIEWERS=OFF",
        "HABITAT_WITH_BULLET=ON",
        "HABITAT_WITH_CUDA=OFF",
        "HABITAT_WITH_AUDIO=OFF",
        "HABITAT_BUILD_TEST=OFF",
        "HABITAT_BUILD_BASIS_COMPRESSOR=OFF",
    ):
        assert setting in DOCKERFILE


def test_final_stage_is_non_root_and_skypilot_bootstrap_capable() -> None:
    final = DOCKERFILE.split("FROM ${BASE_IMAGE} AS runtime", 1)[1]
    assert final.rstrip().endswith(
        'CMD ["python3", "-m", "npa.workflows.habitat_sim_smoke", "--help"]'
    )
    assert "USER ubuntu" in final
    for package in ("netcat-openbsd=", "openssh-server=", "rsync=", "sudo="):
        assert package in final
    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in final
    assert "rm -f /etc/ssh/ssh_host_*" in final
    assert "safe.directory" not in DOCKERFILE
    assert "-name '.gitconfig'" in final


def test_final_stage_has_no_scene_or_vendor_payload_input() -> None:
    for token in (
        "COPY data/",
        "COPY scene",
        "habitat-test-scenes.zip /",
        "FROM nvidia/",
        "nvcr.io",
        "pip install git+",
        "--secret",
    ):
        assert token not in DOCKERFILE
    assert "HABITAT_WITH_CUDA=OFF" in DOCKERFILE
    assert "ffmpeg-*' -delete" in DOCKERFILE


def test_local_builder_outputs_attested_oci_without_push_or_load() -> None:
    script = (PACKAGE / "build.sh").read_text(encoding="utf-8")
    assert "type=oci,dest=$output" in script
    assert "--provenance=mode=max" in script and "--sbom=true" in script
    assert "--push" not in script and "--load" not in script
    assert "git rev-parse --verify HEAD" in script
    assert "^[0-9a-f]{40}$" in script


def test_trusted_public_workflow_refuses_phase_a_candidate() -> None:
    workflow = (ROOT / ".github/workflows/publish-public-images.yml").read_text()
    assert 'if tool == "habitat-sim":' in workflow
    assert "Phase A quarantine refuses a public build" in workflow


def test_release_manifest_has_no_habitat_entry() -> None:
    release = ROOT / "npa/src/npa/deploy/public_release_manifest.json"
    assert "habitat-sim" not in release.read_text(encoding="utf-8")
