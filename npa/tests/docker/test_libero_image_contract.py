"""Static packaging and source-closure contracts for the LIBERO neutral image."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "npa" / "docker" / "workbench" / "libero"
DOCKERFILE = IMAGE_ROOT / "Dockerfile"
LOCK = IMAGE_ROOT / "debian-packages.lock"
MANIFEST = IMAGE_ROOT / "runtime-manifest.json"
REQUIREMENTS = IMAGE_ROOT / "runtime-requirements.txt"
PUBLICATION_WORKFLOW = ROOT / ".github" / "workflows" / "publish-public-images.yml"

BASE_MANIFEST = "sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496"
BASE_ROOTFS = "sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867"
DEBIAN_ROOTFS_MANIFEST = "b06a30ffb450d8ea59edbe54a1e89ecb9f00571d0440684f6449a09f3b0475ec"


def test_dockerfile_is_digest_pinned_nonroot_neutral_bootstrap() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert text.startswith(f"FROM python:3.10-slim-bookworm@{BASE_MANIFEST}\n")
    assert f'org.nebius.npa.base-manifest="{BASE_MANIFEST}"' in text
    assert f'org.nebius.npa.base-rootfs-material="{BASE_ROOTFS}"' in text
    assert 'org.nebius.npa.redistribution="public-neutral-bootstrap"' in text
    assert 'org.nebius.npa.validation-status="quarantined-unvalidated"' in text
    assert "USER ubuntu" in text
    assert "useradd --uid 1000" in text
    assert "ubuntu ALL=(root) NOPASSWD: NPA_SKYPILOT_SSH" in text
    assert "/usr/sbin/sshd" not in text
    assert "NOPASSWD:ALL" not in text.replace(" ", "")
    assert "apt-get upgrade" not in text
    assert '$1 ~ /^(base|dependency|direct)$/' in text
    assert "base|dependency|direct)" in text
    assert "pip install" not in text
    assert "runtime-bootstrap.py ensure" not in text
    for forbidden in (
        "nvidia/cuda:",
        "pytorch/pytorch:",
        "git clone",
        "LIBERO-datasets/resolve",
        "bert-base-cased/resolve",
        "ACCEPT_LIBERO",
    ):
        assert forbidden not in text


def test_debian_lock_closes_selected_binary_and_corresponding_source() -> None:
    text = LOCK.read_text(encoding="utf-8")
    lines = [line.split("\t") for line in text.splitlines() if line and not line.startswith("#")]
    binary_rows = [row for row in lines if row[0] in {"base", "dependency", "direct"}]
    source_rows = [row for row in lines if row[0] == "source"]

    assert f"# Debian rootfs manifest sha256: {DEBIAN_ROOTFS_MANIFEST}" in text
    assert len(binary_rows) == 175
    assert len(source_rows) == 357
    assert len({row[1] for row in binary_rows}) == len(binary_rows)
    direct = {row[1]: row[2] for row in binary_rows if row[0] == "direct"}
    for package in (
        "curl",
        "wget",
        "fuse3",
        "gcc",
        "git",
        "linux-libc-dev",
        "netcat-openbsd",
        "openssh-client",
        "openssh-server",
        "patch",
        "pciutils",
        "procps",
        "rsync",
        "sudo",
    ):
        assert package in direct
    assert direct["linux-libc-dev"] == "6.1.180-1"
    source_names = {row[1] for row in source_rows}
    assert {row[3] for row in binary_rows} <= source_names
    assert all(re.fullmatch(r"[0-9a-f]{64}", row[-2]) for row in lines)
    assert all(
        row[-1].startswith("https://snapshot.debian.org/archive/debian")
        for row in lines
    )


def test_runtime_manifest_is_metadata_only_and_never_an_acceptance_proxy() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest, sort_keys=True)

    assert manifest["runtime_artifact_count"] == 135
    assert len(manifest["runtime_artifacts"]) == 135
    versions = {
        item["name"]: item["version"] for item in manifest["runtime_artifacts"]
    }
    security_refreshed_versions = {
        "future": "0.18.3",
        "hydra-core": "1.3.4",
        "opencv-python": "4.8.1.78",
        "protobuf": "5.29.6",
        "torch": "2.13.0",
        "torchvision": "0.28.0",
        "transformers": "5.10.0",
        "wandb": "0.17.9",
    }
    assert {
        name: versions[name] for name in security_refreshed_versions
    } == security_refreshed_versions
    assert manifest["source"]["revision"] == "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    assert manifest["demonstration"]["license"] == "CC-BY-4.0"
    assert manifest["language_model"]["license"] == "Apache-2.0"
    assert {item["id"] for item in manifest["governing_terms"]} == {
        "libero-mit",
        "dataset-cc-by-4.0",
        "bert-apache-2.0",
        "pytorch-bsd",
        "cuda-eula",
        "nvidia-software-license",
        "cudnn-eula",
    }
    assert all(
        set(item) == {"id", "boundary", "url", "size_bytes", "sha256"}
        and item["url"].startswith("https://")
        and item["size_bytes"] > 0
        and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        for item in manifest["governing_terms"]
    )
    assert "ACCEPT_" not in serialized
    assert "credential" not in serialized.lower()
    assert all(
        set(item) == {
            "name",
            "version",
            "filename",
            "url",
            "sha256",
            "license_expression",
            "metadata_source",
        }
        for item in manifest["runtime_artifacts"]
    )
    lines = [
        line
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert len(lines) == len(manifest["runtime_artifacts"])
    assert lines == [
        f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']} # {item['url']}"
        for item in manifest["runtime_artifacts"]
    ]


def test_image_manifest_binds_runtime_manifest_requirements_and_terms() -> None:
    path = ROOT / "npa" / "src" / "npa" / "deploy" / "libero_image_manifest.json"
    image_manifest = json.loads(path.read_text(encoding="utf-8"))

    assert image_manifest["runtime_manifest_sha256"] == hashlib.sha256(
        MANIFEST.read_bytes()
    ).hexdigest()
    assert image_manifest["runtime_requirements_sha256"] == hashlib.sha256(
        REQUIREMENTS.read_bytes()
    ).hexdigest()
    assert image_manifest["governing_terms_count"] == 7


def test_build_script_requires_exact_sha_tag_and_buildx_attestations() -> None:
    text = (IMAGE_ROOT / "build.sh").read_text(encoding="utf-8")

    assert "ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-$source_sha" in text
    assert "--provenance=mode=max" in text
    assert "--sbom=true" in text
    assert "--metadata-file" in text
    assert "docker push" not in text
    assert "docker history" not in text


def test_publication_workflow_uses_dedicated_scanner_and_published_base_provenance() -> None:
    text = PUBLICATION_WORKFLOW.read_text(encoding="utf-8")

    assert "matrix.tool == 'libero'" in text
    assert "libero-base-provenance.intoto.json" in text
    assert "libero-base-sbom.intoto.json" in text
    assert "0d66ce85e6ecad0d044a1d4bae712afe24ff2eb0a5d89eb224944df3895f226b" in text
    assert "b290dbd3087fc5d2cf4af106f1d253c417a26080b70bdcf440ff96314a12c2bb" in text
    assert text.count("npa/scripts/scan_image_libero_payload.py") == 2
    assert text.count("--exported-rootfs") == 2
    assert text.count("docker export --output") >= 2
    assert text.count('--base-provenance "$RUNNER_TEMP/libero-base-provenance.intoto.json"') == 2
    assert 'if tool != "libero":\n              history = subprocess.check_output(' in text
    assert "--provenance=mode=max" in text
    assert "--sbom=true" in text
