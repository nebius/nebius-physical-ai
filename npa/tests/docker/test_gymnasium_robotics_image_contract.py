from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/gymnasium-robotics"


def test_neutral_dockerfile_is_pinned_non_root_and_gated_before_network() -> None:
    text = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "ubuntu:noble-20260905@sha256:a61567bd31828687156d735ea8eb01ba4e37636e225dd6a48ba94136a70d9d61"
        in text
    )
    assert text.index("./build.sh verify-bootstrap-locks") < text.index(
        "./build.sh install-bootstrap"
    )
    assert "runtime-bootstrap.py" in text
    assert "USER ubuntu" in text
    assert "ENV HOME=/home/ubuntu" in text
    assert "install -d -m 0755 -o ubuntu -g ubuntu /workspace" in text
    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in text
    assert "chmod 0440 /etc/sudoers.d/99-npa-skypilot-runtime" in text
    assert "PasswordAuthentication no" in text
    assert "PermitRootLogin no" in text
    assert "ssh-keygen -A" in text
    assert text.index("rm -f /etc/ssh/ssh_host_*") < text.index("USER ubuntu")
    assert 'ENTRYPOINT ["/usr/local/bin/npa-gymnasium-entrypoint"]' in text
    assert "COPY --from=" not in text
    assert "COPY npa/" not in text
    assert "COPY --chmod=0444 npa/" not in text
    assert "COPY --chmod=0555 npa/" not in text
    assert "/opt/venv" not in text
    assert "pip install" not in text
    assert "nvidia/cuda" not in text.lower()
    assert "nvcr.io" not in text.lower()


def test_candidate_copies_only_neutral_code_metadata_and_notices() -> None:
    text = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    copied = [line for line in text.splitlines() if line.startswith("COPY ")]
    assert copied
    for forbidden in (
        "gymnasium_robotics/",
        ".whl",
        ".tar.gz",
        "runtime-cache",
        "requirements.in",
    ):
        assert all(forbidden not in line for line in copied)
    for required in (
        "runtime-bootstrap.py",
        "capability_smoke.py",
        "asset-lock.json",
        "source-lock.json",
        "runtime-fetch-manifest.json",
        "THIRD_PARTY_NOTICES.md",
    ):
        assert any(required in line for line in copied)


def test_repository_locks_are_complete_exact_and_machine_readable() -> None:
    for name in (
        "source-lock.json",
        "apt-runtime.lock.json",
        "corresponding-source.lock.json",
        "runtime-fetch-manifest.json",
    ):
        payload = json.loads((IMAGE / name).read_text(encoding="utf-8"))
        assert payload["status"] == "complete"
        assert payload["reason"]
    requirements = (IMAGE / "requirements.lock").read_text()
    assert "# status: complete" in requirements
    assert requirements.count("--hash=sha256:") == 19
    fetch = json.loads((IMAGE / "runtime-fetch-manifest.json").read_text())["runtime_fetch"]
    for key in ("source_lock", "corresponding_source_lock"):
        assert fetch[key + "_sha256"] == hashlib.sha256(
            (IMAGE / fetch[key]).read_bytes()
        ).hexdigest()


def test_image_build_accepts_only_the_exact_complete_pre_network_locks() -> None:
    completed = subprocess.run(
        [str(IMAGE / "build.sh"), "verify-bootstrap-locks"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    script = (IMAGE / "build.sh").read_text(encoding="utf-8")
    assert script.index("missing ephemeral TLS trust input") < script.index(
        "apt-get update"
    )
    assert "EXPECTED_FINAL_PACKAGE_MANIFEST_SHA256" in script
    assert "https://snapshot.ubuntu.com/ubuntu/20260905T000000Z" in script


def test_egl_runtime_has_generic_loaders_in_the_pinned_ubuntu_closure() -> None:
    lock = json.loads((IMAGE / "apt-runtime.lock.json").read_text())
    requested = {entry["package"] for entry in lock["requested_runtime_packages"]}
    assert {"libegl1", "libopengl0"} <= requested
    binaries = {entry["package"]: entry for entry in lock["resolved_binary_packages"]}
    sources = {
        (entry["package"], entry["version"])
        for entry in lock["resolved_source_packages"]
    }
    # PyOpenGL's EGL backend loads both EGL and OpenGL through GLVND. Driver
    # injection supplies the vendor implementation, not these generic loaders.
    for name in ("libegl1", "libopengl0", "libglvnd0"):
        package = binaries[name]
        assert package["source_package"] == "libglvnd"
        assert (package["source_package"], package["source_version"]) in sources
        assert package["copyright_sha256"]


def test_apt_reads_only_the_ephemeral_world_readable_ca_secret() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    secret_mount = (
        "--mount=type=secret,id=npa_host_ca_bundle,required=true,"
        "target=/run/npa-host-ca-bundle.crt,mode=0444"
    )
    assert secret_mount in dockerfile
    assert "COPY /run/npa-host-ca-bundle.crt" not in dockerfile
    assert "ADD /run/npa-host-ca-bundle.crt" not in dockerfile


def test_pre_network_locks_are_readable_independent_of_checkout_modes() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    copy_lines = [
        line.strip()
        for line in dockerfile.splitlines()
        if line.strip().startswith("COPY ")
    ]
    assert copy_lines
    assert all("--chmod=04" in line or "--chmod=05" in line for line in copy_lines)
    assert (
        "COPY --chmod=0444 "
        "docker/workbench/gymnasium-robotics/apt-runtime.lock.json "
        "./apt-runtime.lock.json"
    ) in dockerfile
    assert (
        "COPY --chmod=0444 "
        "docker/workbench/gymnasium-robotics/corresponding-source.lock.json "
        "./corresponding-source.lock.json"
    ) in dockerfile
    assert (
        "COPY --chmod=0444 "
        "docker/workbench/gymnasium-robotics/runtime-fetch-manifest.json "
        "./runtime-fetch-manifest.json"
    ) in dockerfile
    assert (
        "COPY --chmod=0555 docker/workbench/gymnasium-robotics/build.sh ./build.sh"
        in dockerfile
    )


def test_installed_notices_are_immutable_and_verified_by_the_default_user() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    notice_directory = "RUN install -d -m 0755 /usr/share/doc/npa-gymnasium-robotics"
    notice_copy = (
        "COPY --chmod=0444 "
        "docker/workbench/gymnasium-robotics/THIRD_PARTY_NOTICES.md "
        "/usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md"
    )
    default_user_verifier = (
        "/usr/bin/python3 -I -B /opt/npa/gymnasium-robotics/verify_image.py"
    )
    assert notice_directory in dockerfile
    assert notice_copy in dockerfile
    assert dockerfile.index(notice_directory) < dockerfile.index(notice_copy)
    assert dockerfile.index("USER ubuntu") < dockerfile.index(default_user_verifier)
    for exact_mode in (
        "$(stat -c '%a' /usr/share/doc/npa-gymnasium-robotics)\" = 755",
        "$(stat -c '%a' /usr/share/doc/npa-gymnasium-robotics/THIRD_PARTY_NOTICES.md)\" = 444",
        "$(stat -c '%a' /usr/share/doc/npa-gymnasium-robotics/REDISTRIBUTION.md)\" = 444",
    ):
        assert exact_mode in dockerfile
    for notice in ("THIRD_PARTY_NOTICES.md", "REDISTRIBUTION.md"):
        path = f"/usr/share/doc/npa-gymnasium-robotics/{notice}"
        assert f"test -r {path}" in dockerfile
        assert f"test ! -w {path}" in dockerfile


def test_packaging_contract_distinguishes_reference_from_accepted_image() -> None:
    contract = yaml.safe_load(
        (ROOT / "npa/docker/workbench/packaging-contract.yaml").read_text()
    )
    entry = contract["images"]["gymnasium-robotics"]
    assert entry["redistribution"] == "public"
    assert entry["phase"] == "development-build-quarantine"
    assert entry["skypilot_bootstrap_contract"] == "skypilot-0.12.2-v1"
    assert "uid 1000" in entry["passwordless_root_exemption"]
    assert "reference build" in entry["notes"]
    assert "not accepted" in entry["notes"]
    assert "complete product scan" in entry["notes"]
    assert "neutral" in entry["notes"].lower()
    assert "accepted_manifest" not in entry


def test_notices_keep_runtime_provenance_without_claiming_a_grant() -> None:
    text = (IMAGE / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for token in (
        "MIT",
        "GPL-2.0-only",
        "Apache-2.0",
        "LICENSE.md",
        "59d6bdf35bd9cf53185a20eb63413fdfe57fe77c",
        "not a license grant",
        "not present in the neutral candidate layers",
    ):
        assert token in text


def test_six_delivery_boundaries_are_classified_independently() -> None:
    lock = json.loads((IMAGE / "source-lock.json").read_text(encoding="utf-8"))
    assert lock["delivery"] == {
        "source": "operator-owned-runtime-cache",
        "baked_runtime": "neutral-bootstrap-only",
        "weights": "none",
        "data_assets": "runtime-cache-only",
        "runtime_cache": "operator-owned-and-external",
        "outputs": "operator-owned-run-artifacts",
    }
    assert lock["decision_sha256"] == (
        "758a29a6fae55075dc4ba879907e81f949b7a4e23fa726b790fd4361241697a3"
    )
