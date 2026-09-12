"""Lock the unbuilt Habitat-Sim image and publication quarantine contract."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess

from npa.deploy import images


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "npa/docker/workbench/habitat-sim"
DOCKERFILE = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")


def _bootstrap_function() -> str:
    start = DOCKERFILE.index("npa_bootstrap_ca()")
    end = DOCKERFILE.index("    npa_bootstrap_ca /\n", start)
    return DOCKERFILE[start:end].replace("\\\n", "\n")


def _run_bootstrap(
    root: Path,
    *,
    certificate_count: int,
    config_bytes: int,
    config_sha256: str,
    bundle_bytes: int,
    bundle_sha256: str,
    path_prefix: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "CA_CERT_COUNT": str(certificate_count),
        "CA_CONFIG_BYTES": str(config_bytes),
        "CA_CONFIG_SHA256": config_sha256,
        "CA_BUNDLE_BYTES": str(bundle_bytes),
        "CA_BUNDLE_SHA256": bundle_sha256,
    }
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}:{env['PATH']}"
    return subprocess.run(
        [
            "bash",
            "-ceu",
            _bootstrap_function() + '\nnpa_bootstrap_ca "$1"',
            "npa-ca-bootstrap",
            str(root),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _create_ca(tmp_path: Path, name: str) -> bytes:
    key = tmp_path / f"{name}.key"
    certificate = tmp_path / f"{name}.crt"
    result = subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            f"/CN={name}",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return certificate.read_bytes()


def test_candidate_is_public_eligible_but_unbuilt_and_unpublishable() -> None:
    assert images.CONTAINER_IMAGE_NAMES["habitat-sim"] == "npa-habitat-sim"
    assert "habitat-sim" not in images.SUPPORTED_TOOL_VERSIONS
    assert images.UNBUILT_CANDIDATE_TOOL_VERSIONS["habitat-sim"].endswith("-unbuilt")
    assert images.supported_tool_version("habitat-sim").endswith("-unbuilt")
    assert images.is_publicly_redistributable("habitat-sim")
    assert "habitat-sim" in images.UNVALIDATED_PUBLICATION_TOOLS
    assert "habitat-sim" not in images.publicly_publishable_tools()


def test_candidate_has_no_generic_build_and_push_remedy() -> None:
    reference = "registry.invalid/task-owned/npa-habitat-sim:missing"

    assert images.build_and_push_command(reference) == ""


def test_dedicated_image_pins_base_snapshot_and_ca_bootstrap() -> None:
    base = "ubuntu:22.04@sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986"
    assert f"ARG BASE_IMAGE={base}" in DOCKERFILE
    assert "ARG APT_SNAPSHOT=20260903T121500Z" in DOCKERFILE
    for expected in (
        "ADD --checksum=sha256:${CA_DEB_SHA256}",
        "ADD --checksum=sha256:${OPENSSL_DEB_SHA256}",
        "ADD --checksum=sha256:${LIBSSL3_DEB_SHA256}",
    ):
        assert expected in DOCKERFILE
    assert (
        "http://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/pool/main/c/ca-certificates/"
        in DOCKERFILE
    )
    assert "URIs: https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/" in DOCKERFILE
    assert "Suites: jammy jammy-updates jammy-security" in DOCKERFILE
    assert "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg" in DOCKERFILE
    for expected in (
        "ARG CA_CERT_COUNT=121",
        "ARG CA_CONFIG_BYTES=4809",
        "ARG CA_CONFIG_SHA256=fe407f6205ff90c56d4f073ffaa2a8abde7835558919c19367a7cd4fd6312dad",
        "ARG CA_BUNDLE_BYTES=182140",
        "ARG CA_BUNDLE_SHA256=9481fcd95f41b221f02f14d896535fe500bec539bc563c4cdca1acee483a8bdd",
    ):
        assert expected in DOCKERFILE
    assert DOCKERFILE.count("npa_bootstrap_ca /") == 1
    trust_stage = DOCKERFILE.index("FROM ${BASE_IMAGE} AS npa-ca-trust")
    bootstrap = DOCKERFILE.index("npa_bootstrap_ca /", trust_stage)
    build_stage = DOCKERFILE.index("FROM ${BASE_IMAGE} AS build")
    assert trust_stage < bootstrap < build_stage
    for stage in ("build", "runtime"):
        stage_start = DOCKERFILE.index(f"FROM ${{BASE_IMAGE}} AS {stage}")
        config_copy = DOCKERFILE.index(
            "COPY --from=npa-ca-trust /etc/ca-certificates.conf", stage_start
        )
        bundle_copy = DOCKERFILE.index(
            "COPY --from=npa-ca-trust /etc/ssl/certs/ca-certificates.crt",
            stage_start,
        )
        https = DOCKERFILE.index(
            "URIs: https://snapshot.ubuntu.com/ubuntu/", stage_start
        )
        assert stage_start < config_copy < bundle_copy < https
    assert DOCKERFILE.count("source=/ca-certificates.deb") == 1
    assert DOCKERFILE.count("source=/openssl.deb") == 1
    assert DOCKERFILE.count("source=/libssl3.deb") == 1
    assert DOCKERFILE.index("dpkg-deb -x /tmp/libssl3.deb") < DOCKERFILE.index(
        "/usr/bin/openssl x509"
    )
    assert DOCKERFILE.index("dpkg-deb -x /tmp/openssl.deb") < DOCKERFILE.index(
        "/usr/bin/openssl x509"
    )
    assert (
        DOCKERFILE.count("openssl x509")
        == DOCKERFILE.count("/usr/bin/openssl x509")
        == 1
    )
    assert "COPY --from=npa-ca-bootstrap" not in DOCKERFILE
    assert not re.search(r"URIs:\s+http://", DOCKERFILE)


def test_ca_bootstrap_creates_exact_nonempty_config_and_bundle(
    tmp_path: Path,
) -> None:
    cert_dir = tmp_path / "usr/share/ca-certificates/mozilla"
    cert_dir.mkdir(parents=True)
    certificates = {
        "Zed.crt": _create_ca(tmp_path, "zed-fixture"),
        "Alpha.crt": _create_ca(tmp_path, "alpha-fixture"),
    }
    for name, payload in certificates.items():
        (cert_dir / name).write_bytes(payload)
    config = b"mozilla/Alpha.crt\nmozilla/Zed.crt\n"
    bundle = certificates["Alpha.crt"] + certificates["Zed.crt"]
    hostile_bin = tmp_path / "hostile-bin"
    hostile_bin.mkdir()
    hostile_openssl = hostile_bin / "openssl"
    hostile_openssl.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    hostile_openssl.chmod(0o755)

    result = _run_bootstrap(
        tmp_path,
        certificate_count=2,
        config_bytes=len(config),
        config_sha256=hashlib.sha256(config).hexdigest(),
        bundle_bytes=len(bundle),
        bundle_sha256=hashlib.sha256(bundle).hexdigest(),
        path_prefix=hostile_bin,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "etc/ca-certificates.conf").read_bytes() == config
    assert (tmp_path / "etc/ssl/certs/ca-certificates.crt").read_bytes() == bundle


def test_ca_bootstrap_refuses_hash_or_incomplete_certificate_input(
    tmp_path: Path,
) -> None:
    cert_dir = tmp_path / "usr/share/ca-certificates/mozilla"
    cert_dir.mkdir(parents=True)
    certificate = b"-----BEGIN CERTIFICATE-----\nfixture\n-----END CERTIFICATE-----\n"
    (cert_dir / "Fixture.crt").write_bytes(certificate)
    config = b"mozilla/Fixture.crt\n"

    bad_hash = _run_bootstrap(
        tmp_path,
        certificate_count=1,
        config_bytes=len(config),
        config_sha256="0" * 64,
        bundle_bytes=len(certificate),
        bundle_sha256=hashlib.sha256(certificate).hexdigest(),
    )
    assert bad_hash.returncode != 0

    bad_count = _run_bootstrap(
        tmp_path,
        certificate_count=2,
        config_bytes=len(config),
        config_sha256=hashlib.sha256(config).hexdigest(),
        bundle_bytes=len(certificate),
        bundle_sha256=hashlib.sha256(certificate).hexdigest(),
    )
    assert bad_count.returncode != 0

    bad_config_size = _run_bootstrap(
        tmp_path,
        certificate_count=1,
        config_bytes=len(config) + 1,
        config_sha256=hashlib.sha256(config).hexdigest(),
        bundle_bytes=len(certificate),
        bundle_sha256=hashlib.sha256(certificate).hexdigest(),
    )
    assert bad_config_size.returncode != 0

    (cert_dir / "Fixture.crt").write_bytes(certificate.rstrip(b"\n"))
    missing_newline = _run_bootstrap(
        tmp_path,
        certificate_count=1,
        config_bytes=len(config),
        config_sha256=hashlib.sha256(config).hexdigest(),
        bundle_bytes=len(certificate) - 1,
        bundle_sha256=hashlib.sha256(certificate.rstrip(b"\n")).hexdigest(),
    )
    assert missing_newline.returncode != 0


def test_ca_bootstrap_refuses_misbound_config_size_then_accepts_exact(
    tmp_path: Path,
) -> None:
    cert_dir = tmp_path / "usr/share/ca-certificates/mozilla"
    cert_dir.mkdir(parents=True)
    certificate = _create_ca(tmp_path, "config-size-fixture")
    (cert_dir / "Fixture.crt").write_bytes(certificate)
    config = b"mozilla/Fixture.crt\n"

    stale_contract = _run_bootstrap(
        tmp_path,
        certificate_count=1,
        config_bytes=len(config) + 1,
        config_sha256=hashlib.sha256(config).hexdigest(),
        bundle_bytes=len(certificate),
        bundle_sha256=hashlib.sha256(certificate).hexdigest(),
    )

    assert stale_contract.returncode != 0
    assert not (tmp_path / "etc/ssl/certs/ca-certificates.crt").exists()

    repaired_contract = _run_bootstrap(
        tmp_path,
        certificate_count=1,
        config_bytes=len(config),
        config_sha256=hashlib.sha256(config).hexdigest(),
        bundle_bytes=len(certificate),
        bundle_sha256=hashlib.sha256(certificate).hexdigest(),
    )
    assert repaired_contract.returncode == 0, repaired_contract.stderr


def test_ca_bootstrap_refuses_malformed_certificate_and_bundle_drift(
    tmp_path: Path,
) -> None:
    cert_dir = tmp_path / "usr/share/ca-certificates/mozilla"
    cert_dir.mkdir(parents=True)
    config = b"mozilla/Fixture.crt\n"
    malformed = (
        b"-----BEGIN CERTIFICATE-----\n"
        b"not-a-valid-x509-certificate\n"
        b"-----END CERTIFICATE-----\n"
    )
    (cert_dir / "Fixture.crt").write_bytes(malformed)
    malformed_result = _run_bootstrap(
        tmp_path,
        certificate_count=1,
        config_bytes=len(config),
        config_sha256=hashlib.sha256(config).hexdigest(),
        bundle_bytes=len(malformed),
        bundle_sha256=hashlib.sha256(malformed).hexdigest(),
    )
    assert malformed_result.returncode != 0

    certificate = _create_ca(tmp_path, "bundle-fixture")
    (cert_dir / "Fixture.crt").write_bytes(certificate)
    bad_bundle_size = _run_bootstrap(
        tmp_path,
        certificate_count=1,
        config_bytes=len(config),
        config_sha256=hashlib.sha256(config).hexdigest(),
        bundle_bytes=len(certificate) + 1,
        bundle_sha256=hashlib.sha256(certificate).hexdigest(),
    )
    assert bad_bundle_size.returncode != 0
    bad_bundle_hash = _run_bootstrap(
        tmp_path,
        certificate_count=1,
        config_bytes=len(config),
        config_sha256=hashlib.sha256(config).hexdigest(),
        bundle_bytes=len(certificate),
        bundle_sha256="0" * 64,
    )
    assert bad_bundle_hash.returncode != 0


def test_https_and_repository_signature_failures_have_no_bypass() -> None:
    for stage in DOCKERFILE.split("FROM ${BASE_IMAGE} AS runtime"):
        if "apt-get update" not in stage:
            continue
        assert (
            stage.index("COPY --from=npa-ca-trust")
            < stage.index("URIs: https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/")
            < stage.index("apt-get update")
        )
    for forbidden in (
        "Acquire::https::Verify-Peer=false",
        "Acquire::https::Verify-Host=false",
        "--allow-unauthenticated",
        "AllowInsecureRepositories",
        "trusted=yes",
        "apt-get update ||",
    ):
        assert forbidden not in DOCKERFILE
    assert (
        DOCKERFILE.count("Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg")
        == 2
    )


def test_generated_bundle_rejects_an_untrusted_tls_certificate(
    tmp_path: Path,
) -> None:
    cert_dir = tmp_path / "root/usr/share/ca-certificates/mozilla"
    cert_dir.mkdir(parents=True)

    trusted = tmp_path / "trusted-fixture.crt"
    untrusted = tmp_path / "untrusted-fixture.crt"
    trusted_payload = _create_ca(tmp_path, "trusted-fixture")
    _create_ca(tmp_path, "untrusted-fixture")
    (cert_dir / "Trusted.crt").write_bytes(trusted_payload)
    config = b"mozilla/Trusted.crt\n"
    root = tmp_path / "root"
    bootstrap = _run_bootstrap(
        root,
        certificate_count=1,
        config_bytes=len(config),
        config_sha256=hashlib.sha256(config).hexdigest(),
        bundle_bytes=len(trusted_payload),
        bundle_sha256=hashlib.sha256(trusted_payload).hexdigest(),
    )
    assert bootstrap.returncode == 0, bootstrap.stderr
    bundle = root / "etc/ssl/certs/ca-certificates.crt"

    trusted_result = subprocess.run(
        ["openssl", "verify", "-CAfile", str(bundle), str(trusted)],
        check=False,
        capture_output=True,
        text=True,
    )
    untrusted_result = subprocess.run(
        ["openssl", "verify", "-CAfile", str(bundle), str(untrusted)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert trusted_result.returncode == 0, trusted_result.stderr
    assert untrusted_result.returncode != 0
    assert "verification failed" in untrusted_result.stderr


def test_snapshot_signature_mismatch_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    distribution = repository / "dists/stable"
    distribution.mkdir(parents=True)
    (distribution / "Release").write_text(
        "Origin: fixture\n"
        "Label: fixture\n"
        "Suite: stable\n"
        "Codename: stable\n"
        "Date: Fri, 11 Sep 2026 00:00:00 UTC\n"
        "Architectures: amd64\n"
        "Components: main\n"
        "Description: signature-refusal fixture\n",
        encoding="utf-8",
    )
    (distribution / "Release.gpg").write_bytes(b"not-a-valid-signature\n")
    keyring = tmp_path / "fixture-keyring.gpg"
    keyring.write_bytes(b"not-a-valid-keyring\n")
    sources = tmp_path / "sources.list"
    sources.write_text(
        f"deb [signed-by={keyring}] file:{repository} stable main\n",
        encoding="utf-8",
    )
    lists = tmp_path / "lists"
    (lists / "partial").mkdir(parents=True)
    cache = tmp_path / "cache"
    (cache / "archives/partial").mkdir(parents=True)

    result = subprocess.run(
        [
            "apt-get",
            "-o",
            f"Dir::Etc::sourcelist={sources}",
            "-o",
            "Dir::Etc::sourceparts=-",
            "-o",
            f"Dir::State::Lists={lists}",
            "-o",
            f"Dir::Cache={cache}",
            "-o",
            "Dir::State::status=/dev/null",
            "-o",
            "Debug::NoLocking=1",
            "-o",
            "Acquire::AllowInsecureRepositories=false",
            "update",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Signed file isn't valid" in output


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
    final = DOCKERFILE.split("FROM ${BASE_IMAGE} AS runtime", 1)[1]
    for archive in ("ca-certificates*.deb", "openssl*.deb", "libssl3*.deb"):
        assert f"-name '{archive}'" in final


def test_local_builder_outputs_attested_oci_without_push_or_load() -> None:
    script = (PACKAGE / "build.sh").read_text(encoding="utf-8")
    assert "type=oci,dest=$output" in script
    assert "--provenance=mode=max" in script and "--sbom=true" in script
    assert "--push" not in script and "--load" not in script
    assert "git rev-parse --verify HEAD" in script
    assert "^[0-9a-f]{40}$" in script


def test_verifier_requires_reviewed_complete_runtime_closure_hashes() -> None:
    verifier = (PACKAGE / "verify_image.py").read_text(encoding="utf-8")
    assert '"--expected-dpkg-inventory-sha256", required=True' in verifier
    assert '"--expected-python-venv-inventory-sha256", required=True' in verifier
    assert '"--expected-native-closure-sha256", required=True' in verifier


def test_trusted_public_workflow_refuses_phase_a_candidate() -> None:
    workflow = (ROOT / ".github/workflows/publish-public-images.yml").read_text()
    assert 'if tool == "habitat-sim":' in workflow
    assert "Phase A quarantine refuses a public build" in workflow


def test_release_manifest_has_no_habitat_entry() -> None:
    release = ROOT / "npa/src/npa/deploy/public_release_manifest.json"
    assert "habitat-sim" not in release.read_text(encoding="utf-8")
