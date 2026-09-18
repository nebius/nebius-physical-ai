"""Lock the unbuilt Habitat-Sim image and publication quarantine contract."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import pytest
import yaml

from npa.deploy import images


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "npa/docker/workbench/habitat-sim"
DOCKERFILE = (PACKAGE / "Dockerfile").read_text(encoding="utf-8")
PACKAGING = yaml.safe_load(
    (ROOT / "npa/docker/workbench/packaging-contract.yaml").read_text(encoding="utf-8")
)
EXPECTED_RUNTIME_IDENTITY_COMMANDS = (
    "groupadd --gid 1000 ubuntu",
    "useradd --uid 1000 --gid 1000 --create-home --home-dir /home/ubuntu "
    "--shell /bin/bash --no-log-init ubuntu",
    'test "$(id -u ubuntu)" -eq 1000',
    'test "$(id -g ubuntu)" -eq 1000',
    "install -d -o ubuntu -g ubuntu -m 0700 /home/ubuntu/.ssh",
    "install -d -m 0755 /etc/ssh/sshd_config.d",
    "printf '%s\\n' 'PasswordAuthentication no' 'PermitRootLogin no' "
    "> /etc/ssh/sshd_config.d/99-npa-worker.conf",
)


def test_habitat_pending_source_closure_agrees_with_catalog_totals() -> None:
    entries = PACKAGING["images"]
    assert entries["habitat-sim"]["redistribution"] == "unvalidated"
    blackwell = json.loads(
        (ROOT / "npa/docker/workbench/blackwell-dc-images.json").read_bytes()
    )
    (habitat,) = [
        row for row in blackwell["images"] if row["name"] == "npa-habitat-sim"
    ]
    assert habitat["redistribution"] == "unvalidated"
    assert habitat["validation"] == "pending-build"
    counts = {
        value: sum(row["redistribution"] == value for row in entries.values())
        for value in ("public", "restricted", "unvalidated")
    }
    assert counts == {"public": 40, "restricted": 2, "unvalidated": 1}
    catalog = (ROOT / "docs/workbench/container-image-catalog.md").read_text()
    assert "43 packaging entries" in catalog
    assert (
        "40 redistribution-eligible, two restricted, and\none with pending source closure"
        in catalog
    )
    assert "habitat-sim" in images.PENDING_REDISTRIBUTION_TOOLS


def test_habitat_capability_metadata_keeps_byte_and_source_qualification_pending() -> (
    None
):
    from npa.smoke.capabilities import GOLDEN_EVAL_CAPABILITIES

    capability = GOLDEN_EVAL_CAPABILITIES["habitat-sim"][0]
    assert "intended exact-byte and corresponding-source" in capability
    assert "closure qualification remains pending" in capability
    assert "complete redistributable dependency closure" not in capability
    entry = yaml.safe_load((ROOT / "npa/src/npa/smoke/golden_evals.yaml").read_text())[
        "containers"
    ]["habitat-sim"]
    notes = entry["safety"]["notes"]
    assert "Unvalidated and publication-quarantined" in notes
    assert "exact-byte redistribution" in notes
    assert "complete corresponding-source qualification remain pending" in notes
    assert "not accepted image/live proof" in notes
    assert entry["golden_eval"]["status"] == "gpu-gated"
    assert "habitat-sim" in images.PENDING_REDISTRIBUTION_TOOLS


def _run_bash(
    script: str,
    *arguments: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-ceu", script, "npa-habitat-contract", *arguments],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


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
    return _run_bash(
        _bootstrap_function() + '\nnpa_bootstrap_ca "$1"',
        str(root),
        env=env,
    )


def _runtime_stage(dockerfile: str) -> str:
    return dockerfile.split("FROM ${BASE_IMAGE} AS runtime", 1)[1]


def _runtime_identity_commands(dockerfile: str) -> list[str]:
    runtime = _runtime_stage(dockerfile)
    start = runtime.index("groupadd --gid 1000 ubuntu")
    end = runtime.index(" && \\\n    printf 'ubuntu ALL=", start)
    command = runtime[start:end].replace("\\\n", " ")
    return [" ".join(part.split()) for part in re.split(r"\s+&&\s+", command)]


def _runtime_identity_contract(dockerfile: str) -> bool:
    runtime = _runtime_stage(dockerfile)
    runtime_users = re.findall(r"(?m)^USER\s+(\S+)\s*$", runtime)
    return tuple(
        _runtime_identity_commands(dockerfile)
    ) == EXPECTED_RUNTIME_IDENTITY_COMMANDS and runtime_users[-1:] == ["ubuntu"]


def _validate_habitat_root_exemption(
    dockerfile: str, exemption: dict[str, object]
) -> None:
    assert exemption["id"] == "habitat-sim-skypilot-0.12.2-v1"
    assert exemption["sources"] == ["habitat-sim/Dockerfile"]
    assert exemption["grants"] == [
        {
            "source": "habitat-sim/Dockerfile",
            "sudoers_entry": "ubuntu ALL=(ALL) NOPASSWD:ALL",
        }
    ]
    rationale = exemption["rationale"]
    for boundary in (
        "package, SSH, rsync",
        "runtime-selected",
        "finite command allowlist cannot preserve",
        "container",
        "trust boundary",
    ):
        assert boundary in rationale
    controls = exemption["compensating_controls"]
    assert controls == {
        "bootstrap_contract": "skypilot-0.12.2-v1",
        "build_time_host_keys_deleted": True,
        "capability_evidence": "runtime_probe_required",
        "entrypoint": "command-passthrough",
        "exact_digest_required": True,
        "final_user": "ubuntu",
        "no_baked_credentials": True,
        "no_default_sshd": True,
        "no_exposed_ssh_port": True,
        "trust_boundary": "ephemeral-workflow-task-container",
    }
    assert exemption["prohibited_uses"] == [
        "default-sshd",
        "public-ingress",
        "baked-credentials",
        "mutable-image-capability-claim",
    ]
    sudoers_writer = (
        "printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\\n' > /etc/sudoers.d/90-npa-skypilot"
    )
    sudoers_validator = (
        'test "$(cat /etc/sudoers.d/90-npa-skypilot)" = \\\n'
        "      'ubuntu ALL=(ALL) NOPASSWD:ALL'"
    )
    assert dockerfile.count(sudoers_writer) == 1
    assert dockerfile.count(sudoers_validator) == 1
    assert "rm -f /etc/ssh/ssh_host_*" in dockerfile
    assert "PasswordAuthentication no" in dockerfile
    assert "PermitRootLogin no" in dockerfile
    assert not re.search(r"(?im)^EXPOSE\s+.*\b22(?:/tcp)?\b", dockerfile)
    assert 'ENTRYPOINT ["/usr/local/bin/npa-habitat-entrypoint"]' in dockerfile
    assert dockerfile.rstrip().endswith(
        'CMD ["python3", "-m", "npa.workflows.habitat_sim_smoke", "--help"]'
    )
    assert re.findall(r"(?m)^USER\s+(\S+)$", dockerfile)[-1] == "ubuntu"
    entrypoint = (PACKAGE / "entrypoint.sh").read_text(encoding="utf-8")
    assert 'exec "$@"' in entrypoint
    assert "sshd" not in entrypoint


def _write_identity_dispatcher(bin_dir: Path) -> None:
    dispatcher = bin_dir / "identity-command"
    dispatcher.write_text(
        """#!/bin/sh
set -eu
state=$NPA_IDENTITY_STATE
case ${0##*/} in
  groupadd)
    test "$*" = "--gid 1000 ubuntu"
    test ! -e "$state"
    printf group > "$state"
    ;;
  useradd)
    test "$(cat "$state")" = group
    test "$*" = "--uid 1000 --gid 1000 --create-home --home-dir /home/ubuntu --shell /bin/bash --no-log-init ubuntu"
    printf user > "$state"
    ;;
  id)
    test "$(cat "$state")" = user
    test "$2" = ubuntu
    case "$1" in -u|-g) printf '1000\\n' ;; *) exit 64 ;; esac
    ;;
  install)
    case "$*" in
      "-d -o ubuntu -g ubuntu -m 0700 /home/ubuntu/.ssh")
        test "$(cat "$state")" = user
        printf ownership > "$state"
        ;;
      "-d -m 0755 $NPA_SSHD_CONFIG_DIR")
        test "$(cat "$state")" = ownership
        mkdir -p "$NPA_SSHD_CONFIG_DIR"
        printf ssh-config-dir > "$state"
        ;;
      *) exit 64 ;;
    esac
    ;;
  *) exit 64 ;;
esac
""",
        encoding="utf-8",
    )
    dispatcher.chmod(0o755)
    for command in ("groupadd", "useradd", "id", "install"):
        (bin_dir / command).symlink_to(dispatcher.name)


def _run_identity_commands(
    commands: list[str], root: Path
) -> subprocess.CompletedProcess[str]:
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    _write_identity_dispatcher(bin_dir)
    sshd_config_dir = root / "etc/ssh/sshd_config.d"
    sandboxed_commands = [
        command.replace("/etc/ssh/sshd_config.d", str(sshd_config_dir))
        for command in commands
    ]
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "NPA_IDENTITY_STATE": str(root / "identity-state"),
        "NPA_SSHD_CONFIG_DIR": str(sshd_config_dir),
    }
    return _run_bash(" && ".join(sandboxed_commands), env=env)


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


def test_candidate_has_pending_source_closure_and_is_unpublishable() -> None:
    assert images.CONTAINER_IMAGE_NAMES["habitat-sim"] == "npa-habitat-sim"
    assert "habitat-sim" not in images.SUPPORTED_TOOL_VERSIONS
    assert images.UNBUILT_CANDIDATE_TOOL_VERSIONS["habitat-sim"].endswith("-unbuilt")
    assert images.supported_tool_version("habitat-sim").endswith("-unbuilt")
    assert not images.is_publicly_redistributable("habitat-sim")
    assert "habitat-sim" in images.PENDING_REDISTRIBUTION_TOOLS
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


def test_runtime_venv_is_created_at_its_final_image_path() -> None:
    build = DOCKERFILE.split("FROM ${BASE_IMAGE} AS build", 1)[1].split(
        "FROM ${BASE_IMAGE} AS runtime", 1
    )[0]

    assert "python3 -m venv /opt/venv" in build
    assert "/opt/venv/bin/pip install" in build
    assert "COPY --from=build /opt/venv /opt/venv" in DOCKERFILE
    assert "/opt/runtime-venv" not in DOCKERFILE


def test_final_notices_exclude_builder_and_verifier_control_inputs() -> None:
    build = DOCKERFILE.split("FROM ${BASE_IMAGE} AS build", 1)[1].split(
        "FROM ${BASE_IMAGE} AS runtime", 1
    )[0]
    notice_copy = re.search(
        r"cp /opt/npa-build/source-manifest\.json .*? /opt/notices/",
        build,
        re.DOTALL,
    )

    assert notice_copy is not None
    assert "requirements-runtime.lock" in notice_copy.group()
    assert "requirements-build.lock" not in notice_copy.group()
    assert "runtime-payload.json" not in notice_copy.group()
    for name in (
        "source-manifest.json",
        "source-projection.json",
        "licenses.json",
        "apt-build.lock",
        "apt-runtime.lock",
        "requirements-runtime.lock",
    ):
        assert f"/opt/npa-build/{name}" in notice_copy.group()
    assert "-r requirements-build.lock" in build
    assert (
        "COPY --from=npa-source-provenance "
        "/inputs/docker/workbench/habitat-sim/runtime-payload.json" in build
    )


def test_openexr_fetchcontent_is_bound_to_local_exact_imath_source() -> None:
    local_override = (
        'CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_IMATH='
        "/opt/habitat-sim/src/deps/imath "
        '-DFETCHCONTENT_FULLY_DISCONNECTED=ON"'
    )
    assert DOCKERFILE.count(local_override) == 1
    build = DOCKERFILE.split("RUN python3 prepare_source.py", 1)[1].split(
        "FROM ${BASE_IMAGE} AS runtime", 1
    )[0]
    assert build.index(local_override) < build.index("/opt/build-venv/bin/pip wheel .")
    assert "src/deps/imath" in (PACKAGE / "source-manifest.json").read_text(
        encoding="utf-8"
    )
    assert "FETCHCONTENT_SOURCE_DIR_IMATH=*" not in DOCKERFILE


def test_final_stage_is_non_root_and_skypilot_bootstrap_capable() -> None:
    final = _runtime_stage(DOCKERFILE)
    runtime_lock = yaml.safe_load((PACKAGE / "apt-runtime.lock").read_text())
    locked_packages = {row["binary"] for row in runtime_lock["packages"]}
    assert final.rstrip().endswith(
        'CMD ["python3", "-m", "npa.workflows.habitat_sim_smoke", "--help"]'
    )
    assert "USER ubuntu" in final
    assert {"netcat-openbsd", "openssh-server", "rsync", "sudo"} <= locked_packages
    assert "npa-habitat-verify-apt verify-direct" in final
    assert "apt-get install -y --no-install-recommends " in final
    assert str(Path("/", "tmp", "npa-runtime-direct-debs", "*.deb")) in final
    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in final
    assert "rm -f /etc/ssh/ssh_host_*" in final
    assert "PasswordAuthentication no" in final
    assert "PermitRootLogin no" in final
    assert "EXPOSE 22" not in final
    assert (
        "org.nebius.npa.sudo-bootstrap-contract="
        '"habitat-sim-skypilot-0.12.2-v1"' in final
    )
    assert "safe.directory" not in DOCKERFILE
    assert "-name '.gitconfig'" in final


def test_skypilot_general_escalation_is_structured_and_mechanically_bounded() -> None:
    docs = (ROOT / "docs/workbench/container-packaging.md").read_text(encoding="utf-8")
    skill = (ROOT / "skills/workflows/byof-onboard/SKILL.md").read_text(
        encoding="utf-8"
    )
    normalized_docs = " ".join(docs.split())
    normalized_skill = " ".join(skill.split())
    assert "root or verified passwordless sudo" in normalized_docs
    assert "installs missing SSH, rsync, and service packages" in normalized_docs
    assert "passwordless `sudo`" in normalized_skill
    exemption = PACKAGING["images"]["habitat-sim"]["passwordless_root_exemption"]
    _validate_habitat_root_exemption(DOCKERFILE, exemption)


@pytest.mark.parametrize(
    ("token", "replacement"),
    [
        ("rm -f /etc/ssh/ssh_host_*", ":"),
        ("PasswordAuthentication no", "PasswordAuthentication yes"),
        ("PermitRootLogin no", "PermitRootLogin yes"),
        ("\nUSER ubuntu\n", "\nUSER root\n"),
        ('ENTRYPOINT ["/usr/local/bin/npa-habitat-entrypoint"]', "EXPOSE 22"),
    ],
)
def test_skypilot_root_exemption_rejects_compensating_control_drift(
    token: str, replacement: str
) -> None:
    exemption = PACKAGING["images"]["habitat-sim"]["passwordless_root_exemption"]
    assert token in DOCKERFILE
    with pytest.raises(AssertionError):
        _validate_habitat_root_exemption(
            DOCKERFILE.replace(token, replacement, 1), exemption
        )

    mutated = copy.deepcopy(exemption)
    mutated["compensating_controls"]["exact_digest_required"] = False
    with pytest.raises(AssertionError):
        _validate_habitat_root_exemption(DOCKERFILE, mutated)


def test_runtime_identity_order_and_final_user_refuse_hostile_mutants(
    tmp_path: Path,
) -> None:
    commands = _runtime_identity_commands(DOCKERFILE)

    assert commands == list(EXPECTED_RUNTIME_IDENTITY_COMMANDS)
    assert _runtime_identity_contract(DOCKERFILE)
    valid_root = tmp_path / "valid"
    valid = _run_identity_commands(commands, valid_root)
    assert valid.returncode == 0, valid.stderr
    assert (
        valid_root / "etc/ssh/sshd_config.d/99-npa-worker.conf"
    ).read_text() == "PasswordAuthentication no\nPermitRootLogin no\n"

    ownership_first = _run_identity_commands(
        [commands[4], *commands[:4], *commands[5:]], tmp_path / "ownership-first"
    )
    assert ownership_first.returncode != 0

    config_before_directory = _run_identity_commands(
        [*commands[:5], commands[6], commands[5]],
        tmp_path / "config-before-directory",
    )
    assert config_before_directory.returncode != 0

    root_mutant = DOCKERFILE.replace(
        "\nUSER ubuntu\nENTRYPOINT", "\nUSER root\nENTRYPOINT"
    )
    assert not _runtime_identity_contract(root_mutant)


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
    assert 'git -C "$repo_root" rev-parse --verify HEAD^{commit}' in script
    assert "^[0-9a-f]{40}$" in script
    assert 'git -C "$repo_root" diff --quiet' in script
    assert 'git -C "$repo_root" diff --cached --quiet' in script
    assert 'git -C "$repo_root" cat-file blob' in script
    assert '--build-context "npa-source-provenance=$projection"' in script
    assert "NPA_SOURCE_MANIFEST_SHA256=$manifest_sha256" in script


def test_verifier_requires_reviewed_complete_runtime_closure_hashes() -> None:
    verifier = (PACKAGE / "verify_image.py").read_text(encoding="utf-8")
    assert '"--analysis-root", type=Path, required=True' in verifier
    assert '"--trusted-root", type=Path, required=True' in verifier
    assert "with W.authorized_roots(args.analysis_root, args.trusted_root)" in verifier
    assert '"--expected-dpkg-inventory-sha256", required=True' in verifier
    assert '"--expected-python-venv-inventory-sha256", required=True' in verifier
    assert '"--expected-native-closure-sha256", required=True' in verifier
    assert '"--expected-npa-source-manifest-sha256", required=True' in verifier


def test_dockerfile_recomputes_and_records_committed_npa_source_manifest() -> None:
    assert (
        "COPY --from=npa-source-provenance / /opt/npa-source-provenance/" in DOCKERFILE
    )
    assert DOCKERFILE.count("sha256sum -c npa-source-manifest.sha256") == 2
    assert "npa-source-expected-paths" in DOCKERFILE
    assert "npa-source-observed-paths" in DOCKERFILE
    assert "cmp /tmp/npa-source-expected-paths" in DOCKERFILE
    assert "/usr/share/doc/npa-habitat-sim/npa-source-provenance" in DOCKERFILE
    assert (
        'org.nebius.npa.source-manifest-sha256="${NPA_SOURCE_MANIFEST_SHA256}"'
        in DOCKERFILE
    )
    assert 'org.opencontainers.image.revision="${NPA_SOURCE_SHA}"' in DOCKERFILE
    assert "COPY src/" not in DOCKERFILE
    assert "COPY docker/workbench/habitat-sim/" not in DOCKERFILE


def test_trusted_public_workflow_refuses_phase_a_candidate() -> None:
    workflow = (ROOT / ".github/workflows/publish-public-images.yml").read_text()
    assert 'if tool == "habitat-sim":' in workflow
    assert "Phase A quarantine refuses a public build" in workflow


def test_release_manifest_has_no_habitat_entry() -> None:
    release = ROOT / "npa/src/npa/deploy/public_release_manifest.json"
    assert "habitat-sim" not in release.read_text(encoding="utf-8")


def test_habitat_golden_requires_real_operator_qualification_bindings() -> None:
    manifest = yaml.safe_load(
        (ROOT / "npa/src/npa/smoke/golden_evals.yaml").read_text(encoding="utf-8")
    )
    golden = manifest["containers"]["habitat-sim"]["golden_eval"]
    command = golden["command"]
    assert golden["status"] == "gpu-gated"
    for variable in (
        "NPA_WORKFLOW_RUN_ID",
        "NPA_RENDERED_PLAN_SHA256",
        "NPA_TASK_IMAGE",
        "NPA_HABITAT_GOLDEN_OUTPUT_URI",
    ):
        assert f'test -n "${variable}"' in command
    assert '--run-id "$NPA_WORKFLOW_RUN_ID"' in command
    assert '--plan-sha256 "$NPA_RENDERED_PLAN_SHA256"' in command
    assert '--output-uri "$NPA_HABITAT_GOLDEN_OUTPUT_URI"' in command
    assert "example-bucket" not in command
