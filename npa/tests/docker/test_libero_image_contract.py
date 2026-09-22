"""Static packaging and source-closure contracts for the LIBERO neutral image."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import yaml


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "npa" / "docker" / "workbench" / "libero"
DOCKERFILE = IMAGE_ROOT / "Dockerfile"
LOCK = IMAGE_ROOT / "debian-packages.lock"
MANIFEST = IMAGE_ROOT / "runtime-manifest.json"
REQUIREMENTS = IMAGE_ROOT / "runtime-requirements.txt"
PUBLICATION_WORKFLOW = ROOT / ".github" / "workflows" / "publish-public-images.yml"

BASE_MANIFEST = (
    "sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496"
)
BASE_ROOTFS = "sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867"
DEBIAN_ROOTFS_MANIFEST = (
    "b06a30ffb450d8ea59edbe54a1e89ecb9f00571d0440684f6449a09f3b0475ec"
)


def test_dockerfile_is_digest_pinned_nonroot_neutral_bootstrap() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert text.startswith(f"FROM python:3.10-slim-bookworm@{BASE_MANIFEST}\n")
    assert f'org.nebius.npa.base-manifest="{BASE_MANIFEST}"' in text
    assert f'org.nebius.npa.base-rootfs-material="{BASE_ROOTFS}"' in text
    assert 'org.nebius.npa.redistribution="public-neutral-bootstrap"' in text
    assert 'org.nebius.npa.validation-status="quarantined-unvalidated"' in text
    assert (
        'org.opencontainers.image.licenses="Apache-2.0 AND '
        'LicenseRef-NPA-LIBERO-Neutral-Third-Party"' in text
    )
    assert (
        'org.nebius.npa.third-party-notices="/opt/npa/libero/'
        'THIRD_PARTY_NOTICES.md"' in text
    )
    assert "USER ubuntu" in text
    assert "useradd --no-log-init --uid 1000" in text
    assert "groupadd --gid 1001 npa-libero-exec" in text
    assert "useradd --no-log-init --uid 1001 --gid npa-libero-exec" in text
    assert (
        "install -d -m 1770 -o ubuntu -g npa-libero-exec /workspace/byof-runs" in text
    )
    assert (
        "install -d -m 0770 -o ubuntu -g npa-libero-exec /workspace/byof-runs"
        not in text
    )
    assert "ubuntu ALL=(npa-libero-exec) NOPASSWD: NPA_LIBERO_EXEC" in text
    assert "NPA_LIBERO_EXEC = /opt/npa/libero/runtime-bootstrap.py execute" in text
    assert "Defaults!NPA_LIBERO_EXEC closefrom_override" in text
    exec_environment = next(
        line
        for line in text.splitlines()
        if "Defaults!NPA_LIBERO_EXEC env_keep" in line
    )
    assert exec_environment.count("NPA_LIBERO_BOOTSTRAP_RECEIPT") == 1
    assert exec_environment.count("NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256") == 1
    assert (
        exec_environment.count("NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT")
        == 1
    )
    assert "NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_ID" not in exec_environment
    assert "execute-python" not in text
    for credential in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        assert credential not in text
    assert "ubuntu ALL=(root) NOPASSWD: NPA_SKYPILOT_SSH" in text
    assert "Cmnd_Alias NPA_SKYPILOT_APT = /usr/local/sbin/apt-get update" in text
    assert "ubuntu ALL=(root) NOPASSWD: NPA_SKYPILOT_APT" in text
    assert "trusted_bootstrap_marker_value=skypilot-apt-v1" in (
        IMAGE_ROOT / "skypilot-bootstrap-guard.sh"
    ).read_text(encoding="utf-8")
    assert "/usr/local/bin/ssh-keygen -A" in text
    assert "rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub" in text
    assert (
        "ln -s /usr/local/sbin/npa-skypilot-bootstrap-guard /usr/local/bin/ssh-keygen"
    ) in text
    assert "/usr/sbin/sshd" not in text
    assert "NOPASSWD:ALL" not in text.replace(" ", "")
    assert "PYTHONPATH" not in text
    assert "LD_PRELOAD" not in text
    assert "apt-get upgrade" not in text
    assert "ARG SOURCE_DATE_EPOCH" in text
    assert "case \"${SOURCE_DATE_EPOCH}\" in ''|*[!0-9]*) exit 64" in text
    assert 'chage --lastday "$((SOURCE_DATE_EPOCH / 86400))" ubuntu' in text
    for volatile_path in (
        "/var/log/apt/*",
        "/var/cache/apt/archives/*",
        "/var/log/dpkg.log",
        "/var/log/alternatives.log",
        "/var/log/lastlog",
        "/var/log/faillog",
        "/var/cache/ldconfig/aux-cache",
    ):
        assert volatile_path in text
    assert "$1 ~ /^(base|dependency|direct)$/" in text
    assert "base|dependency|direct)" in text
    assert "pip install" not in text
    assert "runtime-bootstrap.py ensure" not in text
    assert "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_SHA256" not in text
    assert "type=secret" not in text
    assert "customer-authorization-public-key.b64" not in text
    assert "NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256" in text
    for forbidden in (
        "nvidia/cuda:",
        "pytorch/pytorch:",
        "git clone",
        "LIBERO-datasets/resolve",
        "bert-base-cased/resolve",
        "ACCEPT_LIBERO",
    ):
        assert forbidden not in text


def test_output_storage_trust_root_decode_rejects_trailing_garbage(
    tmp_path: Path,
) -> None:
    encoded = tmp_path / "output-storage-authorization-public-key.b64"
    decoded = tmp_path / "output-storage-authorization-public-key.raw"
    canonical = base64.b64encode(bytes(range(32)))
    script = """
set -eu
base64 -d "$1" > "$2"
test "$(wc -c < "$2")" = 32
test "$(base64 -w 0 < "$2")" = "$(cat "$1")"
"""

    encoded.write_bytes(canonical)
    accepted = subprocess.run(
        ["sh", "-c", script, "sh", str(encoded), str(decoded)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr

    encoded.write_bytes(canonical + b"!")
    rejected = subprocess.run(
        ["sh", "-c", script, "sh", str(encoded), str(decoded)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0


def test_smoke_writes_runtime_configuration_only_under_output_boundary() -> None:
    text = (IMAGE_ROOT / "smoke.sh").read_text(encoding="utf-8")

    assert "umask 027" in text
    assert "output_root=${NPA_SMOKE_OUTPUT_DIR:?" in text
    assert 'LIBERO_CONFIG_PATH="$output_root/.libero-config"' in text
    assert 'LIBERO_CONFIG_PATH="$runtime_root/' not in text
    assert 'rm -f "$LIBERO_CONFIG_PATH/config.yaml"' in text
    assert 'rmdir "$LIBERO_CONFIG_PATH"' in text
    assert 'LIBERO_EXPERIMENT_DIR="$(mktemp -d /tmp/' in text
    assert 'rm -rf -- "$LIBERO_EXPERIMENT_DIR"' in text
    assert 'cfg.experiment_dir = os.environ["LIBERO_EXPERIMENT_DIR"]' in (
        IMAGE_ROOT / "libero_smoke.py"
    ).read_text(encoding="utf-8")
    assert (
        '"cache": "/workspace/.cache/npa/libero/'
        '<customer-run-manifest-scope-sha256>"'
        in (IMAGE_ROOT / "libero_smoke.py").read_text(encoding="utf-8")
    )


def test_skypilot_ssh_key_helper_accepts_only_runtime_host_key_generation(
    tmp_path,
) -> None:
    guard_source = (IMAGE_ROOT / "skypilot-bootstrap-guard.sh").read_text(
        encoding="utf-8"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "ssh-keygen.calls"
    real_keygen = tmp_path / "real-ssh-keygen"
    real_keygen.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$NPA_TEST_KEYGEN_CALLS"\n',
        encoding="utf-8",
    )
    real_keygen.chmod(0o755)
    fake_id = bin_dir / "id"
    fake_id.write_text("#!/bin/sh\nprintf '%s\\n' 0\n", encoding="utf-8")
    fake_id.chmod(0o755)
    skypilot_tmp = Path("/") / "tmp"
    guard_source = guard_source.replace(
        str(skypilot_tmp / "npa-skypilot-bootstrap-contract.failed"),
        str(tmp_path / "bootstrap-contract.failed"),
    ).replace(
        str(skypilot_tmp / "apt-ssh-setup.failed"),
        str(tmp_path / "apt-ssh-setup.failed"),
    )
    ssh_dir = tmp_path / "ssh"
    ssh_dir.mkdir()
    ssh_config = tmp_path / "sshd_config"
    ssh_config.write_text("UsePAM yes\n")
    guard_source = guard_source.replace("/etc/ssh", str(ssh_dir)).replace(
        "/opt/npa/libero/sshd_config", str(ssh_config)
    ).replace("/run/sshd", str(tmp_path / "sshd-run"))
    guard = bin_dir / "npa-skypilot-bootstrap-guard"
    guard.write_text(
        guard_source.replace("/usr/bin/ssh-keygen", str(real_keygen)),
        encoding="utf-8",
    )
    guard.chmod(0o755)
    keygen = bin_dir / "ssh-keygen"
    keygen.symlink_to(guard)
    system_bin = Path(shutil.which("basename") or "/usr/bin/basename").parent
    environment = {
        "PATH": f"{bin_dir}:{system_bin}",
        "NPA_TEST_KEYGEN_CALLS": str(calls),
    }

    accepted = subprocess.run(
        [keygen, "-A"], env=environment, capture_output=True, text=True, check=False
    )
    refused = subprocess.run(
        [keygen, "-f", str(tmp_path / "unexpected")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert accepted.returncode == 0
    assert (ssh_dir / "sshd_config").read_text() == "UsePAM yes\n"
    assert calls.read_text(encoding="utf-8") == "-A\n"
    assert refused.returncode == 87
    assert "unexpected-ssh-keygen-arguments" in refused.stderr
    assert calls.read_text(encoding="utf-8") == "-A\n"


def test_skypilot_failure_sentinels_do_not_follow_preplaced_symlinks(tmp_path) -> None:
    guard_source = (IMAGE_ROOT / "skypilot-bootstrap-guard.sh").read_text(
        encoding="utf-8"
    )
    contract_failure_source = (
        PurePosixPath("/") / "tmp" / "npa-skypilot-bootstrap-contract.failed"
    )
    sky_failure_source = PurePosixPath("/") / "tmp" / "apt-ssh-setup.failed"
    contract_failure = tmp_path / "bootstrap-contract.failed"
    sky_failure = tmp_path / "apt-ssh-setup.failed"
    contract_target = tmp_path / "contract-target"
    sky_target = tmp_path / "sky-target"
    contract_target.write_text("preserve-contract\n", encoding="utf-8")
    sky_target.write_text("preserve-sky\n", encoding="utf-8")
    contract_failure.symlink_to(contract_target)
    sky_failure.symlink_to(sky_target)
    guard_source = guard_source.replace(
        str(contract_failure_source), str(contract_failure)
    ).replace(str(sky_failure_source), str(sky_failure))
    guard_source = guard_source.replace(
        "guard_owner_uid=0", f"guard_owner_uid={os.getuid()}"
    )
    guard = tmp_path / "npa-skypilot-bootstrap-guard"
    guard.write_text(guard_source, encoding="utf-8")
    guard.chmod(0o755)
    keygen = tmp_path / "ssh-keygen"
    keygen.symlink_to(guard)

    result = subprocess.run(
        [keygen, "-f", str(tmp_path / "unexpected")],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    expected = (
        "NPA_SKYPILOT_BOOTSTRAP_FAILED status=87 "
        "detail=unexpected-ssh-keygen-arguments\n"
    )
    assert result.returncode == 87
    assert result.stderr == expected
    assert contract_target.read_text(encoding="utf-8") == "preserve-contract\n"
    assert sky_target.read_text(encoding="utf-8") == "preserve-sky\n"
    assert not contract_failure.is_symlink()
    assert not sky_failure.is_symlink()
    assert contract_failure.read_text(encoding="utf-8") == expected
    assert sky_failure.read_text(encoding="utf-8") == expected


def test_debian_lock_closes_selected_binary_and_corresponding_source() -> None:
    text = LOCK.read_text(encoding="utf-8")
    lines = [
        line.split("\t")
        for line in text.splitlines()
        if line and not line.startswith("#")
    ]
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
    versions = {item["name"]: item["version"] for item in manifest["runtime_artifacts"]}
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
        set(item)
        == {"id", "name", "boundary", "version", "url", "size_bytes", "sha256"}
        and item["name"].strip()
        and item["version"] == f"sha256:{item['sha256']}"
        and item["url"].startswith("https://")
        and item["size_bytes"] > 0
        and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        for item in manifest["governing_terms"]
    )
    assert "ACCEPT_" not in serialized
    assert manifest["customer_runtime_authorization"] == {
        "schema": "npa.libero.customer-runtime-authorization.v2",
        "required": True,
        "credentials_establish_acceptance": False,
    }
    credential_paths: list[tuple[str, ...]] = []

    def collect_credential_paths(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = (*path, str(key))
                if "credential" in str(key).lower():
                    credential_paths.append(child_path)
                collect_credential_paths(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect_credential_paths(child, (*path, str(index)))
        elif "credential" in str(value).lower():
            credential_paths.append(path)

    collect_credential_paths(manifest)
    assert credential_paths == [
        ("customer_runtime_authorization", "credentials_establish_acceptance")
    ]
    assert "HF_TOKEN" not in serialized
    assert "NGC_API_KEY" not in serialized
    assert all(
        set(item)
        == {
            "name",
            "version",
            "filename",
            "url",
            "sha256",
            "size_bytes",
            "license_expression",
            "metadata_source",
        }
        for item in manifest["runtime_artifacts"]
    )
    assert all(item["size_bytes"] > 0 for item in manifest["runtime_artifacts"])
    assert all(
        str(item["license_expression"]).strip()
        for item in manifest["runtime_artifacts"]
    )
    assert sum(item["size_bytes"] for item in manifest["runtime_artifacts"]) == (
        3_277_640_175
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

    assert (
        image_manifest["runtime_manifest_sha256"]
        == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    )
    assert (
        image_manifest["runtime_requirements_sha256"]
        == hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    )
    assert image_manifest["governing_terms_count"] == 7
    assert image_manifest["customer_acceptance"]["terms"] == [
        {
            "id": term["id"],
            "name": term["name"],
            "official_url": term["url"],
            "version": term["version"],
        }
        for term in json.loads(MANIFEST.read_text(encoding="utf-8"))["governing_terms"]
    ]


def test_build_script_requires_exact_sha_tag_and_buildx_attestations() -> None:
    text = (IMAGE_ROOT / "build.sh").read_text(encoding="utf-8")

    assert "ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-$source_sha" in text
    assert "--provenance=mode=max" in text
    assert "--sbom=true" in text
    assert "--metadata-file" in text
    assert "--load" not in text
    assert "type=oci,dest=$oci_archive,tar=true,rewrite-timestamp=true" in text
    assert '--verify-build-oci "$oci_archive"' in text
    assert (
        '"oci-archive:$oci_archive" "docker-archive:$oci_archive.docker.tar:$image"'
        in text
    )
    assert 'docker load --input "$oci_archive.docker.tar"' in text
    assert "_docker_save_config_digest(Path(sys.argv[1]))" in text
    assert "containerimage.config.digest" in text
    assert '--build-arg "SOURCE_DATE_EPOCH=$source_epoch"' in text
    assert "OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY" not in text
    assert "docker push" not in text
    assert "docker history" not in text


def test_publication_plan_accepts_neutral_build_without_runtime_authorization(tmp_path) -> None:
    workflow = yaml.safe_load(PUBLICATION_WORKFLOW.read_text())
    plan = next(step for step in workflow['jobs']['resolve']['steps']
                if step.get('name') == 'Resolve immutable public development plan')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    output = tmp_path / 'github-output'
    result = subprocess.run(
        ['bash', '-c', plan['run']], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, 'TARGET': 'ghcr.io/nebius/nebius-physical-ai',
             'DEVELOPMENT_SHA': head, 'BUILD_TOOLS': 'libero', 'CLEANUP_TOOLS': '',
             'LEROBOT_VERSION': '', 'GITHUB_OUTPUT': str(output)}, check=False,
    )
    assert result.returncode == 0, result.stderr
    values = dict(line.split('=', 1) for line in output.read_text().splitlines())
    matrix = json.loads(values['build_matrix'])
    assert matrix[0]['tool'] == 'libero'
    assert matrix[0]['image'].endswith('npa-libero:dev-' + head)
    assert values['build_count'] == '1'


def test_publication_keeps_pre_and_post_byte_gates_separate_from_runtime() -> None:
    text = PUBLICATION_WORKFLOW.read_text()
    assert 'libero_qualified_image_manifest' not in text
    assert 'NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY' not in text
    assert text.count('npa/scripts/scan_image_libero_payload.py') == 3
    assert text.count('--record-image-inventory') == 1
    assert text.count('--expected-image-inventory-sha256') == 1
    assert text.index('--record-image-inventory') < text.index('docker push "$IMAGE"')
    assert text.index('--expected-image-inventory-sha256') > text.index('docker push "$IMAGE"')
    assert '--verify-build-oci' in text
    assert 'libero-base-provenance.intoto.json' in text
    assert 'libero-base-sbom.intoto.json' in text
    assert '--provenance=mode=max' in text
    assert '--sbom=true' in text
