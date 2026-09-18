"""Static packaging and source-closure contracts for the LIBERO neutral image."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
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
    assert "npa_libero_customer_authorization_public_key_b64" not in text
    assert "customer-authorization-public-key.b64" not in text
    assert (
        "--mount=type=secret,id=npa_libero_output_storage_authorization_public_key_b64,required=true"
        in text
    )
    assert "output-storage-authorization-public-key.b64" in text
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
    assert '"oci-archive:$oci_archive" "docker-daemon:$image"' in text
    assert "containerimage.config.digest" in text
    assert '--build-arg "SOURCE_DATE_EPOCH=$source_epoch"' in text
    assert "docker push" not in text
    assert "docker history" not in text


def test_publication_workflow_uses_dedicated_scanner_and_published_base_provenance() -> (
    None
):
    text = PUBLICATION_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    resolve_steps = workflow["jobs"]["resolve"]["steps"]
    quarantine_index = next(
        index
        for index, step in enumerate(resolve_steps)
        if step.get("name") == "Keep LIBERO public disclosure quarantined"
    )
    quarantine_step = resolve_steps[quarantine_index]

    assert quarantine_index == 1
    assert quarantine_step["env"] == {
        "REQUESTED_TOOLS": "${{ inputs.tool }}",
        "BUILD_TOOLS": "${{ inputs.build_development_tools }}",
        "CLEANUP_TOOLS": "${{ inputs.cleanup_development_tools }}",
    }
    empty_selectors = {
        "REQUESTED_TOOLS": "",
        "BUILD_TOOLS": "",
        "CLEANUP_TOOLS": "",
    }
    quarantine_error = (
        "LIBERO public disclosure remains blocked until a separate "
        "exact-digest live-B200 authorization is implemented and accepted."
    )
    for selector in empty_selectors:
        selector_environment = {**empty_selectors, selector: "libero"}
        result = subprocess.run(
            ["bash", "-c", quarantine_step["run"]],
            check=False,
            capture_output=True,
            env=selector_environment,
            text=True,
        )
        assert result.returncode == 1
        assert quarantine_error in result.stdout

    assert "matrix.tool == 'libero'" in text
    assert "libero-base-provenance.intoto.json" in text
    assert "libero-base-sbom.intoto.json" in text
    assert "0d66ce85e6ecad0d044a1d4bae712afe24ff2eb0a5d89eb224944df3895f226b" in text
    assert "b290dbd3087fc5d2cf4af106f1d253c417a26080b70bdcf440ff96314a12c2bb" in text
    assert text.count("npa/scripts/scan_image_libero_payload.py") == 3
    assert text.count("--verify-build-oci") == 1
    assert text.count("--exported-rootfs") == 2
    assert text.count("--expected-image-inventory-sha256") == 2
    assert text.count("--expected-config-digest") == 2
    assert text.count("--expected-canonical-build-metadata-sha256") == 2
    assert text.count("--expected-base-provenance-sha256") == 2
    assert text.count("--image-inventory-output") == 2
    assert text.count("docker export --output") >= 2
    assert (
        text.count(
            '--base-provenance "$RUNNER_TEMP/libero-base-provenance.intoto.json"'
        )
        == 2
    )
    assert "history = subprocess.check_output(" in text
    assert 'if tool != "libero"' not in text
    assert "--provenance=mode=max" in text
    assert "--sbom=true" in text
    assert "type=oci,dest=$RUNNER_TEMP/libero-build.oci.tar" in text
    archive_verification = '--verify-build-oci "$RUNNER_TEMP/libero-build.oci.tar"'
    archive_refusal = (
        "LIBERO archive import remains disabled while public disclosure is quarantined."
    )
    assert (
        text.index("type=oci,dest=$RUNNER_TEMP/libero-build.oci.tar")
        < text.index(archive_verification)
        < text.index(archive_refusal)
    )
    assert "docker-daemon:" not in text
    assert archive_refusal in text
    assert "Install the LIBERO OCI archive importer" not in text
    skopeo_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "skopeo" in step.get("run", "")
    ]
    assert len(skopeo_steps) == 1
    assert skopeo_steps[0]["if"] == "matrix.tool == 'ncore'"
    assert '[[ "$TOOL" == curobo || "$TOOL" == libero ]]' in text
    assert "libero_qualified_image_manifest" in text
    assert "checked-in qualification development SHA does not match" in text
    assert "only the checked-in qualification manifest may differ" in text
    assert 'git merge-base --is-ancestor "$DEVELOPMENT_SHA" HEAD' in text
    assert 'build_args=(--build-arg "NPA_SOURCE_SHA=$DEVELOPMENT_SHA")' in text
    assert '"npa/src/npa/deploy/libero_image_manifest.json"' in text
    assert "libero_publication_lineage_values" in text
    assert 'crane tag "$exact" "dev-$DEVELOPMENT_SHA"' in text
    assert 'test "$digest" = "$LIBERO_QUALIFIED_OCI_DIGEST"' in text
    assert "Revalidate qualified LIBERO repository and OCI lineage" in text
    assert "LIBERO_QUALIFIED_ATTESTATION_LAYERS" in text
    assert "libero-final-attestation-manifest.json" in text
    assert "LIBERO_QUALIFIED_PUBLICATION_BUNDLE_SHA256" in text
    assert "LIBERO_QUALIFIED_INFRASTRUCTURE_BUNDLE_SHA256" not in text
    assert "inputs.libero_private_image_inventory_sha256" not in text
    assert "inputs.libero_private_config_digest" not in text
    assert (
        "LIBERO_PRIVATE_IMAGE_INVENTORY_SHA256: "
        "${{ matrix.libero_private_image_inventory_sha256 }}"
    ) in text
    assert (
        "LIBERO_PRIVATE_CONFIG_DIGEST: ${{ matrix.libero_private_config_digest }}"
    ) in text
    assert "First publication is unavailable" in text
    assert "refusing before any registry write" in text
    assert "Private destination is genuinely empty" not in text
    assert "group: public-image-registry-mutation" in text
    assert "group: public-image-${{" not in text
    assert re.search(r"^\s*- uses: [^#\n]+@v", text, re.MULTILINE) is None
    assert re.search(r"^\s*uses:\s*[^#\n]+@v[0-9]+\s*$", text, re.MULTILINE) is None
    assert 'visibility="$(gh api "$package_api" --jq .visibility)"' in text
    final_inventory = text.index("libero-final-package-versions.json")
    visibility_refusal = text.index(
        "Public visibility transition is deferred", final_inventory
    )
    assert final_inventory < visibility_refusal
    assert "visibility=public" not in text
    assert "no registry-enforced exclusive-writer or atomic compare-and-set" in text
    assert "no visibility PATCH was attempted" in text
    assert "Repository-local workflow concurrency cannot exclude another" in text
    assert "${TOOL}-public-package-versions.json" in text
    assert "NPA_FIRST_PUBLICATION_REQUIRED=1" in text
    assert "reject every unexpected tag" in text
    assert "Failed LIBERO cleanup cannot isolate every qualified" in text
    for host_gate in (
        "npa/tests/workflows/test_byof_container_verify.py",
        "npa/tests/workflows/test_byof_libero.py",
        "npa/tests/workflows/test_byof_repo.py",
        "npa/tests/workflows/test_byof_solution_smokes.py",
        "npa/tests/workflows/test_byof_source_auth.py",
    ):
        assert host_gate in text
    assert 'test "$LIBERO_PACKAGE_WRITER_REPOSITORY" = "$GITHUB_REPOSITORY"' in text
    assert "Retained the private qualified LIBERO versions" in text
    assert "Refusing non-atomic registry cleanup" in text
    assert "versions and package configuration are retained" in text
    assert "gh api --method DELETE" not in text
    assert "NPA_LIBERO_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_B64" not in text
    assert "LIBERO_QUALIFIED_CUSTOMER_AUTHORIZATION_PUBLIC_KEY_SHA256" not in text
    assert "LIBERO_QUALIFIED_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_SHA256" in text
    assert 'libero_qualification["customer_authorization_public_key_sha256"]' in text
    assert "base64 -d \"$storage_key_file\" | sha256sum | cut -d' ' -f1" in text
    assert "npa_libero_customer_authorization_public_key_b64" not in text
    assert (
        "--secret id=npa_libero_output_storage_authorization_public_key_b64,"
        "env=NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_PUBLIC_KEY_B64"
    ) in text
    assert "refusing before any registry write" in text
    assert '"failure","cancelled"' in text
    for immutable_action in (
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
        "docker/setup-buildx-action@8d2750c68a42422c14e847fe6c8ac0403b4cbd6f",
        "imjasonh/setup-crane@31b88efe9de28ae0ffa220711af4b60be9435f6e",
        "docker/login-action@c94ce9fb468520275223c153574b00df6fe4bcc9",
        "actions/attest-build-provenance@977bb373ede98d70efdf65b84cb5f73e068dcc2a",
        "actions/attest-sbom@4651f806c01d8637787e274ac3bdf724ef169f34",
    ):
        assert immutable_action in text


def test_libero_scratch_cleanup_failure_runs_once_and_fails(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    python = runtime / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o700)
    (runtime / ".complete.json").write_text("{}\n", encoding="utf-8")
    (runtime / "source/libero/libero").mkdir(parents=True)
    output = tmp_path / "output"
    config = output / ".libero-config"
    config.mkdir(parents=True)
    (config / "unexpected").write_text("retain\n", encoding="utf-8")

    completed = subprocess.run(
        [IMAGE_ROOT / "smoke.sh"],
        env={
            "LIBERO_RUNTIME_ROOT": str(runtime),
            "NPA_SMOKE_OUTPUT_DIR": str(output),
            "PATH": os.environ["PATH"],
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stderr.count("LIBERO scratch config cleanup failed") == 1


def test_libero_requested_cleanup_refuses_even_a_complete_exact_graph(
    tmp_path: Path,
) -> None:
    spec = yaml.safe_load(PUBLICATION_WORKFLOW.read_text(encoding="utf-8"))
    job = spec["jobs"]["cleanup-requested"]
    step = next(
        item for item in job["steps"] if item.get("name", "").startswith("Delete")
    )
    script = step["run"]
    sha = "1" * 40
    root = "sha256:" + "a" * 64
    platform = "sha256:" + "b" * 64
    attestation = "sha256:" + "c" * 64
    expected = sorted([root, platform, attestation])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "state=Path(os.environ['FIXTURE_STATE'])\n"
        "if not state.exists(): state.write_text(os.environ['FIXTURE_VERSIONS'])\n"
        "versions=json.loads(state.read_text())\n"
        "if '--paginate' in args:\n"
        " print(json.dumps([versions])); raise SystemExit(0)\n"
        "if '--method' in args and 'DELETE' in args:\n"
        " target=args[-1]\n"
        " if '/versions/' not in target: raise SystemExit('package-wide delete refused')\n"
        " version_id=int(target.rsplit('/',1)[1])\n"
        " if sum(item['id'] == version_id for item in versions) != 1: raise SystemExit('unknown version')\n"
        " state.write_text(json.dumps([item for item in versions if item['id'] != version_id]))\n"
        " open(os.environ['DELETE_RECORD'],'a').write(target+'\\n'); raise SystemExit(0)\n"
        "if '-i' in args:\n"
        " print('HTTP/2.0 404 Not Found'); raise SystemExit(1)\n"
        "print(json.dumps({'visibility':'public','repository':{'full_name':os.environ['GITHUB_REPOSITORY']},'name':'nebius-physical-ai/npa-libero'}))\n",
        encoding="utf-8",
    )
    gh.chmod(0o700)
    crane = bin_dir / "crane"
    crane.write_text(
        f"#!{sys.executable}\nimport os,sys\n"
        "reference=sys.argv[2]\n"
        "print(reference.rsplit('@',1)[1] if '@' in reference else os.environ['LIBERO_QUALIFIED_OCI_DIGEST'])\n",
        encoding="utf-8",
    )
    crane.chmod(0o700)
    delete_record = tmp_path / "deletes"
    common = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "IMAGE": f"ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-{sha}",
        "TOOL": "libero",
        "LIBERO_QUALIFIED_OCI_DIGEST": root,
        "LIBERO_QUALIFIED_PACKAGE_VERSION_DIGESTS": json.dumps(expected),
        "LIBERO_PACKAGE_WRITER_REPOSITORY": "nebius/nebius-physical-ai",
        "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "DELETE_RECORD": str(delete_record),
    }
    exact_versions = [
        {
            "id": 101,
            "name": root,
            "metadata": {"container": {"tags": [f"dev-{sha}"]}},
        },
        {"id": 102, "name": platform, "metadata": {"container": {"tags": []}}},
        {
            "id": 103,
            "name": attestation,
            "metadata": {"container": {"tags": []}},
        },
    ]
    exact_state = tmp_path / "exact-versions.json"
    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={
            **common,
            "FIXTURE_VERSIONS": json.dumps(exact_versions),
            "FIXTURE_STATE": str(exact_state),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0, completed.stderr
    assert "Refusing non-atomic registry cleanup" in completed.stdout
    assert not delete_record.exists()
    assert json.loads(exact_state.read_text(encoding="utf-8")) == exact_versions
    unrelated = [
        *exact_versions,
        {
            "id": 104,
            "name": "sha256:" + "d" * 64,
            "metadata": {"container": {"tags": []}},
        },
    ]
    unrelated_state = tmp_path / "unrelated-versions.json"
    refused = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={
            **common,
            "FIXTURE_VERSIONS": json.dumps(unrelated),
            "FIXTURE_STATE": str(unrelated_state),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert not delete_record.exists()


def test_failed_build_cleanup_runs_for_failure_and_cancellation() -> None:
    spec = yaml.safe_load(PUBLICATION_WORKFLOW.read_text(encoding="utf-8"))
    condition = str(spec["jobs"]["cleanup-failed-build"]["if"])
    assert '"failure","cancelled"' in condition
    assert "always()" in condition


@pytest.mark.parametrize(
    ("failure", "expected_returncode"),
    [
        ("404", 0),
        ("401", 1),
        ("403", 1),
        ("429", 1),
        ("500", 1),
        ("existing", 1),
        ("invalid-json", 1),
    ],
)
def test_failed_build_cleanup_accepts_only_proven_package_absence(
    tmp_path: Path, failure: str, expected_returncode: int
) -> None:
    spec = yaml.safe_load(PUBLICATION_WORKFLOW.read_text(encoding="utf-8"))
    steps = spec["jobs"]["cleanup-failed-build"]["steps"]
    script = next(
        step["run"]
        for step in steps
        if str(step.get("name") or "").startswith("Remove an exact run-owned")
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        "args=sys.argv[1:]\n"
        "failure=os.environ['FIXTURE_FAILURE']\n"
        "if '--paginate' in args:\n"
        " print('{' if failure == 'invalid-json' else '')\n"
        " raise SystemExit(0 if failure == 'invalid-json' else 1)\n"
        "if '-i' in args:\n"
        " status = '200' if failure in {'existing','invalid-json'} else failure\n"
        " print(f'HTTP/2.0 {status} fixture')\n"
        " raise SystemExit(0 if status == '200' else 1)\n"
        "if '--method' in args:\n"
        " open(os.environ['DELETE_RECORD'],'a').write('delete\\n')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    gh.chmod(0o700)
    delete_record = tmp_path / "deletes"
    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FIXTURE_FAILURE": failure,
            "IMAGE": "ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-" + "1" * 40,
            "TOOL": "libero",
            "LIBERO_QUALIFIED_OCI_DIGEST": "sha256:" + "a" * 64,
            "LIBERO_QUALIFIED_PACKAGE_VERSION_DIGESTS": "[]",
            "LIBERO_PACKAGE_WRITER_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "DELETE_RECORD": str(delete_record),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_returncode
    assert not delete_record.exists()
    if failure not in {"404", "existing", "invalid-json"}:
        assert "Failed-build package absence is unverified" in completed.stdout
    if failure in {"existing", "invalid-json"}:
        assert "package exists but its versions could not be verified" in (
            completed.stdout
        )


def test_failed_libero_cleanup_preserves_qualified_and_unrelated_versions(
    tmp_path: Path,
) -> None:
    spec = yaml.safe_load(PUBLICATION_WORKFLOW.read_text(encoding="utf-8"))
    steps = spec["jobs"]["cleanup-failed-build"]["steps"]
    script = next(
        step["run"]
        for step in steps
        if str(step.get("name") or "").startswith("Remove an exact run-owned")
    )
    root = "sha256:" + "a" * 64
    platform = "sha256:" + "b" * 64
    attestation = "sha256:" + "c" * 64
    unrelated = "sha256:" + "d" * 64
    tag = "dev-" + "1" * 40
    state_path = tmp_path / "versions.json"
    state_path.write_text(
        json.dumps(
            [
                {"id": 1, "name": root, "metadata": {"container": {"tags": [tag]}}},
                {"id": 2, "name": platform, "metadata": {"container": {"tags": []}}},
                {"id": 3, "name": attestation, "metadata": {"container": {"tags": []}}},
                {
                    "id": 99,
                    "name": unrelated,
                    "metadata": {"container": {"tags": ["stable-other"]}},
                },
            ]
        ),
        encoding="utf-8",
    )
    delete_record = tmp_path / "deleted.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "args=sys.argv[1:]\n"
        "state_path=os.environ['VERSION_STATE']\n"
        "state=json.load(open(state_path))\n"
        "if '--paginate' in args:\n"
        " print(json.dumps([state]))\n"
        " raise SystemExit(0)\n"
        "if '--method' in args and args[args.index('--method')+1] == 'DELETE':\n"
        " version_id=args[-1].rsplit('/',1)[-1]\n"
        " removed=[row for row in state if str(row['id']) == version_id]\n"
        " if len(removed) != 1: raise SystemExit(1)\n"
        " state=[row for row in state if str(row['id']) != version_id]\n"
        " json.dump(state,open(state_path,'w'))\n"
        " with open(os.environ['DELETE_RECORD'],'a') as stream: stream.write(version_id+'\\n')\n"
        " raise SystemExit(0)\n"
        "if '--jq' in args:\n"
        " query=args[args.index('--jq')+1]\n"
        " print('public' if query == '.visibility' else os.environ['GITHUB_REPOSITORY'])\n"
        " raise SystemExit(0)\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    gh.chmod(0o700)
    repository = "ghcr.io/nebius/nebius-physical-ai/npa-libero"
    predicate_type = "https://slsa.dev/provenance/v1"
    config_blob = b"{}\n"
    config_digest = "sha256:" + hashlib.sha256(config_blob).hexdigest()
    layer_blob = (
        json.dumps(
            {
                "_type": "https://in-toto.io/Statement/v1",
                "predicateType": predicate_type,
                "subject": [
                    {
                        "name": repository,
                        "digest": {"sha256": platform.removeprefix("sha256:")},
                    }
                ],
                "predicate": {},
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    layer_digest = "sha256:" + hashlib.sha256(layer_blob).hexdigest()
    root_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "digest": platform,
                    "platform": {"architecture": "amd64", "os": "linux"},
                },
                {
                    "digest": attestation,
                    "annotations": {
                        "vnd.docker.reference.digest": platform,
                        "vnd.docker.reference.type": "attestation-manifest",
                    },
                },
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    attestation_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "config": {"digest": config_digest, "size": len(config_blob)},
            "layers": [
                {
                    "digest": layer_digest,
                    "size": len(layer_blob),
                    "annotations": {"in-toto.io/predicate-type": predicate_type},
                }
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    crane = bin_dir / "crane"
    crane.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        "args=sys.argv[1:]\n"
        "repository=os.environ['CRANE_REPOSITORY']\n"
        "root=os.environ['ROOT_DIGEST']\n"
        "attestation=os.environ['ATTESTATION_DIGEST']\n"
        "config=os.environ['CONFIG_DIGEST']\n"
        "layer=os.environ['LAYER_DIGEST']\n"
        "if args == ['digest', f'{repository}@{root}']:\n"
        " print(root)\n"
        "elif args == ['manifest', f'{repository}@{root}']:\n"
        " print(os.environ['ROOT_MANIFEST'])\n"
        "elif args == ['digest', f'{repository}@{attestation}']:\n"
        " print(attestation)\n"
        "elif args == ['manifest', f'{repository}@{attestation}']:\n"
        " print(os.environ['ATTESTATION_MANIFEST'])\n"
        "elif args == ['blob', repository, config]:\n"
        " sys.stdout.buffer.write(bytes.fromhex(os.environ['CONFIG_BLOB_HEX']))\n"
        "elif args == ['blob', repository, layer]:\n"
        " sys.stdout.buffer.write(bytes.fromhex(os.environ['LAYER_BLOB_HEX']))\n"
        "else:\n"
        " print(f'refused unexpected crane invocation: {args!r}', file=sys.stderr)\n"
        " raise SystemExit(64)\n",
        encoding="utf-8",
    )
    crane.chmod(0o700)

    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "IMAGE": repository + ":" + tag,
            "TOOL": "libero",
            "LIBERO_QUALIFIED_OCI_DIGEST": root,
            "LIBERO_QUALIFIED_PLATFORM_MANIFEST_DIGEST": platform,
            "LIBERO_QUALIFIED_ATTESTATION_MANIFEST_DIGEST": attestation,
            "LIBERO_QUALIFIED_ATTESTATION_CONFIG_DIGEST": config_digest,
            "LIBERO_QUALIFIED_ATTESTATION_LAYERS": json.dumps(
                [
                    {
                        "predicate_type": predicate_type,
                        "digest": layer_digest,
                        "size_bytes": len(layer_blob),
                    }
                ]
            ),
            "LIBERO_QUALIFIED_PACKAGE_VERSION_DIGESTS": json.dumps(
                sorted([root, platform, attestation])
            ),
            "LIBERO_PACKAGE_WRITER_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "VERSION_STATE": str(state_path),
            "DELETE_RECORD": str(delete_record),
            "CRANE_REPOSITORY": repository,
            "ROOT_DIGEST": root,
            "ATTESTATION_DIGEST": attestation,
            "CONFIG_DIGEST": config_digest,
            "LAYER_DIGEST": layer_digest,
            "ROOT_MANIFEST": root_manifest,
            "ATTESTATION_MANIFEST": attestation_manifest,
            "CONFIG_BLOB_HEX": config_blob.hex(),
            "LAYER_BLOB_HEX": layer_blob.hex(),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert "Refusing non-atomic registry cleanup" in completed.stdout
    assert not delete_record.exists()
    assert json.loads(state_path.read_text(encoding="utf-8")) == [
        {"id": 1, "name": root, "metadata": {"container": {"tags": [tag]}}},
        {"id": 2, "name": platform, "metadata": {"container": {"tags": []}}},
        {"id": 3, "name": attestation, "metadata": {"container": {"tags": []}}},
        {
            "id": 99,
            "name": unrelated,
            "metadata": {"container": {"tags": ["stable-other"]}},
        },
    ]


def test_failed_libero_cleanup_preserves_public_visibility_and_exact_graph(
    tmp_path: Path,
) -> None:
    spec = yaml.safe_load(PUBLICATION_WORKFLOW.read_text(encoding="utf-8"))
    steps = spec["jobs"]["cleanup-failed-build"]["steps"]
    script = next(
        step["run"]
        for step in steps
        if str(step.get("name") or "").startswith("Remove an exact run-owned")
    )
    root = "sha256:" + "a" * 64
    platform = "sha256:" + "b" * 64
    attestation = "sha256:" + "c" * 64
    tag = "dev-" + "1" * 40
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "visibility": "public",
                "deleted": False,
                "versions": [
                    {"id": 1, "name": root, "metadata": {"container": {"tags": [tag]}}},
                    {
                        "id": 2,
                        "name": platform,
                        "metadata": {"container": {"tags": []}},
                    },
                    {
                        "id": 3,
                        "name": attestation,
                        "metadata": {"container": {"tags": []}},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    operations = tmp_path / "operations"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "args=sys.argv[1:]\n"
        "path=os.environ['PACKAGE_STATE']\n"
        "state=json.load(open(path))\n"
        "if '--paginate' in args:\n"
        " print(json.dumps([state['versions']])); raise SystemExit(0)\n"
        "if '-i' in args:\n"
        " print('HTTP/2.0 404 Not Found' if state['deleted'] else 'HTTP/2.0 200 OK')\n"
        " raise SystemExit(1 if state['deleted'] else 0)\n"
        "if '--method' in args and args[args.index('--method')+1] == 'DELETE':\n"
        " assert state['visibility'] == 'public'\n"
        " version_id=args[-1].rsplit('/',1)[-1]\n"
        " removed=[row for row in state['versions'] if str(row['id']) == version_id]\n"
        " assert len(removed) == 1\n"
        " state['versions']=[row for row in state['versions'] if str(row['id']) != version_id]\n"
        " state['deleted']=not state['versions']\n"
        " json.dump(state,open(path,'w'))\n"
        " open(os.environ['OPERATIONS'],'a').write('delete-version-'+version_id+'\\n')\n"
        " raise SystemExit(0)\n"
        "if '--jq' in args:\n"
        " query=args[args.index('--jq')+1]\n"
        " print(state['visibility'] if query == '.visibility' "
        "else os.environ['GITHUB_REPOSITORY'])\n"
        " raise SystemExit(0)\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    gh.chmod(0o700)
    repository = "ghcr.io/nebius/nebius-physical-ai/npa-libero"
    predicate_type = "https://slsa.dev/provenance/v1"
    config_blob = b"{}\n"
    config_digest = "sha256:" + hashlib.sha256(config_blob).hexdigest()
    layer_blob = (
        json.dumps(
            {
                "_type": "https://in-toto.io/Statement/v1",
                "predicateType": predicate_type,
                "subject": [
                    {
                        "name": repository,
                        "digest": {"sha256": platform.removeprefix("sha256:")},
                    }
                ],
                "predicate": {},
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    layer_digest = "sha256:" + hashlib.sha256(layer_blob).hexdigest()
    root_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "digest": platform,
                    "platform": {"architecture": "amd64", "os": "linux"},
                },
                {
                    "digest": attestation,
                    "annotations": {
                        "vnd.docker.reference.digest": platform,
                        "vnd.docker.reference.type": "attestation-manifest",
                    },
                },
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    attestation_manifest = json.dumps(
        {
            "schemaVersion": 2,
            "config": {"digest": config_digest, "size": len(config_blob)},
            "layers": [
                {
                    "digest": layer_digest,
                    "size": len(layer_blob),
                    "annotations": {"in-toto.io/predicate-type": predicate_type},
                }
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    crane = bin_dir / "crane"
    crane.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        "args=sys.argv[1:]\n"
        "repository=os.environ['CRANE_REPOSITORY']\n"
        "root=os.environ['ROOT_DIGEST']\n"
        "attestation=os.environ['ATTESTATION_DIGEST']\n"
        "config=os.environ['CONFIG_DIGEST']\n"
        "layer=os.environ['LAYER_DIGEST']\n"
        "if args == ['digest', f'{repository}@{root}']:\n"
        " print(root)\n"
        "elif args == ['manifest', f'{repository}@{root}']:\n"
        " print(os.environ['ROOT_MANIFEST'])\n"
        "elif args == ['digest', f'{repository}@{attestation}']:\n"
        " print(attestation)\n"
        "elif args == ['manifest', f'{repository}@{attestation}']:\n"
        " print(os.environ['ATTESTATION_MANIFEST'])\n"
        "elif args == ['blob', repository, config]:\n"
        " sys.stdout.buffer.write(bytes.fromhex(os.environ['CONFIG_BLOB_HEX']))\n"
        "elif args == ['blob', repository, layer]:\n"
        " sys.stdout.buffer.write(bytes.fromhex(os.environ['LAYER_BLOB_HEX']))\n"
        "else:\n"
        " print(f'refused unexpected crane invocation: {args!r}', file=sys.stderr)\n"
        " raise SystemExit(64)\n",
        encoding="utf-8",
    )
    crane.chmod(0o700)

    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "IMAGE": repository + ":" + tag,
            "TOOL": "libero",
            "LIBERO_QUALIFIED_OCI_DIGEST": root,
            "LIBERO_QUALIFIED_PLATFORM_MANIFEST_DIGEST": platform,
            "LIBERO_QUALIFIED_ATTESTATION_MANIFEST_DIGEST": attestation,
            "LIBERO_QUALIFIED_ATTESTATION_CONFIG_DIGEST": config_digest,
            "LIBERO_QUALIFIED_ATTESTATION_LAYERS": json.dumps(
                [
                    {
                        "predicate_type": predicate_type,
                        "digest": layer_digest,
                        "size_bytes": len(layer_blob),
                    }
                ]
            ),
            "LIBERO_QUALIFIED_PACKAGE_VERSION_DIGESTS": json.dumps(
                sorted([root, platform, attestation])
            ),
            "LIBERO_PACKAGE_WRITER_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "PACKAGE_STATE": str(state_path),
            "OPERATIONS": str(operations),
            "CRANE_REPOSITORY": repository,
            "ROOT_DIGEST": root,
            "ATTESTATION_DIGEST": attestation,
            "CONFIG_DIGEST": config_digest,
            "LAYER_DIGEST": layer_digest,
            "ROOT_MANIFEST": root_manifest,
            "ATTESTATION_MANIFEST": attestation_manifest,
            "CONFIG_BLOB_HEX": config_blob.hex(),
            "LAYER_BLOB_HEX": layer_blob.hex(),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert "Refusing non-atomic registry cleanup" in completed.stdout
    assert not operations.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["visibility"] == "public"
    assert state["deleted"] is False
    assert state["versions"] == [
        {"id": 1, "name": root, "metadata": {"container": {"tags": [tag]}}},
        {"id": 2, "name": platform, "metadata": {"container": {"tags": []}}},
        {"id": 3, "name": attestation, "metadata": {"container": {"tags": []}}},
    ]


def test_first_publication_retry_preserves_a_closed_private_graph_without_ownership(
    tmp_path: Path,
) -> None:
    spec = yaml.safe_load(PUBLICATION_WORKFLOW.read_text(encoding="utf-8"))
    steps = spec["jobs"]["build-development"]["steps"]
    script = next(
        step["run"]
        for step in steps
        if step.get("name")
        == "Prove destination cannot expose unvalidated tagged bytes"
    )
    root = "sha256:" + "a" * 64
    referrer = "sha256:" + "b" * 64
    tag = "dev-" + "1" * 40
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "deleted": False,
                "versions": [
                    {"id": 1, "name": root, "metadata": {"container": {"tags": [tag]}}},
                    {
                        "id": 2,
                        "name": referrer,
                        "metadata": {"container": {"tags": []}},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "args=sys.argv[1:]\n"
        "path=os.environ['PACKAGE_STATE']\n"
        "state=json.load(open(path))\n"
        "if '--paginate' in args:\n"
        " print(json.dumps([state['versions']])); raise SystemExit(0)\n"
        "if '-i' in args:\n"
        " print('HTTP/2.0 404 Not Found' if state['deleted'] else 'HTTP/2.0 200 OK')\n"
        " raise SystemExit(1 if state['deleted'] else 0)\n"
        "if '--method' in args and args[args.index('--method')+1] == 'DELETE':\n"
        " state['deleted']=True; state['versions']=[]\n"
        " json.dump(state,open(path,'w'))\n"
        " raise SystemExit(0)\n"
        "if '--jq' in args:\n"
        " query=args[args.index('--jq')+1]\n"
        " print('private' if query == '.visibility' "
        "else os.environ['GITHUB_REPOSITORY'])\n"
        " raise SystemExit(0)\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    gh.chmod(0o700)
    crane = bin_dir / "crane"
    crane.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "args=sys.argv[1:]\n"
        "if args[0] == 'digest': print(os.environ['ROOT_DIGEST'])\n"
        "elif args[0] == 'manifest':\n"
        " print(json.dumps({'schemaVersion':2,"
        "'mediaType':'application/vnd.oci.image.manifest.v1+json',"
        "'subject':{'digest':os.environ['ROOT_DIGEST']},"
        "'layers':[{'digest':'sha256:'+'c'*64}]}))\n"
        "else: raise SystemExit(1)\n",
        encoding="utf-8",
    )
    crane.chmod(0o700)
    github_env = tmp_path / "github-env"

    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "IMAGE": "ghcr.io/nebius/nebius-physical-ai/npa-genesis:" + tag,
            "TOOL": "genesis",
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "GITHUB_ENV": str(github_env),
            "PACKAGE_STATE": str(state_path),
            "ROOT_DIGEST": root,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0, completed.stdout + completed.stderr
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["deleted"] is False
    assert state["versions"] == [
        {"id": 1, "name": root, "metadata": {"container": {"tags": [tag]}}},
        {"id": 2, "name": referrer, "metadata": {"container": {"tags": []}}},
    ]
    assert "NPA_FIRST_PUBLICATION_REQUIRED=1" in github_env.read_text(encoding="utf-8")
    assert "First publication is unavailable" in completed.stdout


@pytest.mark.parametrize("destination", ["absent", "empty-private", "private-libero"])
def test_first_publication_refuses_before_any_registry_write(
    tmp_path: Path, destination: str
) -> None:
    spec = yaml.safe_load(PUBLICATION_WORKFLOW.read_text(encoding="utf-8"))
    steps = spec["jobs"]["build-development"]["steps"]
    script = next(
        step["run"]
        for step in steps
        if step.get("name")
        == "Prove destination cannot expose unvalidated tagged bytes"
    )
    operations = tmp_path / "registry-writes"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import os,sys\n"
        "from pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "if '--method' in args or '-X' in args:\n"
        " Path(os.environ['REGISTRY_WRITES']).write_text('unexpected mutation')\n"
        " raise SystemExit(99)\n"
        "if '--include' in args:\n"
        " absent=os.environ['DESTINATION']=='absent'\n"
        " print('HTTP/2.0 404 Not Found' if absent else 'HTTP/2.0 200 OK')\n"
        " raise SystemExit(1 if absent else 0)\n"
        "if args[-2:]==['--jq','.visibility']:\n"
        " print('private'); raise SystemExit(0)\n"
        "raise SystemExit(98)\n",
        encoding="utf-8",
    )
    gh.chmod(0o700)
    github_env = tmp_path / "github-env"
    completed = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            script + '\nprintf push >> "$REGISTRY_WRITES"\n',
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "IMAGE": "ghcr.io/example/test-bootstrap:dev-" + "1" * 40,
            "TOOL": "libero" if destination == "private-libero" else "genesis",
            "GITHUB_ENV": str(github_env),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "DESTINATION": destination,
            "REGISTRY_WRITES": str(operations),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "refusing before any registry write" in completed.stdout
    assert (
        github_env.read_text(encoding="utf-8") == "NPA_FIRST_PUBLICATION_REQUIRED=1\n"
    )
    assert not operations.exists()
