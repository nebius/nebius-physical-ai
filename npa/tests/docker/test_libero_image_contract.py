"""Static packaging and source-closure contracts for the LIBERO neutral image."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

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
    assert "USER ubuntu" in text
    assert "useradd --no-log-init --uid 1000" in text
    assert "groupadd --gid 1001 npa-libero-exec" in text
    assert "useradd --no-log-init --uid 1001 --gid npa-libero-exec" in text
    assert "ubuntu ALL=(npa-libero-exec) NOPASSWD: NPA_LIBERO_EXEC" in text
    assert "NPA_LIBERO_EXEC = /opt/npa/libero/runtime-bootstrap.py execute" in text
    assert "Defaults!NPA_LIBERO_EXEC closefrom_override" in text
    exec_environment = next(
        line for line in text.splitlines() if "Defaults!NPA_LIBERO_EXEC env_keep" in line
    )
    assert exec_environment.count("NPA_LIBERO_BOOTSTRAP_RECEIPT") == 1
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
    assert (
        "--mount=type=secret,id=npa_libero_manager_acceptance_public_key_b64,required=true"
        in text
    )
    assert "manager-acceptance-public-key.b64" in text
    assert "chmod 0444 /opt/npa/libero/manager-acceptance-public-key.b64" in text
    for forbidden in (
        "nvidia/cuda:",
        "pytorch/pytorch:",
        "git clone",
        "LIBERO-datasets/resolve",
        "bert-base-cased/resolve",
        "ACCEPT_LIBERO",
    ):
        assert forbidden not in text


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
        set(item) == {"id", "boundary", "url", "size_bytes", "sha256"}
        and item["url"].startswith("https://")
        and item["size_bytes"] > 0
        and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        for item in manifest["governing_terms"]
    )
    assert "ACCEPT_" not in serialized
    assert "credential" not in serialized.lower()
    assert all(
        set(item)
        == {
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
    assert sum("size_bytes" not in item for item in manifest["runtime_artifacts"]) == 135
    assert sum(
        not str(item.get("license_expression") or "").strip()
        for item in manifest["runtime_artifacts"]
    ) == 65
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


def test_build_script_requires_exact_sha_tag_and_buildx_attestations() -> None:
    text = (IMAGE_ROOT / "build.sh").read_text(encoding="utf-8")

    assert "ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-$source_sha" in text
    assert "--provenance=mode=max" in text
    assert "--sbom=true" in text
    assert "--metadata-file" in text
    assert '--build-arg "SOURCE_DATE_EPOCH=$source_epoch"' in text
    assert "docker push" not in text
    assert "docker history" not in text


def test_publication_workflow_uses_dedicated_scanner_and_published_base_provenance() -> (
    None
):
    text = PUBLICATION_WORKFLOW.read_text(encoding="utf-8")

    assert "matrix.tool == 'libero'" in text
    assert "libero-base-provenance.intoto.json" in text
    assert "libero-base-sbom.intoto.json" in text
    assert "0d66ce85e6ecad0d044a1d4bae712afe24ff2eb0a5d89eb224944df3895f226b" in text
    assert "b290dbd3087fc5d2cf4af106f1d253c417a26080b70bdcf440ff96314a12c2bb" in text
    assert text.count("npa/scripts/scan_image_libero_payload.py") == 2
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
    assert 'history = subprocess.check_output(' in text
    assert 'if tool != "libero"' not in text
    assert "--provenance=mode=max" in text
    assert "--sbom=true" in text
    assert '[[ "$TOOL" == curobo || "$TOOL" == libero ]]' in text
    assert "libero_accepted_image_manifest" in text
    assert "checked-in acceptance development SHA does not match" in text
    assert "only the checked-in acceptance manifest may differ" in text
    assert 'git merge-base --is-ancestor "$DEVELOPMENT_SHA" HEAD' in text
    assert 'build_args=(--build-arg "NPA_SOURCE_SHA=$DEVELOPMENT_SHA")' in text
    assert '"npa/src/npa/deploy/libero_image_manifest.json"' in text
    assert "libero_publication_lineage_values" in text
    assert 'crane tag "$exact" "dev-$DEVELOPMENT_SHA"' in text
    assert 'test "$digest" = "$LIBERO_ACCEPTED_OCI_DIGEST"' in text
    assert "Revalidate accepted LIBERO repository and OCI lineage" in text
    assert "LIBERO_ACCEPTED_ATTESTATION_LAYERS" in text
    assert "libero-final-attestation-manifest.json" in text
    assert "LIBERO_ACCEPTED_PUBLICATION_BUNDLE_SHA256" in text
    assert "LIBERO_ACCEPTED_INFRASTRUCTURE_BUNDLE_SHA256" in text
    assert "inputs.libero_private_image_inventory_sha256" not in text
    assert "inputs.libero_private_config_digest" not in text
    assert (
        "LIBERO_PRIVATE_IMAGE_INVENTORY_SHA256: "
        "${{ matrix.libero_private_image_inventory_sha256 }}"
    ) in text
    assert (
        "LIBERO_PRIVATE_CONFIG_DIGEST: "
        "${{ matrix.libero_private_config_digest }}"
    ) in text
    assert 'version_count="$(jq \'length\' "$versions")"' in text
    assert "Private destination contains image or referrer versions" in text
    assert "Private destination is genuinely empty" in text
    assert "exactly the accepted untagged OCI graph" in text
    assert "tagged_count=" in text
    assert "public-image-${{ inputs.target" in text
    assert (
        "(inputs.release_tag || inputs.development_sha) && "
        "'registry-mutation' || 'registry-mutation'"
    ) in text
    assert re.search(r"^\s*- uses: [^#\n]+@v", text, re.MULTILINE) is None
    assert re.search(
        r"^\s*uses:\s*[^#\n]+@v[0-9]+\s*$", text, re.MULTILINE
    ) is None
    assert 'visibility="$(gh api "$package_api" --jq .visibility)"' in text
    final_inventory = text.index("libero-final-package-versions.json")
    visibility_change = text.index(
        'gh api --method PATCH "$package_api" -f visibility=public',
        final_inventory,
    )
    assert final_inventory < visibility_change
    assert '${TOOL}-public-package-versions.json' in text
    assert "NPA_FIRST_PUBLICATION_REQUIRED=1" in text
    assert "reject every unexpected tag" in text
    assert "Failed LIBERO cleanup cannot isolate every accepted" in text
    for host_gate in (
        "npa/tests/workflows/test_byof_container_verify.py",
        "npa/tests/workflows/test_byof_libero.py",
        "npa/tests/workflows/test_byof_repo.py",
        "npa/tests/workflows/test_byof_solution_smokes.py",
        "npa/tests/workflows/test_byof_source_auth.py",
    ):
        assert host_gate in text
    assert 'test "$LIBERO_PACKAGE_WRITER_REPOSITORY" = "$GITHUB_REPOSITORY"' in text
    assert "Retained the private accepted LIBERO versions" in text
    assert "Deleted only the exact accepted failed public LIBERO OCI versions" in text
    assert 'gh api --method DELETE "${package_api}/versions/${version_id}"' in text
    assert "NPA_LIBERO_MANAGER_ACCEPTANCE_PUBLIC_KEY_B64" in text
    assert (
        "--secret id=npa_libero_manager_acceptance_public_key_b64,"
        "env=NPA_LIBERO_MANAGER_ACCEPTANCE_PUBLIC_KEY_B64"
    ) in text
    assert "Deleted the complete exact requested public LIBERO OCI graph" in text
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


def test_libero_requested_cleanup_executes_complete_exact_graph_or_refuses(
    tmp_path: Path,
) -> None:
    spec = yaml.safe_load(PUBLICATION_WORKFLOW.read_text(encoding="utf-8"))
    job = spec["jobs"]["cleanup-requested"]
    step = next(item for item in job["steps"] if item.get("name", "").startswith("Delete"))
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
        "args=sys.argv[1:]\n"
        "if '--paginate' in args:\n"
        " print(json.dumps([json.loads(os.environ['FIXTURE_VERSIONS'])])); raise SystemExit(0)\n"
        "if '--method' in args and 'DELETE' in args:\n"
        " open(os.environ['DELETE_RECORD'],'a').write(args[-1]+'\\n'); raise SystemExit(0)\n"
        "if '-i' in args:\n"
        " print('HTTP/2.0 404 Not Found'); raise SystemExit(1)\n"
        "query=args[args.index('--jq')+1] if '--jq' in args else ''\n"
        "print('public' if query == '.visibility' else os.environ['GITHUB_REPOSITORY'])\n",
        encoding="utf-8",
    )
    gh.chmod(0o700)
    crane = bin_dir / "crane"
    crane.write_text(
        f"#!{sys.executable}\nimport os\nprint(os.environ['LIBERO_ACCEPTED_OCI_DIGEST'])\n",
        encoding="utf-8",
    )
    crane.chmod(0o700)
    delete_record = tmp_path / "deletes"
    common = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "IMAGE": f"ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-{sha}",
        "TOOL": "libero",
        "LIBERO_ACCEPTED_OCI_DIGEST": root,
        "LIBERO_ACCEPTED_PACKAGE_VERSION_DIGESTS": json.dumps(expected),
        "LIBERO_PACKAGE_WRITER_REPOSITORY": "nebius/nebius-physical-ai",
        "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "DELETE_RECORD": str(delete_record),
    }
    exact_versions = [
        {"name": root, "metadata": {"container": {"tags": [f"dev-{sha}"]}}},
        {"name": platform, "metadata": {"container": {"tags": []}}},
        {"name": attestation, "metadata": {"container": {"tags": []}}},
    ]
    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={**common, "FIXTURE_VERSIONS": json.dumps(exact_versions)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert delete_record.read_text(encoding="utf-8").splitlines() == [
        "/orgs/nebius/packages/container/nebius-physical-ai%2Fnpa-libero"
    ]

    delete_record.unlink()
    unrelated = [
        *exact_versions,
        {"name": "sha256:" + "d" * 64, "metadata": {"container": {"tags": []}}},
    ]
    refused = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={**common, "FIXTURE_VERSIONS": json.dumps(unrelated)},
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
            "IMAGE": "ghcr.io/nebius/nebius-physical-ai/npa-libero:dev-"
            + "1" * 40,
            "TOOL": "libero",
            "LIBERO_ACCEPTED_OCI_DIGEST": "sha256:" + "a" * 64,
            "LIBERO_ACCEPTED_PACKAGE_VERSION_DIGESTS": "[]",
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


def test_failed_libero_cleanup_removes_only_accepted_versions_under_graph_drift(
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
                {"id": 99, "name": unrelated, "metadata": {"container": {"tags": ["stable-other"]}}},
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

    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "IMAGE": "ghcr.io/nebius/nebius-physical-ai/npa-libero:" + tag,
            "TOOL": "libero",
            "LIBERO_ACCEPTED_OCI_DIGEST": root,
            "LIBERO_ACCEPTED_PACKAGE_VERSION_DIGESTS": json.dumps(
                sorted([root, platform, attestation])
            ),
            "LIBERO_PACKAGE_WRITER_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "VERSION_STATE": str(state_path),
            "DELETE_RECORD": str(delete_record),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert set(delete_record.read_text(encoding="utf-8").splitlines()) == {
        "1",
        "2",
        "3",
    }
    assert json.loads(state_path.read_text(encoding="utf-8")) == [
        {
            "id": 99,
            "name": unrelated,
            "metadata": {"container": {"tags": ["stable-other"]}},
        }
    ]
    assert "Deleted only the exact accepted" in (tmp_path / "summary").read_text(
        encoding="utf-8"
    )
