"""Packaging contracts and hermetic archive-verifier behavior for RoboTwin."""

from __future__ import annotations

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
IMAGE_ROOT = ROOT / "npa/docker/workbench/robotwin"
ARCHIVE_DOWNLOAD = (
    "apt-get -o Acquire::https::CaInfo=/run/npa-bootstrap-ca.crt download ${packages};"
)
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


def _archive_verification_shell() -> str:
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    _assert_archive_install_contract(dockerfile)
    start = dockerfile.index('packages="$(awk')
    end = dockerfile.index(ARCHIVE_INSTALL) + len(ARCHIVE_INSTALL)
    shell = dockerfile[start:end].replace("\\\n", "")
    shell = shell.replace("/opt/npa/robotwin/apt-packages.lock", '"${FIXTURE_LOCK}"')
    # Redirect only locations; execute the committed checks and install boundary.
    for name in ("archive_dir", "expected_archives", "verified_archives"):
        shell, count = re.subn(
            rf"\b{name}=[^;]+;", f'{name}="${{{name.upper()}}}";', shell
        )
        assert count == 1
    return "set -eux;\n" + shell


def _inert_archive_set(source: Path) -> list[list[str]]:
    source.mkdir()
    rows = []
    for index in range(75):
        name = f"fixture-package-{index:02d}"
        content = f"Package: {name}\nVersion: 1.0\nArchitecture: amd64\n".encode()
        (source / f"{name}.deb").write_bytes(content)
        rows.append(
            [
                "binary",
                name,
                "1.0",
                "amd64",
                hashlib.sha256(content).hexdigest(),
                str(len(content)),
            ]
        )
    return rows


def _corrupt_archive_fixture(source: Path, rows: list[list[str]], case: str) -> None:
    first = source / f"{rows[0][1]}.deb"
    second = source / f"{rows[1][1]}.deb"
    if case == "hash":
        rows[0][4] = "0" * 64
    elif case == "size":
        rows[0][5] = str(int(rows[0][5]) + 1)
    elif case in {"package", "version", "architecture"}:
        original, replacement = {
            "package": (rows[0][1], "unknown-package"),
            "version": ("1.0", "2.0"),
            "architecture": ("amd64", "arm64"),
        }[case]
        data = first.read_text().replace(original, replacement).encode()
        first.write_bytes(data)
        rows[0][4:6] = [hashlib.sha256(data).hexdigest(), str(len(data))]
    elif case == "duplicate-lock":
        rows[1] = rows[0].copy()
    elif case == "duplicate-archive":
        second.write_bytes(first.read_bytes())
    elif case == "missing":
        second.unlink()
    elif case == "extra":
        (source / "extra.deb").write_bytes(first.read_bytes())
    elif case == "unexpected-file":
        (source / "unexpected.txt").write_text("inert fixture")
    else:
        raise AssertionError(case)


def _archive_command_stubs(directory: Path) -> None:
    directory.mkdir()
    scripts = {
        "apt-get": (
            '#!/bin/sh\nset -eu\n'
            'if [ "$1" = -o ]; then\n'
            'test "$2" = Acquire::https::CaInfo=/run/npa-bootstrap-ca.crt\n'
            'shift 2\nfi\ncase "$1" in\n'
            'download) printf download > "$DOWNLOAD_MARKER"; cp "$FIXTURE_SOURCE/"* . ;;\n'
            'install) printf "%s\\n" "$@" > "$INSTALL_MARKER" ;;\n'
            "*) exit 99 ;;\nesac\n"
        ),
        "dpkg-deb": (
            '#!/bin/sh\nset -eu\ntest "$1" = -f\n'
            "awk -F ': ' -v field=\"$3\" '$1 == field {print $2}' \"$2\"\n"
        ),
    }
    for name, script in scripts.items():
        target = directory / name
        target.write_text(script)
        target.chmod(0o700)


def _run_archive_verifier(
    tmp_path: Path, case: str
) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "source"
    rows = _inert_archive_set(source)
    if case != "valid":
        _corrupt_archive_fixture(source, rows, case)
    lock = tmp_path / "lock.tsv"
    lock.write_text("".join("\t".join(row) + "\n" for row in rows))
    commands = tmp_path / "commands"
    _archive_command_stubs(commands)
    environment = {
        "PATH": f"{commands}:/usr/bin:/bin",
        "LC_ALL": "C",
        "FIXTURE_SOURCE": str(source),
        "FIXTURE_LOCK": str(lock),
        "ARCHIVE_DIR": str(tmp_path / "archives"),
        "EXPECTED_ARCHIVES": str(tmp_path / "expected.tsv"),
        "VERIFIED_ARCHIVES": str(tmp_path / "verified.tsv"),
        "DOWNLOAD_MARKER": str(tmp_path / "downloaded"),
        "INSTALL_MARKER": str(tmp_path / "installed"),
    }
    return subprocess.run(
        ["/bin/sh", "-c", _archive_verification_shell()],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_real_archive_verifier_accepts_complete_locked_fixture_before_install(
    tmp_path: Path,
) -> None:
    result = _run_archive_verifier(tmp_path, "valid")

    assert result.returncode == 0, result.stderr
    install_arguments = (tmp_path / "installed").read_text().splitlines()
    assert "--no-download" in install_arguments
    assert (
        len([argument for argument in install_arguments if argument.endswith(".deb")])
        == 75
    )
    assert (tmp_path / "verified.tsv").read_bytes() == (
        tmp_path / "expected.tsv"
    ).read_bytes()


@pytest.mark.parametrize(
    "case",
    [
        "hash",
        "size",
        "package",
        "version",
        "architecture",
        "duplicate-lock",
        "duplicate-archive",
        "missing",
        "extra",
        "unexpected-file",
    ],
)
def test_real_archive_verifier_rejects_bad_inputs_before_install(
    tmp_path: Path, case: str
) -> None:
    result = _run_archive_verifier(tmp_path, case)

    assert result.returncode != 0
    assert not (tmp_path / "installed").exists()
    assert (tmp_path / "downloaded").exists() is (case != "duplicate-lock")
    if case in {"duplicate-lock", "missing", "extra"}:
        count = 76 if case == "extra" else 74
        assert result.stderr.rstrip().endswith(f"test {count} = 75")
    elif case == "unexpected-file":
        assert result.stderr.rstrip().endswith("/archives/unexpected.txt")
    elif case == "duplicate-archive":
        # Every archive passes its own hash/size/identity check; set equality must fail.
        assert len((tmp_path / "verified.tsv").read_text().splitlines()) == 75
        assert "cmp " in result.stderr
    elif case in {"hash", "size", "package", "version", "architecture"}:
        expected_boundary = {
            "hash": "sha256sum ",
            "size": "stat -c ",
            "package": "locked_identity_count=0",
            "version": "locked_identity_count=0",
            "architecture": "locked_identity_count=0",
        }[case]
        assert expected_boundary in result.stderr
        assert not (tmp_path / "verified.tsv").read_text()


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
