"""Reject CPU-role recipes that retain the observed vulnerable Debian packages."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = "20261002T000000Z"
FLOORS = {
    "perl-base": ("MIN_PERL_BASE_VERSION", "5.40.1-6+deb13u1", "5.40.1-6"),
    "libglib2.0-0t64": ("MIN_LIBGLIB_VERSION", "2.84.4-3~deb13u4", "2.84.4-3~deb13u3"),
    "libmbedcrypto16": (
        "MIN_MBEDCRYPTO_VERSION",
        "3.6.6-0.1~deb13u1",
        "3.6.5-0.1~deb13u1",
    ),
}
ROLE_PACKAGES = {
    "sim2real-control": ("perl-base",),
    "rerun-viewer": tuple(FLOORS),
}


def _recipe(role):
    return (ROOT / "npa/docker/workbench" / role / "Dockerfile").read_text()


def _floor_check(text):
    match = re.search(r"&& for pin in (.*?)\bdone\s*\\", text, re.DOTALL)
    assert match, "Installed package versions must be checked before later build steps"
    return "for pin in " + match.group(1).replace("\\\n", "") + "done"


@pytest.mark.parametrize("role", ROLE_PACKAGES)
def test_cpu_security_snapshot_and_upgrade_are_not_optional(role):
    text = _recipe(role)
    assert f"ARG DEBIAN_SNAPSHOT={SNAPSHOT}" in text
    assert "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" in text
    assert "snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}" in text
    update = text.index("&& apt-get update")
    upgrade = text.index("&& apt-get upgrade -y --no-install-recommends")
    install = text.index("&& apt-get install -y --no-install-recommends")
    check = text.index("&& for pin in")
    assert update < upgrade < install < check < text.index("COPY ")


@pytest.mark.parametrize("role", ROLE_PACKAGES)
def test_observed_fixed_versions_are_enforced_without_scan_exceptions(role):
    text = _recipe(role)
    check = _floor_check(text)
    for package in ROLE_PACKAGES[role]:
        argument, fixed, _ = FLOORS[package]
        assert f"ARG {argument}={fixed}" in text
        assert f'"{package} ${{{argument}}}"' in check
    assert "dpkg-query --show --showformat='${Version}'" in check
    assert 'dpkg --compare-versions "${installed}" ge "$2"' in check
    assert ".trivyignore" not in text
    assert "--ignore-unfixed" not in text
    assert "USER ubuntu" in text
    assert "rm -f /etc/ssh/ssh_host_*" in text


def test_package_floors_match_the_hash_bound_snapshot_readback():
    receipt = (
        ROOT / "docs/workbench/validation/sim2real-cpu-security-snapshot-20261002.json"
    )
    value = json.loads(receipt.read_text())
    assert value["timestamp"] == SNAPSHOT
    assert value["signed_index_authority"] == "open3d-security-snapshot-20261002.json"
    for package, (_, fixed, _) in FLOORS.items():
        assert value["packages"][package]["fixed_floor"] == fixed
        assert re.fullmatch("[0-9a-f]{64}", value["packages"][package]["sha256"])


def _write_query_fixture(directory):
    query = """#!/bin/sh
  if [ "$3" = "$CHECK_PACKAGE" ]; then
    printf '%s' "$CHECK_VERSION"
    exit "$QUERY_STATUS"
  fi
  case "$3" in
    perl-base) printf '%s' "$MIN_PERL_BASE_VERSION" ;;
    libglib2.0-0t64) printf '%s' "$MIN_LIBGLIB_VERSION" ;;
    libmbedcrypto16) printf '%s' "$MIN_MBEDCRYPTO_VERSION" ;;
    *) exit 99 ;;
  esac
"""
    executable = directory / "dpkg-query"
    executable.write_text(query)
    executable.chmod(0o700)


def _run_floor_check(role, package, installed, directory, query_status=0):
    dpkg = shutil.which("dpkg")
    if dpkg is None:
        pytest.skip(
            "Real Debian version comparison requires dpkg; image build remains mandatory"
        )
    environment = {
        "PATH": str(directory) + ":" + str(Path(dpkg).parent),
        **{argument: fixed for argument, fixed, _ in FLOORS.values()},
        "CHECK_PACKAGE": package,
        "CHECK_VERSION": installed,
        "QUERY_STATUS": str(query_status),
    }
    _write_query_fixture(directory)
    command = _floor_check(_recipe(role)) + " && printf 'subsequent-step'"
    return subprocess.run(
        ["/bin/sh", "-c", command],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    "role,package",
    [
        (role, package)
        for role, packages in ROLE_PACKAGES.items()
        for package in packages
    ],
)
@pytest.mark.parametrize(
    "case", ("vulnerable", "fixed", "newer", "query-failure", "empty")
)
def test_actual_build_floor_rejects_vulnerable_or_unreadable_packages(
    role, package, case, tmp_path
):
    _, fixed, vulnerable = FLOORS[package]
    installed = {
        "vulnerable": vulnerable,
        "fixed": fixed,
        "newer": fixed + "+1",
        "query-failure": fixed,
        "empty": "",
    }[case]
    result = _run_floor_check(
        role, package, installed, tmp_path, 1 if case == "query-failure" else 0
    )
    accepted = case in {"fixed", "newer"}
    assert (result.returncode == 0) is accepted, result.stderr
    assert ("subsequent-step" in result.stdout) is accepted
