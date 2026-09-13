"""Verify that FiftyOne's shared OS update step rejects missing security fixes."""

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
UPGRADE = ROOT / "npa/docker/workbench/fiftyone/upgrade_os_packages.sh"


@pytest.fixture
def package_commands(tmp_path):
    """Replace package-manager operations with recorded, deterministic results.

    Args:
        tmp_path: Isolated command and log directory.
    Returns:
        Environment for running the real update script without modifying the host.
    Raises:
        OSError: A fixture command cannot be created.
    """
    commands = {
        "apt-get": 'test "$1" != "${FAILED_APT_STEP:-}"',
        "dpkg-query": 'printf "%s" "${PERL_VERSION:-5.40.1-6+deb13u1}"; exit "${QUERY_STATUS:-0}"',
        "dpkg": 'exit "${VERSION_STATUS:-0}"',
    }
    for name, result in commands.items():
        executable = tmp_path / name
        executable.write_text(
            '#!/bin/sh\n'
            f'printf "%s\\n" "{name} $*" >> "$COMMAND_LOG"\n{result}\n'
        )
        executable.chmod(0o700)
    return {**os.environ, "PATH": str(tmp_path), "COMMAND_LOG": str(tmp_path / "commands")}


def test_updates_are_applied_before_the_fixed_version_is_checked(package_commands):
    """Require refreshed indexes, upgrades and an installed-package version check.

    Args:
        package_commands: Fake package-manager commands and log environment.
    Returns:
        None.
    Raises:
        AssertionError: Updates are skipped or the fixed-version floor changes.
    """
    result = subprocess.run(["/bin/sh", str(UPGRADE)], env=package_commands, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert Path(package_commands["COMMAND_LOG"]).read_text().splitlines() == [
        "apt-get update",
        "apt-get upgrade -y --no-install-recommends",
        "dpkg-query -W -f=${Version} perl-base",
        "dpkg --compare-versions 5.40.1-6+deb13u1 ge 5.40.1-6+deb13u1",
    ]


@pytest.mark.parametrize("failed_step", ["update", "upgrade"])
def test_failed_package_updates_stop_the_build(package_commands, failed_step):
    """Reject unavailable indexes and failed upgrades without reporting success.

    Args:
        package_commands: Fake package-manager commands and log environment.
        failed_step: Package-manager operation to fail.
    Returns:
        None.
    Raises:
        AssertionError: A failed package update reaches the success check.
    """
    package_commands["FAILED_APT_STEP"] = failed_step
    result = subprocess.run(["/bin/sh", str(UPGRADE)], env=package_commands, capture_output=True)
    assert result.returncode != 0
    assert "dpkg" not in Path(package_commands["COMMAND_LOG"]).read_text()


@pytest.mark.parametrize("query_status,version_status", [(1, 0), (0, 1), (0, 2)])
def test_missing_outdated_or_unreadable_package_fails(package_commands, query_status, version_status):
    """Reject missing packages, an unpatched version and comparison errors.

    Args:
        package_commands: Fake package-manager commands and log environment.
        query_status: Installed-package query exit status.
        version_status: Fixed-version comparison exit status.
    Returns:
        None.
    Raises:
        AssertionError: Missing security-fix evidence is accepted.
    """
    package_commands.update(QUERY_STATUS=str(query_status), VERSION_STATUS=str(version_status))
    result = subprocess.run(["/bin/sh", str(UPGRADE)], env=package_commands, capture_output=True)
    assert result.returncode != 0


def test_newer_patched_package_is_accepted(package_commands):
    """Allow a newer vendor security revision above the minimum fixed version.

    Args:
        package_commands: Fake package-manager commands and log environment.
    Returns:
        None.
    Raises:
        AssertionError: The installed version is replaced by a hardcoded value.
    """
    package_commands["PERL_VERSION"] = "5.40.1-6+deb13u2"
    result = subprocess.run(["/bin/sh", str(UPGRADE)], env=package_commands, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "5.40.1-6+deb13u2 ge 5.40.1-6+deb13u1" in Path(package_commands["COMMAND_LOG"]).read_text()
