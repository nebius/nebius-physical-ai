"""Bind the three GPU-role repairs to the actual observed fixed package closures."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from packaging.version import Version


WORKBENCH = Path(__file__).resolve().parents[2] / "docker/workbench"


def test_transfer_security_override_uses_the_verified_fixed_pyjwt_wheel():
    text = (WORKBENCH / "cosmos2-transfer/security-overrides.txt").read_text()
    jwt = next(line for line in text.splitlines() if "/pyjwt-" in line.lower())
    match = re.search(
        r"/pyjwt-(\d+\.\d+\.\d+)-py3-none-any\.whl#sha256=([0-9a-f]{64})$", jwt
    )
    assert match and Version(match[1]) >= Version("2.14.0")
    assert match[1] == "2.14.0"
    # The exact wheel hash is filled from a retained full-byte public download,
    # never derived by replacing the version in the old wheel's URL/hash.
    assert (
        match[2] == "ad0cef71c756a56e74863c2919cf0985f72decbcfcb550ee2f422e7c62b5eedc"
    )


def test_isaac_oss_closure_does_not_retain_vulnerable_pyjwt():
    text = (WORKBENCH / "common/isaac3-oss-deps.txt").read_text()
    match = re.search(r"^pyjwt==(\S+)$", text, re.MULTILINE)
    assert match and Version(match[1]) == Version("2.14.0")


def test_envgen_uses_explicit_fixed_headers_without_changing_other_callers():
    text = (WORKBENCH / "sim2real-envgen/Dockerfile").read_text()
    assert "ARG UBUNTU_SNAPSHOT=20261002T000000Z" in text
    assert "ARG LINUX_LIBC_DEV_VERSION=5.15.0-198.208" in text
    assert (
        'install-workflow-runtime-prereqs "${UBUNTU_SNAPSHOT}" "${LINUX_LIBC_DEV_VERSION}"'
        in text
    )
    assert "dpkg-query --show --showformat='${Version}' linux-libc-dev" in text
    assert '"${LINUX_LIBC_DEV_VERSION}"' in text
    installer = (WORKBENCH / "common/install_workflow_runtime_prereqs.sh").read_text()
    assert 'linux_libc_dev_override="${2:-}"' in installer
    assert 'linux_libc_dev_version="5.15.0-190.200"' in installer
    assert 'linux_libc_dev_version="6.8.0-139.139"' in installer


@pytest.mark.parametrize(
    ("requested", "no_downgrade"),
    [("5.15.0-190.200", False), ("5.15.0-194.204", False), ("5.15.0-198.208", True)],
)
def test_envgen_header_pin_matches_the_frozen_apt_selection(requested, no_downgrade):
    record = json.loads(
        (
            WORKBENCH.parents[2]
            / "docs/workbench/validation/sim2real-envgen-headers-20261002.json"
        ).read_text()
    )
    selected = record["selected_version"]
    text = (WORKBENCH / "sim2real-envgen/Dockerfile").read_text()
    assert f"ARG LINUX_LIBC_DEV_VERSION={selected}" in text
    assert "--allow-downgrades" not in text
    assert selected == record["package"]["Version"]
    assert record["fixed_floor"] == "5.15.0-194.204"
    # Use Debian's actual version ordering, not PEP 440 or a lexical comparison.
    if shutil.which("dpkg") is None:
        pytest.skip(
            "Real Debian version comparison requires dpkg; image build remains mandatory"
        )
    result = subprocess.run(
        ["dpkg", "--compare-versions", requested, "ge", selected], check=False
    )
    assert (result.returncode == 0) is no_downgrade


def test_header_comparison_without_dpkg_skips_after_static_checks(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(pytest.skip.Exception, match="requires dpkg"):
        test_envgen_header_pin_matches_the_frozen_apt_selection("5.15.0-198.208", True)
