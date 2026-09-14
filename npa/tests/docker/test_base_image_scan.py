"""Keep CI's scanned base aligned with the image's OS security preparation."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/image-security-scan.yml"


def _scan_job():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["base-image-cve-scan"]


def _prepare_step():
    return next(
        step for step in _scan_job()["steps"] if step.get("id") == "scan-target"
    )


def _run_preparation(tmp_path, entry, *, build_status=0):
    executable = tmp_path / "docker"
    executable.write_text(
        f"#!{sys.executable}\nimport json, pathlib, sys\n"
        "pathlib.Path('build.json').write_text(json.dumps(sys.argv[1:]))\n"
        "pathlib.Path('Dockerfile').write_text(sys.stdin.read())\n"
        f"raise SystemExit({build_status})\n"
    )
    executable.chmod(0o700)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "BASE_IMAGE": entry["image"],
        "SCAN_NAME": entry["name"],
        "PURGE_LINUX_LIBC_DEV": str(entry["purge_linux_libc_dev"]).lower(),
        "UPGRADE_OS": str(entry.get("upgrade_os", False)).lower(),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }
    return subprocess.run(
        ["bash", "-c", _prepare_step()["run"]],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    "entry",
    _scan_job()["strategy"]["matrix"]["include"],
    ids=lambda entry: entry["name"],
)
def test_scan_target_applies_only_the_declared_preparation(tmp_path, entry):
    result = _run_preparation(tmp_path, entry)
    assert result.returncode == 0, result.stderr
    purge = str(entry["purge_linux_libc_dev"]).lower()
    upgrade = str(entry.get("upgrade_os", False)).lower()
    if purge == "false" and upgrade == "false":
        assert (tmp_path / "outputs").read_text() == f"image={entry['image']}\n"
        assert not (tmp_path / "build.json").exists()
        return
    arguments = json.loads((tmp_path / "build.json").read_text())
    assert arguments == [
        "build",
        "--pull",
        "--no-cache",
        "--build-arg",
        f"BASE_IMAGE={entry['image']}",
        "--build-arg",
        f"PURGE_LINUX_LIBC_DEV={purge}",
        "--build-arg",
        f"UPGRADE_OS={upgrade}",
        "-t",
        f"npa-base-scan:{entry['name']}",
        "-f",
        "-",
        ".",
    ]
    assert (
        tmp_path / "outputs"
    ).read_text() == f"image=npa-base-scan:{entry['name']}\n"
    assert "FROM ${BASE_IMAGE}" in (tmp_path / "Dockerfile").read_text()


def test_failed_preparation_never_emits_a_scan_target(tmp_path):
    entry = next(
        entry
        for entry in _scan_job()["strategy"]["matrix"]["include"]
        if entry.get("upgrade_os") is True
    )
    result = _run_preparation(tmp_path, entry, build_status=17)
    assert result.returncode == 17
    assert not (tmp_path / "outputs").exists()


def _record_package_commands(tmp_path, failed_command):
    for name in ("apt-get", "dpkg", "rm"):
        executable = tmp_path / name
        executable.write_text(
            f"#!{sys.executable}\nimport pathlib, sys\n"
            "with pathlib.Path('commands').open('a') as stream:\n"
            "    stream.write(pathlib.Path(sys.argv[0]).name + ' ' + ' '.join(sys.argv[1:]) + '\\n')\n"
            f"raise SystemExit(23 if {failed_command!r} in [arg.lstrip('-') for arg in sys.argv[1:]] else 0)\n"
        )
        executable.chmod(0o700)


@pytest.mark.parametrize("failed_command", ["update", "upgrade", "purge"])
def test_os_preparation_stops_after_a_package_manager_failure(tmp_path, failed_command):
    entry = {
        "image": "synthetic-base",
        "name": "fixture",
        "upgrade_os": True,
        "purge_linux_libc_dev": True,
    }
    assert _run_preparation(tmp_path, entry).returncode == 0
    dockerfile = (tmp_path / "Dockerfile").read_text()
    command = dockerfile.split("RUN ", 1)[1].replace("\\\n", "")
    _record_package_commands(tmp_path, failed_command)
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": str(tmp_path),
            "UPGRADE_OS": "true",
            "PURGE_LINUX_LIBC_DEV": "true",
        },
        check=False,
    )
    assert result.returncode == 23, result.stderr
    commands = (tmp_path / "commands").read_text().splitlines()
    expected = [
        "apt-get update",
        "apt-get upgrade -y --no-install-recommends",
        "dpkg --purge --force-depends linux-libc-dev",
    ]
    assert (
        commands == expected[: ["update", "upgrade", "purge"].index(failed_command) + 1]
    )


def test_python_scan_matches_fiftyones_pinned_and_upgraded_base():
    dockerfile = (ROOT / "npa/docker/workbench/fiftyone/Dockerfile").read_text()
    base = next(
        line.removeprefix("FROM ")
        for line in dockerfile.splitlines()
        if line.startswith("FROM ")
    )
    entry = next(
        entry
        for entry in _scan_job()["strategy"]["matrix"]["include"]
        if entry["name"] == "python-3-11-slim-trixie"
    )
    assert entry["image"] == base and "@sha256:" in base
    assert entry["upgrade_os"] is True
    assert entry["purge_linux_libc_dev"] is False
    recipe = _prepare_step()["run"]
    for command in (
        "apt-get update",
        "apt-get upgrade -y --no-install-recommends",
        "rm -rf /var/lib/apt/lists/*",
    ):
        assert command in dockerfile and command in recipe
    assert recipe.index("apt-get update") < recipe.index("apt-get upgrade")
    assert _prepare_step()["env"] == {
        "BASE_IMAGE": "${{ matrix.image }}",
        "SCAN_NAME": "${{ matrix.name }}",
        "PURGE_LINUX_LIBC_DEV": "${{ matrix.purge_linux_libc_dev }}",
        "UPGRADE_OS": "${{ matrix.upgrade_os || false }}",
    }


def test_base_cve_gate_remains_blocking_and_scans_the_prepared_target():
    job = _scan_job()
    assert "if" not in job and "continue-on-error" not in job
    gate = next(step for step in job["steps"] if step.get("name") == "Trivy image scan")
    assert "if" not in gate and "continue-on-error" not in gate
    assert gate["with"]["exit-code"] == "1"
    assert gate["with"]["severity"] == "CRITICAL"
    assert gate["with"]["vuln-type"] == "os"
    scans = [step for step in job["steps"] if "trivy-action@" in step.get("uses", "")]
    assert len(scans) == 2
    assert all(
        step["with"]["image-ref"] == "${{ steps.scan-target.outputs.image }}"
        for step in scans
    )
    # PR invocation coverage lives in test_image_security_gate.
