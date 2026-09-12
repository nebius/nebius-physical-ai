"""Exercise daily CI network failures and credential handling without live access."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/dev-vm-daily-tests.yml"
FAKE_COMMAND = r'''
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

name = Path(sys.argv[0]).name
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(json.dumps({"name": name, "args": sys.argv[1:]}) + "\n")
if name == "ssh":
    command = sys.argv[-1]
    if "SSH_DIRECTORY_REPLY" in os.environ:
        print(os.environ["SSH_DIRECTORY_REPLY"])
    elif "mktemp -d /tmp/npa-daily." in command:
        result = subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)
        with open(os.environ["STAGING_ALLOCATIONS"], "a") as allocations:
            allocations.write(result.stdout)
        print(result.stdout, end="")
    elif command.startswith("rm -f -- "):
        subprocess.run(["bash", "-c", command], check=True)
    else:
        sys.exit(int(os.environ.get("REMOTE_TEST_EXIT", "0")))
elif name == "scp":
    shutil.copyfile(sys.argv[-2], sys.argv[-1].split(":", 1)[1])
    sys.exit(int(os.environ.get("COPY_FAILURE", "0")))
'''


def _workflow():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    # YAML 1.1 treats GitHub's `on` key as a boolean.
    if True in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _step(name):
    steps = _workflow()["jobs"]["dev-vm-tests"]["steps"]
    return next(step for step in steps if step["name"] == name)


def _environment(directory):
    binaries = directory / "bin"
    binaries.mkdir()
    fake = binaries / "fake-command"
    fake.write_text(f"#!{sys.executable}\n" + FAKE_COMMAND)
    fake.chmod(0o700)
    for name in ("ssh", "scp"):
        (binaries / name).symlink_to(fake)
    return {
        "PATH": f"{binaries}:/usr/bin:/bin",
        "RUNNER_TEMP": str(directory),
        "COMMAND_LOG": str(directory / "commands.jsonl"),
    }


def _run(name, environment):
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _step(name)["run"]],
        env=environment,
        cwd=WORKFLOW.parents[2],
        text=True,
        capture_output=True,
        check=False,
    )


def _commands(environment):
    path = Path(environment["COMMAND_LOG"])
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_daily_workflow_has_no_pr_trigger_and_always_cleans_credentials():
    workflow = _workflow()
    assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}
    assert workflow["on"]["schedule"] == [{"cron": "0 7 * * *"}]
    assert workflow["permissions"] == {"contents": "read"}
    assert "permissions" not in workflow["jobs"]["dev-vm-tests"]
    assert "tailscale" not in WORKFLOW.read_text().lower()
    assert _step("Clean up SSH key")["if"] == "always()"
    assert _step("Check out repository")["with"]["persist-credentials"] is False
    assert "ssh-keyscan" not in WORKFLOW.read_text()
    missing = _step("Verify required secrets are present")["run"]
    assert '"$DEV_VM_SSH_KNOWN_HOSTS"' in missing
    steps = [step["name"] for step in workflow["jobs"]["dev-vm-tests"]["steps"]]
    cleanup = "Clean up the remote runner script"
    assert "always()" in _step(cleanup)["if"]
    assert "steps.stage.outputs.directory != ''" in _step(cleanup)["if"]
    assert steps.index(cleanup) < steps.index("Clean up SSH key")


@pytest.mark.parametrize("missing", [
    "DEV_VM_SSH_HOST", "DEV_VM_SSH_USER", "DEV_VM_SSH_PRIVATE_KEY", "DEV_VM_SSH_KNOWN_HOSTS",
])
def test_missing_ssh_settings_fail_without_exposing_other_values(tmp_path, missing):
    environment = _environment(tmp_path)
    configured = {
        "DEV_VM_SSH_HOST": "private-host.invalid",
        "DEV_VM_SSH_USER": "private-ci-user",
        "DEV_VM_SSH_PRIVATE_KEY": "synthetic-private-key-canary",
        "DEV_VM_SSH_KNOWN_HOSTS": "synthetic-host-pin-canary",
    }
    environment.update(configured)
    del environment[missing]
    result = _run("Verify required secrets are present", environment)
    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert f"Missing required secret(s): {missing}" in output
    assert all(value not in output for value in configured.values())
    assert not Path(environment["COMMAND_LOG"]).exists()


@pytest.mark.parametrize("vm_reachable", ["0", "1"])
def test_tcp_probe_reports_controls_without_exposing_endpoint(tmp_path, vm_reachable):
    environment = _environment(tmp_path)
    binaries = tmp_path / "bin"
    (binaries / "python3").symlink_to(sys.executable)
    (binaries / "socket.py").write_text(
        "import contextlib, os\n"
        "def create_connection(address, timeout):\n"
        "    if address[0] == os.environ['DEV_VM_SSH_HOST'] "
        "and os.environ['VM_REACHABLE'] == '0':\n"
        "        raise OSError('private endpoint details')\n"
        "    return contextlib.nullcontext()\n"
    )
    environment.update({
        "PYTHONPATH": str(binaries), "DEV_VM_SSH_HOST": "private-endpoint.invalid",
        "DEV_VM_SSH_PORT": "22", "VM_REACHABLE": vm_reachable,
    })
    result = _run("Check direct SSH reachability", environment)
    assert result.returncode == int(vm_reachable == "0")
    output = result.stdout + result.stderr
    assert "private-endpoint.invalid" not in output
    assert "private endpoint details" not in output
    assert ("GitHub HTTPS control: TCP reachable" in output) == (vm_reachable == "0")
    assert ("GitHub SSH control: TCP reachable" in output) == (vm_reachable == "0")
    assert not Path(environment["COMMAND_LOG"]).exists()


@pytest.fixture
def staging_environment(tmp_path):
    environment = _environment(tmp_path)
    output = tmp_path / "outputs"
    allocations = tmp_path / "allocations"
    environment.update({
        "GITHUB_OUTPUT": str(output), "SSH_PORT": "22",
        "STAGING_ALLOCATIONS": str(allocations),
        "DEV_VM_SSH_HOST": "vm.invalid", "DEV_VM_SSH_USER": "runner",
        "TIER": "unit", "REF": "main",
    })
    yield environment
    if allocations.exists():
        for directory in allocations.read_text().splitlines():
            shutil.rmtree(directory, ignore_errors=True)


def _last_staging_directory(environment):
    output = Path(environment["GITHUB_OUTPUT"]).read_text().splitlines()
    return Path(output[-1].split("=", 1)[1])


def test_concurrent_staging_directories_do_not_overwrite_each_other(staging_environment):
    environment = staging_environment
    assert _run("Copy runner script to the dev VM", environment).returncode == 0
    first = _last_staging_directory(environment)
    (first / "runner.sh").write_text("first caller's script")
    assert _run("Copy runner script to the dev VM", environment).returncode == 0
    second = _last_staging_directory(environment)
    assert first != second
    assert first.stat().st_mode & 0o777 == 0o700
    assert second.stat().st_mode & 0o777 == 0o700
    environment["REMOTE_RUN_DIR"] = str(second)
    assert _run("Run test tier on the dev VM", environment).returncode == 0
    command = _commands(environment)[-1]["args"][-1]
    assert str(second / "runner.sh") in command
    assert str(first / "runner.sh") not in command
    assert _run("Clean up the remote runner script", environment).returncode == 0
    assert not second.exists()
    assert (first / "runner.sh").read_text() == "first caller's script"


@pytest.mark.parametrize("failure", ["COPY_FAILURE", "REMOTE_TEST_EXIT"])
def test_staging_cleanup_after_upload_or_test_failure(staging_environment, failure):
    environment = staging_environment
    environment[failure] = "1"
    copied = _run("Copy runner script to the dev VM", environment)
    assert copied.returncode == int(failure == "COPY_FAILURE")
    directory = _last_staging_directory(environment)
    environment["REMOTE_RUN_DIR"] = str(directory)
    if failure == "REMOTE_TEST_EXIT":
        assert _run("Run test tier on the dev VM", environment).returncode == 1
    assert _run("Clean up the remote runner script", environment).returncode == 0
    assert not directory.exists()


@pytest.mark.parametrize("invalid", ["root", "traversal", "command"])
def test_invalid_remote_directory_never_reaches_copy_or_cleanup(staging_environment, tmp_path, invalid):
    environment = staging_environment
    directory = {
        "root": tmp_path.anchor, "traversal": str(tmp_path / ".."), "command": "$(id)",
    }[invalid]
    environment["SSH_DIRECTORY_REPLY"] = directory
    assert _run("Copy runner script to the dev VM", environment).returncode == 1
    assert not Path(environment["GITHUB_OUTPUT"]).exists()
    assert [command["name"] for command in _commands(environment)] == ["ssh"]
    environment["REMOTE_RUN_DIR"] = directory
    assert _run("Clean up the remote runner script", environment).returncode == 1
    assert [command["name"] for command in _commands(environment)] == ["ssh"]
