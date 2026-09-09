"""Exercise daily CI network failures and credential handling without live access."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/dev-vm-daily-tests.yml"
TOKEN_CANARY = "synthetic-oidc-request-token-for-test"
FAKE_COMMAND = r'''
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

name = Path(sys.argv[0]).name
with open(os.environ["COMMAND_LOG"], "a") as log:
    log.write(json.dumps({"name": name, "args": sys.argv[1:]}) + "\n")
if name == "sudo":
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--preserve-env=")]
    os.execvp(args[0], args)
elif name == "sha256sum":
    sys.stdin.read()
    sys.exit(int(os.environ.get("CHECKSUM_FAILURE", "0")))
elif name == "tar":
    directory = Path(sys.argv[sys.argv.index("-C") + 1])
    (directory / "tailscale").symlink_to(sys.argv[0])
    os.chdir(directory)
    with socket.socket(socket.AF_UNIX) as connection:
        connection.bind("tailscaled.sock")
elif name == "tailscale" and "up" in sys.argv:
    assert os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
    print(os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"])
    sys.exit(int(os.environ.get("AUTH_FAILURE", "0")))
elif name == "systemctl" and "stop" in sys.argv:
    sys.exit(int(os.environ.get("STOP_FAILURE", "0")))
elif name == "ssh":
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
    for name in ("curl", "sha256sum", "tar", "sudo", "systemd-run", "systemctl", "ssh", "scp"):
        (binaries / name).symlink_to(fake)
    return {
        "PATH": f"{binaries}:/usr/bin:/bin",
        "RUNNER_TEMP": str(directory),
        "COMMAND_LOG": str(directory / "commands.jsonl"),
        "NPA_DAILY_NETWORK": "tailscale",
        "NPA_DAILY_TAILSCALE_CLIENT_ID": "test-client-id",
        "NPA_DAILY_TAILSCALE_AUDIENCE": "api.tailscale.com/test-client-id",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": TOKEN_CANARY,
        "ACTIONS_ID_TOKEN_REQUEST_URL": "https://oidc.invalid/request",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
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


@pytest.mark.parametrize("network,code", [("direct", 0), ("unknown", 1)])
def test_direct_or_invalid_mode_never_installs_a_network_client(tmp_path, network, code):
    environment = _environment(tmp_path)
    environment["NPA_DAILY_NETWORK"] = network
    result = _run("Connect to the dev VM network", environment)
    assert result.returncode == code
    assert not Path(environment["COMMAND_LOG"]).exists()


@pytest.mark.parametrize("missing", [
    "NPA_DAILY_TAILSCALE_CLIENT_ID", "NPA_DAILY_TAILSCALE_AUDIENCE",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_URL",
])
def test_missing_federation_configuration_fails_before_download(tmp_path, missing):
    environment = _environment(tmp_path)
    del environment[missing]
    result = _run("Connect to the dev VM network", environment)
    assert result.returncode == 1
    assert f"Missing {missing}" in result.stdout
    assert TOKEN_CANARY not in result.stdout + result.stderr
    assert not Path(environment["COMMAND_LOG"]).exists()


def test_checksum_failure_never_executes_downloaded_content(tmp_path):
    environment = _environment(tmp_path)
    environment["CHECKSUM_FAILURE"] = "1"
    result = _run("Connect to the dev VM network", environment)
    assert result.returncode == 1
    assert [command["name"] for command in _commands(environment)] == ["curl", "sha256sum"]
    cleanup = _run("Disconnect from the dev VM network", environment)
    assert cleanup.returncode == 0
    assert not (tmp_path / "npa-daily-tailscale").exists()


@pytest.mark.parametrize("auth_failure", ["0", "1"])
def test_federation_keeps_tokens_private_and_cleans_up(tmp_path, auth_failure):
    environment = _environment(tmp_path)
    environment["AUTH_FAILURE"] = auth_failure
    result = _run("Connect to the dev VM network", environment)
    assert result.returncode == int(auth_failure)
    assert TOKEN_CANARY not in result.stdout + result.stderr
    commands = _commands(environment)
    assert TOKEN_CANARY not in json.dumps(commands)
    invocation = next(command for command in commands if command["name"] == "tailscale")
    assert "--accept-routes" in invocation["args"]
    assert "--advertise-tags=tag:npa-daily" in invocation["args"]
    assert not any(argument.startswith("--id-token") for argument in invocation["args"])
    directory = tmp_path / "npa-daily-tailscale"
    assert directory.stat().st_mode & 0o777 == 0o700
    assert (directory / "auth.log").stat().st_mode & 0o777 == 0o600
    assert TOKEN_CANARY in (directory / "auth.log").read_text()
    cleanup = _run("Disconnect from the dev VM network", environment)
    assert cleanup.returncode == 0
    assert TOKEN_CANARY not in cleanup.stdout + cleanup.stderr
    assert not directory.exists()
    commands = _commands(environment)
    assert any("logout" in command["args"] for command in commands)
    assert any(command["args"] == ["stop", "npa-daily-tailscale"] for command in commands)


def test_daemon_stop_failure_still_removes_private_logs(tmp_path):
    environment = _environment(tmp_path)
    assert _run("Connect to the dev VM network", environment).returncode == 0
    environment["STOP_FAILURE"] = "1"
    result = _run("Disconnect from the dev VM network", environment)
    assert result.returncode == 1
    assert not (tmp_path / "npa-daily-tailscale").exists()
    assert TOKEN_CANARY not in result.stdout + result.stderr


def test_daily_workflow_has_no_pr_trigger_and_always_cleans_credentials():
    workflow = _workflow()
    assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}
    assert workflow["on"]["schedule"] == [{"cron": "0 7 * * *"}]
    assert workflow["jobs"]["dev-vm-tests"]["permissions"] == {
        "contents": "read", "id-token": "write",
    }
    assert _step("Clean up SSH key")["if"] == "always()"
    assert _step("Disconnect from the dev VM network")["if"] == "always()"
    assert _step("Check out repository")["with"]["persist-credentials"] is False
    assert "ssh-keyscan" not in WORKFLOW.read_text()
    missing = _step("Verify required secrets are present")["run"]
    assert '"$DEV_VM_SSH_KNOWN_HOSTS"' in missing
    steps = [step["name"] for step in workflow["jobs"]["dev-vm-tests"]["steps"]]
    cleanup = "Clean up the remote runner script"
    assert "always()" in _step(cleanup)["if"]
    assert "steps.stage.outputs.directory != ''" in _step(cleanup)["if"]
    assert steps.index(cleanup) < steps.index("Clean up SSH key")
    assert steps.index(cleanup) < steps.index("Disconnect from the dev VM network")


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
