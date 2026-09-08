"""Exercise OpenPI command forwarding and require its real final-user build gate."""

from pathlib import Path
import json
import shlex
import subprocess
import sys

import pytest

IMAGE = Path(__file__).resolve().parents[2] / "docker/workbench/openpi"


@pytest.mark.parametrize("exit_code", [0, 7, 78])
def test_entrypoint_preserves_arguments_and_child_exit_status(exit_code):
    arguments = ["two words", "", "--flag", "literal;$value"]
    program = "import json,sys; print(json.dumps(sys.argv[1:])); raise SystemExit(" + str(exit_code) + ")"
    result = subprocess.run(
        ["bash", str(IMAGE / "entrypoint.sh"), sys.executable, "-c", program, *arguments],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == exit_code
    assert json.loads(result.stdout) == arguments
    assert not result.stderr


def test_real_ssh_gate_runs_as_final_nonroot_user_and_removes_generated_keys():
    instructions = (IMAGE / "Dockerfile").read_text().replace("\\\n", " ").splitlines()
    user = None
    gates = []
    for instruction in instructions:
        if instruction.startswith("USER "):
            user = instruction.removeprefix("USER ")
        if instruction.startswith("RUN /usr/local/bin/openpi-entrypoint "):
            gates.append((user, shlex.split(instruction.removeprefix("RUN "))))
    assert len(gates) == 1
    user, command = gates[0]
    assert user == "ubuntu" and command[:3] == ["/usr/local/bin/openpi-entrypoint", "/bin/sh", "-ec"]
    assert command[4:] == ["bootstrap-probe", "two words", "", "--flag"]
    script = command[3]
    checks = ['test "$(id -u)" -ne 0', 'test -w /tmp', 'test -w "$HOME"',
              'test "$#" -eq 3', 'test "$1" = "two words"', 'test -z "$2"', 'test "$3" = "--flag"',
              'command -v rsync', 'command -v service', 'test -x /usr/sbin/sshd', 'sudo -n true',
              'sudo -n install -d -m 0755 /run/sshd', 'sudo -n ssh-keygen -A',
              'sudo -n /usr/sbin/sshd -t', 'grep -qx "passwordauthentication no"', 'grep -qx "permitrootlogin no"',
              'sudo -n service ssh start', 'sudo -n service ssh status', 'sudo -n service ssh stop',
              'sudo -n rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub',
              'test -z "$(find /etc/ssh -maxdepth 1 -name "ssh_host_*" -print -quit)"']
    positions = [script.index(check) for check in checks]
    assert positions == sorted(positions)
    assert "|| true" not in script
