"""Execute generated credential scripts against actual private temporary files."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from npa.cli.agent import _agent_auth_setup_script
from npa.cli.cosmos import (
    _build_install_command,
    _build_serve_command,
    _build_service_env_script,
)


def _executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _tools(tmp_path: Path) -> Path:
    tools = tmp_path / "bin"
    tools.mkdir()
    _executable(tools / "sudo", '#!/bin/sh\nexec "$@"\n')
    (tools / "python3").symlink_to(sys.executable)
    return tools


@pytest.mark.parametrize("operation", ["install", "serve"])
@pytest.mark.parametrize("failure", ["none", "write", "token"])
def test_cosmos_env_private_atomic_and_literal(tmp_path, operation, failure):
    tools = _tools(tmp_path)
    destination = tmp_path / "service" / "env"
    destination.parent.mkdir()
    prior = b'COSMOS_MODEL_ID="previous"\n'
    destination.write_bytes(prior)
    destination.chmod(0o600)
    observations = tmp_path / "observations.json"
    _executable(
        tools / "tee",
        f"#!{sys.executable}\n"
        "import json, pathlib, stat, subprocess, sys\n"
        "path = pathlib.Path(sys.argv[-1])\n"
        "event = {'file_mode': stat.S_IMODE(path.stat().st_mode), "
        "'parent_mode': stat.S_IMODE(path.parent.stat().st_mode)}\n"
        f"pathlib.Path({str(observations)!r}).write_text(json.dumps(event))\n"
        "data = sys.stdin.buffer.read()\n"
        "result = subprocess.run(['/usr/bin/tee', *sys.argv[1:]], input=data, stdout=subprocess.PIPE)\n"
        f"raise SystemExit(17 if {failure == 'write'!r} else result.returncode)\n",
    )
    model = 'example/model-$literal-"quoted"'
    env_block = _build_service_env_script(model, 8080, no_guardrails=True)
    builder = _build_install_command if operation == "install" else _build_serve_command
    # Execute the exact production block wired into both paths, with only its
    # destination redirected to the temporary filesystem.
    rendered = shlex.split(builder(model, 8080, no_guardrails=True))[-1]
    assert env_block in rendered
    if operation == "serve":
        compile(rendered.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0], "server", "exec")
    token = "synthetic '$value' \\\" unicode-é" if failure != "token" else "synthetic\nINJECTED=1"
    completed = subprocess.run(
        ["/bin/bash", "-c", "umask 022\n" + env_block.replace("/etc/npa-cosmos-server", str(destination.parent))],
        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "HF_TOKEN": token},
        capture_output=True,
        text=True,
    )
    assert json.loads(observations.read_text()) == {"file_mode": 0o600, "parent_mode": 0o700}
    assert not list(destination.parent.glob(".env.*"))
    assert token not in completed.stdout + completed.stderr
    if failure != "none":
        assert completed.returncode != 0
        assert destination.read_bytes() == prior
    else:
        assert completed.returncode == 0, completed.stderr
        values = {key: json.loads(value) for key, value in (line.split("=", 1) for line in destination.read_text().splitlines())}
        assert values["HF_TOKEN"] == token
        assert values["COSMOS_MODEL_ID"] == model
        assert values["COSMOS_DISABLE_SAFETY"] == "1"
        assert destination.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("model", ["model\nENV\nexit 0", "model\rname", "model\0name"])
def test_cosmos_rejects_env_control_characters(model):
    for builder in (_build_install_command, _build_serve_command):
        with pytest.raises(ValueError, match="control characters"):
            builder(model, 8080)


def test_cosmos_download_uses_environment_token_not_process_argv():
    rendered = shlex.split(_build_install_command("example/model", 8080))[-1]
    downloads = [line for line in rendered.splitlines() if "huggingface-cli download" in line]
    assert len(downloads) == 1
    assert "--token" not in downloads[0]
    assert "$HF_TOKEN" not in downloads[0]


@pytest.mark.parametrize("exit_code", [0, 17])
def test_cosmos_installer_uses_private_workspace_and_always_cleans_it(tmp_path, exit_code):
    rendered = shlex.split(_build_install_command("example/model", 8080))[-1]
    setup = rendered.split("export DEBIAN_FRONTEND", 1)[0]
    setup = setup.replace("/tmp/npa-cosmos-install.", str(tmp_path / "npa-cosmos-install."))
    # The installer assigns every pip input inside this same private directory
    # before any apt, download or pip operation can fail.
    inspect = """
python3 - <<'PY'
import json, os, pathlib, stat
names = ('cosmos_requirements', 'cosmos_constraints', 'flash_attn_wheel', 'natten_wheel', 'transformer_engine_wheel')
paths = [pathlib.Path(os.environ[name]) for name in names]
stage = pathlib.Path(os.environ['install_stage'])
assert all(path.parent == stage for path in paths)
assert stat.S_IMODE(stage.stat().st_mode) == 0o700
for path in paths:
    path.write_bytes(b'complete installer input')
print(json.dumps({'stage': str(stage), 'inputs': len(paths)}))
PY
"""
    completed = subprocess.run(
        ["/bin/bash", "-c", "umask 022\nset -a\n" + setup + inspect + f"exit {exit_code}\n"],
        env={**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"},
        capture_output=True, text=True,
    )
    assert completed.returncode == exit_code, completed.stderr
    observation = json.loads(completed.stdout)
    assert observation["inputs"] == 5
    assert not Path(observation["stage"]).exists()
    assert not list(tmp_path.glob("npa-cosmos-install.*"))


@pytest.mark.parametrize("fail_hash", [False, True])
def test_agent_auth_private_staging_stdin_and_atomic_failure(tmp_path, fail_hash):
    tools = _tools(tmp_path)
    # Ownership is privileged; the live test verifies root:www-data. All mode,
    # shell, pipe, input and atomic replacement behavior here uses real files.
    _executable(tools / "chown", "#!/bin/sh\nexit 0\n")
    destination = tmp_path / "nginx" / ".npa-agent-htpasswd"
    destination.parent.mkdir()
    prior = b"previous:retained-hash\n"
    destination.write_bytes(prior)
    destination.chmod(0o640)
    observations = tmp_path / "auth-observations.json"
    password = "synthetic ' $() \\" + '" é'
    _executable(
        tools / "htpasswd",
        f"#!{sys.executable}\n"
        "import json, pathlib, stat, sys\n"
        "path = pathlib.Path(sys.argv[-2])\n"
        "event = {'argv': sys.argv[1:], 'stdin': sys.stdin.read(), "
        "'file_mode': stat.S_IMODE(path.stat().st_mode), "
        "'parent_mode': stat.S_IMODE(path.parent.stat().st_mode)}\n"
        f"pathlib.Path({str(observations)!r}).write_text(json.dumps(event))\n"
        "path.write_text('operator:replacement-hash\\n')\n"
        f"raise SystemExit({17 if fail_hash else 0})\n",
    )
    script = _agent_auth_setup_script("operator", password).replace("/etc/nginx", str(destination.parent))
    completed = subprocess.run(
        ["/bin/bash", "-c", "umask 022\n" + script],
        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    event = json.loads(observations.read_text())
    assert event["file_mode"] == 0o600
    assert event["parent_mode"] == 0o700
    assert event["stdin"] == password + "\n"
    assert password not in " ".join(event["argv"])
    assert event["argv"][:3] == ["-iBc", "-C", "12"]
    assert not list(destination.parent.glob(".npa-auth.*"))
    if fail_hash:
        assert completed.returncode == 17
        assert destination.read_bytes() == prior
    else:
        assert completed.returncode == 0, completed.stderr
        assert destination.read_text() == "operator:replacement-hash\n"
        assert destination.stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize("password", ["", "x" * 73, "é" * 37, "line\nother", "line\rother", "nul\0byte"])
def test_agent_auth_rejects_truncation_or_multiline_password(password):
    with pytest.raises(ValueError, match="1..72"):
        _agent_auth_setup_script("operator", password)


@pytest.mark.parametrize("username", ["", "-option", "user:other", "user\nother", "user\0other"])
def test_agent_auth_rejects_malformed_username(username):
    with pytest.raises(ValueError, match="username"):
        _agent_auth_setup_script(username, "synthetic-password")
