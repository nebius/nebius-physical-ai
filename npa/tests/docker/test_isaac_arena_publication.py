"""Exercise Arena publication refusal and smoke cleanup without network access."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "npa/docker/workbench/isaac-arena"


def _executable(path: Path, source: str) -> None:
    path.write_text("#!/bin/bash\nset -eu\n" + source, encoding="utf-8")
    path.chmod(0o700)


@pytest.mark.parametrize(
    "registry",
    ["", "ghcr.io/nebius/nebius-physical-ai", "ghcr.io/nebius/nebius-physical-ai/"],
)
def test_official_public_push_refuses_before_build(
    tmp_path: Path, registry: str
) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    marker = tmp_path / "docker-called"
    _executable(commands / "docker", f"touch {shlex.quote(str(marker))}\nexit 99\n")
    environment = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}")
    environment.pop("REGISTRY", None)
    argv = ["bash", str(SCRIPTS / "build.sh"), "--push", "--tag", "dev-" + "f" * 40]
    if registry:
        argv.extend(["--registry", registry])
    result = subprocess.run(
        argv, env=environment, capture_output=True, text=True, check=False
    )
    assert result.returncode == 2, result.stderr
    assert "publish-public-images.yml" in result.stderr
    assert "pre-publication gates" in result.stderr
    assert not marker.exists()


def _smoke_commands(tmp_path: Path, exit_code: int) -> Path:
    commands = tmp_path / "commands"
    commands.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    mktemp = shutil.which("mktemp")
    assert mktemp is not None
    _executable(
        commands / "mktemp",
        f"exec {shlex.quote(mktemp)} -d {shlex.quote(str(scratch / 'replay.XXXXXX'))}\n",
    )
    _executable(
        commands / "curl",
        'while [ "$1" != --output ]; do shift; done\nprintf fixture > "$2"\n',
    )
    _executable(commands / "sha256sum", "cat >/dev/null\n")
    _executable(
        commands / "npa",
        'while [ "$1" != --input-path ]; do shift; done\n'
        'test -f "$2"\nprintf "%s\\n" "$2" > "$NPA_TEST_REPLAY_RECORD"\n'
        f"exit {exit_code}\n",
    )
    return commands


@pytest.mark.parametrize("exit_code", [0, 19])
def test_smoke_removes_owned_replay_and_preserves_workload_exit(
    tmp_path: Path, exit_code: int
) -> None:
    commands = _smoke_commands(tmp_path, exit_code)
    record = tmp_path / "replay-location.txt"
    environment = dict(
        os.environ,
        PATH=f"{commands}:{os.environ['PATH']}",
        NPA_TEST_REPLAY_RECORD=str(record),
    )
    environment.pop("NPA_ISAAC_ARENA_REPLAY_PATH", None)
    result = subprocess.run(
        ["bash", str(SCRIPTS / "smoke_functional.sh")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    replay = Path(record.read_text().strip())
    assert replay.is_relative_to(tmp_path / "scratch")
    assert not replay.parent.exists()
