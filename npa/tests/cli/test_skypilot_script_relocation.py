# Verify that moving an isolated SkyPilot environment preserves executable pip launchers.
"""Verify that isolated SkyPilot activation preserves genuine pip launchers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from npa.cli.skypilot import _relocate_staged_scripts


def _create_staged_launcher(staging: Path) -> Path:
    """Create a real offline pip launcher for an interpreter-path regression.

    Args:
        staging: Temporary virtual environment to create.
    Returns:
        Path to the generated executable.
    Raises:
        subprocess.CalledProcessError: If environment or launcher creation fails.
        OSError: If the probe module cannot be written.
    """
    # Bundled ensurepip exercises distlib without downloading another runtime.
    subprocess.run([sys.executable, "-m", "venv", str(staging)], check=True, capture_output=True)
    script = staging / "bin" / "sky"
    launcher_program = "\n".join([
        "from pip._vendor.distlib.scripts import ScriptMaker",
        "import sys",
        "maker = ScriptMaker(None, sys.argv[1])",
        "maker.variants = {''}",
        "maker.make('sky=launcher_probe:main')",
    ])
    subprocess.run(
        [str(staging / "bin" / "python"), "-c", launcher_program, str(script.parent)],
        check=True, capture_output=True,
    )
    (script.parent / "launcher_probe.py").write_text(
        "def main():\n"
        "    import json\n"
        "    import sys\n"
        "    print(json.dumps({'prefix': sys.prefix, 'args': sys.argv[1:]}))\n",
        encoding="utf-8",
    )
    return script


@pytest.mark.parametrize("runtime_name", ["runtime", "runtime with spaces"])
def test_pip_long_shebang_launcher_executes_after_staging_rename(tmp_path: Path, runtime_name: str):
    """Run pip's actual trampoline after its environment moves.

    Args:
        tmp_path: Isolated pytest temporary directory.
        runtime_name: Runtime name, including a shell-quoting case.
    Returns:
        None.
    Raises:
        AssertionError: If execution, arguments, payload or mode changes.
    """
    parent = tmp_path / ("long-path-segment-" * 8)
    staging = parent / (".staging-" + runtime_name)
    target = parent / runtime_name
    script = _create_staged_launcher(staging)
    before = script.read_bytes()
    assert before.startswith(b"#!/bin/sh\n'''exec' ")
    assert os.fsencode(str(staging)) in before.splitlines()[1]
    _relocate_staged_scripts(staging, target)
    staging.rename(target)
    result = subprocess.run(
        [str(target / "bin" / "sky"), "--version", "literal argument"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "prefix": str(target), "args": ["--version", "literal argument"],
    }
    after = (target / "bin" / "sky").read_bytes()
    assert after.split(b"\n", 3)[3] == before.split(b"\n", 3)[3]
    assert (target / "bin" / "sky").stat().st_mode & 0o111


def test_relocation_preserves_normal_shebang_and_unrelated_script_content(tmp_path: Path):
    """Leave payloads and external symlink targets outside relocation.

    Args:
        tmp_path: Isolated pytest temporary directory.
    Returns:
        None.
    Raises:
        AssertionError: If unrelated bytes or executable permissions change.
    """
    staging = tmp_path / "stage"
    target = tmp_path / "active"
    scripts = staging / "bin"
    scripts.mkdir(parents=True)
    body = f"print({str(staging)!r})\n".encode()
    regular = scripts / "regular"
    regular.write_bytes(b"#!" + os.fsencode(staging / "bin/python") + b"\n" + body)
    regular.chmod(0o755)
    unrelated = scripts / "unrelated"
    unrelated_body = b"#!/bin/sh\n" + b"echo " + os.fsencode(staging) + b"\n"
    unrelated.write_bytes(unrelated_body)
    external = tmp_path / "external"
    external_body = regular.read_bytes()
    external.write_bytes(external_body)
    (scripts / "linked").symlink_to(external)
    _relocate_staged_scripts(staging, target)
    assert regular.read_bytes() == b"#!" + os.fsencode(target / "bin/python") + b"\n" + body
    assert unrelated.read_bytes() == unrelated_body
    assert external.read_bytes() == external_body
    assert regular.stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize(
    "executable,closing_delimiter",
    [("other/bin/python", b"' '''"), ("bin/not-python", b"' '''"), ("bin/python", b"not-a-trampoline")],
)
def test_relocation_does_not_rewrite_similar_shell_programs(tmp_path: Path, executable, closing_delimiter):
    """Reject lookalike shell headers that are not staged Python launchers.

    Args:
        tmp_path: Isolated pytest temporary directory.
        executable: Interpreter token with a controlled mismatch.
        closing_delimiter: Closing shell/Python delimiter under test.
    Returns:
        None.
    Raises:
        AssertionError: If an unrelated shell program is rewritten.
    """
    staging = tmp_path / "stage"
    target = tmp_path / "active"
    scripts = staging / "bin"
    scripts.mkdir(parents=True)
    script = scripts / "custom"
    original = (
        b"#!/bin/sh\n'''exec' " + os.fsencode(staging / executable)
        + b' "$0" "$@"\n' + closing_delimiter + b"\nprint('body')\n"
    )
    script.write_bytes(original)
    _relocate_staged_scripts(staging, target)
    assert script.read_bytes() == original
