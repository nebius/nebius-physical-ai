"""Exercise the prepared control interpreter across custom stage shells."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from npa.orchestration.npa_workflow.skypilot_render import render_task_run_script


def _isolated_script(tmp_path: Path, command: list[str]) -> str:
    script = render_task_run_script(command)
    # Discover every generated worker-state directory so future additions also
    # stay isolated from other tests and existing setup receipts.
    runtime_paths = set()
    for value in re.findall(r"/[A-Za-z0-9_./-]+", script):
        parts = PurePosixPath(value).parts
        if len(parts) >= 3 and parts[1] == "tmp" and parts[2].startswith("npa-"):
            runtime_paths.add(PurePosixPath(*parts[:3]))
    assert runtime_paths
    for path in sorted(runtime_paths, key=str, reverse=True):
        script = script.replace(str(path), str(tmp_path / path.name))
    return script.replace("/etc/profile.d", str(tmp_path / "profile.d"))


def _run_script(script: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["NPA_CONTROL_PYTHON"] = "/unrelated/inherited/python"
    return subprocess.run(
        ["/bin/bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


def test_control_python_survives_child_shell_path_changes(tmp_path: Path) -> None:
    control_python = tmp_path / "control interpreter"
    control_python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    control_python.chmod(0o700)
    (tmp_path / "npa-python").write_text(str(control_python))
    dependency_check = (
        "import sys; from npa.clients.storage import StorageClient; "
        "import boto3; print(sys.executable)"
    )
    child = (
        "export PATH=/unrelated/policy/bin; "
        f'"$NPA_CONTROL_PYTHON" -c {shlex.quote(dependency_check)}'
    )
    script = _isolated_script(tmp_path, ["/bin/bash", "-c", child])

    result = _run_script(script)

    assert result.stdout.strip() == sys.executable


@pytest.mark.parametrize("recorded", [False, True])
def test_missing_control_python_clears_inherited_value(
    tmp_path: Path, recorded: bool
) -> None:
    if recorded:
        unavailable = tmp_path / "nonexecutable-python"
        unavailable.write_text("not executable\n")
        (tmp_path / "npa-python").write_text(str(unavailable))
    command = ["/bin/bash", "-c", 'test -z "$NPA_CONTROL_PYTHON"']

    result = _run_script(_isolated_script(tmp_path, command))

    assert result.returncode == 0
