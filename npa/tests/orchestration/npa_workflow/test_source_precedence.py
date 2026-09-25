"""Execute rendered stage shells against competing baked and submitted packages."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from npa.orchestration.npa_workflow.skypilot_render import render_task_run_script


@pytest.mark.parametrize("record_baked", [False, True])
def test_submitted_source_wins_over_baked_package(
    tmp_path: Path, record_baked: bool
) -> None:
    overlay = tmp_path / "npa-src-overlay"
    baked = tmp_path / "baked"
    for directory, marker in ((overlay / "src", "submitted"), (baked, "baked")):
        package = directory / "npa"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"SOURCE = {marker!r}\n")
    (tmp_path / "npa-src-root").write_text(str(overlay))
    if record_baked:
        (tmp_path / "npa-baked-pythonpath").write_text(str(baked))
    script = render_task_run_script(
        [sys.executable, "-c", "import npa; assert npa.SOURCE == 'submitted'"]
    )
    # Follow the renderer's real control directory without using shared host files.
    shim = re.search(r"^  mkdir -p (.+/npa-shim)$", script, flags=re.MULTILINE)
    assert shim is not None
    control_prefix = str(PurePosixPath(shim[1]).with_name("npa"))
    script = script.replace(control_prefix, str(tmp_path / "npa"))
    script = script.replace("/etc/profile.d", str(tmp_path / "profiles"))
    environment = {**os.environ, "PYTHONPATH": str(baked), "HOME": str(tmp_path)}
    result = subprocess.run(
        ["bash", "-c", script], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
