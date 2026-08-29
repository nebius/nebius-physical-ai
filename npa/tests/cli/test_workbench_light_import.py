from __future__ import annotations

import os
import subprocess
import sys


def _run_light_import(code: str, *, tool: str = "") -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "NPA_SKIP_EAGER_IMPORTS": "1"}
    if tool:
        env["NPA_LIGHT_WORKBENCH_TOOL"] = tool
    else:
        env.pop("NPA_LIGHT_WORKBENCH_TOOL", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_default_light_workbench_preserves_cosmos2_surface() -> None:
    result = _run_light_import(
        """
import sys
from npa.cli.workbench import app
from npa.cli.workbench.cosmos2 import app as cosmos2_app
assert app is cosmos2_app
assert "npa.cli.groot" not in sys.modules
"""
    )

    assert result.returncode == 0, result.stderr


def test_groot_light_workbench_exposes_only_groot_parent() -> None:
    result = _run_light_import(
        """
import sys
from typer.testing import CliRunner
from npa.cli.workbench import app
assert "npa.cli.groot" in sys.modules
assert "npa.cli.workbench.cosmos2" not in sys.modules
assert "npa.cli.fiftyone" not in sys.modules
result = CliRunner().invoke(app, ["groot", "finetune", "--help"])
assert result.exit_code == 0, result.output
assert "--runtime" in result.output
""",
        tool="groot",
    )

    assert result.returncode == 0, result.stderr
