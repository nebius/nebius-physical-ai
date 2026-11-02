from __future__ import annotations

import os
import subprocess
import sys

import pytest


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


@pytest.mark.parametrize("tool", ["nurec", "rerun-viewer"])
def test_nurec_viewer_light_workbench_exposes_only_nurec_parent(tool: str) -> None:
    result = _run_light_import(
        """
import sys
from typer.testing import CliRunner
from npa.cli.workbench import app
assert "npa.cli.nurec" in sys.modules
assert "npa.cli.workbench.cosmos2" not in sys.modules
assert "npa.cli.fiftyone" not in sys.modules
assert "npa.cli.groot" not in sys.modules
result = CliRunner().invoke(app, ["nurec", "visualize", "--help"])
assert result.exit_code == 0, result.output
assert "--input-uri" in result.output
""",
        tool=tool,
    )

    assert result.returncode == 0, result.stderr


def test_cosmos3_ray_serve_light_workbench_exposes_cosmos3_only() -> None:
    """The cosmos3-ray-serve image gets the cosmos3 surface without Cosmos2 or
    other workbench tools."""
    result = _run_light_import(
        """
import sys
from typer.testing import CliRunner
from npa.cli.workbench import app
# The light surface must have imported cosmos3 (and not cosmos2).
assert "npa.cli.workbench.cosmos3" in sys.modules
assert "npa.cli.workbench.cosmos2" not in sys.modules
# ray-health is the narrow command the container probes.
cli_result = CliRunner().invoke(app, ["cosmos3", "ray-health", "--help"])
assert cli_result.exit_code == 0, cli_result.output
assert "--endpoint" in cli_result.output
""",
        tool="cosmos3-ray-serve",
    )

    assert result.returncode == 0, result.stderr


def test_viewer_light_workbench_reaches_hosted_caption_without_platform_imports() -> (
    None
):
    result = _run_light_import(
        """
import sys
from typer.testing import CliRunner
from npa.cli.workbench import app
from npa.cli.workbench import token_factory
calls = []
def caption(**kwargs):
    calls.append(kwargs)
    raise token_factory.TokenFactoryToolError("caption-boundary")
token_factory.caption_images = caption
result = CliRunner().invoke(app, [
    "token-factory", "caption", "--input-path", "s3://example-bucket/frames/",
    "--output-path", "s3://example-bucket/captions/", "--model", "fixture-vision",
    "--instruction", "Describe visible robot parts", "--max-images", "8",
])
assert result.exit_code == 1 and "caption-boundary" in result.output, result.output
assert len(calls) == 1 and calls[0]["model"] == "fixture-vision"
assert calls[0]["input_path"] == "s3://example-bucket/frames/"
assert calls[0]["max_images"] == 8
assert "npa.cli.fiftyone" not in sys.modules
assert "npa.cli.groot" not in sys.modules
assert "npa.cli.workbench.cosmos2" not in sys.modules
assert "mujoco" not in sys.modules
""",
        tool="rerun-viewer",
    )

    assert result.returncode == 0, result.stderr
