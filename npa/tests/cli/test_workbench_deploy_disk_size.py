from __future__ import annotations

import pytest
from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()


@pytest.mark.parametrize("tool", ["lerobot", "genesis", "isaac-lab", "cosmos", "fiftyone"])
def test_workbench_deploys_expose_disk_size_flag(tool: str) -> None:
    result = runner.invoke(app, ["workbench", tool, "deploy", "--help"])

    assert result.exit_code == 0
    assert "--disk-size" in result.output
