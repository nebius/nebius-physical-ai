from __future__ import annotations

import json
import os

import pytest
from typer.testing import CliRunner

from npa.cli.main import app


pytestmark = pytest.mark.ngc_e2e


def test_groot_status_reports_ngc_ready_when_credentials_are_deployed() -> None:
    if os.environ.get("NPA_TEST_GROOT_NGC_E2E") != "1":
        pytest.skip("set NPA_TEST_GROOT_NGC_E2E=1 to run the live GR00T NGC status check")
    project = os.environ.get("NPA_TEST_GROOT_PROJECT")
    name = os.environ.get("NPA_TEST_GROOT_NAME")
    if not project or not name:
        pytest.skip("set NPA_TEST_GROOT_PROJECT and NPA_TEST_GROOT_NAME for the live GR00T workbench")

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "groot",
            "-p",
            project,
            "-n",
            name,
            "status",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ngc_credentials_configured"] is True
    assert payload["readiness"]["ngc_credentials_configured"] is True
    assert payload["readiness"]["ready"] is True
