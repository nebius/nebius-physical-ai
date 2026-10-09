"""Opt-in live read proof for one Cosmos Evaluator report object."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

from npa.sdk.workbench import cosmos_evaluator


pytestmark = pytest.mark.e2e


def test_live_report_inspection_uses_one_explicit_object() -> None:
    """Compare the SDK and installed CLI summaries of one supplied report object."""
    report_uri = _required_report_uri()
    sdk_summary = cosmos_evaluator.report(input_path=report_uri)
    cli_json = _run_cli(report_uri, "json")
    cli_summary = json.loads(cli_json.stdout)
    cli_text = _run_cli(report_uri, "text")

    assert cli_summary == sdk_summary
    assert cli_summary["reported_gate"]["score"] == sdk_summary["reported_gate"]["score"]
    assert cli_summary["reported_gate"]["passed"] is sdk_summary["reported_gate"]["passed"]
    assert f"gate score: {sdk_summary['reported_gate']['score']}" in cli_text.stdout
    assert "score is not calibrated confidence" in cli_text.stdout
    assert "multi-view assessment is not evaluated" in cli_text.stdout


def _required_report_uri() -> str:
    report_uri = os.environ.get("NPA_COSMOS_EVALUATOR_REPORT_URI", "").strip()
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not report_uri:
        pytest.skip(
            "requires NPA_INTEGRATION_E2E=1 and one exact "
            "NPA_COSMOS_EVALUATOR_REPORT_URI object"
        )
    parsed = urlparse(report_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        pytest.fail("NPA_COSMOS_EVALUATOR_REPORT_URI must name one exact S3 object")
    if parsed.path.endswith("/") or parsed.query or parsed.fragment:
        pytest.fail("NPA_COSMOS_EVALUATOR_REPORT_URI must not be a prefix or URL variant")
    return report_uri


def _run_cli(report_uri: str, output_format: str) -> subprocess.CompletedProcess[str]:
    command = Path(sys.executable).with_name("npa")
    if not command.is_file():
        pytest.skip("requires the installed npa CLI beside the active interpreter")
    result = subprocess.run(
        [
            str(command),
            "workbench",
            "cosmos-evaluator",
            "report",
            "--input-path",
            report_uri,
            "--output-format",
            output_format,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail("installed report CLI failed to inspect the supplied artifact")
    return result
