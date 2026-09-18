"""Verify model provenance against an operator-provisioned GPU VLM endpoint."""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app


@pytest.mark.e2e
@pytest.mark.gpu
def test_self_hosted_result_retains_served_model() -> None:
    config_path = os.environ.get("NPA_VLM_PROVENANCE_LIVE_CONFIG", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not config_path:
        pytest.skip("provide an operator-owned NPA_VLM_PROVENANCE_LIVE_CONFIG")
    config = json.loads(Path(config_path).read_text())

    result = CliRunner().invoke(
        app,
        [
            "workbench", "vlm-eval", "run",
            "--input-path", config["input_path"],
            "--output-path", config["output_path"],
            "--task", config["task"],
            "--backend", "self-hosted",
            "--endpoint-url", config["endpoint_url"],
            "--model", config["model"],
            "--api-key-env", config.get("api_key_env", "VLM_EVAL_API_KEY"),
            "--frame-selection", "keyframes",
            "--max-frames", "8",
            "--success-threshold", "0.8",
            "--output", "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["served_model"] == config["expected_served_model"]
    assert payload["model"] == config["model"]
    assert 0 <= payload["score"] <= 1
    assert payload["backend"] == "self-hosted" and not payload["dry_run"]
    assert payload.pop("written_uri") == config["output_path"]
    saved = json.loads(Path(config["output_path"]).read_text())
    assert saved == payload
