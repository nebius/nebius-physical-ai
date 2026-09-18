"""Verify saved-project selection through a real operator-owned GPU workflow."""

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import default_project_name
from npa.clients.project_credentials import s3_client_for_project


def _live_config():
    path = os.environ.get("NPA_RUNTIME_DEFAULT_PROJECT_LIVE_CONFIG", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not path:
        pytest.skip("provide an operator-owned runtime default-project live config")
    return json.loads(Path(path).read_text())


def _submit_arguments(config):
    arguments = [
        "workbench", "workflow", "submit", config["spec_path"],
        "--runtime", "--run-id", config["run_id"],
        "--infra", config["infra"], "--output-format", "json",
        "--isolated-config-dir", config["isolated_config_dir"],
    ]
    for key in ["sky_bin", "config_path"]:
        if config.get(key):
            arguments.extend(["--" + key.replace("_", "-"), config[key]])
    return arguments


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.timeout(0)
def test_runtime_uses_saved_project_for_gpu_execution() -> None:
    config = _live_config()
    project = default_project_name()
    assert project and project != "default"
    result = CliRunner().invoke(app, _submit_arguments(config))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "succeeded" and payload["waves"]
    for wave in payload["waves"]:
        assert wave["status"] == "succeeded" and wave["job_id"]
        assert wave["credentials"]["source"] == "project:" + project
    location = urlparse(config["output_uri"])
    response = s3_client_for_project(project).get_object(
        Bucket=location.netloc, Key=location.path.lstrip("/"),
    )
    proof = json.loads(response["Body"].read())
    assert proof["cuda"] and proof["gpu_count"] >= 1
    assert proof["finite"] and proof["max_abs_error"] < 1e-5
