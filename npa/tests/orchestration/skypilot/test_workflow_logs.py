"""Verify managed-job log requests reach the selected isolated controller."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest
import yaml

from npa.orchestration.skypilot import _bin, local_api
from npa.orchestration.skypilot.workflow_state import tail_live_job_logs


@pytest.fixture
def recording_sky(tmp_path: Path) -> Path:
    executable = tmp_path / "sky"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), "
        "'home': os.environ.get('HOME'), "
        "'endpoint': os.environ.get('SKYPILOT_API_SERVER_ENDPOINT'), "
        "'config': os.environ.get('SKYPILOT_GLOBAL_CONFIG')}))\n"
    )
    executable.chmod(0o700)
    return executable


@pytest.mark.parametrize(("stage", "follow"), [("0", False), ("", True)])
def test_logs_use_saved_isolated_connection(
    tmp_path: Path, monkeypatch, recording_sky: Path, stage: str, follow: bool,
) -> None:
    isolated = tmp_path / "isolated controller"
    sky_config = tmp_path / "sky-config.yaml"
    config = tmp_path / "npa-config.yaml"
    sky_config.write_text("{}\n")
    config.write_text(yaml.safe_dump({"skypilot": {
        "isolated_config_dir": str(isolated), "global_config_path": str(sky_config),
    }}))
    monkeypatch.setattr(_bin, "CONFIG_PATH", config)
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "http://127.0.0.1:1111")
    calls = []

    def selected_connection(root, environment):
        calls.append(root)
        return {**environment, "SKYPILOT_API_SERVER_ENDPOINT": "http://127.0.0.1:2222"}

    monkeypatch.setattr(local_api, "isolated_api_environment", selected_connection)
    result = tail_live_job_logs(sky_bin=str(recording_sky), job_id="61", stage=stage, follow=follow)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert calls == [isolated]
    assert output["endpoint"] == "http://127.0.0.1:2222"
    assert output["home"] == str(isolated / "home")
    assert Path(output["cwd"]).resolve() == isolated.resolve()
    assert output["config"] == str(sky_config)
    assert output["argv"] == ["jobs", "logs", "61", *([stage] if stage else []),
                              "--follow" if follow else "--no-follow"]


def test_logs_preserve_explicit_connection_without_isolation(
    tmp_path: Path, monkeypatch, recording_sky: Path,
) -> None:
    monkeypatch.setattr(_bin, "CONFIG_PATH", tmp_path / "absent-config.yaml")
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "http://127.0.0.1:3333")
    result = tail_live_job_logs(sky_bin=str(recording_sky), job_id="61")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["endpoint"] == "http://127.0.0.1:3333"
    assert output["home"] == os.environ.get("HOME")
