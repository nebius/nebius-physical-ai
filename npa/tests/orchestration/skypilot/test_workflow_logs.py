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


@pytest.mark.parametrize("stage", ["rollout", "99"])
@pytest.mark.parametrize("split_streams", [False, True])
@pytest.mark.parametrize("valid_ids", ["0", "0-7"])
def test_logs_propagate_remote_task_selection_failure(
    tmp_path: Path, monkeypatch, stage: str, split_streams: bool, valid_ids: str,
) -> None:
    monkeypatch.setattr(_bin, "CONFIG_PATH", tmp_path / "absent.yaml")
    task = int(stage) if stage.isdecimal() else stage
    diagnostic = f"No task found matching {task!r} in job 61. Valid task IDs are {valid_ids}.\n"
    trailer = "command terminated with exit code 102\n"
    stdout = "\x1b[31m" + diagnostic + "\x1b[0m" + ("" if split_streams else trailer)
    stderr = trailer if split_streams else ""
    executable = tmp_path / "sky"
    executable.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"sys.stdout.write({stdout!r})\nsys.stderr.write({stderr!r})\n"
    )
    executable.chmod(0o700)

    result = tail_live_job_logs(sky_bin=str(executable), job_id="61", stage=stage)

    assert result.returncode == 102
    assert result.stdout == stdout and result.stderr == stderr


@pytest.mark.parametrize("output", [
    "command terminated with exit code 102\n",
    "No task found matching 'rollout' in job 62. Valid task IDs are 0.\ncommand terminated with exit code 102\n",
    "No task found matching 'other' in job 61. Valid task IDs are 0.\ncommand terminated with exit code 102\n",
    "(worker pid=1) No task found matching 'rollout' in job 61. Valid task IDs are 0.\ncommand terminated with exit code 102\n",
    "application diagnostic follows\nNo task found matching 'rollout' in job 61. Valid task IDs are 0.\ncommand terminated with exit code 102\n",
    "No task found matching 'rollout' in job 61. Valid task IDs are 0.\ncommand terminated with exit code 102\napplication continued\n",
    "Traceback: application FAILED\n",
])
def test_logs_do_not_reclassify_application_diagnostics(
    tmp_path: Path, monkeypatch, output: str,
) -> None:
    monkeypatch.setattr(_bin, "CONFIG_PATH", tmp_path / "absent.yaml")
    executable = tmp_path / "sky"
    executable.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.write({output!r})\n")
    executable.chmod(0o700)

    result = tail_live_job_logs(sky_bin=str(executable), job_id="61", stage="rollout")

    assert result.returncode == 0 and result.stdout == output
