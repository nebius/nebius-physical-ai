"""Verify managed-job log requests reach the selected isolated controller."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from npa.orchestration.skypilot import _bin, local_api
from npa.orchestration.skypilot.workflow_state import tail_live_job_logs


_COMPATIBLE_RUNTIME_EVIDENCE = "3.10 33.1.0"


def _write_sky(executable: Path, body: str) -> Path:
    executable.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "if sys.argv[1:] == ['--version']:\n"
        f"    print('skypilot, version {_bin.REQUIRED_SKYPILOT_VERSION}')\n"
        "    raise SystemExit(0)\n"
        + body
    )
    executable.chmod(0o700)
    interpreter = executable.with_name("python")
    # The test runner need not match the isolated SkyPilot runtime it models.
    interpreter.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "if sys.argv[1:2] == ['-c']:\n"
        f"    print({_COMPATIBLE_RUNTIME_EVIDENCE!r})\n"
        "    raise SystemExit(0)\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    interpreter.chmod(0o700)
    return executable


def test_synthetic_sky_runtime_reports_compatible_evidence(tmp_path: Path) -> None:
    executable = _write_sky(tmp_path / "sky", "")
    result = subprocess.run(
        [str(executable.with_name("python")), "-c", "ignored"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == _COMPATIBLE_RUNTIME_EVIDENCE


@pytest.fixture
def recording_sky(tmp_path: Path) -> Path:
    return _write_sky(
        tmp_path / "sky", "import json, os\n"
        "print(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), "
        "'home': os.environ.get('HOME'), "
        "'endpoint': os.environ.get('SKYPILOT_API_SERVER_ENDPOINT'), "
        "'config': os.environ.get('SKYPILOT_GLOBAL_CONFIG')}))\n"
    )


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
@pytest.mark.parametrize("banners", [False, True])
def test_logs_propagate_remote_task_selection_failure(
    tmp_path: Path, monkeypatch, stage: str, split_streams: bool, valid_ids: str, banners: bool,
) -> None:
    monkeypatch.setattr(_bin, "CONFIG_PATH", tmp_path / "absent.yaml")
    task = int(stage) if stage.isdecimal() else stage
    diagnostic = f"No task found matching {task!r} in job 61. Valid task IDs are {valid_ids}.\n"
    trailer = "command terminated with exit code 102\n"
    stdout = "\x1b[31m" + diagnostic + "\x1b[0m" + ("" if split_streams else trailer)
    stderr = trailer if split_streams else ""
    if banners:
        stdout = "SkyPilot: fetching task logs\n" + stdout
        stderr += "SkyPilot: log request finished\n"
    executable = tmp_path / "sky"
    _write_sky(
        executable,
        f"sys.stdout.write({stdout!r})\nsys.stderr.write({stderr!r})\n"
    )

    result = tail_live_job_logs(sky_bin=str(executable), job_id="61", stage=stage)

    assert result.returncode == 102
    assert result.stdout == stdout and result.stderr == stderr


@pytest.mark.parametrize("output", [
    "command terminated with exit code 102\n",
    "No task found matching 'rollout' in job 62. Valid task IDs are 0.\ncommand terminated with exit code 102\n",
    "No task found matching 'other' in job 61. Valid task IDs are 0.\ncommand terminated with exit code 102\n",
    "(worker pid=1) No task found matching 'rollout' in job 61. Valid task IDs are 0.\ncommand terminated with exit code 102\n",
    "application diagnostic: No task found matching 'rollout' in job 61. Valid task IDs are 0.\ncommand terminated with exit code 102\n",
    "No task found matching 'rollout' in job 61. Valid task IDs are 0.\ncommand terminated with exit code 1024\n",
    "Traceback: application FAILED\n",
])
def test_logs_do_not_reclassify_application_diagnostics(
    tmp_path: Path, monkeypatch, output: str,
) -> None:
    monkeypatch.setattr(_bin, "CONFIG_PATH", tmp_path / "absent.yaml")
    executable = tmp_path / "sky"
    _write_sky(executable, f"sys.stdout.write({output!r})\n")

    result = tail_live_job_logs(sky_bin=str(executable), job_id="61", stage="rollout")

    assert result.returncode == 0 and result.stdout == output
