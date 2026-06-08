from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from npa.orchestration.skypilot import _bin as bin_module
from npa.orchestration.skypilot import workflow as workflow_module
from npa.orchestration.skypilot.workflow import (
    SkyPilotSubmitError,
    _status_from_queue_payload,
    submit_workflow,
    workflow_status,
)


def _fake_sky(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sky = bin_dir / "sky"
    sky.write_text("#!/bin/sh\n", encoding="utf-8")
    sky.chmod(0o755)
    return sky


def _is_status_cmd(cmd: list[str]) -> bool:
    return len(cmd) >= 2 and cmd[1] == "status"


def _healthy_status(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")


@pytest.fixture(autouse=True)
def _skip_version_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(workflow_module, "ensure_skypilot_version", lambda sky_bin: Path(sky_bin))
    monkeypatch.setattr(bin_module, "CONFIG_PATH", tmp_path / "missing-config.yaml")
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.delenv("SKYPILOT_GLOBAL_CONFIG", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", raising=False)


def test_submit_workflow_loads_yaml_applies_controller_and_calls_subprocess(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Job submitted, ID: 42\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = submit_workflow(yaml_path, "run-abc", isolated_config_dir=tmp_path / "sky-state", sky_bin=sky_bin)

    assert result.status == "SUBMITTED"
    assert result.job_id == "42"
    assert calls[0][0] == [str(sky_bin), "status", "--refresh", "--output", "json"]
    cmd, kwargs = calls[1]
    assert cmd[:5] == [str(sky_bin), "jobs", "launch", "--name", "run-abc"]
    assert "--config" not in cmd
    assert "--detach-run" in cmd
    assert kwargs["env"]["HOME"] == str(tmp_path / "sky-state" / "home")
    assert kwargs["env"]["SKYPILOT_GLOBAL_CONFIG"] == result.log_paths["config"]
    config = yaml.safe_load((tmp_path / "sky-state" / "submissions" / "run-abc" / "skypilot-config.yaml").read_text())
    assert config["jobs"]["controller"]["resources"] == {
        "cloud": "kubernetes",
        "cpus": 4,
        "memory": 16,
        "autostop": False,
    }


def test_submit_workflow_network_failure_raises_typed_error(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="network connection failed")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SkyPilotSubmitError, match="sky jobs launch failed.*network connection failed"):
        submit_workflow(yaml_path, "run-fail", isolated_config_dir=tmp_path / "sky", sky_bin=sky_bin)


def test_submit_workflow_auth_failure_raises_typed_error(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Authentication failed: credentials expired")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SkyPilotSubmitError, match="auth failure.*credentials expired"):
        submit_workflow(yaml_path, "run-auth-fail", isolated_config_dir=tmp_path / "sky", sky_bin=sky_bin)


def test_submit_workflow_yaml_parse_error_raises_typed_error(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: [unterminated\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        raise AssertionError("malformed YAML should fail before sky jobs launch")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SkyPilotSubmitError, match="workflow submission failed"):
        submit_workflow(yaml_path, "run-yaml-fail", isolated_config_dir=tmp_path / "sky", sky_bin=sky_bin)


def test_submit_workflow_cleans_owned_temp_dir_on_timeout(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    owned_dir = tmp_path / "owned-submission"

    def fake_mkdtemp(prefix: str) -> str:
        owned_dir.mkdir()
        return str(owned_dir)

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(workflow_module.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SkyPilotSubmitError, match="timed out"):
        submit_workflow(yaml_path, "run-timeout", sky_bin=sky_bin)

    assert not owned_dir.exists()


def test_submit_workflow_can_emit_nebius_controller_fallback(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Job submitted, ID: 12\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = submit_workflow(
        yaml_path,
        "run-nebius",
        isolated_config_dir=tmp_path / "sky-state",
        sky_bin=sky_bin,
        controller_backend="nebius",
    )

    config = yaml.safe_load(Path(result.log_paths["config"]).read_text())
    resources = config["jobs"]["controller"]["resources"]
    assert resources["cloud"] == "nebius"
    assert resources["instance_type"] == "cpu-e2_2vcpu-8gb"
    assert resources["autostop"] is False


def test_submit_workflow_passes_configured_secret_env_names(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Job submitted, ID: 10\n", stderr="")

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)

    submit_workflow(
        yaml_path,
        "run-secrets",
        isolated_config_dir=tmp_path / "sky-state",
        sky_bin=sky_bin,
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    )

    cmd = calls[0]
    assert ["--secret", "AWS_ACCESS_KEY_ID"] == cmd[-3:-1]
    assert "test-access-key" not in cmd
    assert "AWS_SECRET_ACCESS_KEY" not in cmd


def test_submit_workflow_honors_isolated_config_dir(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs["env"])
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Job submitted, ID: 9", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    submit_workflow(yaml_path, "run-env", isolated_config_dir=tmp_path / "isolated", sky_bin=sky_bin)

    assert captured_env["HOME"] == str(tmp_path / "isolated" / "home")
    assert captured_env["SKY_RUNTIME_DIR"] == str(tmp_path / "isolated" / "sky-runtime")


def test_submit_workflow_require_controller_up_uses_canonical_preflight(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:4] == ["status", "--refresh", "--output"]:
            stdout = '[{"name": "sky-jobs-controller-abc123", "status": "UP"}]'
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="Job submitted, ID: 77\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = submit_workflow(
        yaml_path,
        "run-guard",
        isolated_config_dir=tmp_path / "sky-state",
        sky_bin=sky_bin,
        require_controller_up=True,
    )

    assert result.job_id == "77"
    assert calls[0] == [str(sky_bin), "status", "--refresh", "--output", "json"]
    assert calls[1][:5] == [str(sky_bin), "jobs", "launch", "--name", "run-guard"]


def test_submit_workflow_require_controller_up_blocks_missing_controller(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if _is_status_cmd(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
        raise AssertionError("launch should be blocked until a controller exists")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SkyPilotSubmitError, match="no jobs-controller found"):
        submit_workflow(
            yaml_path,
            "run-require-controller",
            isolated_config_dir=tmp_path / "sky-state",
            sky_bin=sky_bin,
            require_controller_up=True,
            controller_preflight_timeout=0,
            controller_preflight_interval=0,
        )

    assert calls == [[str(sky_bin), "status", "--refresh", "--output", "json"]]


def test_submit_workflow_blocks_unhealthy_existing_jobs_controller(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    owned_dir = tmp_path / "owned-autostop-submission"
    calls = []

    def fake_mkdtemp(prefix: str) -> str:
        owned_dir.mkdir()
        return str(owned_dir)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if _is_status_cmd(cmd):
            stdout = '[{"name": "sky-jobs-controller-64ce57a0", "status": "AUTOSTOPPING"}]'
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        raise AssertionError("launch should be blocked until controller is healthy")

    monkeypatch.setattr(workflow_module.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SkyPilotSubmitError, match="sky-jobs-controller-64ce57a0=AUTOSTOPPING"):
        submit_workflow(
            yaml_path,
            "run-autostop",
            sky_bin=sky_bin,
            controller_preflight_timeout=0,
            controller_preflight_interval=0,
        )

    assert calls == [[str(sky_bin), "status", "--refresh", "--output", "json"]]
    assert not owned_dir.exists()


def test_submit_workflow_controller_preflight_parses_warning_prefixed_json(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if not _is_status_cmd(cmd):
            raise AssertionError("launch should be blocked until controller is healthy")
        stdout = (
            "\x1b[33mCluster 'sky-jobs-controller-abc123' is autostopping.\x1b[0m\n\n"
            '[{"name": "sky-jobs-controller-abc123", "status": "AUTOSTOPPING"}]'
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(
        SkyPilotSubmitError,
        match="sky-jobs-controller-abc123=AUTOSTOPPING",
    ):
        submit_workflow(
            yaml_path,
            "run-guard",
            isolated_config_dir=tmp_path / "sky-state",
            sky_bin=sky_bin,
            controller_preflight_timeout=0,
            controller_preflight_interval=0,
        )

    assert calls == [[str(sky_bin), "status", "--refresh", "--output", "json"]]


def test_workflow_status_reads_json_queue(monkeypatch, tmp_path) -> None:
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        stdout = '[{"job_id": 42, "name": "run", "status": "SUCCEEDED"}]'
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = workflow_status("42", sky_bin=sky_bin)

    assert result.status == "SUCCEEDED"
    assert result.job_id == "42"


def test_status_from_queue_payload_waits_for_all_dag_tasks() -> None:
    payload = [
        {"job_id": 1, "task_id": 0, "status": "SUCCEEDED"},
        {"job_id": 1, "task_id": 1, "status": "STARTING"},
        {"job_id": 1, "task_id": 2, "status": "PENDING"},
    ]

    assert _status_from_queue_payload(json.dumps(payload), "1") == "STARTING"


def test_status_from_queue_payload_reports_success_after_all_dag_tasks() -> None:
    payload = [
        {"job_id": 1, "task_id": 0, "status": "SUCCEEDED"},
        {"job_id": 1, "task_id": 1, "status": "SUCCEEDED"},
        {"job_id": 1, "task_id": 2, "status": "SUCCEEDED"},
    ]

    assert _status_from_queue_payload(json.dumps(payload), "1") == "SUCCEEDED"


def test_status_from_queue_payload_failure_wins() -> None:
    payload = [
        {"job_id": 1, "task_id": 0, "status": "SUCCEEDED"},
        {"job_id": 1, "task_id": 1, "status": "FAILED"},
        {"job_id": 1, "task_id": 2, "status": "PENDING"},
    ]

    assert _status_from_queue_payload(json.dumps(payload), "1") == "FAILED"
