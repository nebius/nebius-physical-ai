from __future__ import annotations

import subprocess

import yaml

from npa.orchestration.skypilot.workflow import submit_workflow, workflow_status


def test_submit_workflow_loads_yaml_applies_controller_and_calls_subprocess(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="Job submitted, ID: 42\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = submit_workflow(yaml_path, "run-abc", isolated_config_dir=tmp_path / "sky")

    assert result.status == "SUBMITTED"
    assert result.job_id == "42"
    cmd, kwargs = calls[0]
    assert cmd[:5] == ["sky", "jobs", "launch", "--name", "run-abc"]
    assert "--config" not in cmd
    assert "--detach-run" in cmd
    assert kwargs["env"]["HOME"] == str(tmp_path / "sky" / "home")
    assert kwargs["env"]["SKYPILOT_GLOBAL_CONFIG"] == result.log_paths["config"]
    config = yaml.safe_load((tmp_path / "sky" / "submissions" / "run-abc" / "skypilot-config.yaml").read_text())
    assert config["jobs"]["controller"]["resources"]["instance_type"] == "cpu-e2_2vcpu-8gb"
    assert config["jobs"]["controller"]["resources"]["autostop"]["down"] is False


def test_submit_workflow_failure_returns_workflow_result(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="failed")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = submit_workflow(yaml_path, "run-fail", isolated_config_dir=tmp_path / "sky")

    assert result.status == "FAILED_SUBMIT"
    assert result.returncode == 2
    assert result.error == "failed"


def test_submit_workflow_honors_isolated_config_dir(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout="Job submitted, ID: 9", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    submit_workflow(yaml_path, "run-env", isolated_config_dir=tmp_path / "isolated")

    assert captured_env["HOME"] == str(tmp_path / "isolated" / "home")
    assert captured_env["SKY_RUNTIME_DIR"] == str(tmp_path / "isolated" / "sky-runtime")


def test_workflow_status_reads_json_queue(monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        stdout = '[{"job_id": 42, "name": "run", "status": "SUCCEEDED"}]'
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = workflow_status("42")

    assert result.status == "SUCCEEDED"
    assert result.job_id == "42"
