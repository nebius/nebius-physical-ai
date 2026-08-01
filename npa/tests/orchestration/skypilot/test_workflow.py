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
    _stable_sky_cwd,
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


def test_submit_workflow_strips_name_from_global_config(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8")
    global_config = tmp_path / "global.yaml"
    global_config.write_text(
        "name: human-readable-config\nkubernetes:\n  pod_config:\n    spec: {}\n",
        encoding="utf-8",
    )
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Job submitted, ID: 11\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = submit_workflow(
        yaml_path,
        "run-global-config",
        config_path=global_config,
        isolated_config_dir=tmp_path / "sky-state",
        sky_bin=sky_bin,
    )

    rendered = yaml.safe_load(Path(result.log_paths["config"]).read_text(encoding="utf-8"))
    assert "name" not in rendered
    assert "kubernetes" in rendered


def test_submit_workflow_runs_sky_from_stable_cwd(monkeypatch, tmp_path) -> None:
    """All sky invocations must run from a durable cwd so the auto-started
    API server daemon never inherits an ephemeral (later-deleted) directory."""
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    isolated = tmp_path / "sky-state"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Job submitted, ID: 7\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    submit_workflow(yaml_path, "run-cwd", isolated_config_dir=isolated, sky_bin=sky_bin)

    assert calls, "expected sky to be invoked"
    for cmd, kwargs in calls:
        cwd = kwargs.get("cwd")
        assert cwd, f"sky command missing stable cwd: {cmd}"
        assert Path(cwd).is_dir(), f"sky cwd is not an existing directory: {cwd}"
        assert cwd == str(isolated)


def test_stable_sky_cwd_falls_back_to_home_when_dir_missing(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    assert _stable_sky_cwd(missing) == str(Path.home())
    assert _stable_sky_cwd(None) == str(Path.home())


def test_stable_sky_cwd_prefers_existing_isolated_dir(tmp_path) -> None:
    isolated = tmp_path / "sky-state"
    isolated.mkdir()
    assert _stable_sky_cwd(isolated) == str(isolated)


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


def test_submit_workflow_secrets_can_come_from_extra_env(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    calls = []
    captured_env = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs["env"])
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Job submitted, ID: 10\n", stderr="")

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)

    submit_workflow(
        yaml_path,
        "run-config-secrets",
        isolated_config_dir=tmp_path / "sky-state",
        sky_bin=sky_bin,
        infra="k8s/npa-rtxpro-mk8s",
        secret_envs=("AWS_ACCESS_KEY_ID",),
        extra_env={"AWS_ACCESS_KEY_ID": "from-config"},
    )

    cmd = calls[0]
    assert "--infra" in cmd
    assert cmd[cmd.index("--infra") + 1] == "k8s/npa-rtxpro-mk8s"
    assert ["--secret", "AWS_ACCESS_KEY_ID"] == cmd[cmd.index("--secret") : cmd.index("--secret") + 2]
    assert "from-config" not in cmd
    assert captured_env["AWS_ACCESS_KEY_ID"] == "from-config"


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


# --- per-task timelines (JobGroup / pipeline evidence) -----------------------

_QUEUE_JSON = """
[
  {"job_id": 75, "task_id": 0, "task_name": "caption-shard-a", "job_name": "wf-01-caption",
   "status": "SUCCEEDED", "submitted_at": 1785297417.2239287, "start_at": null,
   "end_at": 1785297483.0891774, "is_job_group": false, "execution": null,
   "cluster_name_on_cloud": null, "current_cluster_name": null},
  {"job_id": 75, "task_id": 2, "task_name": "caption-shard-c", "job_name": "wf-01-caption",
   "status": "SUCCEEDED", "submitted_at": 1785297417.2361672, "start_at": null,
   "end_at": 1785297477.9160478, "is_job_group": false, "execution": null},
  {"job_id": 75, "task_id": 1, "task_name": "caption-shard-b", "job_name": "wf-01-caption",
   "status": "RUNNING", "submitted_at": 1785297417.229656, "start_at": null,
   "end_at": null, "is_job_group": false, "execution": null},
  {"job_id": 76, "task_id": 0, "task_name": "wf-02-aggregate", "job_name": "wf-02-aggregate",
   "status": "SUCCEEDED", "submitted_at": 1785297577.643265, "start_at": null,
   "end_at": 1785297631.8969326, "is_job_group": false, "execution": null}
]
"""


def test_parse_task_statuses_returns_ordered_rows_for_one_job() -> None:
    from npa.orchestration.skypilot.workflow import parse_task_statuses

    rows = parse_task_statuses(_QUEUE_JSON, "75")

    assert [row["task_id"] for row in rows] == [0, 1, 2]
    assert [row["task_name"] for row in rows] == [
        "caption-shard-a",
        "caption-shard-b",
        "caption-shard-c",
    ]
    assert {row["job_id"] for row in rows} == {"75"}
    assert rows[0]["status"] == "SUCCEEDED"
    assert rows[1]["status"] == "RUNNING"
    assert rows[0]["submitted_at"] == 1785297417.2239287
    assert rows[0]["end_at"] == 1785297483.0891774
    assert rows[1]["end_at"] is None


def test_parse_task_statuses_isolates_other_jobs_and_bad_payloads() -> None:
    from npa.orchestration.skypilot.workflow import parse_task_statuses

    assert [row["task_name"] for row in parse_task_statuses(_QUEUE_JSON, "76")] == [
        "wf-02-aggregate"
    ]
    assert parse_task_statuses(_QUEUE_JSON, "999") == []
    assert parse_task_statuses("not json", "75") == []
    assert parse_task_statuses("", "75") == []


def test_workflow_task_statuses_returns_empty_on_command_failure(mocker) -> None:
    import subprocess

    from npa.orchestration.skypilot import workflow as workflow_mod

    mocker.patch.object(
        workflow_mod,
        "resolve_config",
        return_value=mocker.Mock(sky_bin="sky", isolated_config_dir=None, global_config_path=None),
    )
    mocker.patch.object(workflow_mod, "ensure_skypilot_version", return_value="sky")
    mocker.patch.object(
        workflow_mod.subprocess,
        "run",
        return_value=subprocess.CompletedProcess(["sky"], 1, stdout="", stderr="boom"),
    )

    assert workflow_mod.workflow_task_statuses("75") == []


def test_parse_job_ids_by_name_returns_newest_first() -> None:
    from npa.orchestration.skypilot.workflow import parse_job_ids_by_name

    payload = """
    [
      {"job_id": 140, "job_name": "wave-01", "task_name": "a", "status": "CANCELLED"},
      {"job_id": 141, "job_name": "wave-01", "task_name": "a", "status": "RUNNING"},
      {"job_id": 141, "job_name": "wave-01", "task_name": "b", "status": "RUNNING"},
      {"job_id": 142, "job_name": "other", "task_name": "a", "status": "RUNNING"}
    ]
    """
    assert parse_job_ids_by_name(payload, "wave-01") == ["141", "140"]
    assert parse_job_ids_by_name(payload, "other") == ["142"]
    assert parse_job_ids_by_name(payload, "missing") == []
    assert parse_job_ids_by_name("not json", "wave-01") == []


def test_verified_job_id_prefers_the_name_lookup(mocker) -> None:
    """A stale scraped id must not win over the queue's view of the job name."""

    import subprocess

    from npa.orchestration.skypilot import workflow as workflow_mod

    queue = """
    [{"job_id": 164, "job_name": "wave-x", "task_name": "a", "status": "RUNNING"}]
    """
    mocker.patch.object(
        workflow_mod.subprocess,
        "run",
        return_value=subprocess.CompletedProcess(["sky"], 0, stdout=queue, stderr=""),
    )
    verified = workflow_mod._verified_job_id(
        "163", "wave-x", env={}, sky_executable="sky", cwd=None
    )
    assert verified == "164"

    # When the queue agrees, the parsed id is kept.
    assert (
        workflow_mod._verified_job_id("164", "wave-x", env={}, sky_executable="sky", cwd=None)
        == "164"
    )


def test_verified_job_id_falls_back_when_the_lookup_fails(mocker) -> None:
    from npa.orchestration.skypilot import workflow as workflow_mod

    mocker.patch.object(workflow_mod.subprocess, "run", side_effect=OSError("no sky"))
    assert (
        workflow_mod._verified_job_id("163", "wave-x", env={}, sky_executable="sky", cwd=None)
        == "163"
    )


def test_verified_job_id_can_be_disabled(monkeypatch, mocker) -> None:
    """The extra queue round-trip is opt-out for latency-sensitive callers."""

    from npa.orchestration.skypilot import workflow as workflow_mod

    run = mocker.patch.object(workflow_mod.subprocess, "run")
    monkeypatch.setenv("NPA_SKYPILOT_VERIFY_JOB_ID", "0")
    assert (
        workflow_mod._verified_job_id("163", "wave-x", env={}, sky_executable="sky", cwd=None)
        == "163"
    )
    run.assert_not_called()
