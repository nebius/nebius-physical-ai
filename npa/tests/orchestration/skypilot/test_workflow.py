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


def _controller_status_run(status: str):
    payload = json.dumps(
        {"clusters": [{"name": "sky-jobs-controller-abc123", "status": status}]}
    )

    def _run(cmd, **_kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=payload, stderr="")

    return _run


def test_wait_for_controller_proceeds_when_stopped(monkeypatch) -> None:
    # A STOPPED (autostopped) controller must not block launch: `sky jobs launch`
    # restarts it. Regression for the stale-controller submit block.
    monkeypatch.setattr(
        workflow_module.subprocess, "run", _controller_status_run("STOPPED")
    )
    # Returns (no raise) even with a tiny timeout because STOPPED is ready.
    workflow_module._wait_for_healthy_jobs_controller(
        "sky", env={}, timeout=0, interval=0.01
    )


def test_wait_for_controller_proceeds_when_up(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow_module.subprocess, "run", _controller_status_run("UP")
    )
    workflow_module._wait_for_healthy_jobs_controller(
        "sky", env={}, timeout=0, interval=0.01
    )


def test_wait_for_controller_blocks_on_transient_init(monkeypatch) -> None:
    # A transient INIT/provisioning controller is still treated as not-ready.
    monkeypatch.setattr(
        workflow_module.subprocess, "run", _controller_status_run("INIT")
    )
    with pytest.raises(SkyPilotSubmitError) as exc:
        workflow_module._wait_for_healthy_jobs_controller(
            "sky", env={}, timeout=0, interval=0.01
        )
    assert "INIT" in str(exc.value)
    assert "sky down" in str(exc.value)


def _failing_status_run(stderr: str):
    def _run(cmd, **_kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr=stderr
        )

    return _run


def test_stale_controller_kubeconfig_failure_explains_the_fix(monkeypatch) -> None:
    """A controller cached against a missing kubeconfig must say how to recover.

    Regression: submitting against a stale `sky-jobs-controller-*` from another
    NPA setup only produced a raw `sky status` stack trace, with nothing about
    purging the controller or repointing KUBECONFIG.
    """
    stderr = (
        "RuntimeError: Failed to load kubeconfig "
        "/home/op/.npa/clusters/npa-rtxpro-mk8s/kubeconfig: "
        "No such file or directory"
    )
    monkeypatch.setattr(workflow_module.subprocess, "run", _failing_status_run(stderr))

    with pytest.raises(SkyPilotSubmitError) as exc:
        workflow_module._wait_for_healthy_jobs_controller(
            "sky", env={}, timeout=0, interval=0.01
        )

    message = str(exc.value)
    assert "controller health check failed" in message
    # Names the exact kubeconfig it is stuck on ...
    assert "/home/op/.npa/clusters/npa-rtxpro-mk8s/kubeconfig" in message
    # ... and every recovery lever. `sky status` in SkyPilot 0.12 has no `--all`
    # flag (it errors with "Did you mean --all-users?"), so the remedy must use
    # `sky status -r` / plain `sky status`.
    assert "sky status -r" in message
    assert "sky status --all" not in message
    assert "sky down sky-jobs-controller-" in message
    assert "provision-if-absent" in message
    assert "--infra k8s/<context>" in message


def test_unrelated_status_failure_keeps_the_raw_error(monkeypatch) -> None:
    """Don't bolt stale-controller advice onto unrelated failures."""
    monkeypatch.setattr(
        workflow_module.subprocess,
        "run",
        _failing_status_run("error: quota exceeded for account"),
    )

    with pytest.raises(SkyPilotSubmitError) as exc:
        workflow_module._wait_for_healthy_jobs_controller(
            "sky", env={}, timeout=0, interval=0.01
        )

    message = str(exc.value)
    assert "quota exceeded" in message
    assert "sky down" not in message


def test_controller_health_remedy_without_a_kubeconfig_path() -> None:
    remedy = workflow_module._controller_health_remedy(
        "kubernetes.config.config_exception: Invalid kube-context specified"
    )

    assert "KUBECONFIG" in remedy
    assert "sky down sky-jobs-controller-" in remedy


def test_unhealthy_controller_timeout_names_the_controller(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow_module.subprocess, "run", _controller_status_run("INIT")
    )

    with pytest.raises(SkyPilotSubmitError) as exc:
        workflow_module._wait_for_healthy_jobs_controller(
            "sky", env={}, timeout=0, interval=0.01
        )

    assert "sky down sky-jobs-controller-abc123" in str(exc.value)


def test_unhealthy_controller_timeout_names_the_unhealthy_one(monkeypatch) -> None:
    """With several cached controllers, don't send `sky down` at a healthy one."""
    payload = json.dumps(
        {
            "clusters": [
                {"name": "sky-jobs-controller-healthy", "status": "UP"},
                {"name": "sky-jobs-controller-stuck", "status": "INIT"},
            ]
        }
    )
    monkeypatch.setattr(
        workflow_module.subprocess,
        "run",
        lambda cmd, **_k: subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=payload, stderr=""
        ),
    )

    with pytest.raises(SkyPilotSubmitError) as exc:
        workflow_module._wait_for_healthy_jobs_controller(
            "sky", env={}, timeout=0, interval=0.01
        )

    message = str(exc.value)
    assert "sky down sky-jobs-controller-stuck" in message
    assert "sky-jobs-controller-healthy`" not in message


def test_missing_controller_timeout_does_not_advise_tearing_one_down(monkeypatch) -> None:
    """With no controller at all, teardown advice is nonsense — a launch creates it."""
    payload = json.dumps({"clusters": []})
    monkeypatch.setattr(
        workflow_module.subprocess,
        "run",
        lambda cmd, **_k: subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=payload, stderr=""
        ),
    )

    with pytest.raises(SkyPilotSubmitError) as exc:
        workflow_module._wait_for_healthy_jobs_controller(
            "sky", env={}, timeout=0, interval=0.01, require_existing=True
        )

    message = str(exc.value)
    assert "no jobs-controller found" in message
    assert "sky down" not in message
    assert "<controller-name>" not in message


def test_generic_connection_errors_alone_are_not_stale_controller_evidence() -> None:
    """A dead API server also says "connection refused"; don't advise a purge."""
    assert (
        workflow_module._controller_health_remedy(
            "urllib3.exceptions.NewConnectionError: [Errno 111] Connection refused"
        )
        == ""
    )
    assert (
        workflow_module._controller_health_remedy(
            "FileNotFoundError: [Errno 2] No such file or directory: '/home/op/.sky/config.yaml'"
        )
        == ""
    )
    # The same phrase alongside a controller/context signal still gets the remedy.
    assert "sky down sky-jobs-controller-" in workflow_module._controller_health_remedy(
        "Cached cluster sky-jobs-controller-64ce57a0: connection refused"
    )


def test_launch_failure_on_a_cached_controller_gets_the_same_remedy() -> None:
    """`sky status --refresh` exits 0 while only *warning* about dead clusters.

    Reproduced live: the health check passes and `sky jobs launch` is what fails,
    with CachedClusterUnavailable naming an NPA cluster kubeconfig that is gone.
    """
    detail = (
        "sky.exceptions.CachedClusterUnavailable: Cached jobs controller cluster "
        "sky-jobs-controller-64ce57a0 cannot be refreshed.\n"
        "Reason: ValueError: Failed to load Kubernetes configuration for "
        "'npa-rtxpro-mk8s'. Please check if your kubeconfig file exists at "
        "/home/op/.npa/clusters/npa-rtxpro-mk8s/kubeconfig and is valid."
    )
    result = subprocess.CompletedProcess(
        args=["sky", "jobs", "launch"], returncode=1, stdout="", stderr=detail
    )

    message = workflow_module._format_submit_error(["sky", "jobs", "launch"], result)

    assert "sky jobs launch failed" in message
    assert "/home/op/.npa/clusters/npa-rtxpro-mk8s/kubeconfig" in message
    assert "sky down sky-jobs-controller-" in message
    assert "--infra k8s/<context>" in message


def test_launch_failure_unrelated_to_the_controller_stays_raw() -> None:
    result = subprocess.CompletedProcess(
        args=["sky", "jobs", "launch"],
        returncode=1,
        stdout="",
        stderr="ValueError: task yaml is invalid",
    )

    message = workflow_module._format_submit_error(["sky", "jobs", "launch"], result)

    assert "task yaml is invalid" in message
    assert "sky down" not in message


def test_referenced_kubeconfig_path_prefers_the_real_path() -> None:
    """SkyPilot says "kubeconfig" several times before naming the file.

    Regression from a live run: the remedy printed "the referenced kubeconfig is
    kubeconfig" because the first regex match was the bare word.
    """
    detail = (
        "ValueError: Failed to load Kubernetes configuration for 'npa-rtxpro-mk8s'. "
        "Please check if your kubeconfig file exists at "
        "/home/op/.npa/clusters/npa-rtxpro-mk8s/kubeconfig and is valid.\n"
        "Invalid kube-config file. No configuration found.\n"
        "Hint: Kubernetes attempted to query the current-context set in kubeconfig."
    )

    assert (
        workflow_module._referenced_kubeconfig_path(detail)
        == "/home/op/.npa/clusters/npa-rtxpro-mk8s/kubeconfig"
    )
    assert workflow_module._referenced_kubeconfig_path("no paths here") == ""

    remedy = workflow_module._controller_health_remedy(detail)
    assert "referenced kubeconfig is /home/op/.npa/clusters" in remedy
    assert "referenced kubeconfig is kubeconfig" not in remedy
