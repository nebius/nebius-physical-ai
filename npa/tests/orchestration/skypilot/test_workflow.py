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
from npa.orchestration.skypilot.launch_transaction import (
    FailureCategory,
    LaunchState,
    LaunchTransactionError,
    LaunchTransactionResult,
)


_REAL_RUN_LAUNCH_TRANSACTION = workflow_module.run_launch_transaction


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
    monkeypatch.setattr(
        workflow_module, "ensure_skypilot_version", lambda sky_bin: Path(sky_bin)
    )
    monkeypatch.setattr(bin_module, "CONFIG_PATH", tmp_path / "missing-config.yaml")
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.delenv("SKYPILOT_GLOBAL_CONFIG", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", raising=False)
    # Keep these command-shape tests independent of the host runner.  GitHub's
    # image includes kubectl while the local unit-test environment may not; the
    # Kubernetes probe/transaction suites exercise the installed-kubectl path.
    monkeypatch.setattr(workflow_module.shutil, "which", lambda _name: None)

    # Most tests in this module predate the transaction and isolate YAML/config/
    # streaming mechanics with one generic subprocess stub. Keep those unit seams
    # narrow; transaction/reconciliation behavior has dedicated tests below and in
    # test_launch_transaction.py.
    def legacy_transaction(**kwargs):
        result = LaunchTransactionResult(LaunchState.SUBMITTED, kwargs["logical_id"])
        try:
            launch_pair = kwargs["launch"]()
        except BaseException as exc:
            result.state = LaunchState.TERMINAL_FAILURE
            result.category = FailureCategory.UNKNOWN
            result.primary_error = str(exc)
            raise LaunchTransactionError(str(exc), result) from exc
        command_result = launch_pair[0]
        result.job_id = workflow_module._parse_job_id(
            f"{command_result.stdout}\n{command_result.stderr}"
        )
        result.launch_sequence = 1
        result.launch_result = launch_pair
        return result

    monkeypatch.setattr(workflow_module, "run_launch_transaction", legacy_transaction)


def test_submit_workflow_loads_yaml_applies_controller_and_calls_subprocess(
    monkeypatch, tmp_path
) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text(
        "name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8"
    )
    sky_bin = _fake_sky(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Job submitted, ID: 42\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = submit_workflow(
        yaml_path,
        "run-abc",
        isolated_config_dir=tmp_path / "sky-state",
        sky_bin=sky_bin,
    )

    assert result.status == "SUBMITTED"
    assert result.job_id == "42"
    assert calls[0][0] == [str(sky_bin), "status", "--refresh", "--output", "json"]
    cmd, kwargs = calls[1]
    assert cmd[:5] == [str(sky_bin), "jobs", "launch", "--name", "run-abc"]
    assert "--config" not in cmd
    assert "--detach-run" in cmd
    assert kwargs["env"]["HOME"] == str(tmp_path / "sky-state" / "home")
    assert kwargs["env"]["SKYPILOT_GLOBAL_CONFIG"] == result.log_paths["config"]
    config = yaml.safe_load(
        (
            tmp_path / "sky-state" / "submissions" / "run-abc" / "skypilot-config.yaml"
        ).read_text()
    )
    assert config["jobs"]["controller"]["resources"] == {
        "cloud": "kubernetes",
        "cpus": 2,
        "memory": 8,
        "autostop": False,
    }


def test_submit_workflow_strips_name_from_global_config(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text(
        "name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8"
    )
    global_config = tmp_path / "global.yaml"
    global_config.write_text(
        "name: human-readable-config\nkubernetes:\n  pod_config:\n    spec: {}\n",
        encoding="utf-8",
    )
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Job submitted, ID: 11\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = submit_workflow(
        yaml_path,
        "run-global-config",
        config_path=global_config,
        isolated_config_dir=tmp_path / "sky-state",
        sky_bin=sky_bin,
    )

    rendered = yaml.safe_load(
        Path(result.log_paths["config"]).read_text(encoding="utf-8")
    )
    assert "name" not in rendered
    assert "kubernetes" in rendered


def test_submit_workflow_runs_sky_from_stable_cwd(monkeypatch, tmp_path) -> None:
    """All sky invocations must run from a durable cwd so the auto-started
    API server daemon never inherits an ephemeral (later-deleted) directory."""
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text(
        "name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8"
    )
    sky_bin = _fake_sky(tmp_path)
    isolated = tmp_path / "sky-state"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Job submitted, ID: 7\n", stderr=""
        )

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


def test_submit_workflow_network_failure_raises_typed_error(
    monkeypatch, tmp_path
) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(
            cmd, 2, stdout="", stderr="network connection failed"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(
        SkyPilotSubmitError, match="sky jobs launch failed.*network connection failed"
    ):
        submit_workflow(
            yaml_path, "run-fail", isolated_config_dir=tmp_path / "sky", sky_bin=sky_bin
        )


def test_submit_workflow_auth_failure_raises_typed_error(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="Authentication failed: credentials expired"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SkyPilotSubmitError, match="auth failure.*credentials expired"):
        submit_workflow(
            yaml_path,
            "run-auth-fail",
            isolated_config_dir=tmp_path / "sky",
            sky_bin=sky_bin,
        )


def test_submit_workflow_yaml_parse_error_raises_typed_error(
    monkeypatch, tmp_path
) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: [unterminated\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        raise AssertionError("malformed YAML should fail before sky jobs launch")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SkyPilotSubmitError, match="workflow submission failed"):
        submit_workflow(
            yaml_path,
            "run-yaml-fail",
            isolated_config_dir=tmp_path / "sky",
            sky_bin=sky_bin,
        )


def test_submit_workflow_cleans_owned_temp_dir_on_timeout(
    monkeypatch, tmp_path
) -> None:
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


def test_submit_workflow_can_emit_nebius_controller_fallback(
    monkeypatch, tmp_path
) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text(
        "name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8"
    )
    sky_bin = _fake_sky(tmp_path)

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Job submitted, ID: 12\n", stderr=""
        )

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


def test_submit_workflow_passes_configured_secret_env_names(
    monkeypatch, tmp_path
) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        if _is_status_cmd(cmd):
            return _healthy_status(cmd)
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Job submitted, ID: 10\n", stderr=""
        )

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
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Job submitted, ID: 10\n", stderr=""
        )

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
    assert ["--secret", "AWS_ACCESS_KEY_ID"] == cmd[
        cmd.index("--secret") : cmd.index("--secret") + 2
    ]
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
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Job submitted, ID: 9", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    submit_workflow(
        yaml_path, "run-env", isolated_config_dir=tmp_path / "isolated", sky_bin=sky_bin
    )

    assert captured_env["HOME"] == str(tmp_path / "isolated" / "home")
    assert captured_env["SKY_RUNTIME_DIR"] == str(tmp_path / "isolated" / "sky-runtime")


def test_sky_environment_preserves_nebius_exec_auth_without_copying(
    monkeypatch, tmp_path
) -> None:
    from npa.orchestration.skypilot.cleanup import sky_environment

    operator_home = tmp_path / "operator"
    provider_config = operator_home / ".nebius"
    provider_config.mkdir(parents=True)
    (provider_config / "config.yaml").write_text("profiles: {}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(operator_home))

    isolated = tmp_path / "isolated"
    env = sky_environment(isolated)

    linked = isolated / "home" / ".nebius"
    assert linked.is_symlink()
    assert linked.resolve() == provider_config.resolve()
    assert env["HOME"] == str(isolated / "home")


def test_submit_workflow_require_controller_up_uses_canonical_preflight(
    monkeypatch, tmp_path
) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text("name: demo\n", encoding="utf-8")
    sky_bin = _fake_sky(tmp_path)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:4] == ["status", "--refresh", "--output"]:
            stdout = '[{"name": "sky-jobs-controller-abc123", "status": "UP"}]'
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Job submitted, ID: 77\n", stderr=""
        )

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


def test_submit_workflow_require_controller_up_blocks_missing_controller(
    monkeypatch, tmp_path
) -> None:
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


def test_submit_workflow_blocks_unhealthy_existing_jobs_controller(
    monkeypatch, tmp_path
) -> None:
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
            stdout = (
                '[{"name": "sky-jobs-controller-64ce57a0", "status": "AUTOSTOPPING"}]'
            )
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        raise AssertionError("launch should be blocked until controller is healthy")

    monkeypatch.setattr(workflow_module.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(
        SkyPilotSubmitError, match="sky-jobs-controller-64ce57a0=AUTOSTOPPING"
    ):
        submit_workflow(
            yaml_path,
            "run-autostop",
            sky_bin=sky_bin,
            controller_preflight_timeout=0,
            controller_preflight_interval=0,
        )

    assert calls == [[str(sky_bin), "status", "--refresh", "--output", "json"]]
    assert not owned_dir.exists()


def test_submit_workflow_controller_preflight_parses_warning_prefixed_json(
    monkeypatch, tmp_path
) -> None:
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


def test_workflow_status_treats_real_empty_queue_as_verified_absence(
    monkeypatch, tmp_path
) -> None:
    sky_bin = _fake_sky(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="sky.exceptions.ClusterNotUpError: No in-progress managed jobs.\n",
        ),
    )

    result = workflow_status("42", sky_bin=sky_bin)

    assert result.status == "ABSENT"
    assert result.error == ""


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('[{"job_id": 42, "status": "SUCCEEDED"}]', "SUCCEEDED"),
        ('harmless warning\n[{"job_id": 42, "status": "FAILED"}]', "FAILED"),
        ('[{"job_id": 42, "status": "RUNNING"}]\n[]', ""),
        ("diagnostic only", ""),
        ("", ""),
        ('{"unexpected": []}', ""),
    ],
)
def test_queue_status_parser_is_preamble_tolerant_but_never_guesses(
    output: str, expected: str
) -> None:
    assert _status_from_queue_payload(output, "42") == expected


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
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=payload, stderr=""
        )

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
    monkeypatch.setattr(workflow_module.subprocess, "run", _controller_status_run("UP"))
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


def test_missing_controller_timeout_does_not_advise_tearing_one_down(
    monkeypatch,
) -> None:
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


def test_optional_nebius_profile_failure_is_one_informational_fallback() -> None:
    messages: list[str] = []
    streamer = workflow_module._LaunchStreamer(
        messages.append, optional_nebius_profile=True
    )

    streamer._emit("Unable to create Nebius profile: provider helper unavailable")
    streamer._emit("Unable to create Nebius profile: provider helper unavailable")
    streamer._emit("Jobs controller launched successfully")

    assert (
        sum("optional SkyPilot Nebius provider-profile" in item for item in messages)
        == 1
    )
    assert not any(
        item.startswith("Unable to create Nebius profile") for item in messages
    )
    assert "Jobs controller launched successfully" in messages
    assert "Kubernetes-controller/context execution path" in messages[0]


def test_mandatory_nebius_profile_failure_is_not_suppressed() -> None:
    messages: list[str] = []
    streamer = workflow_module._LaunchStreamer(
        messages.append, optional_nebius_profile=False
    )

    streamer._emit("Unable to create Nebius profile: authentication failed")

    assert messages[0] == "Unable to create Nebius profile: authentication failed"


def test_nebius_profile_optional_only_for_verified_kubernetes_task_path() -> None:
    optional = workflow_module._nebius_profile_is_optional

    assert optional(
        [{"resources": {"cloud": "kubernetes"}}],
        controller_backend="kubernetes",
        infra="",
    )
    assert optional(
        [{"resources": {}}],
        controller_backend="kubernetes",
        infra="k8s/synthetic-context",
    )
    assert not optional(
        [{"resources": {"cloud": "nebius"}}],
        controller_backend="kubernetes",
        infra="",
    )
    assert not optional(
        [{"resources": {}}],
        controller_backend="kubernetes",
        infra="",
    )
    assert not optional(
        [{"resources": {"cloud": "kubernetes"}}],
        controller_backend="nebius",
        infra="",
    )


def test_launch_failure_pod_config_kubernetes_bug_gets_a_fix_hint() -> None:
    """The SkyPilot/kubernetes pod_config bug retries forever; surface it once with a fix."""
    detail = "RuntimeError: Invalid pod_config: ... No module named 'kubernetes.client.models.dict[str, str]'"
    result = subprocess.CompletedProcess(
        args=["sky", "jobs", "launch"], returncode=1, stdout="", stderr=detail
    )

    message = workflow_module._format_submit_error(["sky", "jobs", "launch"], result)

    assert "kubernetes-client incompatibility" in message
    assert "npa skypilot uninstall && npa skypilot bootstrap" in message
    assert "retries it indefinitely" in message


def test_submission_dir_and_secret_files_are_owner_only(tmp_path) -> None:
    """The submission dir + its secret-bearing files must not be world-readable.

    The rendered task YAML / generated SkyPilot config can carry a registry IAM
    token (SKYPILOT_DOCKER_PASSWORD) and S3 creds; write_text/mkdir honor the
    umask, so submit tightens them explicitly (security bug 9).
    """
    import shutil

    owned = workflow_module._submission_dir("run-owner", None)
    try:
        assert (owned.stat().st_mode & 0o077) == 0
    finally:
        shutil.rmtree(owned, ignore_errors=True)

    isolated = tmp_path / "sky-config"
    scoped = workflow_module._submission_dir("run-scoped", isolated)
    assert (scoped.stat().st_mode & 0o077) == 0

    secret_file = scoped / "workflow.yaml"
    secret_file.write_text("envs:\n  SKYPILOT_DOCKER_PASSWORD: tok\n")
    secret_file.chmod(0o644)
    workflow_module._chmod_owner_only(secret_file)
    assert (secret_file.stat().st_mode & 0o077) == 0


def test_pod_config_classifier_ignores_unrelated_errors() -> None:
    assert workflow_module._looks_like_pod_config_error("some random error") is False
    assert (
        workflow_module._looks_like_pod_config_error("Invalid pod_config: bad") is True
    )
    assert (
        workflow_module._looks_like_pod_config_error(
            "No module named 'kubernetes.client.models.dict[str, str]'"
        )
        is True
    )


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
        return_value=mocker.Mock(
            sky_bin="sky", isolated_config_dir=None, global_config_path=None
        ),
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


def test_exact_managed_job_lookup_preserves_absent_vs_unavailable(
    monkeypatch, tmp_path
) -> None:
    from npa.orchestration.skypilot.workflow import lookup_managed_job

    sky_bin = _fake_sky(tmp_path)
    responses = iter(
        [
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    [
                        {
                            "job_id": 41,
                            "job_name": "exact-run-other",
                            "task_id": 0,
                            "status": "RUNNING",
                        }
                    ]
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                [], 1, stdout="", stderr="fixture provider unavailable"
            ),
        ]
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(responses))

    absent = lookup_managed_job("exact-run", sky_bin=sky_bin)
    unavailable = lookup_managed_job("exact-run", sky_bin=sky_bin)

    assert absent.outcome == "absent"
    assert unavailable.outcome == "unavailable"
    assert "provider unavailable" in unavailable.error


def test_exact_managed_job_lookup_accepts_real_0_12_2_empty_queue(
    monkeypatch, tmp_path
) -> None:
    from npa.orchestration.skypilot.workflow import lookup_managed_job

    sky_bin = _fake_sky(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="sky.exceptions.ClusterNotUpError: No in-progress managed jobs.\n",
        ),
    )

    evidence = lookup_managed_job("exact-run", sky_bin=sky_bin)

    assert evidence.outcome == "absent"


def test_exact_managed_job_lookup_refuses_ambiguous_name_without_immutable_id(
    monkeypatch, tmp_path
) -> None:
    from npa.orchestration.skypilot.workflow import lookup_managed_job

    sky_bin = _fake_sky(tmp_path)
    payload = [
        {
            "job_id": 42,
            "job_name": "exact-run",
            "task_id": 0,
            "task_name": "annotate",
            "status": "SUCCEEDED",
        },
        {
            "job_id": 43,
            "job_name": "exact-run",
            "task_id": 0,
            "task_name": "annotate",
            "status": "RUNNING",
            "retry_count": 2,
        },
        {"job_id": 44, "job_name": "exact-run-nested", "status": "RUNNING"},
    ]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    evidence = lookup_managed_job("exact-run", sky_bin=sky_bin)

    assert evidence.outcome == "unavailable"
    assert evidence.job_id == ""
    assert "ambiguous" in evidence.error
    assert "42, 43" in evidence.error

    exact = lookup_managed_job("exact-run", job_id="43", sky_bin=sky_bin)
    assert exact.outcome == "found"
    assert exact.job_id == "43"
    assert exact.status == "RUNNING"
    assert exact.task_rows[0]["retry_count"] == 2


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
        workflow_mod._verified_job_id(
            "164", "wave-x", env={}, sky_executable="sky", cwd=None
        )
        == "164"
    )


def test_verified_job_id_falls_back_when_the_lookup_fails(mocker) -> None:
    from npa.orchestration.skypilot import workflow as workflow_mod

    mocker.patch.object(workflow_mod.subprocess, "run", side_effect=OSError("no sky"))
    assert (
        workflow_mod._verified_job_id(
            "163", "wave-x", env={}, sky_executable="sky", cwd=None
        )
        == "163"
    )


def test_verified_job_id_can_be_disabled(monkeypatch, mocker) -> None:
    """The extra queue round-trip is opt-out for latency-sensitive callers."""

    from npa.orchestration.skypilot import workflow as workflow_mod

    run = mocker.patch.object(workflow_mod.subprocess, "run")
    monkeypatch.setenv("NPA_SKYPILOT_VERIFY_JOB_ID", "0")
    assert (
        workflow_mod._verified_job_id(
            "163", "wave-x", env={}, sky_executable="sky", cwd=None
        )
        == "163"
    )
    run.assert_not_called()


def test_submit_streams_launch_output_and_names_a_known_hang(
    tmp_path, monkeypatch
) -> None:
    """A retrying controller must not look like a silent hang.

    ``sky jobs launch`` can retry for the full submit timeout without exiting, so
    buffering its output to a pipe leaves the operator with a blank terminal.
    """

    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text(
        "name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8"
    )
    sky_bin = _fake_sky(tmp_path)
    lines: list[str] = []

    def fake_run(cmd, **kwargs):
        if "launch" in cmd:
            stdout = kwargs.get("stdout")
            # Real sky writes progress as it goes; the streamer tails the file.
            stdout.write("Launching managed job 'demo'\n")
            stdout.write(
                "Invalid pod_config. Details: Validation error in metadata.labels: "
                "No module named 'kubernetes.client.models.dict[str, str]'\n"
            )
            stdout.flush()
            return subprocess.CompletedProcess(cmd, 1, stdout=None, stderr=None)
        return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SkyPilotSubmitError) as excinfo:
        submit_workflow(yaml_path, "run-stream", sky_bin=sky_bin, echo=lines.append)

    assert "Launching managed job 'demo'" in lines
    assert any("kubernetes_client_pod_config" in line for line in lines)
    assert "npa skypilot bootstrap" in str(excinfo.value)


def test_submit_can_run_without_streaming(tmp_path, monkeypatch) -> None:
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text(
        "name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8"
    )
    sky_bin = _fake_sky(tmp_path)
    seen: list[object] = []

    def fake_run(cmd, **kwargs):
        if "launch" in cmd:
            seen.append(kwargs.get("stdout"))
            return subprocess.CompletedProcess(cmd, 0, stdout="Job ID: 7", stderr="")
        if "queue" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps([{"job_id": 7, "job_name": "run-plain"}]),
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = submit_workflow(
        yaml_path, "run-plain", sky_bin=sky_bin, stream_output=False
    )

    assert result.status == "SUBMITTED"
    assert seen == [subprocess.PIPE]


def test_submit_transaction_recovers_controller_creation_refusal(
    monkeypatch, tmp_path
) -> None:
    """Hermetic reproduction of the first-PAIDF-submit TOCTOU boundary."""

    from npa.orchestration.skypilot.launch_transaction import (
        EvidenceState,
        ProbeObservation,
        RecoveryPolicy,
        StabilityPolicy,
    )

    monkeypatch.setattr(
        workflow_module, "run_launch_transaction", _REAL_RUN_LAUNCH_TRANSACTION
    )
    yaml_path = tmp_path / "workflow.yaml"
    yaml_path.write_text(
        "name: demo\nresources:\n  cloud: kubernetes\n", encoding="utf-8"
    )
    sky_bin = _fake_sky(tmp_path)
    clock = {"now": 0.0}
    launch_calls = 0
    exact_job = ""

    def now() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        clock["now"] += max(seconds, 0.1)

    def probe() -> ProbeObservation:
        return ProbeObservation(
            EvidenceState.READY,
            observed_at=f"t{clock['now']}",
            monotonic_at=clock["now"],
        )

    def fake_run(cmd, **_kwargs):
        nonlocal launch_calls, exact_job
        if _is_status_cmd(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
        if cmd[1:3] == ["jobs", "queue"]:
            rows = (
                [
                    {
                        "job_id": int(exact_job),
                        "job_name": "paidf-wave-1",
                        "status": "PENDING",
                    }
                ]
                if exact_job
                else []
            )
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(rows), stderr=""
            )
        if cmd[1:3] == ["jobs", "launch"]:
            launch_calls += 1
            if launch_calls == 1:
                return subprocess.CompletedProcess(
                    cmd, 1, stdout="", stderr="dial tcp: connection refused"
                )
            exact_job = "501"
            return subprocess.CompletedProcess(
                cmd, 0, stdout="Job submitted, ID: 501\n", stderr=""
            )
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = submit_workflow(
        yaml_path,
        "paidf-wave-1",
        isolated_config_dir=tmp_path / "sky-state",
        sky_bin=sky_bin,
        infra="k8s/exact-context",
        stream_output=False,
        stability_probe=probe,
        stability_policy=StabilityPolicy(2, 0, 0, 1),
        recovery_policy=RecoveryPolicy(10, 1, 1, 2, 0),
        transaction_clock=now,
        transaction_sleeper=sleep,
        transaction_random=lambda: 0.5,
        launch_lock_root=tmp_path / "locks",
    )
    assert result.status == "SUBMITTED"
    assert result.job_id == "501"
    assert launch_calls == 2
    assert result.launch_transaction["launch_sequence"] == 2
    assert (
        result.launch_transaction["recovery_decision"]
        == "submitted_and_reconciled"
    )
    assert result.launch_transaction["controller"]["state"] == "absent"
