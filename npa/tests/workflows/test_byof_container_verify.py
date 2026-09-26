# npa: publication-enforcement=libero
from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "npa" / "scripts" / "run_byof_container_verify.py"
YAML_PATH = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-container-smoke-rtxpro.yaml"
)
LIBERO_YAML_PATH = YAML_PATH.with_name("byof-solution-smoke-libero-b200-gpu.yaml")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_byof_container_verify", SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_robotwin_real_wrapper_preserves_the_strict_file_and_environment_bridge(
    monkeypatch, tmp_path
):
    from npa.orchestration.npa_workflow.robotwin_preflight import (
        validate_confidential_submit_bridge,
    )
    from tests.orchestration.skypilot.test_workflow import _robotwin_bridge_fixture

    module = _load_module()
    _, outer, _, _, _ = _robotwin_bridge_fixture(monkeypatch, tmp_path)
    authorization = outer.authorization
    spec = importlib.util.spec_from_file_location(
        "robotwin_wrapper_bridge_test", ROOT / "npa/scripts/run_byof_repo.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "synthetic-bridge-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-bridge-secret")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.eu-north1.nebius.cloud")
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/synthetic-host-bin")
    environment = runner._authorized_live_env(
        authorization,
        {"report_sha256": "b" * 64, "archives_scanned": 2},
        project=authorization.project,
        image=authorization.bootstrap_image,
    )
    workflow = yaml.safe_load(
        (ROOT / "workflows/testing/byof-robotwin.yaml").read_text()
    )
    checked = []

    class BoundaryChecked(Exception):
        pass

    def bootstrap(_directory, control):
        assert control["HOME"] == environment["HOME"]
        assert control["PATH"] == environment["PATH"]
        return "/synthetic/sky"

    def submit(path, _run_id, **kwargs):
        documents = module._load_yaml_documents(path)
        checked.append(
            validate_confidential_submit_bridge(
                kwargs["robotwin_submit_context"],
                documents=documents,
                infra=kwargs["infra"],
                config_path=kwargs["config_path"],
                secret_envs=kwargs["secret_envs"],
                extra_env=kwargs["extra_env"],
                execution_target=kwargs["execution_target"],
                execution_report=kwargs["execution_preflight_report"],
            )
        )
        assert len(documents) == 2
        assert not {"HOME", "PATH", "LANG", "LC_ALL"} & kwargs["extra_env"].keys()
        raise BoundaryChecked

    monkeypatch.setattr(module, "_bootstrap_robotwin_sky", bootstrap)
    monkeypatch.setattr(module, "submit_workflow", submit)
    monkeypatch.setattr(module, "stop_isolated_api", Mock())
    monkeypatch.setattr(module, "install_teardown_signal_handlers", lambda *_args: {})
    monkeypatch.setattr(module, "restore_signal_handlers", lambda *_args: None)
    with pytest.raises(BoundaryChecked):
        module.run_authorized_robotwin(
            [
                "--yaml",
                str(
                    ROOT
                    / "npa/src/npa/workflows/byof/profiles"
                    / (workflow["config"]["resource_profile_yaml"] + ".yaml")
                ),
                "--solution-name",
                "robotwin",
                "--smoke-command",
                workflow["config"]["smoke_command"],
                "--capability-name",
                workflow["config"]["capability_name"],
                "--smoke-artifact-name",
                "robotwin-smoke.json",
            ],
            authorization=authorization,
            environment=environment,
        )
    assert checked == [authorization]


@pytest.fixture
def inert_wrapper(monkeypatch):
    """Replace every wrapper filesystem, process and resource effect with a mock."""
    module = _load_module()
    effects = {}
    for name, result in {
        "render_workflow": [{"name": "synthetic"}],
        "_write_yaml_documents": None,
        "preflight_output_storage": None,
        "_ensure_infra_enabled": None,
        "resolve_sky_bin": "/synthetic/sky",
        "_bootstrap_robotwin_sky": "/synthetic/sky",
        "restore_signal_handlers": None,
        "install_teardown_signal_handlers": {},
    }.items():
        effects[name] = Mock(return_value=result)
        monkeypatch.setattr(module, name, effects[name])
    effects["freshness"] = Mock()
    monkeypatch.setattr(
        module, "require_customer_authorization_fresh", effects["freshness"]
    )
    monkeypatch.setattr(
        module, "_normalize_kubeconfig_current_context", lambda _p, values=None: values
    )
    for name, target, attribute, result in (
        ("temp", module.tempfile, "mkdtemp", "/synthetic-private/run"),
        ("remove", module.shutil, "rmtree", None),
    ):
        effects[name] = Mock(return_value=result)
        monkeypatch.setattr(target, attribute, effects[name])
    effects["mkdir"] = Mock()

    def mkdir(path, *args, **kwargs):
        return effects["mkdir"](path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(
        module.subprocess, "run", Mock(side_effect=AssertionError("real subprocess"))
    )
    return module, effects


def _inert_authorized_context(module, monkeypatch):
    authorization = SimpleNamespace(
        run_id="synthetic-run",
        inner_launch_id="robotwin-inner-synthetic",
        output_root="s3://synthetic-bucket/output",
        skypilot_config_source="/synthetic-private/authorized-config",
        kubeconfig_source="/synthetic-private/kubeconfig",
        kubernetes_context="synthetic-context",
        project="synthetic-project",
        customer_authorization_expires_at="2099-01-01T00:00:00Z",
    )
    context = SimpleNamespace(authorization=authorization)
    monkeypatch.setattr(module, "prepare_inner_submit", Mock(return_value=context))
    environment = {
        module.CHILD_IMAGE_ENV: "example.invalid/robotwin@sha256:" + "a" * 64,
        "HOME": "/synthetic-home",
        "PATH": "/bin",
        "AWS_ACCESS_KEY_ID": "synthetic-access",
        "AWS_SECRET_ACCESS_KEY": "synthetic-secret",
        "NPA_SKYPILOT_ISOLATED_CONFIG_DIR": "/synthetic-shared-state",
    }
    return context, environment


@pytest.mark.parametrize(
    "outcome",
    (
        "complete",
        "missing-identity",
        "changed-identity",
        "config-change",
        "config-missing",
        "signal",
        "poll-error",
    ),
)
def test_authorized_default_wrapper_uses_one_private_native_state(
    inert_wrapper, monkeypatch, outcome
):
    from npa.orchestration.skypilot._managed_job_api import NativeLaunchResult

    module, effects = inert_wrapper
    context, environment = _inert_authorized_context(module, monkeypatch)
    root = Path("/synthetic-private/run")
    expected = root / "skypilot-state"
    generated_config = expected / "submissions" / "config.yaml"
    input_config = Path("/synthetic-private/authorized-config")
    assert generated_config != input_config
    # This is an inert callback carrier, not a native/provider ownership receipt.
    cleanup = SimpleNamespace(
        config_path=generated_config,
        verified=False,
        request=Mock(),
        cwd=expected,
        run_id=context.authorization.inner_launch_id,
        job_id="42",
        active=True,
        submitting=False,
        requested=False,
        native_result=NativeLaunchResult(
            "a" * 64,
            "00000000-0000-4000-8000-000000000001",
            "42",
            (0,),
            "c" * 64,
        ),
        native_verified=True,
        environment={
            "HOME": str(expected / "home"),
            "PATH": "/bin",
            "KUBECONFIG": str(generated_config),
            "SKYPILOT_API_SERVER_ENDPOINT": "https://synthetic-api",
            "AWS_ACCESS_KEY_ID": "synthetic-secret",
            "NPA_EXECUTION_OUTPUTS": "synthetic-output",
        },
    )
    cleanup._lookup = lambda: module.ReconciliationEvidence(
        module.ReconciliationState.FOUND,
        job_id="42",
        status="RUNNING",
        workload_observable=True,
        observed_task_ids=(0,),
    )
    calls = []

    def submit(_yaml, run_id, **kwargs):
        module.prepare_inner_submit.assert_called_once_with(
            context.authorization, environment
        )
        effects["mkdir"].assert_called_once_with(expected, mode=0o700)
        assert kwargs["isolated_config_dir"] == expected
        assert kwargs["config_path"] == input_config
        assert isinstance(kwargs["isolated_config_dir"], Path)
        assert kwargs["robotwin_submit_context"] is context
        assert run_id == context.authorization.inner_launch_id
        assert "NPA_SKYPILOT_ISOLATED_CONFIG_DIR" not in kwargs["extra_env"]
        cleanup.cwd = kwargs["isolated_config_dir"]
        kwargs["on_launch_ready"](cleanup)
        # The original authorization input may change after submit; polling
        # must retain the producer-captured generated configuration instead.
        context.authorization.kubeconfig_source = (
            "/synthetic-private/changed-after-submit"
        )
        calls.append(kwargs)
        if outcome == "signal":
            effects["install_teardown_signal_handlers"].call_args.args[0]()
        if outcome in {"missing-identity", "changed-identity", "signal"}:
            raise module.SkyPilotConfigError("synthetic native identity unavailable")
        log_paths = {"submission_dir": str(expected / "submissions")}
        if outcome != "config-missing":
            log_paths["config"] = (
                str(root / "unrelated")
                if outcome == "config-change"
                else str(generated_config)
            )
        return SimpleNamespace(
            status="SUBMITTED",
            job_id="42",
            returncode=0,
            error="",
            launch_transaction={
                "state": "submitted",
                "identity_source": "native_request_result",
                "logical_launch_id": run_id,
                "job_id": "42",
            },
            log_paths=log_paths,
        )

    original_wait = module._wait_for_terminal
    status = Mock(
        side_effect=[
            SimpleNamespace(status="RUNNING", returncode=0),
            SimpleNamespace(status="SUCCEEDED", returncode=0),
        ]
    )
    monkeypatch.setattr(module, "workflow_status", status)
    monkeypatch.setattr(module.time, "sleep", Mock())

    def wait(_run_id, **kwargs):
        assert kwargs["isolated_config_dir"] is calls[0]["isolated_config_dir"]
        assert cleanup.cwd is kwargs["isolated_config_dir"]
        assert kwargs["config_path"] == generated_config
        assert kwargs["environment"] == {
            "HOME": str(expected / "home"),
            "PATH": "/bin",
            "KUBECONFIG": str(generated_config),
            "SKYPILOT_API_SERVER_ENDPOINT": "https://synthetic-api",
        }
        if outcome == "poll-error":
            raise module.SkyPilotConfigError("synthetic post-return polling failure")
        return original_wait(_run_id, **kwargs)

    launch, poll = Mock(side_effect=submit), Mock(side_effect=wait)
    monkeypatch.setattr(module, "submit_workflow", launch)
    monkeypatch.setattr(module, "_wait_for_terminal", poll)
    argv = ["--solution-name", "robotwin"]
    assert module._parse_args(argv).isolated_config_dir == ""
    if outcome == "complete":
        # Mock completion does not prove cleanup; retain the private recovery root.
        assert (
            module.run_authorized_robotwin(
                argv, authorization=context.authorization, environment=environment
            )
            == 1
        )
        poll.assert_called_once()
        assert status.call_count == 2
        for call in status.call_args_list:
            assert call.args == ("42",)
            assert call.kwargs["config_path"] == generated_config
            assert call.kwargs["isolated_config_dir"] is calls[0]["isolated_config_dir"]
            assert call.kwargs["environment"] == poll.call_args.kwargs["environment"]
    else:
        with pytest.raises(module.SkyPilotConfigError):
            module.run_authorized_robotwin(
                argv, authorization=context.authorization, environment=environment
            )
        if outcome == "poll-error":
            poll.assert_called_once()
        else:
            poll.assert_not_called()
        status.assert_not_called()
    launch.assert_called_once()
    assert cleanup.cwd is calls[0]["isolated_config_dir"]
    assert cleanup.config_path == generated_config and not cleanup.verified
    cleanup.request.assert_called()
    effects["remove"].assert_not_called()
    effects["_ensure_infra_enabled"].assert_not_called()
    module.subprocess.run.assert_not_called()


def test_authorized_wrapper_refusal_precedes_isolated_state(inert_wrapper, monkeypatch):
    module, effects = inert_wrapper
    context, environment = _inert_authorized_context(module, monkeypatch)
    module.prepare_inner_submit.side_effect = ValueError("synthetic refusal")
    with pytest.raises(ValueError, match="synthetic refusal"):
        module.run_authorized_robotwin(
            ["--solution-name", "robotwin"],
            authorization=context.authorization,
            environment=environment,
        )
    for effect in effects.values():
        effect.assert_not_called()
    module.subprocess.run.assert_not_called()


def test_authorized_wrapper_rechecks_customer_freshness_before_submit(
    inert_wrapper, monkeypatch
):
    from npa.orchestration.npa_workflow.robotwin_preflight import (
        RobotwinPreflightError,
    )

    module, effects = inert_wrapper
    context, environment = _inert_authorized_context(module, monkeypatch)
    stale = Mock(side_effect=RobotwinPreflightError("customer-authorization-stale"))
    monkeypatch.setattr(module, "require_customer_authorization_fresh", stale)
    submit = Mock()
    monkeypatch.setattr(module, "submit_workflow", submit)

    with pytest.raises(RobotwinPreflightError, match="customer-authorization-stale"):
        module.run_authorized_robotwin(
            ["--solution-name", "robotwin"],
            authorization=context.authorization,
            environment=environment,
        )

    stale.assert_called_once_with(context.authorization)
    submit.assert_not_called()
    effects["_bootstrap_robotwin_sky"].assert_called_once()


@pytest.mark.parametrize(
    ("logical_id", "result_job_id", "native_job_id", "expected"),
    (
        ("robotwin-inner-owned", "42", "42", "42"),
        ("robotwin-inner-owned", "robotwin-inner-owned", "42", None),
        ("robotwin-inner-owned", "42", "43", None),
        ("robotwin-inner-other", "42", "42", None),
    ),
)
def test_native_polling_identity_requires_bound_native_receipt(
    logical_id, result_job_id, native_job_id, expected
):
    from npa.orchestration.skypilot._managed_job_api import NativeLaunchResult

    module = _load_module()
    cleanup = SimpleNamespace(
        run_id=logical_id,
        job_id=native_job_id,
        active=True,
        submitting=False,
        requested=False,
        native_verified=True,
        environment={
            "HOME": "/synthetic-private/home",
            "PATH": "/bin",
            "KUBECONFIG": "/synthetic-private/generated-kubeconfig",
        },
        native_result=NativeLaunchResult(
            "a" * 64,
            "00000000-0000-4000-8000-000000000004",
            native_job_id,
            (0,),
            "b" * 64,
        ),
        _lookup=lambda: module.ReconciliationEvidence(
            module.ReconciliationState.FOUND,
            job_id=native_job_id,
            status="RUNNING",
            workload_observable=True,
            observed_task_ids=(0,),
        ),
    )
    result = SimpleNamespace(
        status="SUBMITTED",
        job_id=result_job_id,
        returncode=0,
        error="",
        launch_transaction={
            "state": "submitted",
            "identity_source": "native_request_result",
            "logical_launch_id": "robotwin-inner-owned",
            "job_id": result_job_id,
        },
    )
    assert module._native_polling_identity(result, cleanup, logical_id) == expected

    if expected is None:
        guard = module._SubmitTeardown(cleanup_on_failure=True)
        guard.bind(cleanup)
        with pytest.raises(module.SkyPilotConfigError, match="native polling identity"):
            guard.native_job_id(result, logical_id)
        assert guard.retain_context and guard.pending


@pytest.mark.parametrize(
    "evidence",
    (
        # A matching ID without a controller-observed workload is not ownership.
        lambda module: module.ReconciliationEvidence(
            module.ReconciliationState.FOUND,
            job_id="42",
            status="RUNNING",
            workload_observable=False,
            observed_task_ids=(0,),
        ),
        # UNKNOWN is not a recognized native status, even when IDs match.
        lambda module: module.ReconciliationEvidence(
            module.ReconciliationState.FOUND,
            job_id="42",
            status="UNKNOWN",
            workload_observable=True,
            observed_task_ids=(0,),
        ),
        # A partial or unrelated task population cannot establish this launch.
        lambda module: module.ReconciliationEvidence(
            module.ReconciliationState.FOUND,
            job_id="42",
            status="RUNNING",
            workload_observable=True,
            observed_task_ids=(1,),
        ),
        lambda module: module.ReconciliationEvidence(
            module.ReconciliationState.ABSENT,
            job_id="42",
            status="RUNNING",
            workload_observable=True,
            observed_task_ids=(0,),
        ),
    ),
)
def test_native_polling_identity_retains_context_without_complete_controller_evidence(
    evidence,
):
    from npa.orchestration.skypilot._managed_job_api import NativeLaunchResult

    module = _load_module()
    cleanup = SimpleNamespace(
        run_id="robotwin-inner-owned",
        job_id="42",
        active=True,
        submitting=False,
        requested=False,
        native_verified=True,
        native_result=NativeLaunchResult(
            "a" * 64,
            "00000000-0000-4000-8000-000000000005",
            "42",
            (0,),
            "b" * 64,
        ),
    )
    cleanup._lookup = lambda: evidence(module)
    result = SimpleNamespace(
        status="SUBMITTED",
        job_id="42",
        returncode=0,
        error="",
        launch_transaction={
            "state": "submitted",
            "identity_source": "native_request_result",
            "logical_launch_id": "robotwin-inner-owned",
            "job_id": "42",
        },
    )

    assert module._native_polling_identity(result, cleanup, cleanup.run_id) is None


def test_generic_wrapper_preserves_explicit_isolated_state(monkeypatch, tmp_path):
    module = _load_module()
    signal_teardown = module.SignalTeardown
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    monkeypatch.setattr(module, "SignalTeardown", signal_teardown)
    args.config_path = str(tmp_path / "skypilot.yaml")
    selected = tmp_path / "selected-state"
    args.isolated_config_dir = str(selected)
    launch = Mock(return_value=SimpleNamespace(job_id="73", log_paths={}))
    poll = Mock(return_value=(SimpleNamespace(status="SUCCEEDED", returncode=0), {}))
    monkeypatch.setattr(module, "submit_workflow", launch)
    monkeypatch.setattr(module, "_wait_for_terminal", poll)
    bootstrap = Mock(
        side_effect=AssertionError("generic wrapper entered RoboTwin bootstrap")
    )
    monkeypatch.setattr(module, "_bootstrap_robotwin_sky", bootstrap)

    assert module._submit_and_wait(args) == 0

    assert launch.call_args.kwargs["isolated_config_dir"] == selected
    assert poll.call_args.kwargs["isolated_config_dir"] == selected
    assert launch.call_args.kwargs["config_path"] == Path(args.config_path)
    assert poll.call_args.kwargs["config_path"] == Path(args.config_path)
    assert launch.call_args.kwargs.get("robotwin_submit_context") is None
    assert poll.call_args.args == ("73",)
    launch.assert_called_once()
    bootstrap.assert_not_called()


@pytest.mark.parametrize("confidential", (False, True))
@pytest.mark.parametrize(
    "failure", ("refused", "accepted", "signal", "unverified", "post-return")
)
def test_managed_submit_cleanup_ownership_boundary(
    monkeypatch, tmp_path, confidential, failure
):
    from npa.orchestration.skypilot import workflow as submit_module
    from npa.orchestration.skypilot._managed_job_api import NativeLaunchResult

    module = _load_module()
    monkeypatch.setattr(module, "require_customer_authorization_fresh", Mock())
    context = (
        SimpleNamespace(
            authorization=SimpleNamespace(
                inner_launch_id="robotwin-inner-synthetic",
                project="synthetic-project",
                kubeconfig_source=str(tmp_path / "kubeconfig"),
                customer_authorization_expires_at="2099-01-01T00:00:00Z",
            )
        )
        if confidential
        else None
    )
    monkeypatch.setattr(
        module, "render_workflow", lambda *_a, **_k: [{"name": "synthetic"}]
    )
    monkeypatch.setattr(module, "_write_yaml_documents", lambda *_a: None)
    monkeypatch.setattr(module, "preflight_output_storage", lambda **_k: None)
    monkeypatch.setattr(module, "_ensure_infra_enabled", lambda **_k: None)
    monkeypatch.setattr(module, "resolve_sky_bin", lambda *_a: "/synthetic-sky")
    monkeypatch.setattr(module, "_bootstrap_robotwin_sky", lambda *_a: "/synthetic-sky")
    monkeypatch.setattr(
        module, "_normalize_kubeconfig_current_context", lambda _p, values=None: values
    )
    monkeypatch.setattr(module, "restore_signal_handlers", lambda *_a: None)
    handlers, handles, workdirs, mutations = [], [], [], []
    monkeypatch.setattr(
        module,
        "install_teardown_signal_handlers",
        lambda callback: handlers.append(callback) or {},
    )
    create = module.tempfile.mkdtemp

    def private_temp(**kwargs):
        directory = create(dir=tmp_path, **kwargs)
        workdirs.append(Path(directory))
        return directory

    monkeypatch.setattr(module.tempfile, "mkdtemp", private_temp)

    def submit(_yaml, _run_id, **kwargs):
        if failure == "refused":
            raise RuntimeError("synthetic authorization refusal")
        cleanup = submit_module._SubmissionCleanup(
            _run_id,
            {"KUBECONFIG": str(tmp_path / "kubeconfig")},
            "/synthetic-sky",
            str(tmp_path),
            0,
            kwargs["config_path"],
            active=True,
        )
        if failure == "post-return":
            cleanup.native_result = NativeLaunchResult(
                "e" * 64,
                "00000000-0000-4000-8000-000000000003",
                "42",
                (0,),
                "f" * 64,
            )
            cleanup.job_id = "42"
            cleanup.native_verified = True
            cleanup.context_check = lambda: "f" * 64
        handles.append(cleanup)
        kwargs["on_launch_ready"](cleanup)
        if failure == "signal":
            handlers[0]()
            assert not mutations
        cleanup.finish_submit(failed=failure != "post-return")
        if failure == "post-return":
            return SimpleNamespace(
                status="SUBMITTED",
                job_id="42",
                returncode=0,
                error="",
                launch_transaction={
                    "state": "submitted",
                    "identity_source": "native_request_result",
                    "logical_launch_id": _run_id,
                    "job_id": "42",
                },
                log_paths={
                    "config": str(kwargs["config_path"]),
                    "submission_dir": str(workdirs[0] / "submission"),
                },
            )
        raise RuntimeError("synthetic accepted failure")

    def wait(*_a, **_k):
        assert failure == "post-return"
        raise RuntimeError("synthetic post-return failure")

    monkeypatch.setattr(module, "_wait_for_terminal", wait)

    status = ["RUNNING"]

    def lookup(*_a, **_k):
        state = submit_module.ReconciliationState
        return submit_module.ReconciliationEvidence(
            state.UNAVAILABLE if failure == "unverified" else state.FOUND,
            job_id="42",
            status=status[0],
            workload_observable=True,
            observed_task_ids=(0,),
        )

    def run(command, **kwargs):
        assert command == ["/synthetic-sky", "jobs", "cancel", "--yes", "42"]
        assert kwargs["env"] == handles[0].environment
        mutations.append(command)
        status[0] = "CANCELLED"
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "submit_workflow", submit)
    monkeypatch.setattr(submit_module, "_reconcile_managed_job_env", lookup)
    monkeypatch.setattr(subprocess, "run", run)
    args = module._parse_args(
        [
            "--yaml",
            str(YAML_PATH),
            "--solution-name",
            "robotwin" if confidential else "synthetic",
            "--run-id",
            "synthetic-owned",
            "--output-root",
            "s3://synthetic-bucket/output",
            "--config-path",
            str(tmp_path / "config"),
            "--no-direct-launch",
            "--cleanup",
        ]
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        module._submit_robotwin_and_wait(
            args, robotwin_submit_context=context, authorized_env={}
        )
    if failure == "post-return":
        assert mutations == [["/synthetic-sky", "jobs", "cancel", "--yes", "42"]]
        assert status == ["CANCELLED"]
        assert all(not path.exists() for path in workdirs)
    else:
        assert not mutations and status == ["RUNNING"]
        assert all(path.exists() == (failure != "refused") for path in workdirs)
    if failure == "post-return":
        assert all(handle.job_id == "42" and handle.verified for handle in handles)
        assert all(not handle.result.errors for handle in handles)
    else:
        assert all(not handle.job_id and not handle.verified for handle in handles)
        assert all(handle.result.errors for handle in handles)


def test_submit_cleanup_configuration_refinement_cannot_retarget(tmp_path):
    module = _load_module()
    cleanup = SimpleNamespace(config_path=tmp_path / "verified", verified=False)
    guard = module._SubmitTeardown(cleanup_on_failure=True)
    guard.bind(cleanup)
    guard.refine_config(tmp_path / "verified")
    with pytest.raises(module.SkyPilotConfigError, match="configuration mismatch"):
        guard.refine_config(tmp_path / "unrelated")
    assert guard.cleanup.config_path == tmp_path / "verified"


@pytest.mark.parametrize("confidential", (False, True))
def test_native_wrapper_keeps_context_when_controller_binding_changes(
    monkeypatch, tmp_path, confidential
):
    from npa.orchestration.skypilot import workflow as submit_module
    from npa.orchestration.skypilot._managed_job_api import NativeLaunchResult

    module = _load_module()
    environment = {"KUBECONFIG": str(tmp_path / "synthetic-kubeconfig")}
    if confidential:
        environment["SYNTHETIC_PRIVATE_CONTROL"] = "synthetic-private-value"
    cleanup = submit_module._SubmissionCleanup(
        "synthetic",
        environment,
        "/synthetic-sky",
        str(tmp_path),
        0,
        tmp_path / "config",
        job_id="41",
        active=True,
        submitting=False,
        native_result=NativeLaunchResult(
            "attempt", "00000000-0000-4000-8000-000000000001", "41", (0,), "c" * 64
        ),
        native_verified=True,
        context_check=lambda: "d" * 64,
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: pytest.fail("unrelated mutation")
    )
    guard = module._SubmitTeardown(cleanup_on_failure=True)
    guard.bind(cleanup)
    guard.refine_config(tmp_path / "config")
    cleanup.request()
    assert not cleanup.verified and cleanup.result.errors
    assert cleanup.environment == environment and tmp_path.exists()


def _indirect_submit_args(module, monkeypatch, tmp_path):
    config_path = tmp_path / "skypilot.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("NPA_BYOF_REFRESH_SKY_API", "0")
    monkeypatch.setattr(module, "resolve_sky_bin", lambda *_a, **_k: "/opt/sky")
    monkeypatch.setattr(
        module,
        "render_workflow",
        lambda *_a, **_k: [{"name": "task", "envs": {}, "resources": {}}],
    )
    monkeypatch.setattr(
        module, "_normalize_kubeconfig_current_context", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        module,
        "_write_default_k8s_config",
        lambda *_a, **_k: str(config_path),
    )
    monkeypatch.setattr(
        module,
        "verify_solution_payload_service_accounts",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(module, "preflight_output_storage", lambda **_k: None)
    monkeypatch.setattr(module, "_ensure_infra_enabled", lambda **_k: None)
    guard = SimpleNamespace(
        isolated_config_dir=None,
        mark_launched=lambda **_k: None,
        teardown=lambda: module.CleanupResult(),
    )
    monkeypatch.setattr(module, "SignalTeardown", lambda **_k: guard)
    monkeypatch.setattr(module, "install_teardown_signal_handlers", lambda *_a: None)
    monkeypatch.setattr(module, "restore_signal_handlers", lambda *_a: None)
    return module._parse_args(
        [
            "--yaml",
            str(YAML_PATH),
            "--run-id",
            "human-run-name",
            "--output-root",
            "s3://bucket/prefix",
            "--no-direct-launch",
            "--no-cleanup",
        ]
    )


def test_render_workflow_injects_solution_smoke_metadata(monkeypatch) -> None:
    from npa.execution_preflight import skypilot_output_destinations

    module = _load_module()
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setattr(module, "_resolved_storage_env", lambda: {})
    docs = module.render_workflow(
        YAML_PATH,
        run_id="byof-demo",
        output_root="s3://bucket/prefix",
        image="registry.example/npa-byof:demo",
        smoke_command="python -c 'print(42)'",
        solution_name="demo-solution",
        capability_name="demo-capability",
        smoke_artifact_name="demo_artifact.json",
    )

    task = docs[1]
    envs = task["envs"]
    assert envs["BYOF_SMOKE_COMMAND"] == "python -c 'print(42)'"
    assert envs["BYOF_SOLUTION_NAME"] == "demo-solution"
    assert envs["BYOF_CAPABILITY_NAME"] == "demo-capability"
    assert envs["BYOF_SMOKE_ARTIFACT_NAME"] == "demo_artifact.json"
    assert envs["BYOF_IMAGE"] == "registry.example/npa-byof:demo"
    assert envs["S3_OUTPUT_PREFIX"] == "s3://bucket/prefix/byof-demo/"
    assert json.loads(envs["NPA_EXECUTION_OUTPUTS"]) == [
        {"uri": "s3://bucket/prefix/byof-demo/", "kind": "directory"}
    ]
    assert skypilot_output_destinations(docs) == {
        "s3://bucket/prefix/byof-demo/": "directory"
    }
    assert envs["NPA_S3_BUCKET"] == "bucket"
    assert envs["AWS_ENDPOINT_URL"] == "https://storage.example"
    assert "AWS_ACCESS_KEY_ID" not in envs
    assert "AWS_SECRET_ACCESS_KEY" not in envs
    assert "AWS_SESSION_TOKEN" not in envs
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in envs
    assert task["resources"]["image_id"] == "docker:registry.example/npa-byof:demo"


def test_render_workflow_materializes_libero_payload_account(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_resolved_storage_env", lambda: {})

    docs = module.render_workflow(
        LIBERO_YAML_PATH,
        run_id="libero-demo",
        output_root="s3://bucket/prefix",
        solution_name="libero",
    )

    task = docs[1]
    assert "kubernetes" not in task["resources"]
    assert (
        task["config"]["kubernetes"]["pod_config"]["spec"]["serviceAccountName"]
        == "npa-byof-libero-payload"
    )


def test_libero_profile_refuses_missing_or_mistyped_solution_name(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "render_workflow",
        lambda *_a, **_k: [{"execution": "serial"}, {"name": "task"}],
    )
    for solution_name in ("", "not-libero", "LIBERO", " libero", "libero "):
        args = module._parse_args(
            [
                "--yaml",
                str(LIBERO_YAML_PATH),
                "--run-id",
                "libero-identity-refusal",
                "--output-root",
                "s3://bucket/prefix",
                "--solution-name",
                solution_name,
                "--render-only",
            ]
        )
        with pytest.raises(ValueError, match="requires --solution-name libero"):
            module._submit_and_wait(args)


@pytest.mark.parametrize("run_id", ["short", "libero.bad", "../libero-escape"])
def test_libero_profile_refuses_unsafe_run_id_before_render_output(
    monkeypatch, run_id
) -> None:
    module = _load_module()
    args = module._parse_args(
        [
            "--yaml",
            str(LIBERO_YAML_PATH),
            "--run-id",
            run_id,
            "--solution-name",
            "libero",
            "--render-only",
        ]
    )
    with pytest.raises(ValueError, match="SkyPilot run_id"):
        module._submit_and_wait(args)


def test_libero_payload_account_is_not_self_attested_authorization() -> None:
    module = _load_module()
    args = SimpleNamespace(solution_name="", yaml_path=Path("generic.yaml"))
    documents = [
        {
            "config": {
                "kubernetes": {
                    "pod_config": {
                        "spec": {"serviceAccountName": "npa-byof-libero-payload"}
                    }
                }
            }
        }
    ]

    assert module._is_libero_invocation(args, documents) is False
    assert module._uses_libero_payload_service_account(documents) is True


def test_official_libero_image_forces_refusal_for_renamed_profile(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_module()
    renamed_profile = tmp_path / "renamed-profile.yaml"
    renamed_profile.write_bytes(YAML_PATH.read_bytes())
    candidate = "ghcr.io/nebius/nebius-physical-ai/npa-libero@sha256:" + "9" * 64
    documents = [
        {"execution": "serial"},
        {
            "name": "renamed-profile",
            "resources": {"image_id": f"docker:{candidate}"},
            "envs": {"BYOF_IMAGE": candidate},
            "run": "true",
        },
    ]
    monkeypatch.setattr(module, "render_workflow", lambda *_args, **_kwargs: documents)
    args = module._parse_args(
        [
            "--yaml",
            str(renamed_profile),
            "--run-id",
            "libero-image-refusal",
            "--solution-name",
            "not-libero",
            "--render-only",
        ]
    )

    with pytest.raises(ValueError, match="requires --solution-name libero"):
        module._submit_and_wait(args)


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/example/npa-libero@sha256:" + "9" * 64,
        "ghcr.io/nebius/nebius-physical-ai/npa-libero-copy@sha256:" + "9" * 64,
        "mirror.invalid/ghcr.io/nebius/nebius-physical-ai/npa-libero@sha256:"
        + "9" * 64,
    ],
)
def test_libero_image_classifier_rejects_other_repositories_and_substrings(
    image,
) -> None:
    module = _load_module()
    args = SimpleNamespace(solution_name="not-libero", yaml_path=Path("renamed.yaml"))
    documents = [
        {
            "name": "renamed-profile",
            "resources": {"image_id": f"docker:{image}"},
            "envs": {"BYOF_IMAGE": image},
            "run": "true",
        }
    ]

    assert module._is_libero_invocation(args, documents) is False


def test_reserved_libero_payload_account_requires_explicit_solution(
    monkeypatch,
) -> None:
    module = _load_module()
    args = module._parse_args(
        [
            "--yaml",
            "byof-container-smoke-rtxpro",
            "--run-id",
            "generic-run",
            "--render-only",
        ]
    )
    monkeypatch.setattr(
        module,
        "render_workflow",
        lambda *_args, **_kwargs: [
            {},
            {
                "config": {
                    "kubernetes": {
                        "pod_config": {
                            "spec": {"serviceAccountName": "npa-byof-libero-payload"}
                        }
                    }
                }
            },
        ],
    )

    with pytest.raises(ValueError, match="requires an explicit LIBERO"):
        module._submit_and_wait(args)


def test_libero_isolated_scheduler_state_must_be_owner_private(tmp_path) -> None:
    module = _load_module()
    run_id = "libero-isolated-state"
    state = tmp_path / run_id
    state.mkdir()
    state.chmod(0o755)
    with pytest.raises(ValueError, match="owner-private"):
        module._libero_isolated_state_root(state, run_id)

    state.chmod(0o700)
    assert module._libero_isolated_state_root(state, run_id) == state.resolve()
    with pytest.raises(ValueError, match="exact-run scoped"):
        module._libero_isolated_state_root(state, "different-run-id")
    with pytest.raises(ValueError, match="requires an isolated"):
        module._libero_isolated_state_root(None, run_id)


def test_libero_runtime_binding_refuses_disabled_cleanup() -> None:
    module = _load_module()
    args = SimpleNamespace(
        solution_name="libero",
        yaml_path=LIBERO_YAML_PATH,
        direct_launch=False,
        cleanup=False,
    )
    documents = [{"execution": "serial"}, {"name": "task"}]

    with pytest.raises(ValueError, match="requires verified managed cleanup"):
        module._bind_libero_runtime_contract(
            args,
            documents,
            global_config={},
            infra="k8s/context",
            run_id="libero-cleanup-test",
        )


def test_libero_runtime_binding_refuses_disabled_api_lifecycle(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_BYOF_REFRESH_SKY_API", "0")
    args = SimpleNamespace(
        solution_name="libero",
        yaml_path=LIBERO_YAML_PATH,
        direct_launch=False,
        cleanup=True,
    )

    with pytest.raises(ValueError, match="verified isolated Sky API shutdown"):
        module._bind_libero_runtime_contract(
            args,
            [{"execution": "serial"}, {"name": "task"}],
            global_config={},
            infra="k8s/context",
            run_id="libero-api-lifecycle",
        )


def test_libero_runtime_binding_requires_managed_exact_node_and_separate_context(
    monkeypatch, tmp_path
) -> None:
    from npa.execution_preflight import (
        libero_executable_profile_bytes,
        libero_executable_profile_sha256,
    )

    module = _load_module()
    run_id = "libero-runtime-binding"
    runtime_manifest_sha256 = module.hashlib.sha256(
        module.LIBERO_RUNTIME_MANIFEST.read_bytes()
    ).hexdigest()
    candidate = "ghcr.io/nebius/nebius-physical-ai/npa-libero@sha256:" + "9" * 64
    customer_authorization = {
        "schema": "npa.libero.customer-runtime-authorization.v2",
        "solution": "libero",
        "status": "authorized",
        "authorization_id": "libero-customer-authorization-test-0001",
        "customer_identity_sha256": "7" * 64,
        "candidate_image": candidate,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "workflow_profile_sha256": "f" * 64,
        "upstream_source_revision": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
        "terms": [],
        "run_id": run_id,
        "issuer": "customer",
        "evidence_type": "customer-controlled-signature",
        "customer_signer_public_key_b64": "fixture-customer-key",
        "acknowledged_at": datetime.now(timezone.utc).isoformat(),
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "nonce": "unique-libero-test-nonce-0000000001",
        "signature": {},
    }
    output_prefix = f"s3://fixture-bucket/byof/{run_id}/"
    access_key = "fixture-temporary-access"
    secret_key = "fixture-temporary-secret"
    session_token = "fixture-temporary-session"
    policy_sha256 = "d" * 64
    endpoint = "https://storage.fixture.invalid"
    storage_authorization = {
        "schema": "npa.libero.output-storage-authorization.v3",
        "issuer": "npa-output-storage-control-plane",
        "capability_id": "libero-output-capability-test-0001",
        "customer_identity_sha256": customer_authorization["customer_identity_sha256"],
        "run_id": run_id,
        "candidate_image": candidate,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "output_prefix": output_prefix,
        "endpoint_url": endpoint,
        "access_key_id_sha256": module.hashlib.sha256(access_key.encode()).hexdigest(),
        "secret_access_key_sha256": module.hashlib.sha256(
            secret_key.encode()
        ).hexdigest(),
        "session_token_sha256": module.hashlib.sha256(
            session_token.encode()
        ).hexdigest(),
        "policy_sha256": policy_sha256,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": customer_authorization["expires_at"],
        "nonce": "unique-libero-storage-nonce-0000000001",
        "signature": {
            "algorithm": "ed25519",
            "public_key_sha256": "e" * 64,
            "signature_b64": "synthetic-signature",
        },
    }
    storage_bytes = (json.dumps(storage_authorization, sort_keys=True) + "\n").encode()
    storage_sha256 = module.hashlib.sha256(storage_bytes).hexdigest()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", access_key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret_key)
    monkeypatch.setenv("AWS_SESSION_TOKEN", session_token)
    monkeypatch.setenv(
        "NPA_LIBERO_OUTPUT_STORAGE_AUTHORIZATION_B64",
        module.base64.b64encode(storage_bytes).decode("ascii"),
    )
    kubeconfig = tmp_path / "payload-kubeconfig"
    kubeconfig.write_text("payload proof\n", encoding="utf-8")
    kubeconfig.chmod(0o600)
    execution_kubeconfig = tmp_path / "execution-kubeconfig"
    execution_kubeconfig.write_text("execution proof\n", encoding="utf-8")
    execution_kubeconfig.chmod(0o600)
    monkeypatch.setenv("KUBECONFIG", str(execution_kubeconfig))
    monkeypatch.setattr(module, "_libero_payload_kubeconfig", lambda: kubeconfig)
    monkeypatch.setattr(
        module,
        "_libero_context_contract",
        lambda path, **_kwargs: (
            (
                "payload-proof-context",
                "isolated-namespace",
                "6" * 64,
            )
            if path == kubeconfig
            else ("execution-context", "isolated-namespace", "6" * 64)
        ),
    )
    rbac = {
        "service_account_uid_sha256": "1" * 64,
        "role_uid_sha256": "2" * 64,
        "role_binding_uid_sha256": "3" * 64,
        "rbac_spec_sha256": "4" * 64,
        "namespace_sha256": "5" * 64,
        "namespace_uid_sha256": "7" * 64,
        "namespace_inventory_sha256": "8" * 64,
        "external_rbac_inventory_sha256": "e" * 64,
    }
    access_state = module.LiberoAccessState(
        kubeconfig=kubeconfig,
        context="payload-proof-context",
        namespace="isolated-namespace",
        namespace_uid="namespace-uid",
        service_account_uid="service-account-uid",
        role_uid="role-uid",
        role_binding_uid="binding-uid",
        controller_service_account_uid="controller-account-uid",
        controller_role_uid="controller-role-uid",
        controller_role_binding_uid="controller-binding-uid",
        run_id=run_id,
    )
    monkeypatch.setattr(
        module, "_libero_rbac_evidence", lambda *_a: (dict(rbac), access_state)
    )
    controller_evidence = {
        "controller_service_account_uid_sha256": "9" * 64,
        "controller_role_uid_sha256": "a" * 64,
        "controller_role_binding_uid_sha256": "b" * 64,
        "controller_rbac_spec_sha256": "d" * 64,
        "external_rbac_inventory_sha256": "e" * 64,
    }
    controller_identities = {
        "controller_service_account_uid": "controller-account-uid",
        "controller_role_uid": "controller-role-uid",
        "controller_role_binding_uid": "controller-binding-uid",
    }
    monkeypatch.setattr(
        module,
        "_libero_controller_rbac_evidence",
        lambda *_a: (dict(controller_evidence), dict(controller_identities)),
    )
    expected_evidence = {
        **rbac,
        **controller_evidence,
        "cluster_identity_sha256": "6" * 64,
        "allowed_node_sha256": module.hashlib.sha256(b"worker").hexdigest(),
        "payload_kubeconfig_sha256": module.hashlib.sha256(
            kubeconfig.read_bytes()
        ).hexdigest(),
        "execution_kubeconfig_sha256": module.hashlib.sha256(
            execution_kubeconfig.read_bytes()
        ).hexdigest(),
        "skypilot_config_sha256": module._sha256_json(
            {"kubernetes": {"allowed_nodes": {"names": ["worker"]}}}
        ),
    }
    expected_envs = {
        "S3_OUTPUT_PREFIX": output_prefix,
        "AWS_ENDPOINT_URL": endpoint,
    }
    expected_envs.update(
        {
            f"NPA_LIBERO_EXPECTED_{key.upper()}": value
            for key, value in expected_evidence.items()
        }
    )
    expected_envs.update(
        {
            "NPA_LIBERO_EXPECTED_CANONICAL_BUILD_METADATA_SHA256": "c" * 64,
            "NPA_LIBERO_EXPECTED_CUSTOMER_AUTHORIZATION_EXPIRES_AT": customer_authorization[
                "expires_at"
            ],
            "NPA_LIBERO_EXPECTED_PUBLICATION_BUNDLE_SHA256": "a" * 64,
            "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_AUTHORIZATION_SHA256": storage_sha256,
            "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_PREFIX_SHA256": module.hashlib.sha256(
                output_prefix.encode()
            ).hexdigest(),
            "NPA_LIBERO_EXPECTED_OUTPUT_STORAGE_POLICY_SHA256": policy_sha256,
        }
    )
    expected_envs["NPA_LIBERO_EXPECTED_INFRASTRUCTURE_BUNDLE_SHA256"] = (
        module._sha256_json(
            {
                "schema": "npa.libero.infrastructure-bundle.v2",
                "run_id": run_id,
                **expected_evidence,
                "output_storage_authorization_sha256": storage_sha256,
                "output_storage_prefix_sha256": module.hashlib.sha256(
                    output_prefix.encode()
                ).hexdigest(),
                "output_storage_policy_sha256": policy_sha256,
            }
        )
    )

    def runtime_documents(envs, *, bound=False):
        volumes, mounts = module._libero_reviewed_mount_contract("7" * 64)
        resources = {"cloud": "kubernetes"}
        if bound:
            resources["region"] = "execution-context"
        return [
            {"execution": "serial"},
            {
                "name": "byof-solution-smoke-libero-b200-gpu",
                "resources": resources,
                "envs": envs,
                "config": {
                    "kubernetes": {
                        "pod_config": {
                            "spec": {
                                "volumes": volumes,
                                "containers": [{"volumeMounts": mounts}],
                            }
                        }
                    }
                },
            },
        ]

    authorization_bytes = (
        json.dumps(customer_authorization, sort_keys=True) + "\n"
    ).encode()
    authorization_sha256 = module.hashlib.sha256(authorization_bytes).hexdigest()
    monkeypatch.setenv(
        "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64",
        module.base64.b64encode(authorization_bytes).decode("ascii"),
    )
    monkeypatch.setenv("NPA_LIBERO_CUSTOMER_AUTHORIZATION_SHA256", authorization_sha256)
    monkeypatch.setenv(
        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256",
        customer_authorization["customer_identity_sha256"],
    )
    caller_bytes = b'{"synthetic":"caller"}\n'
    caller_sha256 = module.hashlib.sha256(caller_bytes).hexdigest()
    monkeypatch.setenv(
        "NPA_LIBERO_AUTHENTICATED_CALLER_B64",
        module.base64.b64encode(caller_bytes).decode("ascii"),
    )
    monkeypatch.setenv("NPA_LIBERO_AUTHENTICATED_CALLER_SHA256", caller_sha256)
    qualification = {
        "development_sha": "a" * 40,
        "candidate_image": candidate,
        "canonical_build_metadata_sha256": "c" * 64,
        "publication_bundle_sha256": "a" * 64,
        "runtime_manifest_sha256": runtime_manifest_sha256,
    }
    image_manifest = {"schema": "fixture", "qualification": qualification}
    monkeypatch.setattr(module, "libero_image_manifest", lambda: image_manifest)
    monkeypatch.setattr(
        module,
        "validate_libero_qualified_image_manifest",
        lambda payload: payload["qualification"],
    )
    monkeypatch.setattr(
        module,
        "validate_libero_authenticated_caller_assertion",
        lambda *_args, **_kwargs: (
            {
                "customer_identity_sha256": customer_authorization[
                    "customer_identity_sha256"
                ],
                "customer_signer_public_key_sha256": "f" * 64,
            },
            caller_sha256,
        ),
    )
    monkeypatch.setattr(
        module,
        "validate_libero_customer_runtime_authorization",
        lambda *_args, **_kwargs: (customer_authorization, authorization_sha256),
    )

    def validate_storage(*_args, **kwargs):
        if kwargs["endpoint_url"] != endpoint:
            raise RuntimeError(
                "LIBERO output storage authorization is invalid or expired"
            )
        return storage_authorization, storage_sha256

    monkeypatch.setattr(
        module, "validate_libero_output_storage_authorization", validate_storage
    )
    lineage_calls = []
    monkeypatch.setattr(
        module,
        "libero_publication_lineage_values",
        lambda value, repository_root, *, development_sha: lineage_calls.append(
            (value, repository_root, development_sha)
        ),
    )
    args = SimpleNamespace(
        solution_name="libero", direct_launch=False, cleanup=True, image=candidate
    )
    documents = runtime_documents(
        {"S3_OUTPUT_PREFIX": output_prefix, "AWS_ENDPOINT_URL": endpoint}
    )
    volumes, mounts = module._libero_reviewed_mount_contract("7" * 64)
    documents[1]["config"] = {
        "kubernetes": {
            "pod_config": {
                "spec": {
                    "volumes": volumes,
                    "containers": [{"volumeMounts": mounts}],
                }
            }
        }
    }
    registration = tmp_path / (("7" * 64) + ".b64")
    registration.write_text(customer_authorization["customer_signer_public_key_b64"])
    registration.chmod(0o600)
    monkeypatch.setenv(
        "NPA_LIBERO_CUSTOMER_SIGNER_REGISTRATION_FILE", str(registration)
    )
    expected_envs.update(
        {
            "NPA_LIBERO_EXPECTED_CUSTOMER_SIGNER_PUBLIC_KEY_SHA256": "f" * 64,
            "NPA_LIBERO_EXPECTED_EXECUTABLE_PROFILE_SHA256": libero_executable_profile_sha256(
                documents
            ),
            "NPA_LIBERO_EXECUTABLE_PROFILE_B64": module.base64.b64encode(
                libero_executable_profile_bytes(documents)
            ).decode("ascii"),
        }
    )

    binding = module._bind_libero_runtime_contract(
        args,
        documents,
        global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
        infra="k8s/execution-context",
        run_id=run_id,
    )

    assert binding.evidence == expected_evidence
    assert binding.access_state == module.replace(
        access_state,
        **controller_identities,
        execution_kubeconfig=execution_kubeconfig,
        execution_context="execution-context",
    )
    assert documents[1]["envs"] == expected_envs
    assert documents[1]["resources"] == {
        "cloud": "kubernetes",
        "region": "execution-context",
    }
    assert "NPA_LIBERO_CUSTOMER_AUTHORIZATION_B64" not in documents[1]["envs"]
    assert json.loads(module.base64.b64decode(binding.customer_authorization_b64)) == (
        customer_authorization
    )
    assert lineage_calls == [
        (
            qualification,
            module.Path(module.__file__).resolve().parents[2],
            qualification["development_sha"],
        )
    ]

    with pytest.raises(ValueError, match="invalid or expired"):
        module._bind_libero_runtime_contract(
            args,
            runtime_documents(
                {
                    "S3_OUTPUT_PREFIX": output_prefix,
                    "AWS_ENDPOINT_URL": "https://other.invalid",
                }
            ),
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
            run_id=run_id,
        )
    with pytest.raises(ValueError, match="origin-only HTTPS"):
        module._bind_libero_runtime_contract(
            args,
            runtime_documents(
                {
                    "S3_OUTPUT_PREFIX": output_prefix,
                    "AWS_ENDPOINT_URL": endpoint + "/path?redirect=1",
                }
            ),
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
            run_id=run_id,
        )

    mismatched_controller = {
        **controller_evidence,
        "external_rbac_inventory_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        module,
        "_libero_controller_rbac_evidence",
        lambda *_a: (dict(mismatched_controller), dict(controller_identities)),
    )
    with pytest.raises(ValueError, match="observe different external RBAC"):
        module._bind_libero_runtime_contract(
            args,
            runtime_documents(
                {"S3_OUTPUT_PREFIX": output_prefix, "AWS_ENDPOINT_URL": endpoint}
            ),
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
            run_id=run_id,
        )
    monkeypatch.setattr(
        module,
        "_libero_controller_rbac_evidence",
        lambda *_a: (dict(controller_evidence), dict(controller_identities)),
    )

    monkeypatch.setenv("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="authenticated caller identity differs"):
        module._bind_libero_runtime_contract(
            args,
            runtime_documents(
                {"S3_OUTPUT_PREFIX": output_prefix, "AWS_ENDPOINT_URL": endpoint}
            ),
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
            run_id=run_id,
        )
    monkeypatch.setenv(
        "NPA_LIBERO_CUSTOMER_IDENTITY_SHA256",
        customer_authorization["customer_identity_sha256"],
    )

    for direct, allowed in (
        (True, {"names": ["worker"]}),
        (False, None),
        (False, {"names": ["worker-a", "worker-b"]}),
        (False, {"names": [" worker"]}),
        (False, {"names": ["worker "]}),
    ):
        args.direct_launch = direct
        with pytest.raises(ValueError):
            module._bind_libero_runtime_contract(
                args,
                runtime_documents(
                    {"S3_OUTPUT_PREFIX": output_prefix, "AWS_ENDPOINT_URL": endpoint}
                ),
                global_config={"kubernetes": {"allowed_nodes": allowed}},
                infra="k8s/execution-context",
                run_id=run_id,
            )

    args.direct_launch = False
    monkeypatch.setattr(
        module,
        "_libero_context_contract",
        lambda path, **_kwargs: (
            ("payload-proof-context", "isolated-namespace", "6" * 64)
            if path == kubeconfig
            else ("execution-context", "isolated-namespace", "7" * 64)
        ),
    )
    with pytest.raises(ValueError, match="different clusters"):
        module._bind_libero_runtime_contract(
            args,
            runtime_documents(
                {"S3_OUTPUT_PREFIX": output_prefix, "AWS_ENDPOINT_URL": endpoint}
            ),
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
            run_id=run_id,
        )

    monkeypatch.setattr(
        module,
        "_libero_context_contract",
        lambda path, **_kwargs: (
            ("payload-proof-context", "isolated-namespace", "6" * 64)
            if path == kubeconfig
            else ("execution-context", "different-namespace", "6" * 64)
        ),
    )
    with pytest.raises(ValueError, match="exact run namespace"):
        module._bind_libero_runtime_contract(
            args,
            runtime_documents(
                {"S3_OUTPUT_PREFIX": output_prefix, "AWS_ENDPOINT_URL": endpoint}
            ),
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
            run_id=run_id,
        )

    monkeypatch.setattr(
        module,
        "_libero_context_contract",
        lambda _path, **_kwargs: (
            "execution-context",
            "isolated-namespace",
            "6" * 64,
        ),
    )
    with pytest.raises(ValueError, match="explicitly separated"):
        module._bind_libero_runtime_contract(
            args,
            runtime_documents(
                {"S3_OUTPUT_PREFIX": output_prefix, "AWS_ENDPOINT_URL": endpoint}
            ),
            global_config={"kubernetes": {"allowed_nodes": {"names": ["worker"]}}},
            infra="k8s/execution-context",
            run_id=run_id,
        )


def test_libero_runtime_binding_refuses_local_enforcement_drift_before_access(
    monkeypatch,
) -> None:
    module = _load_module()
    qualification = {"development_sha": "a" * 40}
    monkeypatch.setattr(
        module, "libero_image_manifest", lambda: {"qualification": qualification}
    )
    monkeypatch.setattr(
        module,
        "validate_libero_qualified_image_manifest",
        lambda _payload: qualification,
    )

    def refuse(*_args, **_kwargs):
        raise RuntimeError("LIBERO neutral build inputs differ from qualification")

    monkeypatch.setattr(module, "libero_publication_lineage_values", refuse)
    monkeypatch.setattr(
        module,
        "_libero_payload_kubeconfig",
        lambda: pytest.fail("access state must not be read after lineage drift"),
    )
    args = SimpleNamespace(
        solution_name="libero", direct_launch=False, cleanup=True, image="unused"
    )

    with pytest.raises(ValueError, match="neutral build inputs differ"):
        module._bind_libero_runtime_contract(
            args,
            [
                {"execution": "serial"},
                {"name": "byof-solution-smoke-libero-b200-gpu", "envs": {}},
            ],
            global_config={},
            infra="k8s/execution-context",
            run_id="libero-lineage-drift",
        )


def test_libero_payload_kubeconfig_must_be_private_regular_file(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    kubeconfig = tmp_path / "payload-kubeconfig"
    kubeconfig.write_text("proof\n", encoding="utf-8")
    kubeconfig.chmod(0o644)
    monkeypatch.setenv("NPA_LIBERO_PAYLOAD_KUBECONFIG", str(kubeconfig))

    with pytest.raises(ValueError, match="owner-private regular file"):
        module._libero_payload_kubeconfig()

    kubeconfig.chmod(0o600)
    assert module._libero_payload_kubeconfig() == kubeconfig.resolve()

    monkeypatch.setattr(module.os, "getuid", lambda: kubeconfig.stat().st_uid + 1)
    with pytest.raises(ValueError, match="owner-private regular file"):
        module._libero_payload_kubeconfig()


def test_libero_sky_config_uses_only_explicit_mode_private_owner_input(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    args = SimpleNamespace(config_path="")
    config = tmp_path / "skypilot-config"
    config.write_text("kubernetes: {}\n", encoding="utf-8")
    config.chmod(0o644)
    monkeypatch.setenv("NPA_LIBERO_SKYPILOT_CONFIG", str(config))

    with pytest.raises(ValueError, match="owner-private regular file"):
        module._libero_global_config_path(args)

    config.chmod(0o600)
    assert module._libero_global_config_path(args) == str(config.resolve())
    monkeypatch.delenv("NPA_LIBERO_SKYPILOT_CONFIG")
    with pytest.raises(ValueError, match="owner-supplied"):
        module._libero_global_config_path(args)


def test_libero_context_contract_binds_server_and_ca_without_exposing_them(
    monkeypatch,
) -> None:
    module = _load_module()
    config = {
        "current-context": "payload-context",
        "contexts": [
            {
                "name": "payload-context",
                "context": {"cluster": "cluster", "namespace": "isolated"},
            }
        ],
        "clusters": [
            {
                "name": "cluster",
                "cluster": {
                    "server": "https://cluster.example",
                    "certificate-authority-data": "base64-ca",
                },
            }
        ],
    }
    seen: dict[str, object] = {}

    def kubectl_json(arguments, *, purpose, kubeconfig):
        seen.update(arguments=arguments, purpose=purpose, kubeconfig=kubeconfig)
        return config

    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    path = Path("/private/payload-kubeconfig")
    context, namespace, identity = module._libero_context_contract(
        path, require_namespace=True
    )

    assert (context, namespace) == ("payload-context", "isolated")
    assert identity == module._sha256_json(
        {
            "server": "https://cluster.example",
            "certificate_authority_data": "base64-ca",
        }
    )
    assert seen == {
        "arguments": ["config", "view", "--minify", "--flatten", "--raw"],
        "purpose": "selected-context",
        "kubeconfig": path,
    }

    config["clusters"][0]["cluster"]["insecure-skip-tls-verify"] = True
    with pytest.raises(RuntimeError, match="strict TLS identity"):
        module._libero_context_contract(path, require_namespace=True)


def test_libero_rbac_evidence_refuses_role_or_binding_drift(monkeypatch) -> None:
    module = _load_module()
    namespace = "isolated-namespace"
    run_id = "libero-rbac-evidence"
    objects = {
        "serviceaccount": {
            "metadata": {"uid": "account-uid"},
        },
        "role": {
            "metadata": {"uid": "role-uid"},
            "rules": module._libero_payload_rules(
                module.LIBERO_PAYLOAD_UNBOUND_RESOURCE_NAME
            ),
        },
        "rolebinding": {
            "metadata": {"uid": "binding-uid"},
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "npa-byof-libero-payload",
                    "namespace": namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": "npa-byof-libero-pod-reader",
            },
        },
    }
    monkeypatch.setattr(
        module,
        "_libero_resource",
        lambda _kubeconfig, _context, _namespace, kind, _name: objects[kind],
    )

    def kubectl_json(arguments, **_kwargs):
        if arguments[-1] in {"clusterrolebindings", "rolebindings"}:
            return {"items": []}
        return {
            "metadata": {
                "name": namespace,
                "uid": "namespace-uid",
                "labels": {
                    "npa.nebius.ai/solution": "libero",
                    "npa.nebius.ai/run-id-sha256": module.hashlib.sha256(
                        run_id.encode()
                    ).hexdigest(),
                },
            }
        }

    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    monkeypatch.setattr(
        module,
        "_libero_namespaced_inventory",
        lambda *_a: {
            "pods": [],
            "secrets": [],
            "serviceaccounts": ["default", "npa-byof-libero-payload"],
            "roles": ["npa-byof-libero-pod-reader"],
            "rolebindings": ["npa-byof-libero-payload-pod-reader"],
            "configmaps": [],
            "services": [],
        },
    )

    kubeconfig = Path("/private/payload-kubeconfig")
    evidence, access_state = module._libero_rbac_evidence(
        kubeconfig, "payload-context", namespace, run_id
    )
    assert (
        evidence["service_account_uid_sha256"]
        == module.hashlib.sha256(b"account-uid").hexdigest()
    )
    assert access_state.namespace_uid == "namespace-uid"

    objects["role"]["rules"][0]["verbs"] = ["get", "list"]
    with pytest.raises(RuntimeError, match="exact Pod resourceName"):
        module._libero_rbac_evidence(kubeconfig, "payload-context", namespace, run_id)
    objects["role"]["rules"][0]["verbs"] = ["get"]
    objects["rolebinding"]["subjects"][0]["name"] = "default"
    with pytest.raises(RuntimeError, match="differs"):
        module._libero_rbac_evidence(kubeconfig, "payload-context", namespace, run_id)


def test_libero_post_submit_rbac_recheck_refuses_any_service(monkeypatch) -> None:
    module = _load_module()
    namespace = "isolated-namespace"
    run_id = "libero-post-submit-rbac"
    resources = {
        "serviceaccount": {"metadata": {"uid": "account-uid"}},
        "role": {
            "metadata": {"uid": "role-uid"},
            "rules": module._libero_payload_rules("payload-pod"),
        },
        "rolebinding": {
            "metadata": {"uid": "binding-uid"},
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": module.LIBERO_PAYLOAD_SERVICE_ACCOUNT,
                    "namespace": namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": module.LIBERO_PAYLOAD_ROLE,
            },
        },
    }
    monkeypatch.setattr(
        module,
        "_libero_resource",
        lambda _kubeconfig, _context, _namespace, kind, _name: resources[kind],
    )
    monkeypatch.setattr(
        module, "_libero_external_rbac_inventory_sha256", lambda *_args: "a" * 64
    )
    services: list[dict[str, object]] = []

    def kubectl_json(arguments, **_kwargs):
        if arguments[-1] == "services":
            return {"items": services}
        return {
            "metadata": {
                "name": namespace,
                "uid": "namespace-uid",
                "labels": {
                    "npa.nebius.ai/solution": "libero",
                    "npa.nebius.ai/run-id-sha256": module.hashlib.sha256(
                        run_id.encode()
                    ).hexdigest(),
                },
            }
        }

    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    module._libero_rbac_evidence(
        Path("/private/payload-kubeconfig"),
        "payload-context",
        namespace,
        run_id,
        require_empty_inventory=False,
        expected_pod_name="payload-pod",
    )

    services.append({"metadata": {"name": "unreviewed-selector"}})
    with pytest.raises(RuntimeError, match="free of unreviewed Services"):
        module._libero_rbac_evidence(
            Path("/private/payload-kubeconfig"),
            "payload-context",
            namespace,
            run_id,
            require_empty_inventory=False,
            expected_pod_name="payload-pod",
        )


def _libero_payload_pod_fixture(module, *, job_id: str = "73") -> dict[str, object]:
    task_name = module.LIBERO_PROFILE_TASK_NAME
    display_name = module._libero_managed_job_display_name(task_name, job_id)
    cluster_name = f"sky-cluster-{job_id}-fixture-user"
    pod_name = f"{cluster_name}-head"
    candidate = "ghcr.io/nebius/nebius-physical-ai/npa-libero@sha256:" + "9" * 64
    pod = {
        "metadata": {
            "name": pod_name,
            "namespace": "isolated-namespace",
            "uid": "payload-pod-uid",
            "creationTimestamp": "2026-09-12T00:00:00Z",
            "labels": {
                "skypilot-cluster-name": cluster_name,
                "ray-node-type": "head",
                "component": pod_name,
            },
            "annotations": {"skypilot-cluster-name": display_name},
        },
        "spec": {
            "serviceAccountName": module.LIBERO_PAYLOAD_SERVICE_ACCOUNT,
            "automountServiceAccountToken": True,
            "securityContext": {
                "runAsNonRoot": True,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "nodeName": "worker",
            "containers": [
                {
                    "name": "ray-node",
                    "image": candidate,
                    "resources": {
                        "requests": {"nvidia.com/gpu": "1"},
                        "limits": {"nvidia.com/gpu": "1"},
                    },
                    "securityContext": {
                        "privileged": False,
                        "allowPrivilegeEscalation": False,
                        "readOnlyRootFilesystem": True,
                        "capabilities": {"drop": ["ALL"]},
                    },
                }
            ],
        },
        "status": {
            "containerStatuses": [
                {
                    "name": "ray-node",
                    "imageID": "docker-pullable://" + candidate,
                    "allocatedResourcesStatus": [
                        {
                            "name": "nvidia.com/b200",
                            "resources": [{"resourceID": "GPU-fixture-b200"}],
                        }
                    ],
                }
            ]
        },
    }

    # Exercise the actual checked-in volume/mount contract, not a stripped Pod.
    document = list(module.yaml.safe_load_all(LIBERO_YAML_PATH.read_text()))[1]
    spec = document["resources"]["kubernetes"]["pod_config"]["spec"]
    pod["spec"]["volumes"] = spec["volumes"]
    mounts = spec["containers"][0]["volumeMounts"]
    for mount in mounts:
        for field in ("mountPath", "subPath"):
            if field in mount:
                mount[field] = mount[field].replace(
                    "<customer-identity-sha256>", "8" * 64
                )
    pod["spec"]["containers"][0]["volumeMounts"] = mounts
    return pod


def test_libero_managed_job_name_matches_pinned_skypilot_algorithm() -> None:
    module = _load_module()

    assert (
        module._libero_managed_job_display_name(module.LIBERO_PROFILE_TASK_NAME, "73")
        == "byof-solution-smoke-li-d9-73"
    )
    with pytest.raises(RuntimeError, match="scheduler job ID"):
        module._libero_managed_job_display_name(
            module.LIBERO_PROFILE_TASK_NAME, "73-foreign"
        )


def test_libero_payload_pod_record_is_exact_and_singleton(monkeypatch) -> None:
    module = _load_module()
    pod = _libero_payload_pod_fixture(module)
    candidate = pod["spec"]["containers"][0]["image"]
    binding = module.LiberoRuntimeBinding(
        customer_identity_sha256="8" * 64,
        evidence={},
        access_state=module.LiberoAccessState(
            kubeconfig=Path("/private/payload-kubeconfig"),
            context="payload-context",
            namespace="isolated-namespace",
            namespace_uid="namespace-uid",
            service_account_uid="service-account-uid",
            role_uid="role-uid",
            role_binding_uid="binding-uid",
            run_id="libero-payload-record",
        ),
        candidate_image=candidate,
        task_name=module.LIBERO_PROFILE_TASK_NAME,
    )
    items = [pod]
    monkeypatch.setattr(module, "_kubectl_json", lambda *_a, **_k: {"items": items})

    observed, evidence = module._libero_payload_pod_record(
        binding, scheduler_job_id="73"
    )
    assert observed is pod
    assert (
        evidence["payload_pod_name_sha256"]
        == module.hashlib.sha256(pod["metadata"]["name"].encode()).hexdigest()
    )

    items.append(json.loads(json.dumps(pod)))
    with pytest.raises(RuntimeError, match="exactly one payload Pod"):
        module._libero_payload_pod_record(binding, scheduler_job_id="73")
    items.pop()
    pod["spec"]["containers"].append({"name": "foreign", "image": candidate})
    with pytest.raises(RuntimeError, match="accepted candidate image"):
        module._libero_payload_pod_record(binding, scheduler_job_id="73")
    pod["spec"]["containers"].pop()
    with pytest.raises(RuntimeError, match="submitted scheduler job"):
        module._libero_payload_pod_record(binding, scheduler_job_id="74")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("privileged", "security context"),
        ("host-network", "host or runtime boundary"),
        ("host-path", "forbidden volume"),
        ("host-port", "security context"),
        ("capability", "security context"),
        ("writable-root", "shadows a reviewed mount"),
        ("runtime-class", "host or runtime boundary"),
        ("release-token", "release token contract"),
        ("foreign-configmap", "forbidden volume"),
        ("missing-root", "missing its exact reviewed mounts"),
        ("root-writable", "unreviewed required mount"),
        ("wrong-registration", "unreviewed required mount"),
        ("extra-writable-run", "broad writable mount"),
    ],
)
def test_libero_payload_pod_rejects_every_confinement_bypass(
    monkeypatch, mutation, expected
) -> None:
    module = _load_module()
    pod = _libero_payload_pod_fixture(module)
    container = pod["spec"]["containers"][0]
    if mutation == "privileged":
        container["securityContext"]["privileged"] = True
    elif mutation == "host-network":
        pod["spec"]["hostNetwork"] = True
    elif mutation == "host-path":
        pod["spec"]["volumes"] = [
            {"name": "host", "hostPath": {"path": "/", "type": "Directory"}}
        ]
    elif mutation == "host-port":
        container["ports"] = [{"containerPort": 8080, "hostPort": 8080}]
    elif mutation == "capability":
        container["securityContext"]["capabilities"]["add"] = ["SYS_ADMIN"]
    elif mutation == "writable-root":
        pod["spec"]["volumes"] = [{"name": "data", "emptyDir": {}}]
        container["volumeMounts"] = [{"name": "data", "mountPath": "/etc"}]
    elif mutation == "runtime-class":
        pod["spec"]["runtimeClassName"] = "unreviewed"
    elif mutation == "foreign-configmap":
        pod["spec"]["volumes"].append(
            {"name": "foreign", "configMap": {"name": "foreign"}}
        )
    elif mutation == "missing-root":
        container["volumeMounts"] = [
            mount
            for mount in container["volumeMounts"]
            if mount["name"] != "libero-caller-verification"
        ]
    elif mutation == "root-writable":
        next(
            m
            for m in container["volumeMounts"]
            if m["name"] == "libero-caller-verification"
        )["readOnly"] = False
    elif mutation == "wrong-registration":
        next(
            m
            for m in container["volumeMounts"]
            if m["name"] == "libero-customer-registration"
        )["subPath"] = "9" * 64 + ".b64"
    elif mutation == "extra-writable-run":
        pod["spec"]["volumes"].append({"name": "foreign", "emptyDir": {}})
        container["volumeMounts"].append(
            {"name": "foreign", "mountPath": "/run/foreign"}
        )
    else:
        pod["spec"]["automountServiceAccountToken"] = False
    binding = module.LiberoRuntimeBinding(
        customer_identity_sha256="8" * 64,
        evidence={},
        access_state=module.LiberoAccessState(
            kubeconfig=Path("/private/payload-kubeconfig"),
            context="payload-context",
            namespace="isolated-namespace",
            namespace_uid="namespace-uid",
            service_account_uid="service-account-uid",
            role_uid="role-uid",
            role_binding_uid="binding-uid",
            run_id="libero-payload-confinement",
        ),
        candidate_image=container["image"],
        task_name=module.LIBERO_PROFILE_TASK_NAME,
    )
    monkeypatch.setattr(module, "_kubectl_json", lambda *_a, **_k: {"items": [pod]})

    with pytest.raises(RuntimeError, match=expected):
        module._libero_payload_pod_record(binding, scheduler_job_id="73")


@pytest.mark.parametrize(
    "drift", ["none", "node", "b200-resource", "gpu-request", "image-status"]
)
def test_libero_manager_live_evidence_is_independent_and_exact(
    monkeypatch, drift
) -> None:
    module = _load_module()
    pod = _libero_payload_pod_fixture(module)
    candidate = pod["spec"]["containers"][0]["image"]
    state = module.LiberoAccessState(
        kubeconfig=Path("/private/payload-kubeconfig"),
        context="payload-context",
        namespace="isolated-namespace",
        namespace_uid="namespace-uid",
        service_account_uid="service-account-uid",
        role_uid="role-uid",
        role_binding_uid="binding-uid",
        execution_kubeconfig=Path("/private/execution-kubeconfig"),
        execution_context="execution-context",
        run_id="libero-manager-evidence",
    )
    binding = module.LiberoRuntimeBinding(
        customer_identity_sha256="8" * 64,
        evidence={"allowed_node_sha256": module.hashlib.sha256(b"worker").hexdigest()},
        access_state=state,
        candidate_image=candidate,
        task_name=module.LIBERO_PROFILE_TASK_NAME,
    )
    node = {
        "metadata": {
            "name": "worker",
            "uid": "worker-uid",
            "creationTimestamp": "2026-09-12T00:00:00Z",
            "labels": {"nvidia.com/gpu.product": "NVIDIA-B200"},
        },
        "spec": {"providerID": "nebius://fixture-instance"},
        "status": {
            "allocatable": {"nvidia.com/gpu": "8", "nvidia.com/b200": "8"},
            "nodeInfo": {
                "machineID": "fixture-machine",
                "systemUUID": "fixture-system-uuid",
            },
        },
    }
    if drift == "node":
        pod["spec"]["nodeName"] = "foreign-worker"
    elif drift == "b200-resource":
        node["status"]["allocatable"].pop("nvidia.com/b200")
    elif drift == "gpu-request":
        pod["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = "2"
    elif drift == "image-status":
        pod["status"]["containerStatuses"][0]["imageID"] = (
            "docker-pullable://example.invalid/other@sha256:" + "8" * 64
        )

    def kubectl_json(arguments, **_kwargs):
        if "pods" in arguments:
            return {"items": [pod]}
        return node

    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    if drift != "none":
        with pytest.raises(RuntimeError):
            module._libero_manager_live_evidence(binding, "73")
        return

    evidence = module._libero_manager_live_evidence(binding, "73")
    assert evidence["schema"] == "npa.libero.manager-live-evidence.v1"
    assert evidence["pod_observed_image_digest"] == candidate.rsplit("@", 1)[1]
    assert evidence["gpu_family"] == "B200"
    assert evidence["pod_gpu_count"] == 1
    assert evidence["node_allocatable_gpu_count"] == 8
    assert evidence["node_allocatable_b200_count"] == 8
    assert evidence["provider_identity_sha256"]
    assert evidence["device_resource_id_sha256"]


def test_libero_payload_role_binds_uid_atomically_to_observed_pod(
    monkeypatch,
) -> None:
    module = _load_module()
    pod = _libero_payload_pod_fixture(module)
    candidate = pod["spec"]["containers"][0]["image"]
    state = module.LiberoAccessState(
        kubeconfig=Path("/private/payload-kubeconfig"),
        context="payload-context",
        namespace="isolated-namespace",
        namespace_uid="namespace-uid",
        service_account_uid="service-account-uid",
        role_uid="role-uid",
        role_binding_uid="binding-uid",
        execution_kubeconfig=Path("/private/execution-kubeconfig"),
        execution_context="execution-context",
        run_id="libero-payload-binding",
    )
    base_evidence = {
        "service_account_uid_sha256": "1" * 64,
        "role_uid_sha256": "2" * 64,
        "role_binding_uid_sha256": "3" * 64,
        "rbac_spec_sha256": "4" * 64,
        "namespace_sha256": "5" * 64,
        "namespace_uid_sha256": "6" * 64,
    }
    binding = module.LiberoRuntimeBinding(
        customer_identity_sha256="8" * 64,
        evidence=base_evidence,
        access_state=state,
        candidate_image=candidate,
        task_name=module.LIBERO_PROFILE_TASK_NAME,
    )
    patches: list[list[dict[str, object]]] = []

    def kubectl_json(arguments, **_kwargs):
        if "patch" not in arguments:
            return {"items": [pod]}
        patch = json.loads(arguments[arguments.index("--patch") + 1])
        patches.append(patch)
        if "pod" in arguments:
            pod["metadata"].setdefault("annotations", {})[
                module.LIBERO_PAYLOAD_RELEASE_ANNOTATION
            ] = patch[1]["value"]
            return pod
        return {
            "metadata": {"uid": "role-uid"},
            "rules": module._libero_payload_rules(pod["metadata"]["name"]),
        }

    bound_rbac = {
        **base_evidence,
        "rbac_spec_sha256": "7" * 64,
    }
    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    monkeypatch.setattr(
        module,
        "_libero_rbac_evidence",
        lambda *_a, **_k: (bound_rbac, state),
    )

    bound = module._bind_libero_payload_pod_access(
        binding, "73", timeout=1, poll_interval=1
    )

    assert patches[0][0] == {
        "op": "test",
        "path": "/metadata/uid",
        "value": "role-uid",
    }
    assert patches[0][1]["value"] == module._libero_payload_rules(
        module.LIBERO_PAYLOAD_UNBOUND_RESOURCE_NAME
    )
    assert patches[0][2]["value"] == module._libero_payload_rules(
        pod["metadata"]["name"]
    )
    assert bound.access_state.payload_pod_name == pod["metadata"]["name"]
    assert bound.access_state.payload_pod_uid == "payload-pod-uid"
    assert bound.evidence["bound_rbac_spec_sha256"] == "7" * 64
    assert (
        bound.evidence["scheduler_job_id_sha256"]
        == module.hashlib.sha256(b"73").hexdigest()
    )
    assert patches[1][0] == {
        "op": "test",
        "path": "/metadata/uid",
        "value": "payload-pod-uid",
    }
    assert patches[1][1]["path"].endswith("libero-release")


def test_libero_inventory_refuses_cluster_role_binding_for_namespace(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_module()
    namespace = "isolated-namespace"
    expected = {
        "pods": [],
        "serviceaccounts": [
            "default",
            "npa-byof-libero-payload",
            "skypilot-service-account",
        ],
        "roles": [
            "npa-byof-libero-pod-reader",
            "skypilot-service-account-role",
        ],
        "rolebindings": [
            "npa-byof-libero-payload-pod-reader",
            "skypilot-service-account-role-binding",
        ],
        "secrets": [],
        "configmaps": sorted(
            [
                "kube-root-ca.crt",
                module.LIBERO_STORAGE_VERIFICATION_CONFIGMAP,
                module.LIBERO_CALLER_VERIFICATION_CONFIGMAP,
                module.LIBERO_CUSTOMER_REGISTRATION_CONFIGMAP,
            ]
        ),
        "services": [],
    }

    caller_key = module.base64.b64encode(bytes(range(32, 64))).decode()
    registration_key = module.base64.b64encode(bytes(range(64, 96))).decode()
    identity = "8" * 64
    caller_path = tmp_path / "caller.b64"
    caller_path.write_text(caller_key)
    caller_path.chmod(0o600)
    registration_path = tmp_path / (identity + ".b64")
    registration_path.write_text(registration_key)
    registration_path.chmod(0o600)
    monkeypatch.setenv(
        "NPA_LIBERO_AUTHENTICATED_CALLER_PUBLIC_KEY_FILE", str(caller_path)
    )
    monkeypatch.setenv(
        "NPA_LIBERO_CUSTOMER_SIGNER_REGISTRATION_FILE", str(registration_path)
    )
    monkeypatch.setenv("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", identity)
    runtime_roots = [
        {
            "metadata": {
                "name": module.LIBERO_CALLER_VERIFICATION_CONFIGMAP,
                "namespace": namespace,
                "uid": "caller-uid",
            },
            "immutable": True,
            "data": {"authenticated-caller-public-key.b64": caller_key},
        },
        {
            "metadata": {
                "name": module.LIBERO_CUSTOMER_REGISTRATION_CONFIGMAP,
                "namespace": namespace,
                "uid": "registration-uid",
            },
            "immutable": True,
            "data": {identity + ".b64": registration_key},
        },
    ]
    storage_key = bytes(range(32))
    monkeypatch.setattr(module, "libero_image_manifest", lambda: {})
    monkeypatch.setattr(
        module,
        "validate_libero_qualified_image_manifest",
        lambda _manifest: {
            "output_storage_authorization_public_key_sha256": module.hashlib.sha256(
                storage_key
            ).hexdigest()
        },
    )

    def kubectl_json(arguments, **_kwargs):
        if arguments[-1] == "configmaps":
            return {
                "items": [
                    *runtime_roots,
                    {
                        "metadata": {
                            "name": "kube-root-ca.crt",
                            "namespace": namespace,
                            "uid": "root-ca-uid",
                        },
                        "data": {"ca.crt": "synthetic public certificate"},
                    },
                    {
                        "metadata": {
                            "name": module.LIBERO_STORAGE_VERIFICATION_CONFIGMAP,
                            "namespace": namespace,
                            "uid": "storage-key-uid",
                        },
                        "immutable": True,
                        "data": {
                            "output-storage-authorization-public-key.b64": module.base64.b64encode(
                                storage_key
                            ).decode()
                        },
                    },
                ]
            }
        kind = arguments[-1]
        return {"items": [{"metadata": {"name": name}} for name in expected[kind]]}

    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    inventory = module._libero_namespaced_inventory(
        Path("/private/payload-kubeconfig"), "payload-context", namespace
    )
    assert len(inventory.pop("configmap_records")) == 4
    assert inventory == expected

    expected["services"] = ["selector-trap"]
    with pytest.raises(RuntimeError, match="reviewed services inventory"):
        module._libero_namespaced_inventory(
            Path("/private/payload-kubeconfig"), "payload-context", namespace
        )
    expected["services"] = []

    def broadened(arguments, **kwargs):
        if arguments[-1] == "rolebindings":
            return {"items": []}
        if arguments[-1] != "clusterrolebindings":
            return kubectl_json(arguments, **kwargs)
        return {
            "items": [
                {
                    "metadata": {"name": "unexpected-binding"},
                    "subjects": [
                        {
                            "kind": "ServiceAccount",
                            "name": "npa-byof-libero-payload",
                            "namespace": namespace,
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(module, "_kubectl_json", broadened)
    with pytest.raises(RuntimeError, match="may not receive ClusterRoleBindings"):
        module._libero_external_rbac_inventory_sha256(
            Path("/private/payload-kubeconfig"), "payload-context", namespace
        )


def test_libero_controller_rbac_is_exact_and_namespace_scoped(monkeypatch) -> None:
    module = _load_module()
    namespace = "isolated-namespace"

    def metadata(name: str) -> dict[str, str]:
        return {
            "name": name,
            "namespace": namespace,
            "uid": f"uid-{name}",
            "creationTimestamp": "2026-09-12T00:00:00Z",
        }

    objects = {
        "skypilot-service-account": {"metadata": metadata("skypilot-service-account")},
        "skypilot-service-account-role": {
            "metadata": metadata("skypilot-service-account-role"),
            "rules": module.LIBERO_CONTROLLER_RULES,
        },
        "skypilot-service-account-role-binding": {
            "metadata": metadata("skypilot-service-account-role-binding"),
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": "skypilot-service-account",
                    "namespace": namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": "skypilot-service-account-role",
            },
        },
    }

    def kubectl_json(arguments, **_kwargs):
        if arguments[-1] in {"clusterrolebindings", "rolebindings"}:
            return {"items": []}
        return objects[arguments[-1]]

    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    evidence, identities = module._libero_controller_rbac_evidence(
        Path("/private/execution-kubeconfig"), "execution-context", namespace
    )
    assert set(evidence) == {
        "controller_service_account_uid_sha256",
        "controller_role_uid_sha256",
        "controller_role_binding_uid_sha256",
        "controller_rbac_spec_sha256",
        "external_rbac_inventory_sha256",
    }
    assert identities == {
        "controller_service_account_uid": "uid-skypilot-service-account",
        "controller_role_uid": "uid-skypilot-service-account-role",
        "controller_role_binding_uid": "uid-skypilot-service-account-role-binding",
    }
    objects["skypilot-service-account-role"]["rules"] = [
        {"apiGroups": ["*"], "resources": ["*"], "verbs": ["*"]}
    ]
    with pytest.raises(RuntimeError, match="namespace-only contract"):
        module._libero_controller_rbac_evidence(
            Path("/private/execution-kubeconfig"), "execution-context", namespace
        )
    objects["skypilot-service-account-role"]["rules"] = module.LIBERO_CONTROLLER_RULES

    objects["skypilot-service-account-role-binding"]["roleRef"]["kind"] = "ClusterRole"
    with pytest.raises(RuntimeError, match="RoleBinding differs"):
        module._libero_controller_rbac_evidence(
            Path("/private/execution-kubeconfig"), "execution-context", namespace
        )
    objects["skypilot-service-account-role-binding"]["roleRef"]["kind"] = "Role"

    def cluster_bound(arguments, **kwargs):
        if arguments[-1] != "clusterrolebindings":
            return kubectl_json(arguments, **kwargs)
        return {
            "items": [
                {
                    "subjects": [
                        {
                            "kind": "ServiceAccount",
                            "name": "skypilot-service-account",
                            "namespace": namespace,
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(module, "_kubectl_json", cluster_bound)
    with pytest.raises(RuntimeError, match="may not receive ClusterRoleBindings"):
        module._libero_controller_rbac_evidence(
            Path("/private/execution-kubeconfig"), "execution-context", namespace
        )


@pytest.mark.parametrize(
    "subject",
    [
        {"kind": "Group", "name": "system:serviceaccounts"},
        {"kind": "Group", "name": "system:serviceaccounts:isolated-namespace"},
        {
            "kind": "User",
            "name": "system:serviceaccount:isolated-namespace:any-account",
        },
    ],
)
def test_libero_rejects_indirect_cluster_binding_to_namespace_accounts(
    monkeypatch, subject
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_kubectl_json",
        lambda *_args, **_kwargs: {
            "items": [
                {
                    "subjects": [subject],
                    "roleRef": {"kind": "ClusterRole", "name": "cluster-admin"},
                }
            ]
        },
    )

    with pytest.raises(RuntimeError, match="may not receive ClusterRoleBindings"):
        module._reject_libero_cluster_role_bindings(
            Path("/private/payload-kubeconfig"),
            "payload-context",
            "isolated-namespace",
        )


def test_libero_rejects_workload_access_for_authenticated_group(monkeypatch) -> None:
    module = _load_module()

    def kubectl_json(arguments, **_kwargs):
        if arguments[-1] == "clusterrolebindings":
            return {
                "items": [
                    {
                        "metadata": {
                            "name": "unsafe-binding",
                            "uid": "unsafe-binding-uid",
                        },
                        "subjects": [{"kind": "Group", "name": "system:authenticated"}],
                        "roleRef": {
                            "apiGroup": "rbac.authorization.k8s.io",
                            "kind": "ClusterRole",
                            "name": "unsafe-pod-reader",
                        },
                    }
                ]
            }
        assert arguments[-2:] == ["clusterrole", "unsafe-pod-reader"]
        return {
            "metadata": {"uid": "unsafe-role-uid"},
            "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}],
        }

    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    with pytest.raises(RuntimeError, match="may not receive workload resource"):
        module._reject_libero_cluster_role_bindings(
            Path("/private/payload-kubeconfig"),
            "payload-context",
            "isolated-namespace",
        )


def test_libero_allows_only_discovery_for_authenticated_group(monkeypatch) -> None:
    module = _load_module()

    def kubectl_json(arguments, **_kwargs):
        if arguments[-1] == "clusterrolebindings":
            return {
                "items": [
                    {
                        "metadata": {
                            "name": "discovery-binding",
                            "uid": "discovery-binding-uid",
                        },
                        "subjects": [{"kind": "Group", "name": "system:authenticated"}],
                        "roleRef": {
                            "apiGroup": "rbac.authorization.k8s.io",
                            "kind": "ClusterRole",
                            "name": "system:discovery",
                        },
                    }
                ]
            }
        assert arguments[-2:] == ["clusterrole", "system:discovery"]
        return {
            "metadata": {"uid": "discovery-role-uid"},
            "rules": [{"nonResourceURLs": ["/api", "/apis"], "verbs": ["get"]}],
        }

    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    module._reject_libero_cluster_role_bindings(
        Path("/private/payload-kubeconfig"),
        "payload-context",
        "isolated-namespace",
    )


@pytest.mark.parametrize("non_resource_url", ["/metrics", "/*"])
def test_libero_rejects_broad_public_group_non_resource_urls(
    non_resource_url,
) -> None:
    module = _load_module()

    assert not module._libero_safe_public_group_rules(
        [{"nonResourceURLs": [non_resource_url], "verbs": ["get"]}]
    )


@pytest.mark.parametrize(
    "subject",
    [
        {
            "kind": "ServiceAccount",
            "name": "npa-byof-libero-payload",
            "namespace": "isolated-namespace",
        },
        {
            "kind": "User",
            "name": "system:serviceaccount:isolated-namespace:npa-byof-libero-payload",
        },
        {"kind": "Group", "name": "system:serviceaccounts"},
        {"kind": "Group", "name": "system:serviceaccounts:isolated-namespace"},
    ],
)
def test_libero_rejects_cross_namespace_role_binding_to_run_identity(
    monkeypatch, subject
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_kubectl_json",
        lambda *_args, **_kwargs: {
            "items": [
                {
                    "metadata": {
                        "name": "pre-staged-access",
                        "namespace": "other-namespace",
                        "uid": "pre-staged-binding-uid",
                    },
                    "subjects": [subject],
                    "roleRef": {
                        "apiGroup": "rbac.authorization.k8s.io",
                        "kind": "Role",
                        "name": "pod-reader",
                    },
                }
            ]
        },
    )

    with pytest.raises(RuntimeError, match="cross-namespace RoleBindings"):
        module._reject_libero_cross_namespace_role_bindings(
            Path("/private/payload-kubeconfig"),
            "payload-context",
            "isolated-namespace",
        )


def test_libero_external_rbac_inventory_is_hash_bound_and_drift_sensitive(
    monkeypatch,
) -> None:
    module = _load_module()
    role_uid = "safe-role-uid"

    def kubectl_json(arguments, **_kwargs):
        if arguments[-1] == "clusterrolebindings":
            return {
                "items": [
                    {
                        "metadata": {"name": "discovery", "uid": "binding-uid"},
                        "subjects": [{"kind": "Group", "name": "system:authenticated"}],
                        "roleRef": {
                            "apiGroup": "rbac.authorization.k8s.io",
                            "kind": "ClusterRole",
                            "name": "system:discovery",
                        },
                    }
                ]
            }
        if arguments[-1] == "rolebindings":
            return {"items": []}
        assert arguments[-2:] == ["clusterrole", "system:discovery"]
        return {
            "metadata": {"uid": role_uid},
            "rules": [{"nonResourceURLs": ["/api"], "verbs": ["get"]}],
        }

    monkeypatch.setattr(module, "_kubectl_json", kubectl_json)
    first = module._libero_external_rbac_inventory_sha256(
        Path("/private/payload-kubeconfig"),
        "payload-context",
        "isolated-namespace",
    )
    role_uid = "replacement-role-uid"
    second = module._libero_external_rbac_inventory_sha256(
        Path("/private/payload-kubeconfig"),
        "payload-context",
        "isolated-namespace",
    )

    assert first != second


def test_libero_cleanup_preserves_kubeconfig_outside_run_state(tmp_path) -> None:
    module = _load_module()
    run_id = "libero-local-cleanup"
    state_root = tmp_path / run_id
    state_root.mkdir(mode=0o700)
    kubeconfig = tmp_path / "durable-operator-kubeconfig"
    kubeconfig.write_text("fixture\n", encoding="utf-8")
    kubeconfig.chmod(0o600)

    result = module._cleanup_libero_local_state(
        isolated_state_root=state_root,
        payload_kubeconfig=kubeconfig,
        run_id=run_id,
    )

    assert result.ok is False
    assert kubeconfig.is_file()
    assert state_root.is_dir()


@pytest.mark.parametrize(
    ("failure_stage", "failure_recheck"),
    [
        ("write", None),
        ("storage", None),
        ("provider", None),
        ("payload", 1),
        ("payload", 2),
        ("controller", 1),
        ("controller", 2),
    ],
    ids=[
        "write",
        "storage",
        "provider",
        "payload-first",
        "payload-second",
        "controller-first",
        "controller-second",
    ],
)
def test_libero_pre_submit_failure_always_runs_no_scheduler_access_cleanup(
    monkeypatch, tmp_path, failure_stage, failure_recheck
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    args.cleanup = True
    args.solution_name = "libero"
    payload_kubeconfig = tmp_path / "payload-kubeconfig-source"
    payload_kubeconfig.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    payload_kubeconfig.chmod(0o600)
    monkeypatch.setenv("NPA_LIBERO_PAYLOAD_KUBECONFIG", str(payload_kubeconfig))
    execution_kubeconfig = tmp_path / "execution-kubeconfig"
    execution_kubeconfig.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    execution_kubeconfig.chmod(0o600)
    monkeypatch.setenv("KUBECONFIG", str(execution_kubeconfig))
    sky_config = tmp_path / "skypilot.yaml"
    sky_config.chmod(0o600)
    args.config_path = str(sky_config)
    isolated = tmp_path / args.run_id
    isolated.mkdir(mode=0o700)
    args.isolated_config_dir = str(isolated)
    binding = module.LiberoRuntimeBinding(
        customer_identity_sha256="8" * 64,
        customer_authorization_b64="signed-customer-authorization",
        evidence={},
        access_state=module.LiberoAccessState(
            kubeconfig=payload_kubeconfig,
            context="payload-context",
            namespace="isolated-namespace",
            namespace_uid="namespace-uid",
            service_account_uid="service-account-uid",
            role_uid="role-uid",
            role_binding_uid="role-binding-uid",
            execution_kubeconfig=Path(os.environ["KUBECONFIG"]),
            execution_context="execution-context",
            run_id=args.run_id,
        ),
    )
    events: list[str] = []
    installed: list[object] = []
    submission_calls: list[str] = []
    scheduler_cleanup_ids: list[str] = []
    bound_objects = {
        "namespace",
        "payload-service-account",
        "payload-role",
        "payload-role-binding",
        "controller-service-account",
        "controller-role",
        "controller-role-binding",
    }
    guard = SimpleNamespace(
        run_id=args.run_id,
        timeout=max(int(args.submit_timeout), 1),
        isolated_config_dir=isolated,
        mark_launched=lambda **_k: None,
    )

    monkeypatch.setattr(module, "_is_libero_invocation", lambda *_a: True)
    monkeypatch.setattr(
        module, "_libero_global_config_path", lambda selected: selected.config_path
    )
    monkeypatch.setattr(
        module,
        "_bind_libero_runtime_contract",
        lambda *_a, **_k: events.append("binding") or binding,
    )
    monkeypatch.setattr(
        module,
        "install_teardown_signal_handlers",
        lambda callback: (
            events.append("handlers") or installed.append(callback) or None
        ),
    )
    monkeypatch.setattr(module, "restore_signal_handlers", lambda *_a: None)
    monkeypatch.setattr(module, "SignalTeardown", lambda **_k: guard)

    def cleanup_access(*_args, **_kwargs):
        events.append("access")
        bound_objects.clear()
        return _verified_cleanup(module, "all-bound-access")

    monkeypatch.setattr(module, "_cleanup_libero_access_objects", cleanup_access)
    monkeypatch.setattr(module, "_stop_sky_api", lambda **_k: events.append("api"))
    monkeypatch.setattr(
        module,
        "_cleanup_libero_local_state",
        lambda **_k: (
            events.append("local") or _verified_cleanup(module, "libero-local")
        ),
    )

    def cleanup_scheduler(job_id, **_kwargs):
        scheduler_cleanup_ids.append(job_id)
        return module.CleanupResult()

    monkeypatch.setattr(module, "_cancel_then_teardown_managed_job", cleanup_scheduler)

    def submit(*_args, **_kwargs):
        submission_calls.append("submit")
        pytest.fail("submission must not start after a failed integrity recheck")

    monkeypatch.setattr(module, "submit_workflow", submit)

    def fail() -> None:
        raise RuntimeError(f"fixture {failure_stage} failure")

    recheck_calls = {"payload": 0, "controller": 0}

    def recheck(stage: str) -> None:
        events.append(stage)
        recheck_calls[stage] += 1
        if failure_stage == stage and recheck_calls[stage] == failure_recheck:
            fail()

    if failure_stage == "write":
        monkeypatch.setattr(
            module,
            "_write_yaml_documents",
            lambda *_a: events.append("write") or fail(),
        )
    elif failure_stage == "storage":
        monkeypatch.setattr(
            module,
            "preflight_output_storage",
            lambda **_k: events.append("storage") or fail(),
        )
    elif failure_stage == "provider":
        monkeypatch.setattr(
            module,
            "_ensure_infra_enabled",
            lambda **_k: events.append("provider") or fail(),
        )
    else:
        monkeypatch.setattr(
            module,
            "_verify_libero_payload_unchanged",
            lambda *_a, **_k: recheck("payload"),
        )
        monkeypatch.setattr(
            module,
            "_verify_libero_controller_unchanged",
            lambda *_a, **_k: recheck("controller"),
        )
    if failure_stage not in {"payload", "controller"}:
        monkeypatch.setattr(
            module,
            "_verify_libero_payload_unchanged",
            lambda *_a, **_k: events.append("payload"),
        )
        monkeypatch.setattr(
            module,
            "_verify_libero_controller_unchanged",
            lambda *_a, **_k: events.append("controller"),
        )

    if failure_stage != "provider":
        monkeypatch.setattr(
            module, "_ensure_infra_enabled", lambda **_k: events.append("provider")
        )
    if failure_stage != "storage":
        monkeypatch.setattr(
            module,
            "preflight_output_storage",
            lambda **_k: (
                events.append("storage")
                or module.OutputPrefixLease(
                    "bucket", "prefix/.npa-output-lease", '"etag"', "version"
                )
            ),
        )
    if failure_stage != "write":
        monkeypatch.setattr(
            module,
            "_write_yaml_documents",
            lambda *_a: events.append("write"),
        )
    monkeypatch.setattr(module, "_release_output_prefix_lease", lambda *_a: None)

    if failure_recheck == 2:
        assert module._submit_and_wait(args) == 1
    else:
        with pytest.raises(RuntimeError, match=f"fixture {failure_stage} failure"):
            module._submit_and_wait(args)

    assert len(installed) == 1
    assert submission_calls == []
    assert events.index("handlers") < events.index("provider")
    if "storage" in events:
        assert events.index("provider") < events.index("storage")
    if "write" in events:
        assert events.index("storage") < events.index("write")
    if "payload" in events and "storage" in events:
        assert events.index("payload") < events.index("storage")
    if "controller" in events and "storage" in events:
        assert events.index("controller") < events.index("storage")
    assert events[-3:] == ["access", "api", "local"]
    assert not bound_objects
    assert all(job_id == "" for job_id in scheduler_cleanup_ids)


def test_libero_payload_grant_recheck_refuses_hash_and_identity_drift(
    monkeypatch,
) -> None:
    module = _load_module()
    state = module.LiberoAccessState(
        kubeconfig=Path("/private/payload-kubeconfig"),
        context="payload-context",
        namespace="isolated-namespace",
        namespace_uid="namespace-uid",
        service_account_uid="service-account-uid",
        role_uid="role-uid",
        role_binding_uid="binding-uid",
        run_id="libero-payload-recheck",
    )
    binding = module.LiberoRuntimeBinding(
        customer_identity_sha256="8" * 64,
        evidence={
            "role_uid_sha256": "a" * 64,
            "namespace_inventory_sha256": "b" * 64,
        },
        access_state=state,
    )
    calls = []

    def unchanged(*_args, require_empty_inventory, expected_pod_name):
        calls.append(require_empty_inventory)
        assert expected_pod_name == module.LIBERO_PAYLOAD_UNBOUND_RESOURCE_NAME
        evidence = {"role_uid_sha256": "a" * 64}
        if require_empty_inventory:
            evidence["namespace_inventory_sha256"] = "b" * 64
        return evidence, state

    monkeypatch.setattr(module, "_libero_rbac_evidence", unchanged)
    module._verify_libero_payload_unchanged(binding, require_empty_inventory=True)
    module._verify_libero_payload_unchanged(binding, require_empty_inventory=False)
    assert calls == [True, False]

    monkeypatch.setattr(
        module,
        "_libero_rbac_evidence",
        lambda *_args, **_kwargs: ({"role_uid_sha256": "c" * 64}, state),
    )
    with pytest.raises(RuntimeError, match="payload RBAC changed"):
        module._verify_libero_payload_unchanged(binding, require_empty_inventory=False)

    changed_identity = module.replace(state, role_uid="replacement-role-uid")
    monkeypatch.setattr(
        module,
        "_libero_rbac_evidence",
        lambda *_args, **_kwargs: (
            {"role_uid_sha256": "a" * 64},
            changed_identity,
        ),
    )
    with pytest.raises(RuntimeError, match="payload RBAC identity changed"):
        module._verify_libero_payload_unchanged(binding, require_empty_inventory=False)


def test_libero_cleanup_deletes_uid_bound_access_and_namespace(monkeypatch) -> None:
    module = _load_module()
    calls: list[tuple[str, str, str]] = []

    class ApiException(Exception):
        def __init__(self, status: int):
            super().__init__(status)
            self.status = status

    class Preconditions:
        def __init__(self, *, uid: str):
            self.uid = uid

    class DeleteOptions:
        def __init__(self, *, preconditions: Preconditions):
            self.preconditions = preconditions

    def absent(*_args, **_kwargs):
        raise ApiException(404)

    class Core:
        def __init__(self, _client):
            pass

        def delete_namespaced_service_account(self, name, namespace, *, body):
            calls.append(("serviceaccount", name, body.preconditions.uid))

        read_namespaced_service_account = staticmethod(absent)

        def delete_namespace(self, name, *, body):
            calls.append(("namespace", name, body.preconditions.uid))

        read_namespace = staticmethod(absent)

    class Rbac:
        def __init__(self, _client):
            pass

        def delete_namespaced_role_binding(self, name, namespace, *, body):
            calls.append(("rolebinding", name, body.preconditions.uid))

        read_namespaced_role_binding = staticmethod(absent)

        def delete_namespaced_role(self, name, namespace, *, body):
            calls.append(("role", name, body.preconditions.uid))

        read_namespaced_role = staticmethod(absent)

    client_module = ModuleType("kubernetes.client")
    client_module.CoreV1Api = Core
    client_module.RbacAuthorizationV1Api = Rbac
    client_module.V1DeleteOptions = DeleteOptions
    client_module.V1Preconditions = Preconditions
    exceptions_module = ModuleType("kubernetes.client.exceptions")
    exceptions_module.ApiException = ApiException
    config_module = ModuleType("kubernetes.config")
    config_module.new_client_from_config = lambda **_kwargs: object()
    package = ModuleType("kubernetes")
    package.client = client_module
    package.config = config_module
    monkeypatch.setitem(sys.modules, "kubernetes", package)
    monkeypatch.setitem(sys.modules, "kubernetes.client", client_module)
    monkeypatch.setitem(sys.modules, "kubernetes.client.exceptions", exceptions_module)
    monkeypatch.setitem(sys.modules, "kubernetes.config", config_module)
    state = module.LiberoAccessState(
        kubeconfig=Path("/private/payload-kubeconfig"),
        context="payload-context",
        namespace="isolated-namespace",
        namespace_uid="namespace-uid",
        service_account_uid="service-account-uid",
        role_uid="role-uid",
        role_binding_uid="binding-uid",
        controller_service_account_uid="controller-account-uid",
        controller_role_uid="controller-role-uid",
        controller_role_binding_uid="controller-binding-uid",
    )

    result = module._cleanup_libero_access_objects(state, timeout=1)

    assert result.ok is True
    assert result.verified is True
    assert result.remote_absence_verified is True
    assert calls == [
        (
            "rolebinding",
            module.LIBERO_CONTROLLER_ROLE_BINDING,
            "controller-binding-uid",
        ),
        ("role", module.LIBERO_CONTROLLER_ROLE, "controller-role-uid"),
        (
            "serviceaccount",
            module.SKYPILOT_ENGINE_SERVICE_ACCOUNT,
            "controller-account-uid",
        ),
        ("rolebinding", module.LIBERO_PAYLOAD_ROLE_BINDING, "binding-uid"),
        ("role", module.LIBERO_PAYLOAD_ROLE, "role-uid"),
        (
            "serviceaccount",
            module.LIBERO_PAYLOAD_SERVICE_ACCOUNT,
            "service-account-uid",
        ),
        ("namespace", "isolated-namespace", "namespace-uid"),
    ]


@pytest.mark.parametrize("failure_operation", ["delete", "read"])
@pytest.mark.parametrize("failure_index", range(7))
def test_libero_cleanup_attempts_every_uid_bound_position_after_failure(
    monkeypatch, failure_operation, failure_index
) -> None:
    module = _load_module()
    delete_calls: list[str] = []
    read_calls: list[str] = []
    labels = [
        "controller-rolebinding",
        "controller-role",
        "controller-serviceaccount",
        "rolebinding",
        "role",
        "serviceaccount",
        "namespace",
    ]

    class ApiException(Exception):
        def __init__(self, status: int):
            super().__init__(status)
            self.status = status

    class Preconditions:
        def __init__(self, *, uid: str):
            self.uid = uid

    class DeleteOptions:
        def __init__(self, *, preconditions: Preconditions):
            self.preconditions = preconditions

    def operation(kind: str, label: str) -> None:
        calls = delete_calls if kind == "delete" else read_calls
        calls.append(label)
        if kind == failure_operation and labels.index(label) == failure_index:
            raise RuntimeError(f"injected {kind} failure at {label}")
        if kind == "read":
            raise ApiException(404)

    class Core:
        def __init__(self, _client):
            pass

        def delete_namespaced_service_account(self, name, *_args, **_kwargs):
            operation(
                "delete",
                "controller-serviceaccount"
                if name == module.SKYPILOT_ENGINE_SERVICE_ACCOUNT
                else "serviceaccount",
            )

        def read_namespaced_service_account(self, name, *_args, **_kwargs):
            operation(
                "read",
                "controller-serviceaccount"
                if name == module.SKYPILOT_ENGINE_SERVICE_ACCOUNT
                else "serviceaccount",
            )

        def delete_namespace(self, *_args, **_kwargs):
            operation("delete", "namespace")

        def read_namespace(self, *_args, **_kwargs):
            operation("read", "namespace")

    class Rbac:
        def __init__(self, _client):
            pass

        def delete_namespaced_role_binding(self, name, *_args, **_kwargs):
            operation(
                "delete",
                "controller-rolebinding"
                if name == module.LIBERO_CONTROLLER_ROLE_BINDING
                else "rolebinding",
            )

        def read_namespaced_role_binding(self, name, *_args, **_kwargs):
            operation(
                "read",
                "controller-rolebinding"
                if name == module.LIBERO_CONTROLLER_ROLE_BINDING
                else "rolebinding",
            )

        def delete_namespaced_role(self, name, *_args, **_kwargs):
            operation(
                "delete",
                "controller-role" if name == module.LIBERO_CONTROLLER_ROLE else "role",
            )

        def read_namespaced_role(self, name, *_args, **_kwargs):
            operation(
                "read",
                "controller-role" if name == module.LIBERO_CONTROLLER_ROLE else "role",
            )

    client_module = ModuleType("kubernetes.client")
    client_module.CoreV1Api = Core
    client_module.RbacAuthorizationV1Api = Rbac
    client_module.V1DeleteOptions = DeleteOptions
    client_module.V1Preconditions = Preconditions
    exceptions_module = ModuleType("kubernetes.client.exceptions")
    exceptions_module.ApiException = ApiException
    config_module = ModuleType("kubernetes.config")
    config_module.new_client_from_config = lambda **_kwargs: object()
    package = ModuleType("kubernetes")
    package.client = client_module
    package.config = config_module
    monkeypatch.setitem(sys.modules, "kubernetes", package)
    monkeypatch.setitem(sys.modules, "kubernetes.client", client_module)
    monkeypatch.setitem(sys.modules, "kubernetes.client.exceptions", exceptions_module)
    monkeypatch.setitem(sys.modules, "kubernetes.config", config_module)
    state = module.LiberoAccessState(
        kubeconfig=Path("/private/payload-kubeconfig"),
        context="payload-context",
        namespace="isolated-namespace",
        namespace_uid="namespace-uid",
        service_account_uid="service-account-uid",
        role_uid="role-uid",
        role_binding_uid="binding-uid",
        controller_service_account_uid="controller-account-uid",
        controller_role_uid="controller-role-uid",
        controller_role_binding_uid="controller-binding-uid",
    )

    result = module._cleanup_libero_access_objects(state, timeout=1)

    assert delete_calls == labels
    assert read_calls == labels
    assert result.ok is False
    assert result.verified is False
    assert result.remote_absence_verified is False


def test_libero_cleanup_removes_only_exact_run_local_state(tmp_path) -> None:
    module = _load_module()
    run_id = "libero-local-cleanup"
    state_root = tmp_path / run_id
    state_root.mkdir(mode=0o700)
    (state_root / "state").write_text("fixture\n", encoding="utf-8")
    kubeconfig = state_root / "payload-kubeconfig"
    kubeconfig.write_text("fixture\n", encoding="utf-8")
    kubeconfig.chmod(0o600)

    result = module._cleanup_libero_local_state(
        isolated_state_root=state_root,
        payload_kubeconfig=kubeconfig,
        run_id=run_id,
    )

    assert result.ok is True
    assert result.verified is True
    assert result.remote_absence_verified is True
    assert not state_root.exists()
    assert not kubeconfig.exists()


def _verified_cleanup(module, resource: str) -> object:
    result = module.CleanupResult(resources_removed=[resource])
    result.verified = True
    result.remote_absence_verified = True
    return result


def test_complete_libero_cleanup_orders_remote_api_and_local_absence(
    monkeypatch,
) -> None:
    module = _load_module()
    events: list[str] = []
    binding = SimpleNamespace(
        access_state=SimpleNamespace(kubeconfig=Path("/private/payload-kubeconfig"))
    )
    monkeypatch.setattr(
        module,
        "_cleanup_libero_access_objects",
        lambda *_a, **_k: (
            events.append("access") or _verified_cleanup(module, "libero-access")
        ),
    )
    monkeypatch.setattr(module, "_stop_sky_api", lambda **_k: events.append("api"))
    monkeypatch.setattr(
        module,
        "_cleanup_libero_local_state",
        lambda **_k: (
            events.append("local") or _verified_cleanup(module, "libero-local")
        ),
    )

    result, attempted, stopped = module._complete_libero_cleanup(
        _verified_cleanup(module, "managed"),
        binding=binding,
        timeout=1,
        sky_bin="sky",
        isolated_config_dir=Path("/private/exact-run"),
        config_path=Path("/private/skypilot.yaml"),
        run_id="exact-run",
    )

    assert events == ["access", "api", "local"]
    assert attempted is True
    assert stopped is True
    assert result.ok is True
    assert result.verified is True
    assert result.remote_absence_verified is True
    assert result.resources_removed == [
        "managed",
        "libero-access",
        "libero-skypilot-api",
        "libero-local",
    ]


def test_complete_libero_cleanup_preserves_recovery_state_on_access_failure(
    monkeypatch,
) -> None:
    module = _load_module()
    binding = SimpleNamespace(
        access_state=SimpleNamespace(kubeconfig=Path("/private/payload-kubeconfig"))
    )
    failed = module.CleanupResult(errors=["access absence is unverified"])
    monkeypatch.setattr(
        module, "_cleanup_libero_access_objects", lambda *_a, **_k: failed
    )
    monkeypatch.setattr(
        module,
        "_stop_sky_api",
        lambda **_k: pytest.fail("API must stay available for recovery"),
    )
    monkeypatch.setattr(
        module,
        "_cleanup_libero_local_state",
        lambda **_k: pytest.fail("local recovery state must be preserved"),
    )

    result, attempted, stopped = module._complete_libero_cleanup(
        _verified_cleanup(module, "managed"),
        binding=binding,
        timeout=1,
        sky_bin="sky",
        isolated_config_dir=Path("/private/exact-run"),
        config_path=Path("/private/skypilot.yaml"),
        run_id="exact-run",
    )

    assert attempted is False
    assert stopped is False
    assert result.ok is False
    assert result.errors == ["access absence is unverified"]


def test_complete_libero_cleanup_preserves_state_on_api_stop_failure(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    run_id = "exact-run"
    isolated_state = tmp_path / run_id
    isolated_state.mkdir(mode=0o700)
    state_file = isolated_state / "state"
    state_file.write_text("recovery fixture\n", encoding="utf-8")
    payload_kubeconfig = isolated_state / "payload-kubeconfig"
    payload_kubeconfig.write_text("recovery fixture\n", encoding="utf-8")
    payload_kubeconfig.chmod(0o600)
    binding = SimpleNamespace(
        access_state=SimpleNamespace(kubeconfig=payload_kubeconfig)
    )
    events: list[str] = []
    monkeypatch.setattr(
        module,
        "_cleanup_libero_access_objects",
        lambda *_a, **_k: (
            events.append("access") or _verified_cleanup(module, "libero-access")
        ),
    )

    def fail_api_stop(**_kwargs) -> None:
        events.append("api")
        raise module.SkyPilotConfigError("fixture API stop failure")

    monkeypatch.setattr(module, "_stop_sky_api", fail_api_stop)
    monkeypatch.setattr(
        module,
        "_cleanup_libero_local_state",
        lambda **_k: pytest.fail("local recovery state must be preserved"),
    )

    result, attempted, stopped = module._complete_libero_cleanup(
        _verified_cleanup(module, "managed"),
        binding=binding,
        timeout=1,
        sky_bin="sky",
        isolated_config_dir=isolated_state,
        config_path=tmp_path / "skypilot.yaml",
        run_id=run_id,
    )

    assert events == ["access", "api"]
    assert attempted is True
    assert stopped is False
    assert result.ok is False
    assert result.verified is False
    assert result.remote_absence_verified is True
    assert result.errors == [
        "LIBERO SkyPilot API stop failed; local recovery state preserved"
    ]
    assert isolated_state.is_dir()
    assert state_file.read_text(encoding="utf-8") == "recovery fixture\n"
    assert payload_kubeconfig.read_text(encoding="utf-8") == "recovery fixture\n"


def test_libero_signal_callback_completes_every_cleanup_layer(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    args.cleanup = True
    args.solution_name = "libero"
    payload_kubeconfig = tmp_path / "payload-kubeconfig-source"
    payload_kubeconfig.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    payload_kubeconfig.chmod(0o600)
    monkeypatch.setenv("NPA_LIBERO_PAYLOAD_KUBECONFIG", str(payload_kubeconfig))
    execution_kubeconfig = tmp_path / "execution-kubeconfig"
    execution_kubeconfig.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    execution_kubeconfig.chmod(0o600)
    monkeypatch.setenv("KUBECONFIG", str(execution_kubeconfig))
    sky_config = tmp_path / "skypilot.yaml"
    sky_config.chmod(0o600)
    args.config_path = str(sky_config)
    isolated = tmp_path / args.run_id
    isolated.mkdir(mode=0o700)
    args.isolated_config_dir = str(isolated)
    binding = module.LiberoRuntimeBinding(
        customer_identity_sha256="8" * 64,
        customer_authorization_b64="signed-customer-authorization",
        evidence={},
        access_state=module.LiberoAccessState(
            kubeconfig=payload_kubeconfig,
            context="payload-context",
            namespace="isolated-namespace",
            namespace_uid="namespace-uid",
            service_account_uid="service-account-uid",
            role_uid="role-uid",
            role_binding_uid="role-binding-uid",
            execution_kubeconfig=Path(os.environ["KUBECONFIG"]),
            execution_context="execution-context",
            run_id=args.run_id,
        ),
    )
    output_prefix_lease = module.OutputPrefixLease(
        "bucket", "prefix/.npa-output-lease", '"etag"', "version"
    )
    events: list[str] = []
    installed: dict[str, object] = {}
    guard = SimpleNamespace(
        run_id=args.run_id,
        timeout=10,
        isolated_config_dir=isolated,
        mark_launched=lambda **_k: None,
    )

    monkeypatch.setattr(module, "_is_libero_invocation", lambda *_a: True)
    monkeypatch.setattr(
        module, "_libero_global_config_path", lambda selected: selected.config_path
    )
    monkeypatch.setattr(
        module, "_bind_libero_runtime_contract", lambda *_a, **_k: binding
    )
    monkeypatch.setattr(
        module, "_verify_libero_payload_unchanged", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        module, "_verify_libero_controller_unchanged", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        module, "_bind_libero_payload_pod_access", lambda *_a, **_k: binding
    )
    monkeypatch.setattr(
        module, "preflight_output_storage", lambda **_k: output_prefix_lease
    )

    def release_output_prefix_lease(lease):
        assert lease is output_prefix_lease
        events.append("lease")

    monkeypatch.setattr(
        module, "_release_output_prefix_lease", release_output_prefix_lease
    )
    monkeypatch.setattr(module, "resolve_secret_envs", lambda *_a, **_k: [])
    monkeypatch.setattr(module, "SignalTeardown", lambda **_k: guard)

    def install(callback):
        installed["callback"] = callback
        return None

    monkeypatch.setattr(module, "install_teardown_signal_handlers", install)
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id="73", log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "_cancel_then_teardown_managed_job",
        lambda *_a, **_k: (
            events.append("managed") or _verified_cleanup(module, "managed")
        ),
    )
    monkeypatch.setattr(
        module,
        "_cleanup_libero_access_objects",
        lambda *_a, **_k: (
            events.append("access") or _verified_cleanup(module, "libero-access")
        ),
    )
    monkeypatch.setattr(module, "_stop_sky_api", lambda **_k: events.append("api"))
    monkeypatch.setattr(
        module,
        "_cleanup_libero_local_state",
        lambda **_k: (
            events.append("local") or _verified_cleanup(module, "libero-local")
        ),
    )

    def interrupt_wait(*_args, **_kwargs):
        callback = installed["callback"]
        assert callable(callback)
        callback()
        raise SystemExit(143)

    monkeypatch.setattr(module, "_wait_for_terminal", interrupt_wait)

    with pytest.raises(SystemExit, match="143"):
        module._submit_and_wait(args)

    assert events == ["managed", "access", "api", "local", "lease"]


def test_runtime_secret_channel_has_no_invented_wan_consent(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "probe-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "probe-secret")

    assert module.resolve_secret_envs(None, solution_name="wan2.2") == [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]
    # Wan's own NVIDIA gate was removed upstream, so it has no entry and nothing
    # is invented for it; HF_TOKEN is unset here and drops out too.
    assert module.resolve_secret_envs(["HF_TOKEN"], solution_name="wan2.2") == []


def test_openpi_runtime_acceptance_uses_secret_channel(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", "YES")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "probe-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "probe-secret")

    assert module.resolve_secret_envs(None, solution_name="openpi") == [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "NPA_OPENPI_ACCEPT_GEMMA_TERMS",
    ]
    assert module.resolve_secret_envs(["HF_TOKEN"], solution_name="openpi") == [
        "NPA_OPENPI_ACCEPT_GEMMA_TERMS"
    ]


def test_robotwin_gate_evidence_uses_secret_channel(monkeypatch) -> None:
    module = _load_module()
    evidence_names = [
        "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_ARCHIVES",
        "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_SHA256",
        "NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256",
    ]
    for name in evidence_names:
        monkeypatch.setenv(name, "2" if name.endswith("ARCHIVES") else "a" * 64)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    assert module.resolve_secret_envs(None, solution_name="robotwin") == list(
        dict.fromkeys(
            (
                *module.OPERATOR_RUNTIME_ENVS_BY_SOLUTION["robotwin"],
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_ENDPOINT_URL",
                "NEBIUS_S3_ENDPOINT",
            )
        )
    )


def test_robotwin_secret_set_is_exact_and_session_token_is_the_only_optional_name(
    monkeypatch,
) -> None:
    module = _load_module()
    environment = {
        name: f"private-{name.lower()}"
        for name in (
            *module.OPERATOR_RUNTIME_ENVS_BY_SOLUTION["robotwin"],
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ENDPOINT_URL",
            "NEBIUS_S3_ENDPOINT",
            "AWS_SESSION_TOKEN",
            "AWS_SECURITY_TOKEN",
            "UNBOUND_PRIVATE_ALIAS",
        )
    }

    assert module.resolve_secret_envs(
        ["UNBOUND_PRIVATE_ALIAS"],
        solution_name="robotwin",
        environment=environment,
    ) == list(
        dict.fromkeys(
            (
                *module.OPERATOR_RUNTIME_ENVS_BY_SOLUTION["robotwin"],
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_ENDPOINT_URL",
                "NEBIUS_S3_ENDPOINT",
                "AWS_SESSION_TOKEN",
            )
        )
    )


def test_robotwin_inner_coordinates_use_only_secret_values_and_ignore_ambient_controls(
    monkeypatch,
) -> None:
    module = _load_module()
    authorization = SimpleNamespace(
        run_id="robotwin-private-run-canary",
        output_root="s3://private-bucket-canary/output",
        skypilot_config_source="/owner-only/robotwin-skypilot.yaml",
        kubernetes_context="private-context-canary",
    )
    environment = {
        module.CHILD_IMAGE_ENV: "registry.example/private/image@sha256:" + "a" * 64,
        "AWS_ENDPOINT_URL": "https://storage.eu-north1.nebius.cloud",
        "NEBIUS_S3_ENDPOINT": "https://storage.eu-north1.nebius.cloud",
    }
    context = object()
    monkeypatch.setattr(
        module,
        "prepare_inner_submit",
        lambda observed, values: (
            context
            if observed is authorization and values is environment
            else pytest.fail("authorization was not carried in process")
        ),
    )
    for name, value in {
        "NPA_BYOF_DIRECT_LAUNCH": "1",
        "NPA_BYOF_INFRA": "k8s/unauthorized-context",
        "NPA_SKYPILOT_INFRA": "k8s/unauthorized-context",
        "NPA_SKYPILOT_BIN": "/untrusted/sky",
        "NPA_BYOF_S3_ENDPOINT": "https://unauthorized.invalid",
        "NPA_ISAAC_LAB_ACCEPT_PRECHECK_FAILURE": "1",
    }.items():
        monkeypatch.setenv(name, value)

    args = module._parse_args(
        [
            "--yaml",
            str(YAML_PATH),
            "--solution-name",
            "robotwin",
            "--infra",
            "k8s/argv-override",
            "--sky-bin",
            "/untrusted/argv-sky",
            "--direct-launch",
        ]
    )
    assert (
        module._apply_robotwin_authorization(args, authorization, environment)
        is context
    )

    assert args.infra == "k8s/private-context-canary"
    assert args.sky_bin == ""
    assert args.direct_launch is False
    assert args.image == environment[module.CHILD_IMAGE_ENV]
    assert args.output_root == authorization.output_root
    assert args.run_id == authorization.run_id
    assert args.config_path == authorization.skypilot_config_source

    rendered = yaml.safe_dump_all(
        module.render_workflow(
            YAML_PATH,
            run_id=args.run_id,
            output_root=args.output_root,
            image=args.image,
            solution_name="robotwin",
            runtime_env=environment,
        ),
        sort_keys=False,
    )
    for value in (
        args.image,
        args.output_root,
        args.run_id,
        args.config_path,
        authorization.kubernetes_context,
    ):
        assert value not in rendered
    for name in (
        module.CHILD_BUCKET_ENV,
        module.CHILD_IMAGE_ENV,
        module.CHILD_OUTPUT_PREFIX_ENV,
        module.CHILD_RUN_ID_ENV,
    ):
        assert name in rendered

    assert module.main(["--solution-name", "robotwin"]) == 2


def test_authorized_robotwin_uses_explicit_environment_without_global_mutation(
    monkeypatch,
) -> None:
    module = _load_module()
    authorization = SimpleNamespace(kubeconfig_source="/owner-only/kubeconfig")
    context = object()
    supplied_environment = {"AUTHORIZED_CANARY": "yes"}
    hostile_names = (
        "NPA_BYOF_DIRECT_LAUNCH",
        "NPA_BYOF_INFRA",
        "NPA_SKYPILOT_INFRA",
        "NPA_SKYPILOT_BIN",
        "NPA_BYOF_S3_ENDPOINT",
        "NPA_ISAAC_LAB_ACCEPT_PRECHECK_FAILURE",
        "NPA_BYOF_SKIP_SKY_CHECK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    )
    for name in hostile_names:
        monkeypatch.setenv(name, f"hostile-{name.lower()}")
    monkeypatch.setattr(
        module,
        "_apply_robotwin_authorization",
        lambda _args, observed, values: (
            context
            if observed is authorization and values is supplied_environment
            else pytest.fail("authorization was not preserved in process")
        ),
    )

    def submit(_args, *, robotwin_submit_context, authorized_env):
        assert robotwin_submit_context is context
        assert authorized_env == supplied_environment
        assert all(
            os.environ[name] == f"hostile-{name.lower()}" for name in hostile_names
        )
        return 0

    monkeypatch.setattr(module, "_submit_and_wait", submit)

    assert (
        module.run_authorized_robotwin(
            ["--solution-name", "robotwin"],
            authorization=authorization,
            environment=supplied_environment,
        )
        == 0
    )
    assert all(os.environ[name] == f"hostile-{name.lower()}" for name in hostile_names)


def test_two_authorized_robotwin_runs_cannot_exchange_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    barrier = __import__("threading").Barrier(2)
    observed: list[tuple[object, dict[str, str]]] = []
    ambient = dict(os.environ)

    monkeypatch.setattr(
        module,
        "_apply_robotwin_authorization",
        lambda _args, authorization, _environment: authorization,
    )

    def submit(_args, *, robotwin_submit_context, authorized_env):
        barrier.wait(timeout=5)
        observed.append((robotwin_submit_context, dict(authorized_env)))
        return 0

    monkeypatch.setattr(module, "_submit_and_wait", submit)
    authorizations = (SimpleNamespace(name="one"), SimpleNamespace(name="two"))
    environments = ({"RUN_SECRET": "one"}, {"RUN_SECRET": "two"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda item: module.run_authorized_robotwin(
                    ["--solution-name", "robotwin"],
                    authorization=item[0],
                    environment=item[1],
                ),
                zip(authorizations, environments, strict=True),
            )
        )

    assert results == [0, 0]
    assert {(context.name, env["RUN_SECRET"]) for context, env in observed} == {
        ("one", "one"),
        ("two", "two"),
    }
    assert dict(os.environ) == ambient


def test_authorized_robotwin_failure_leaves_process_environment_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    authorization = SimpleNamespace(name="failure")
    monkeypatch.setattr(
        module,
        "_apply_robotwin_authorization",
        lambda *_args, **_kwargs: authorization,
    )
    monkeypatch.setattr(
        module,
        "_submit_and_wait",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failure")),
    )
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="failure"):
        module.run_authorized_robotwin(
            ["--solution-name", "robotwin"],
            authorization=authorization,
            environment={"RUN_SECRET": "failure"},
        )
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    ("status", "returncode", "expected_rc"),
    (("SUCCEEDED", 0, 0), ("FAILED", 1, 1), ("RUNNING", 0, 1)),
)
def test_robotwin_terminal_summary_never_discloses_destination_or_customer_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: str,
    returncode: int,
    expected_rc: int,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "require_customer_authorization_fresh", Mock())
    run_canary = "robotwin-customer-run-private-canary"
    root_canary = "s3://private-destination-canary/output"
    launch_id = "robotwin-inner-" + "1" * 20
    authorization = SimpleNamespace(
        inner_launch_id=launch_id,
        project="private-project-canary",
        kubeconfig_source=str(tmp_path / "private-kubeconfig"),
        customer_authorization_expires_at="2099-01-01T00:00:00Z",
    )
    context = SimpleNamespace(authorization=authorization)
    environment = {"PRIVATE_CANARY": "private-value"}
    confidential_dir = tmp_path / "confidential-submission"
    confidential_dir.mkdir()

    monkeypatch.setattr(
        module,
        "render_workflow",
        lambda *_args, **_kwargs: [{"name": "meta"}, {"name": "task"}],
    )
    monkeypatch.setattr(module, "_write_yaml_documents", lambda *_args: None)
    monkeypatch.setattr(module, "_bootstrap_robotwin_sky", lambda *_args: "/sky")
    monkeypatch.setattr(
        module,
        "_normalize_kubeconfig_current_context",
        lambda _path, values: values,
    )
    monkeypatch.setattr(
        module,
        "_robotwin_control_environment",
        lambda *_args: {
            "HOME": str(tmp_path / "home"),
            "PATH": "/bin",
            "KUBECONFIG": str(tmp_path / "generated-kubeconfig"),
        },
    )
    monkeypatch.setattr(module, "install_teardown_signal_handlers", lambda *_args: {})
    monkeypatch.setattr(module, "restore_signal_handlers", lambda *_args: None)
    api_events: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        module,
        "stop_isolated_api",
        lambda path: api_events.append(("stop", Path(path))),
    )
    monkeypatch.setattr(
        module.shutil,
        "rmtree",
        lambda path, **_kwargs: api_events.append(("remove", Path(path))),
    )
    stop_commands: list[list[str]] = []

    def stop_sky_api(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert command == ["/sky", "api", "stop"]
        stop_commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", stop_sky_api)

    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_args, **kwargs: _summary_submit_result(
            module, kwargs, confidential_dir
        ),
    )
    monkeypatch.setattr(
        module,
        "_wait_for_terminal",
        lambda *_args, **_kwargs: (
            SimpleNamespace(status=status, returncode=returncode),
            {"private": root_canary},
        ),
    )
    args = module._parse_args(
        [
            "--yaml",
            str(YAML_PATH),
            "--solution-name",
            "robotwin",
            "--run-id",
            run_canary,
            "--output-root",
            root_canary,
            "--infra",
            "k8s/private-context-canary",
            "--config-path",
            str(tmp_path / "private-skypilot.yaml"),
            "--no-direct-launch",
        ]
    )

    assert (
        module._submit_and_wait(
            args,
            robotwin_submit_context=context,
            authorized_env=environment,
        )
        == expected_rc
    )
    assert api_events and api_events[0][0] == "stop"
    assert stop_commands == []
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload == {
        "launch_id": launch_id,
        "returncode": returncode,
        "status": status.lower(),
    }
    assert run_canary not in output
    assert root_canary not in output
    assert "private-destination-canary" not in output


def test_robotwin_owned_api_stop_failure_retains_private_root(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "skypilot-state"
    root.mkdir()
    monkeypatch.setattr(
        module,
        "stop_isolated_api",
        Mock(side_effect=module.SkyPilotConfigError("synthetic stop refusal")),
    )

    assert not module._stop_owned_robotwin_api(root)
    assert root.exists()


def _summary_submit_result(module, kwargs, confidential_dir):
    from npa.orchestration.skypilot._managed_job_api import NativeLaunchResult

    logical_id = kwargs["logical_launch_id"]
    cleanup = SimpleNamespace(
        config_path=None,
        verified=True,
        request=Mock(),
        run_id=logical_id,
        job_id="42",
        active=True,
        submitting=False,
        requested=False,
        native_result=NativeLaunchResult(
            "b" * 64,
            "00000000-0000-4000-8000-000000000002",
            "42",
            (0,),
            "d" * 64,
        ),
        native_verified=True,
        environment={
            "HOME": str(confidential_dir.parent),
            "PATH": "/bin",
            "KUBECONFIG": str(confidential_dir / "generated-kubeconfig"),
        },
    )
    cleanup._lookup = lambda: module.ReconciliationEvidence(
        module.ReconciliationState.FOUND,
        job_id="42",
        status="RUNNING",
        workload_observable=True,
        observed_task_ids=(0,),
    )
    kwargs["on_launch_ready"](cleanup)
    return SimpleNamespace(
        status="SUBMITTED",
        job_id="42",
        returncode=0,
        error="",
        launch_transaction={
            "state": "submitted",
            "identity_source": "native_request_result",
            "logical_launch_id": logical_id,
            "job_id": "42",
        },
        log_paths={"submission_dir": str(confidential_dir)},
    )


def test_robotwin_render_only_summary_is_opaque(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    run_canary = "robotwin-render-customer-run-private-canary"
    root_canary = "s3://private-render-destination-canary/output"
    launch_id = "robotwin-inner-" + "2" * 20
    context = SimpleNamespace(
        authorization=SimpleNamespace(
            inner_launch_id=launch_id,
            customer_authorization_expires_at="2099-01-01T00:00:00Z",
        )
    )
    monkeypatch.setattr(
        module,
        "render_workflow",
        lambda *_args, **_kwargs: [{"name": "meta"}, {"name": "task"}],
    )
    monkeypatch.setattr(module, "_write_yaml_documents", lambda *_args: None)
    args = module._parse_args(
        [
            "--yaml",
            str(YAML_PATH),
            "--solution-name",
            "robotwin",
            "--run-id",
            run_canary,
            "--output-root",
            root_canary,
            "--render-only",
        ]
    )

    assert (
        module._submit_and_wait(
            args,
            robotwin_submit_context=context,
            authorized_env={},
        )
        == 0
    )
    output = capsys.readouterr().out
    assert json.loads(output) == {"launch_id": launch_id, "status": "rendered"}
    assert run_canary not in output
    assert root_canary not in output


def test_robotwin_sky_bootstrap_is_worker_local_and_pinned(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    sky_bin = tmp_path / "runtime" / "skypilot-venv" / "bin" / "sky"
    observed: dict[str, object] = {}

    def run(argv, **kwargs):
        observed.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout=f"{sky_bin}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    environment = {"HOME": str(tmp_path), "PATH": "/bin"}
    assert module._bootstrap_robotwin_sky(tmp_path / "runtime", environment) == str(
        sky_bin
    )
    assert observed["kwargs"]["env"] == environment
    assert observed["argv"][-1] == str(tmp_path / "runtime" / "skypilot-venv")


def test_one_solutions_operator_answers_do_not_widen_anothers(monkeypatch) -> None:
    """Vendor answers are per-image, and a shared tuple made them global.

    With one tuple for every BYOF image, adding LTX's variables also forwarded
    them — and HF_TOKEN — into wan2-2 and open-dreamer runs whenever they were
    set in the operator's shell. Nothing broke visibly, which is why it needs a
    test rather than a review.
    """

    module = _load_module()
    for name in (
        "NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS",
        "NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS",
        "HF_TOKEN",
    ):
        monkeypatch.setenv(name, "set")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    wan = module.resolve_secret_envs(None, solution_name="wan2.2")
    ltx = module.resolve_secret_envs(None, solution_name="ltx2.5")
    other = module.resolve_secret_envs(None, solution_name="open-dreamer")

    assert not [name for name in wan if name.startswith("NPA_LTX_")]
    assert "NPA_WAN_ACCEPT_NVIDIA_RUNTIME_TERMS" not in ltx
    assert "NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS" in ltx
    # The entitlement the LTX container needs for both of its fetches.
    assert "HF_TOKEN" in ltx
    # A solution with no vendor answers of its own forwards none, including the
    # token: it has no gate that reads one.
    assert other == []


def test_output_storage_preflight_writes_reads_and_deletes(monkeypatch) -> None:
    module = _load_module()
    calls: list[tuple[str, str, str]] = []

    class FakeS3:
        def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
            calls.append(("list", Bucket, Prefix))
            assert MaxKeys == 1
            return {"Contents": []}

        def put_object(self, *, Bucket, Key, **_kwargs):
            assert _kwargs["IfNoneMatch"] == "*"
            calls.append(("put", Bucket, Key))
            return {}

        def head_object(self, *, Bucket, Key):
            calls.append(("head", Bucket, Key))
            return {"ContentLength": 24}

        def delete_object(self, *, Bucket, Key):
            calls.append(("delete", Bucket, Key))

    monkeypatch.setenv("NPA_E2E_PROJECT", "demo-project")
    monkeypatch.setattr(
        module,
        "s3_client_for_project",
        lambda project, *, allow_host_creds, endpoint_url: (
            FakeS3()
            if project == "demo-project"
            and allow_host_creds
            and endpoint_url == "https://storage.override"
            else pytest.fail("unexpected S3 credential scope")
        ),
    )
    monkeypatch.setenv("NPA_BYOF_S3_ENDPOINT", "https://storage.override")

    module.preflight_output_storage(
        output_root="s3://bucket/prefix", run_id="byof-demo"
    )

    assert calls == [
        ("list", "bucket", "prefix/byof-demo/"),
        ("put", "bucket", "prefix/byof-demo/.npa-write-preflight"),
        ("head", "bucket", "prefix/byof-demo/.npa-write-preflight"),
        ("delete", "bucket", "prefix/byof-demo/.npa-write-preflight"),
    ]


def test_render_storage_env_honors_explicit_regional_endpoint(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_E2E_PROJECT", "demo-project")
    monkeypatch.setenv("NPA_BYOF_S3_ENDPOINT", "https://storage.correct-region")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.stale-region")

    def storage_env(project, *, allow_host_creds, endpoint_url):
        assert project == "demo-project"
        assert allow_host_creds is True
        assert endpoint_url == "https://storage.correct-region"
        return {"AWS_ENDPOINT_URL": endpoint_url}

    monkeypatch.setattr(module, "storage_env_for_project", storage_env)

    assert module._resolved_storage_env() == {
        "AWS_ENDPOINT_URL": "https://storage.correct-region"
    }

    docs = module.render_workflow(
        YAML_PATH,
        run_id="regional-endpoint",
        output_root="s3://project-bucket/byof",
    )
    assert docs[1]["envs"]["AWS_ENDPOINT_URL"] == "https://storage.correct-region"
    assert docs[1]["envs"]["NEBIUS_S3_ENDPOINT"] == "https://storage.correct-region"


def test_libero_output_preflight_first_call_uses_exact_authorized_triplet(
    monkeypatch,
) -> None:
    import boto3

    module = _load_module()
    calls: list[tuple[str, str, str]] = []
    clients: list[dict] = []

    class FakeS3:
        def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
            calls.append(("list", Bucket, Prefix))
            assert MaxKeys == 1
            return {"Contents": []}

        def put_object(self, *, Bucket, Key, **_kwargs):
            calls.append(("put", Bucket, Key))
            assert _kwargs["IfNoneMatch"] == "*"
            return {"VersionId": "lease-version", "ETag": '"lease-etag"'}

        def head_object(self, *, Bucket, Key, VersionId):
            calls.append(("head", Bucket, Key))
            assert VersionId == "lease-version"
            return {
                "ContentLength": 24,
                "VersionId": VersionId,
                "ETag": '"lease-etag"',
            }

        def delete_object(self, *, Bucket, Key):
            calls.append(("delete", Bucket, Key))

    monkeypatch.setattr(
        module,
        "s3_client_for_project",
        lambda *_args, **_kwargs: pytest.fail("saved project storage is forbidden"),
    )

    def client(service, **kwargs):
        clients.append({"service": service, **kwargs})
        return FakeS3()

    monkeypatch.setattr(boto3, "client", client)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "manager-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "manager-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "manager-session")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example")
    monkeypatch.setenv("NPA_PROJECT", "saved-project-with-different-credentials")

    lease = module.preflight_output_storage(
        output_root="s3://manager-bucket/accepted-prefix",
        run_id="libero-authorized-probe",
        libero=True,
    )

    assert len(clients) == 1
    assert clients[0]["service"] == "s3"
    assert clients[0]["endpoint_url"] == "https://storage.example"
    assert clients[0]["aws_access_key_id"] == "manager-access"
    assert clients[0]["aws_secret_access_key"] == "manager-secret"
    assert clients[0]["aws_session_token"] == "manager-session"
    assert lease == module.OutputPrefixLease(
        "manager-bucket",
        "accepted-prefix/libero-authorized-probe/.npa-output-lease",
        '"lease-etag"',
        "lease-version",
    )
    assert calls[0] == (
        "list",
        "manager-bucket",
        "accepted-prefix/libero-authorized-probe/",
    )


def test_libero_output_preflight_preserves_commit_interleaving(
    monkeypatch,
) -> None:
    import boto3

    module = _load_module()
    run_id = "libero-authorized-probe"
    prefix = f"accepted-prefix/{run_id}/"
    artifact_key = prefix + "libero-smoke.json"
    commit_key = prefix + "npa_upload_receipt.json"
    objects = {artifact_key: b"listed bytes"}
    deletes: list[str] = []

    class FakeS3:
        def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
            assert Bucket == "manager-bucket"
            assert Prefix == prefix
            listed = [{"Key": key} for key in sorted(objects)[:MaxKeys]]
            objects[artifact_key] = b"replacement bytes"
            objects[commit_key] = b"committed transaction"
            return {"Contents": listed}

        def delete_object(self, *, Bucket, Key):
            assert Bucket == "manager-bucket"
            deletes.append(Key)
            objects.pop(Key, None)

        def put_object(self, **_kwargs):
            pytest.fail("preflight wrote into a nonempty run prefix")

        def head_object(self, **_kwargs):
            pytest.fail("preflight observed a marker after refusing the prefix")

    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: FakeS3())
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "manager-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "manager-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "manager-session")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example")

    with pytest.raises(RuntimeError, match="output storage preflight failed"):
        module.preflight_output_storage(
            output_root="s3://manager-bucket/accepted-prefix",
            run_id=run_id,
            libero=True,
        )

    assert deletes == []
    assert objects == {
        artifact_key: b"replacement bytes",
        commit_key: b"committed transaction",
    }


def test_libero_output_prefix_refusal_is_retryable_after_external_cleanup(
    monkeypatch,
) -> None:
    import boto3

    module = _load_module()
    run_id = "libero-authorized-probe"
    prefix = f"accepted-prefix/{run_id}/"
    partial_key = prefix + "npa_runtime_metadata.json"
    objects: dict[str, tuple[str, str]] = {partial_key: ("", "")}
    deletes: list[str] = []

    class FakeS3:
        def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
            return {"Contents": [{"Key": key} for key in sorted(objects)[:MaxKeys]]}

        def delete_object(self, *, Bucket, Key, VersionId=None):
            deletes.append(Key)
            objects.pop(Key, None)

        def put_object(self, *, Bucket, Key, **_kwargs):
            objects[Key] = ('"lease-etag"', "lease-version")
            return {"ETag": '"lease-etag"', "VersionId": "lease-version"}

        def head_object(self, *, Bucket, Key, VersionId=None):
            etag, version = objects[Key]
            assert VersionId == version
            return {"ContentLength": 24, "ETag": etag, "VersionId": version}

    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: FakeS3())
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "manager-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "manager-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "manager-session")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example")

    with pytest.raises(RuntimeError, match="output storage preflight failed"):
        module.preflight_output_storage(
            output_root="s3://manager-bucket/accepted-prefix",
            run_id=run_id,
            libero=True,
        )
    assert objects == {partial_key: ("", "")}
    assert deletes == []

    objects.clear()
    lease = module.preflight_output_storage(
        output_root="s3://manager-bucket/accepted-prefix",
        run_id=run_id,
        libero=True,
    )
    assert lease is not None
    assert objects == {prefix + ".npa-output-lease": ('"lease-etag"', "lease-version")}
    assert deletes == []


@pytest.mark.parametrize("suffix", ["npa_upload_receipt.json", "foreign.json"])
def test_libero_output_preflight_never_reconciles_committed_or_unknown_objects(
    monkeypatch, suffix
) -> None:
    import boto3

    module = _load_module()
    deleted = False

    class FakeS3:
        def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
            return {"Contents": [{"Key": Prefix + suffix}]}

        def delete_object(self, **_kwargs):
            nonlocal deleted
            deleted = True

    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: FakeS3())
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "manager-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "manager-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "manager-session")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example")

    with pytest.raises(RuntimeError, match="output storage preflight failed"):
        module.preflight_output_storage(
            output_root="s3://manager-bucket/accepted-prefix",
            run_id="libero-authorized-probe",
            libero=True,
        )
    assert deleted is False


@pytest.mark.parametrize(
    "missing", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]
)
def test_libero_output_preflight_rejects_partial_triplets_before_provider(
    monkeypatch, missing
) -> None:
    import boto3

    module = _load_module()
    values = {
        "AWS_ACCESS_KEY_ID": "manager-access",
        "AWS_SECRET_ACCESS_KEY": "manager-secret",
        "AWS_SESSION_TOKEN": "manager-session",
    }
    for name, value in values.items():
        if name == missing:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example")
    monkeypatch.setattr(
        boto3,
        "client",
        lambda *_args, **_kwargs: pytest.fail("provider must not be called"),
    )

    with pytest.raises(ValueError, match="complete.*triplet"):
        module.preflight_output_storage(
            output_root="s3://manager-bucket/accepted-prefix",
            run_id="libero-authorized-probe",
            libero=True,
        )


def test_output_storage_preflight_fails_before_launch(monkeypatch) -> None:
    module = _load_module()

    class DeniedS3:
        def list_objects_v2(self, **_kwargs):
            return {"Contents": []}

        def put_object(self, **_kwargs):
            raise PermissionError("Access denied")

    monkeypatch.setattr(
        module,
        "s3_client_for_project",
        lambda *_args, **_kwargs: DeniedS3(),
    )

    with pytest.raises(RuntimeError, match="output storage preflight failed"):
        module.preflight_output_storage(
            output_root="s3://bucket/prefix", run_id="byof-demo"
        )


def test_output_storage_preflight_rejects_reused_run_prefix(monkeypatch) -> None:
    module = _load_module()

    class ExistingS3:
        def list_objects_v2(self, **_kwargs):
            return {"Contents": [{"Key": "prefix/byof-demo/result.json"}]}

    monkeypatch.setattr(
        module,
        "s3_client_for_project",
        lambda *_args, **_kwargs: ExistingS3(),
    )

    with pytest.raises(
        RuntimeError, match="refusing to reuse a non-empty BYOF run prefix"
    ):
        module.preflight_output_storage(
            output_root="s3://bucket/prefix", run_id="byof-demo"
        )


def test_wait_timeout_zero_checks_status_once(monkeypatch) -> None:
    module = _load_module()
    calls: list[str] = []
    status = type("Status", (), {"status": "RUNNING"})()
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_args, **_kwargs: calls.append("status") or status,
    )
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda *_args: (_ for _ in ()).throw(AssertionError("slept")),
    )

    final, diagnostics = module._wait_for_terminal(
        "run", sky_bin="sky", wait_timeout=0, poll_interval=1
    )
    assert final.status == "RUNNING"
    assert calls == ["status"]
    assert diagnostics == {
        "mode": "immediate",
        "polls": 1,
        "statuses": ["RUNNING"],
        "terminal": False,
        "deadline_exhausted": False,
        "stuck_state": "RUNNING",
        "hint": "workflow is not terminal; inspect SkyPilot controller/job and pod events",
    }


def test_wait_preserves_isolated_scheduler_identity(monkeypatch, tmp_path) -> None:
    module = _load_module()
    isolated = tmp_path / "isolated"
    config = tmp_path / "config.yaml"
    observed: dict[str, object] = {}

    def status(job_id, **kwargs):
        observed.update(job_id=job_id, **kwargs)
        return SimpleNamespace(status="SUCCEEDED")

    monkeypatch.setattr(module, "workflow_status", status)
    module._wait_for_terminal(
        "73",
        sky_bin="sky",
        isolated_config_dir=isolated,
        config_path=config,
        wait_timeout=0,
        poll_interval=1,
    )

    assert observed == {
        "job_id": "73",
        "sky_bin": "sky",
        "isolated_config_dir": isolated,
        "config_path": config,
    }


def test_wait_treats_verified_absence_as_terminal_failure(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: SimpleNamespace(status="ABSENT"),
    )

    final, diagnostics = module._wait_for_terminal(
        "73", sky_bin="sky", wait_timeout=-1, poll_interval=1
    )

    assert final.status == "ABSENT"
    assert diagnostics["terminal"] is True
    assert diagnostics["polls"] == 1


def test_positive_wait_is_bounded_and_reports_stuck_state(monkeypatch) -> None:
    module = _load_module()
    clock = {"now": 100.0}
    status = type("Status", (), {"status": "PENDING"})()
    monkeypatch.setattr(module, "workflow_status", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(module.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    _, diagnostics = module._wait_for_terminal(
        "run", sky_bin="sky", wait_timeout=2, poll_interval=1
    )
    assert diagnostics["mode"] == "bounded"
    assert diagnostics["polls"] == 3
    assert diagnostics["deadline_exhausted"] is True
    assert diagnostics["stuck_state"] == "PENDING"


def test_negative_one_waits_until_terminal(monkeypatch) -> None:
    module = _load_module()
    statuses = iter(["PENDING", "RUNNING", "SUCCEEDED"])
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_args, **_kwargs: type("Status", (), {"status": next(statuses)})(),
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    final, diagnostics = module._wait_for_terminal(
        "run", sky_bin="sky", wait_timeout=-1, poll_interval=1
    )
    assert final.status == "SUCCEEDED"
    assert diagnostics["mode"] == "indefinite"
    assert diagnostics["statuses"] == ["PENDING", "RUNNING", "SUCCEEDED"]
    assert diagnostics["terminal"] is True


@pytest.mark.parametrize(
    "terminal_status",
    ["FAILED_RUNTIME", "FAILED_NO_RESOURCE", "CANCELED", "STOPPED", "FAILED_CUSTOM"],
)
def test_wait_for_terminal_recognizes_canonical_failure_variants(
    monkeypatch, terminal_status: str
) -> None:
    module = _load_module()
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_args, **_kwargs: (
            calls.append("status") or type("Status", (), {"status": terminal_status})()
        ),
    )
    monkeypatch.setattr(
        module.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("slept")),
    )

    final, diagnostics = module._wait_for_terminal(
        "run", sky_bin="sky", wait_timeout=-1, poll_interval=1
    )

    assert final.status == terminal_status
    assert calls == ["status"]
    assert diagnostics["terminal"] is True
    assert diagnostics["polls"] == 1


def test_wait_runs_access_guard_at_every_status_observation(monkeypatch) -> None:
    module = _load_module()
    statuses = iter(["PENDING", "RUNNING", "SUCCEEDED"])
    guarded: list[str] = []
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_args, **_kwargs: SimpleNamespace(status=next(statuses)),
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    module._wait_for_terminal(
        "73",
        sky_bin="sky",
        wait_timeout=-1,
        poll_interval=1,
        observation_guard=guarded.append,
    )

    assert guarded == ["PENDING", "RUNNING", "SUCCEEDED"]


def test_wait_timeout_less_than_negative_one_is_rejected() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="must be -1"):
        module._wait_for_terminal(
            "run", sky_bin="sky", wait_timeout=-2, poll_interval=1
        )


def test_managed_cleanup_cancels_and_drains_exact_job_before_down(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    calls: list[object] = []
    statuses = iter(["RUNNING", "CANCELLING", "CANCELLED"])
    isolated = tmp_path / "isolated"
    config = tmp_path / "config.yaml"

    def status(job_id, **kwargs):
        calls.append(("status", job_id, kwargs))
        return SimpleNamespace(status=next(statuses))

    def cancel(**kwargs):
        calls.append(("cancel", kwargs))
        return {"cancel_returncode": 0}

    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: calls.append("down") or module.CleanupResult(),
    )
    monkeypatch.setattr(module, "workflow_status", status)
    monkeypatch.setattr(module, "cancel_workflow_job", cancel)

    def verified_absence(**_kwargs):
        calls.append("verify-absent")
        result = module.CleanupResult()
        result.verified = True
        result.remote_absence_verified = True
        return result

    monkeypatch.setattr(
        module,
        "_verify_managed_clusters_absent",
        verified_absence,
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=isolated,
        config_path=config,
        poll_interval=1,
    )

    assert result.ok is True
    assert result.verified is True
    assert result.remote_absence_verified is True
    assert calls[-2:] == ["down", "verify-absent"]
    cancel_call = next(call for call in calls if call[0] == "cancel")[1]
    assert cancel_call == {
        "sky_bin": "sky",
        "job_id": "73",
        "run_id": "human-run-name",
        "isolated_config_dir": isolated,
        "config_path": config,
        "timeout": 10,
        "poll_seconds": 1.0,
        "also_down_cluster": False,
    }
    assert [
        call[1] for call in calls if isinstance(call, tuple) and call[0] == "status"
    ] == [
        "73",
        "73",
        "73",
    ]


def test_managed_cleanup_preserves_clusters_when_exact_cancel_fails(
    monkeypatch,
) -> None:
    module = _load_module()
    down = []
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: SimpleNamespace(status="RUNNING"),
    )
    monkeypatch.setattr(
        module, "cancel_workflow_job", lambda **_k: {"cancel_returncode": 1}
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: down.append(True) or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is False
    assert result.errors == ["exact managed-job cancellation failed"]
    assert down == []


def test_managed_cleanup_preserves_clusters_when_exact_cancel_raises(
    monkeypatch,
) -> None:
    module = _load_module()
    down = []
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: SimpleNamespace(status="RUNNING"),
    )
    monkeypatch.setattr(
        module,
        "cancel_workflow_job",
        lambda **_k: (_ for _ in ()).throw(TypeError("cancel unavailable")),
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: down.append(True) or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.errors == ["exact managed-job cancellation raised unexpectedly"]
    assert down == []


def test_managed_cleanup_reports_teardown_exception_without_absence_claim(
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: SimpleNamespace(status="CANCELLED"),
    )
    monkeypatch.setattr(
        module,
        "_verify_managed_clusters_absent",
        lambda **_k: pytest.fail("absence cannot be checked after teardown failure"),
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: (_ for _ in ()).throw(OSError("teardown failed")),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.errors == ["run-cluster teardown raised unexpectedly"]
    assert result.remote_absence_verified is False


def test_managed_cleanup_preserves_clusters_on_ambiguous_controller_status(
    monkeypatch,
) -> None:
    module = _load_module()
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: SimpleNamespace(status="FAILED_CONTROLLER"),
    )
    monkeypatch.setattr(
        module,
        "cancel_workflow_job",
        lambda **_k: calls.append("cancel") or {"cancel_returncode": 0},
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: calls.append("down") or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is False
    assert result.errors == [
        "managed job did not reach a verified terminal or absent state; "
        "preserving its clusters"
    ]
    assert calls == ["cancel"]


def test_managed_cleanup_cancels_after_status_exception_then_proves_drain(
    monkeypatch,
) -> None:
    module = _load_module()
    calls: list[str] = []
    statuses = iter(
        [subprocess.TimeoutExpired(["sky", "jobs", "queue"], 1), "CANCELLED"]
    )

    def status(*_args, **_kwargs):
        value = next(statuses)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(status=value)

    monkeypatch.setattr(module, "workflow_status", status)
    monkeypatch.setattr(
        module,
        "cancel_workflow_job",
        lambda **_k: calls.append("cancel") or {"cancel_returncode": 0},
    )
    monkeypatch.setattr(
        module,
        "_verify_managed_clusters_absent",
        lambda **_k: calls.append("verify") or module.CleanupResult(),
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: calls.append("down") or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is True
    assert calls == ["cancel", "down", "verify"]


def test_managed_cleanup_preserves_clusters_after_persistent_status_exception(
    monkeypatch,
) -> None:
    module = _load_module()
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("status unavailable")),
    )
    monkeypatch.setattr(
        module,
        "cancel_workflow_job",
        lambda **_k: calls.append("cancel") or {"cancel_returncode": 0},
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: calls.append("down") or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is False
    assert "could not be verified" in result.errors[0]
    assert calls == ["cancel"]


@pytest.mark.parametrize(
    "status_failure",
    [None, SimpleNamespace(), TypeError("malformed status")],
)
def test_managed_cleanup_cancels_on_malformed_or_unlisted_status_failure(
    monkeypatch, status_failure
) -> None:
    module = _load_module()
    calls: list[str] = []

    def status(*_args, **_kwargs):
        if isinstance(status_failure, Exception):
            raise status_failure
        return status_failure

    monkeypatch.setattr(module, "workflow_status", status)
    monkeypatch.setattr(
        module,
        "cancel_workflow_job",
        lambda **_k: calls.append("cancel") or {"cancel_returncode": 0},
    )
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: calls.append("down") or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "73",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is False
    assert "could not be verified" in result.errors[0]
    assert calls == ["cancel"]


@pytest.mark.parametrize(
    ("stdout", "expected_error"),
    [
        ("not-json", "was not exact JSON"),
        ('{"unexpected": []}', "invalid schema"),
        ('{"clusters": [], "error": "denied"}', "invalid schema"),
        ('[{"status": "UP"}]', "invalid row"),
        ('[{"name": " human-run-name-worker "}]', "invalid row"),
        ('[{"name": "human-run-name-worker\\u0000"}]', "invalid row"),
        (
            '[{"name": "unrelated", "cluster": "human-run-name-worker"}]',
            "ambiguous name row",
        ),
        (
            '[{"name": "unrelated", "error": "permission denied"}]',
            "contradictory error metadata",
        ),
        ('[{"name": "unrelated", "error": ""}]', "contradictory error metadata"),
        ('[{"name": "unrelated", "errors": []}]', "contradictory error metadata"),
        ('[{"name": "unrelated", "exception": null}]', "contradictory error metadata"),
        ('[{"name": "human-run-name-worker"}]', "still contains"),
    ],
)
def test_post_teardown_inventory_refuses_ambiguous_or_present_state(
    monkeypatch, stdout, expected_error
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            ["sky", "status"], 0, stdout=stdout, stderr=""
        ),
    )

    result = module._verify_managed_clusters_absent(
        run_id="human-run-name",
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        timeout=10,
    )

    assert result.ok is False
    assert expected_error in result.errors[0]
    assert result.remote_absence_verified is False


def test_post_teardown_inventory_proves_exact_run_absence(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    isolated = tmp_path / "isolated"
    config = tmp_path / "config.yaml"
    observed = {}

    def run(cmd, **kwargs):
        observed.update(cmd=cmd, kwargs=kwargs)
        return subprocess.CompletedProcess(
            cmd, 0, stdout='[{"name": "unrelated-cluster"}]', stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    result = module._verify_managed_clusters_absent(
        run_id="human-run-name",
        sky_bin="sky",
        isolated_config_dir=isolated,
        config_path=config,
        timeout=10,
    )

    assert result.ok is True
    assert result.verified is True
    assert result.remote_absence_verified is True
    assert observed["cmd"] == [
        "sky",
        "status",
        "--config",
        str(config),
        "--refresh",
        "--output",
        "json",
    ]
    assert observed["kwargs"]["env"]["HOME"] == str(isolated / "home")


def test_post_teardown_inventory_exception_keeps_absence_unverified(
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["sky", "status"], 1)
        ),
    )

    result = module._verify_managed_clusters_absent(
        run_id="human-run-name",
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        timeout=1,
    )

    assert result.errors == ["post-teardown SkyPilot cluster inventory raised"]
    assert result.remote_absence_verified is False


def test_managed_cleanup_preserves_resources_without_scheduler_id() -> None:
    module = _load_module()
    down = []
    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        teardown=lambda: down.append(True) or module.CleanupResult(),
    )

    result = module._cancel_then_teardown_managed_job(
        "",
        teardown_guard=guard,
        sky_bin="sky",
        isolated_config_dir=None,
        config_path=None,
        poll_interval=1,
    )

    assert result.ok is False
    assert "preserving" in result.errors[0]
    assert down == []


def test_submit_waits_on_scheduler_id_not_human_run_name(monkeypatch, tmp_path) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    isolated = tmp_path / args.run_id
    isolated.mkdir()
    isolated.chmod(0o700)
    args.isolated_config_dir = str(isolated)
    observed: dict[str, object] = {"resolve_calls": 0}

    def resolve_isolated(value):
        observed["resolve_calls"] = int(observed["resolve_calls"]) + 1
        assert value == str(isolated)
        return isolated.resolve()

    def guard_factory(**kwargs):
        observed["guard_isolated"] = kwargs["isolated_config_dir"]
        return SimpleNamespace(
            run_id="human-run-name",
            timeout=10,
            isolated_config_dir=kwargs["isolated_config_dir"],
            mark_launched=lambda **_k: None,
            teardown=lambda: module.CleanupResult(),
        )

    def submit(_path, run_id, **kwargs):
        observed["submitted_run_id"] = run_id
        observed["submitted_isolated"] = kwargs["isolated_config_dir"]
        return SimpleNamespace(job_id="73", log_paths={})

    def wait(scheduler_job_id, **kwargs):
        observed["waited_job_id"] = scheduler_job_id
        observed["waited_isolated"] = kwargs["isolated_config_dir"]
        return SimpleNamespace(status="SUCCEEDED"), {"terminal": True}

    monkeypatch.setattr(module, "resolve_isolated_config_dir", resolve_isolated)
    monkeypatch.setattr(module, "SignalTeardown", guard_factory)
    monkeypatch.setattr(module, "submit_workflow", submit)
    monkeypatch.setattr(module, "_wait_for_terminal", wait)

    assert module._submit_and_wait(args) == 0
    assert observed == {
        "submitted_run_id": "human-run-name",
        "submitted_isolated": isolated.resolve(),
        "waited_job_id": "73",
        "waited_isolated": isolated.resolve(),
        "guard_isolated": isolated.resolve(),
        "resolve_calls": 1,
    }


def test_libero_refuses_isaac_lab_precheck_failure_override(
    monkeypatch, tmp_path, capsys
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    isolated = tmp_path / args.run_id
    isolated.mkdir()
    isolated.chmod(0o700)
    args.isolated_config_dir = str(isolated)
    args.solution_name = "libero"
    payload_kubeconfig = tmp_path / "payload-kubeconfig-source"
    payload_kubeconfig.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    payload_kubeconfig.chmod(0o600)
    monkeypatch.setenv("NPA_LIBERO_PAYLOAD_KUBECONFIG", str(payload_kubeconfig))
    execution_kubeconfig = tmp_path / "execution-kubeconfig"
    execution_kubeconfig.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    execution_kubeconfig.chmod(0o600)
    monkeypatch.setenv("KUBECONFIG", str(execution_kubeconfig))
    sky_config = tmp_path / "skypilot.yaml"
    sky_config.chmod(0o600)
    args.config_path = str(sky_config)
    monkeypatch.setenv("NPA_ISAAC_LAB_ACCEPT_PRECHECK_FAILURE", "1")
    monkeypatch.setattr(module, "_is_libero_invocation", lambda *_a: True)
    monkeypatch.setattr(
        module, "_libero_global_config_path", lambda _args: _args.config_path
    )
    monkeypatch.setattr(module, "_bind_libero_runtime_contract", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id="73", log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "_wait_for_terminal",
        lambda *_a, **_k: (
            SimpleNamespace(status="FAILED_PRECHECKS"),
            {"terminal": True},
        ),
    )

    assert module._submit_and_wait(args) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["final"]["status"] == "FAILED_PRECHECKS"


def test_submit_refuses_empty_scheduler_id(monkeypatch, tmp_path, capsys) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id="", log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "_wait_for_terminal",
        lambda *_a, **_k: pytest.fail("waiter must not run without a scheduler ID"),
    )

    assert module._submit_and_wait(args) == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["submit"] == {
        "error_type": "ValueError",
        "scheduler_job_id_recovered": False,
        "status": "failed",
    }


@pytest.mark.parametrize("job_id", [" 73", "73 ", "human-run-name", "0", -1])
def test_submit_refuses_nonexact_numeric_scheduler_id(
    monkeypatch, tmp_path, capsys, job_id
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id=job_id, log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "_wait_for_terminal",
        lambda *_a, **_k: pytest.fail("waiter must receive only an exact job ID"),
    )

    assert module._submit_and_wait(args) == 2
    assert json.loads(capsys.readouterr().out)["submit"]["status"] == "failed"


def test_submit_error_recovers_exact_scheduler_id_and_config_for_cleanup(
    monkeypatch, tmp_path, capsys
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    args.cleanup = True
    isolated = tmp_path / "isolated"
    generated_config = (
        isolated / "submissions" / "human-run-name" / "skypilot-config.yaml"
    )
    generated_config.parent.mkdir(parents=True)
    generated_config.write_text("kubernetes: {}\n", encoding="utf-8")
    generated_config.chmod(0o600)
    args.isolated_config_dir = str(isolated)
    observed: dict[str, object] = {}
    transaction = SimpleNamespace(job_id="73")

    def submit(*_args, **_kwargs):
        raise module.SkyPilotSubmitError(
            "reconciled failure",
            transaction=transaction,
        )

    def cleanup(scheduler_job_id, **kwargs):
        observed.update(job_id=scheduler_job_id, config_path=kwargs["config_path"])
        result = module.CleanupResult()
        result.verified = True
        result.remote_absence_verified = True
        return result

    monkeypatch.setattr(module, "submit_workflow", submit)
    monkeypatch.setattr(module, "_cancel_then_teardown_managed_job", cleanup)

    assert module._submit_and_wait(args) == 2
    summary = json.loads(capsys.readouterr().out)
    assert observed == {"job_id": "73", "config_path": generated_config}
    assert summary["submit"]["scheduler_job_id_recovered"] is True
    assert summary["submit"]["reconciled_scheduler_job_id"] is True
    assert summary["submit"]["generated_config_recovered"] is True
    assert summary["cleanup"]["verified"] is True
    assert summary["cleanup"]["remote_absence_verified"] is True


def test_submit_error_preserves_when_generated_config_cannot_be_recovered(
    monkeypatch, tmp_path, capsys
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    args.cleanup = True
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    args.isolated_config_dir = str(isolated)
    observed: list[str] = []

    def submit(*_args, **_kwargs):
        raise module.SkyPilotSubmitError(
            "reconciled failure",
            transaction=SimpleNamespace(job_id="73"),
        )

    def cleanup(scheduler_job_id, **_kwargs):
        observed.append(scheduler_job_id)
        result = module.CleanupResult()
        result.errors.append("exact config unavailable; resources preserved")
        return result

    monkeypatch.setattr(module, "submit_workflow", submit)
    monkeypatch.setattr(module, "_cancel_then_teardown_managed_job", cleanup)

    assert module._submit_and_wait(args) == 1
    summary = json.loads(capsys.readouterr().out)
    assert observed == [""]
    assert summary["submit"]["reconciled_scheduler_job_id"] is True
    assert summary["submit"]["scheduler_job_id_recovered"] is False
    assert summary["submit"]["generated_config_recovered"] is False
    assert summary["cleanup"]["remote_absence_verified"] is False


def test_wait_exception_emits_failure_and_cleans_exact_scheduler_id(
    monkeypatch, tmp_path, capsys
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    args.cleanup = True
    observed: list[str] = []
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id="73", log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "_wait_for_terminal",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("status failed")),
    )

    def cleanup(scheduler_job_id, **_kwargs):
        observed.append(scheduler_job_id)
        result = module.CleanupResult()
        result.verified = True
        result.remote_absence_verified = True
        return result

    monkeypatch.setattr(module, "_cancel_then_teardown_managed_job", cleanup)

    assert module._submit_and_wait(args) == 2
    summary = json.loads(capsys.readouterr().out)
    assert observed == ["73"]
    assert summary["submit"]["error_type"] == "RuntimeError"
    assert summary["cleanup"]["verified"] is True


def test_submit_returns_failure_when_exact_cleanup_is_not_verified(
    monkeypatch, tmp_path, capsys
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    args.cleanup = True
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id="73", log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "workflow_status",
        lambda *_a, **_k: SimpleNamespace(status="SUCCEEDED"),
    )

    def failed_teardown():
        result = module.CleanupResult()
        result.errors.append("cluster absence was not verified")
        return result

    guard = SimpleNamespace(
        run_id="human-run-name",
        timeout=10,
        isolated_config_dir=None,
        config_path=None,
        mark_launched=lambda **_k: None,
        teardown=failed_teardown,
    )
    monkeypatch.setattr(module, "SignalTeardown", lambda **_k: guard)

    assert module._submit_and_wait(args) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["cleanup"] == {
        "errors": ["cluster absence was not verified"],
        "ok": False,
        "remote_absence_verified": False,
        "resources_removed": [],
        "verified": False,
    }


def test_submit_requires_and_emits_verified_remote_cleanup(
    monkeypatch, tmp_path, capsys
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    args.cleanup = True
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id="73", log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "_wait_for_terminal",
        lambda *_a, **_k: (SimpleNamespace(status="SUCCEEDED"), {"terminal": True}),
    )
    cleanup = module.CleanupResult()
    cleanup.verified = True
    cleanup.remote_absence_verified = True
    monkeypatch.setattr(
        module, "_cancel_then_teardown_managed_job", lambda *_a, **_k: cleanup
    )

    assert module._submit_and_wait(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["cleanup"] == {
        "errors": [],
        "ok": True,
        "remote_absence_verified": True,
        "resources_removed": [],
        "verified": True,
    }


def test_submit_rejects_error_free_but_unverified_cleanup(
    monkeypatch, tmp_path, capsys
) -> None:
    module = _load_module()
    args = _indirect_submit_args(module, monkeypatch, tmp_path)
    args.cleanup = True
    monkeypatch.setattr(
        module,
        "submit_workflow",
        lambda *_a, **_k: SimpleNamespace(job_id="73", log_paths={}),
    )
    monkeypatch.setattr(
        module,
        "_wait_for_terminal",
        lambda *_a, **_k: (SimpleNamespace(status="SUCCEEDED"), {"terminal": True}),
    )
    monkeypatch.setattr(
        module,
        "_cancel_then_teardown_managed_job",
        lambda *_a, **_k: module.CleanupResult(),
    )

    assert module._submit_and_wait(args) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["cleanup"]["ok"] is False
    assert summary["cleanup"]["verified"] is False
    assert summary["cleanup"]["remote_absence_verified"] is False


def test_render_workflow_normalizes_docker_image_for_summary(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_resolved_storage_env", lambda: {})
    docs = module.render_workflow(
        YAML_PATH,
        run_id="byof-demo",
        image="docker:registry.example/npa-byof:demo",
    )
    task = docs[1]
    assert task["envs"]["BYOF_IMAGE"] == "registry.example/npa-byof:demo"
    assert task["resources"]["image_id"] == "docker:registry.example/npa-byof:demo"


def test_render_workflow_rejects_unresolved_endpoint_placeholder(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("AWS_ENDPOINT_URL", "${AWS_ENDPOINT_URL}")
    monkeypatch.setattr(
        module,
        "_resolved_storage_env",
        lambda: {"AWS_ENDPOINT_URL": "https://storage.from-project"},
    )
    docs = module.render_workflow(
        YAML_PATH,
        run_id="byof-demo",
        output_root="s3://bucket/prefix",
    )
    assert docs[1]["envs"]["AWS_ENDPOINT_URL"] == "https://storage.from-project"


def test_normalize_output_root_strips_double_s3_prefix(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_resolved_storage_env", lambda: {})
    assert (
        module._normalize_s3_bucket("s3://lerobot-demo/checkpoints/") == "lerobot-demo"
    )
    assert (
        module._normalize_output_root("s3://s3://lerobot-demo/checkpoints/")
        == "s3://lerobot-demo/checkpoints"
    )
    assert (
        module._normalize_output_root("s3://lerobot-demo/checkpoints/")
        == "s3://lerobot-demo/checkpoints"
    )
    docs = module.render_workflow(
        YAML_PATH,
        run_id="byof-demo",
        output_root="s3://s3://lerobot-demo/checkpoints/",
    )
    assert (
        docs[1]["envs"]["S3_OUTPUT_PREFIX"]
        == "s3://lerobot-demo/checkpoints/byof-demo/"
    )
    assert docs[1]["envs"]["NPA_S3_BUCKET"] == "lerobot-demo"


def test_default_infra_uses_resolved_kubernetes_context(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_BYOF_K8S_CONTEXT", "customer-mk8s")
    monkeypatch.delenv("NPA_BYOF_INFRA", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_INFRA", raising=False)
    assert module._default_infra() == "k8s/customer-mk8s"


def test_ensure_infra_enabled_runs_sky_check_for_kubernetes(monkeypatch) -> None:
    module = _load_module()
    seen: list[list[str]] = []
    environments: list[tuple[Path | None, dict[str, str]]] = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        environments.append((isolated, dict(kwargs["env"])))
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"default": {"Kubernetes": ["compute"]}}', stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module,
        "sky_environment",
        lambda _isolated: {},
    )
    isolated = Path("/owner/isolated-sky-state")
    module._ensure_infra_enabled(
        sky_bin="/opt/sky",
        infra="k8s/customer-mk8s",
        config_path="/tmp/skypilot.yaml",
        isolated_config_dir=isolated,
    )

    assert seen == [
        ["/opt/sky", "api", "stop"],
        [
            "/opt/sky",
            "check",
            "kubernetes",
            "-o",
            "json",
            "--config",
            "/tmp/skypilot.yaml",
        ],
    ]
    assert [root for root, _env in environments] == [isolated, isolated]
    assert environments[0][1]["SKYPILOT_GLOBAL_CONFIG"] == "/tmp/skypilot.yaml"


def test_stop_sky_api_fails_closed_and_binds_exact_config(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    isolated = tmp_path / "isolated"
    config = tmp_path / "generated.yaml"
    observed: dict[str, object] = {}
    monkeypatch.setattr(module, "sky_environment", lambda root: {"ROOT": str(root)})

    def fail(cmd, **kwargs):
        observed.update(cmd=list(cmd), env=dict(kwargs["env"]))
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="failed")

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(module.SkyPilotConfigError, match="absence is unverified"):
        module._stop_sky_api(
            sky_bin="/opt/sky",
            isolated_config_dir=isolated,
            config_path=config,
        )
    assert observed["cmd"] == ["/opt/sky", "api", "stop"]
    assert observed["env"] == {
        "ROOT": str(isolated),
        "SKYPILOT_GLOBAL_CONFIG": str(config),
    }


def test_ensure_infra_enabled_skips_non_kubernetes(monkeypatch) -> None:
    module = _load_module()
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="aws/us-east-1")
    assert called is False


def test_ensure_infra_enabled_rejects_zero_exit_with_disabled_provider(
    monkeypatch,
) -> None:
    module = _load_module()

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout="{}", stderr=""
        ),
    )

    with pytest.raises(module.SkyPilotConfigError, match="did not enable compute"):
        module._ensure_infra_enabled(sky_bin="/opt/sky", infra="k8s/customer-mk8s")


def test_ensure_infra_enabled_parses_json_after_api_startup_prose(
    monkeypatch,
) -> None:
    module = _load_module()
    calls = 0

    def fake_run(cmd, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="",
            stderr=(
                "Failed to connect to local API server; starting one.\n"
                '{"default": {"Kubernetes": ["compute"]}}\n'
            ),
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="k8s/customer-mk8s")


def test_ensure_infra_enabled_examines_all_kubernetes_entries(monkeypatch) -> None:
    module = _load_module()
    calls = 0

    def fake_run(cmd, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "Kubernetes": {"enabled": False, "capabilities": []},
                    "profiles": [
                        {
                            "selected": {
                                "Kubernetes": {
                                    "enabled": True,
                                    "capabilities": ["compute"],
                                }
                            }
                        }
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="k8s/customer-mk8s")


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        ("", ""),
        ("not json", "still not json"),
        ('{"Kubernetes": []}', ""),
        ('{"Kubernetes": {"enabled": false, "capabilities": ["compute"]}}', ""),
        ('{"status": "error", "Kubernetes": ["compute"]}', ""),
        ('{"error": "authentication failed", "Kubernetes": ["compute"]}', ""),
        (
            '{"result": {"status": "error", "Kubernetes": ["compute"]}}',
            "",
        ),
        (
            '{"error": "stale provider state"}\n'
            '{"result": {"Kubernetes": ["compute"]}}',
            "",
        ),
    ],
)
def test_ensure_infra_enabled_rejects_empty_disabled_malformed_and_error_output(
    monkeypatch, stdout, stderr
) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_BYOF_REFRESH_SKY_API", "0")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout=stdout, stderr=stderr
        ),
    )

    with pytest.raises(module.SkyPilotConfigError, match="did not enable compute"):
        module._ensure_infra_enabled(sky_bin="/opt/sky", infra="k8s/customer-mk8s")


def test_ensure_infra_enabled_accepts_enabled_json_from_stdout_or_stderr(
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setenv("NPA_BYOF_REFRESH_SKY_API", "0")
    outputs = iter(
        [
            ('[{"Kubernetes": ["compute"]}]', ""),
            ("", 'startup prose\n{"default": {"Kubernetes": ["compute"]}}'),
        ]
    )

    def fake_run(cmd, **_kwargs):
        stdout, stderr = next(outputs)
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="kubernetes")
    module._ensure_infra_enabled(sky_bin="/opt/sky", infra="kubernetes")


def test_direct_launch_uses_sky_launch_with_down(monkeypatch, tmp_path, capsys) -> None:
    module = _load_module()
    rendered_yaml = tmp_path / "workflow.yaml"
    rendered_yaml.write_text("name: demo\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    rc = module._direct_launch(
        rendered_yaml=rendered_yaml,
        run_id="byof-demo",
        outputs={"summary": "s3://bucket/summary.json"},
        sky_bin="/opt/sky",
        infra="k8s/customer-mk8s",
        config_path="/tmp/skypilot.yaml",
        cleanup=True,
        secret_envs=["HF_TOKEN"],
    )

    assert rc == 0
    assert seen["cmd"] == [
        "/opt/sky",
        "launch",
        "--yes",
        "--cluster",
        "byof-demo",
        "--name",
        "byof-demo",
        "--down",
        "--infra",
        "k8s/customer-mk8s",
        "--config",
        "/tmp/skypilot.yaml",
        "--secret",
        "HF_TOKEN",
        str(rendered_yaml),
    ]
    output = capsys.readouterr().out
    assert '"mode": "direct-launch"' in output


def test_write_default_k8s_config_adds_pull_secrets(tmp_path) -> None:
    module = _load_module()
    config_path = module._write_default_k8s_config(tmp_path, "k8s/customer-mk8s")

    assert config_path
    text = Path(config_path).read_text(encoding="utf-8")
    assert "imagePullSecrets" in text
    assert "agent-sa" in text
    assert "serviceAccountName: skypilot-service-account" in text
    assert "kubectl create secret docker-registry" not in text
    assert "allowed_contexts" in text
    assert "customer-mk8s" in text


def test_normalize_kubeconfig_current_context(monkeypatch, tmp_path) -> None:
    module = _load_module()
    source = tmp_path / "source-kubeconfig"
    source.write_text(
        """
apiVersion: v1
kind: Config
current-context: old-context
contexts:
- name: target-context
  context: {}
clusters: []
users: []
""".strip(),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("KUBECONFIG", str(source))
    monkeypatch.setenv("KUBECONTEXT", "target-context")

    module._normalize_kubeconfig_current_context(out)

    updated = Path(os.environ["KUBECONFIG"]).read_text(encoding="utf-8")
    assert "current-context: target-context" in updated
    assert str(out) in os.environ["KUBECONFIG"]


def test_libero_normalizes_only_from_an_immutable_exact_run_copy(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    source = tmp_path / "source-kubeconfig"
    source.write_text(
        "apiVersion: v1\nkind: Config\ncurrent-context: old-context\n",
        encoding="utf-8",
    )
    source.chmod(0o600)
    out = tmp_path / "out"
    out.mkdir(mode=0o700)
    monkeypatch.setenv("KUBECONFIG", str(source))
    monkeypatch.setenv("KUBECONTEXT", "target-context")

    module._normalize_kubeconfig_current_context(out, immutable=True)
    source.write_text("attacker replacement\n", encoding="utf-8")

    snapshot = out / "execution-kubeconfig.source"
    normalized = Path(os.environ["KUBECONFIG"])
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o400
    assert stat.S_IMODE(normalized.stat().st_mode) == 0o400
    assert "current-context: old-context" in snapshot.read_text(encoding="utf-8")
    assert "current-context: target-context" in normalized.read_text(encoding="utf-8")
    assert "attacker replacement" not in normalized.read_text(encoding="utf-8")


def test_submit_and_wait_restores_kubeconfig_after_direct_launch(
    monkeypatch, tmp_path
) -> None:
    """Temp kubeconfig under TemporaryDirectory must not leak into later sky jobs."""
    module = _load_module()
    original = str(tmp_path / "original-kubeconfig")
    Path(original).write_text("kind: Config\n", encoding="utf-8")
    monkeypatch.setenv("KUBECONFIG", original)
    monkeypatch.setenv("NPA_BYOF_REFRESH_SKY_API", "1")
    monkeypatch.setattr(module, "resolve_sky_bin", lambda *_a, **_k: "/opt/sky")
    isolated = tmp_path / "isolated-state"
    isolated.mkdir()
    monkeypatch.setattr(module, "resolve_isolated_config_dir", lambda _value: isolated)
    monkeypatch.setattr(module, "_default_run_id", lambda: "byof-restore")
    monkeypatch.setattr(
        module,
        "render_workflow",
        lambda *_a, **_k: [
            {"name": "meta"},
            {"name": "task", "envs": {}, "resources": {}},
        ],
    )
    monkeypatch.setattr(module, "_write_yaml_documents", lambda *_a, **_k: None)

    def _leak_kubeconfig(tmp: Path, **_kwargs) -> None:
        os.environ["KUBECONFIG"] = str(tmp / "leaked")

    monkeypatch.setattr(
        module, "_normalize_kubeconfig_current_context", _leak_kubeconfig
    )
    monkeypatch.setattr(module, "_default_infra", lambda: "k8s/demo")
    config_path = tmp_path / "skypilot.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module, "_write_default_k8s_config", lambda *_a, **_k: str(config_path)
    )
    monkeypatch.setattr(module, "_ensure_infra_enabled", lambda **_k: None)
    monkeypatch.setattr(module, "preflight_output_storage", lambda **_k: None)
    monkeypatch.setattr(module, "_direct_launch", lambda **_k: 0)
    seen_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        del kwargs
        seen_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    environment_roots: list[Path | None] = []
    monkeypatch.setattr(
        module,
        "sky_environment",
        lambda root: environment_roots.append(root) or os.environ.copy(),
    )

    args = module._parse_args(
        [
            "--yaml",
            str(YAML_PATH),
            "--direct-launch",
            "--isolated-config-dir",
            str(isolated),
            "--output-root",
            "s3://bucket/prefix",
        ]
    )
    assert module._submit_and_wait(args) == 0
    assert os.environ.get("KUBECONFIG") == original
    assert ["/opt/sky", "api", "stop"] in seen_cmds
    assert environment_roots == [isolated]


@pytest.mark.parametrize(
    "mutation", ["mutable", "extra-data", "wrong-key", "invalid-key"]
)
def test_libero_storage_configmap_refuses_unqualified_mount(
    monkeypatch, mutation
) -> None:
    module = _load_module()
    key = bytes(range(32))
    item = {
        "immutable": True,
        "data": {
            "output-storage-authorization-public-key.b64": module.base64.b64encode(
                key
            ).decode()
        },
    }
    monkeypatch.setattr(module, "libero_image_manifest", lambda: {})
    monkeypatch.setattr(
        module,
        "validate_libero_qualified_image_manifest",
        lambda _manifest: {
            "output_storage_authorization_public_key_sha256": module.hashlib.sha256(
                key
            ).hexdigest()
        },
    )
    if mutation == "mutable":
        item["immutable"] = False
    elif mutation == "extra-data":
        item["data"]["unexpected"] = "value"
    elif mutation == "wrong-key":
        item["data"]["output-storage-authorization-public-key.b64"] = (
            module.base64.b64encode(bytes(32)).decode()
        )
    else:
        item["data"]["output-storage-authorization-public-key.b64"] = "!"
    with pytest.raises(RuntimeError, match="storage verification ConfigMap"):
        module._verify_libero_storage_configmap(item)


@pytest.mark.parametrize("kind", ["caller", "customer"])
@pytest.mark.parametrize(
    "mutation", ["none", "mutable", "extra", "wrong-key", "missing-file"]
)
def test_libero_provided_runtime_key_configmaps_are_exact(
    monkeypatch, tmp_path, kind, mutation
):
    module = _load_module()
    identity = "8" * 64
    key_name = (
        "authenticated-caller-public-key.b64" if kind == "caller" else identity + ".b64"
    )
    name = (
        module.LIBERO_CALLER_VERIFICATION_CONFIGMAP
        if kind == "caller"
        else module.LIBERO_CUSTOMER_REGISTRATION_CONFIGMAP
    )
    encoded = module.base64.b64encode(bytes(range(32))).decode()
    path = tmp_path / key_name
    path.write_text(encoded)
    path.chmod(0o600)
    env_name = (
        "NPA_LIBERO_AUTHENTICATED_CALLER_PUBLIC_KEY_FILE"
        if kind == "caller"
        else "NPA_LIBERO_CUSTOMER_SIGNER_REGISTRATION_FILE"
    )
    monkeypatch.setenv(env_name, str(path))
    monkeypatch.setenv("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", identity)
    item = {"metadata": {"name": name}, "immutable": True, "data": {key_name: encoded}}
    if mutation == "mutable":
        item["immutable"] = False
    elif mutation == "extra":
        item["data"]["foreign.b64"] = encoded
    elif mutation == "wrong-key":
        item["data"][key_name] = module.base64.b64encode(bytes(32)).decode()
    elif mutation == "missing-file":
        path.unlink()
    if mutation == "none":
        module._verify_libero_runtime_key_configmap(item)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            module._verify_libero_runtime_key_configmap(item)


def test_libero_render_binds_customer_registration_before_signing(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", "8" * 64)
    monkeypatch.setattr(module, "_resolved_storage_env", lambda: {})
    arguments = {"run_id": "libero-render-signing", "solution_name": "libero"}
    first = module.render_workflow(LIBERO_YAML_PATH, **arguments)
    rendered = json.dumps(first)
    assert "<customer-identity-sha256>" not in rendered
    assert "/run/npa/libero/customer-signer-roots/" + "8" * 64 + ".b64" in rendered
    first_sha256 = module.hashlib.sha256(
        module.libero_executable_profile_bytes(first)
    ).hexdigest()
    monkeypatch.setenv("NPA_LIBERO_CUSTOMER_IDENTITY_SHA256", "9" * 64)
    second = module.render_workflow(LIBERO_YAML_PATH, **arguments)
    assert (
        module.hashlib.sha256(
            module.libero_executable_profile_bytes(second)
        ).hexdigest()
        != first_sha256
    )
