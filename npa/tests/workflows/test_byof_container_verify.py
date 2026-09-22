from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
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
    workflow = yaml.safe_load((ROOT / "workflows/testing/byof-robotwin.yaml").read_text())
    checked = []

    class BoundaryChecked(Exception):
        pass

    def bootstrap(_directory, control):
        assert control["HOME"] == environment["HOME"]
        assert control["PATH"] == environment["PATH"]
        return "/synthetic/sky"

    def submit(path, _run_id, **kwargs):
        documents = module._load_yaml_documents(path)
        checked.append(validate_confidential_submit_bridge(
            kwargs["robotwin_submit_context"],
            documents=documents,
            infra=kwargs["infra"],
            config_path=kwargs["config_path"],
            secret_envs=kwargs["secret_envs"],
            extra_env=kwargs["extra_env"],
            execution_target=kwargs["execution_target"],
            execution_report=kwargs["execution_preflight_report"],
        ))
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
                "--yaml", str(
                    ROOT / "npa/src/npa/workflows/byof/profiles"
                    / (workflow["config"]["resource_profile_yaml"] + ".yaml")
                ),
                "--solution-name", "robotwin",
                "--smoke-command", workflow["config"]["smoke_command"],
                "--capability-name", workflow["config"]["capability_name"],
                "--smoke-artifact-name", "robotwin-smoke.json",
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


def test_generic_wrapper_preserves_explicit_isolated_state(inert_wrapper, monkeypatch):
    module, effects = inert_wrapper
    selected = "/synthetic-generic-state"
    launch = Mock(return_value=SimpleNamespace(log_paths={}))
    poll = Mock(return_value=(SimpleNamespace(status="SUCCEEDED", returncode=0), {}))
    monkeypatch.setattr(module, "submit_workflow", launch)
    monkeypatch.setattr(module, "_wait_for_terminal", poll)
    assert (
        module.main(
            [
                "--run-id",
                "synthetic",
                "--no-direct-launch",
                "--isolated-config-dir",
                selected,
                "--config-path",
                "/synthetic-config",
            ]
        )
        == 0
    )
    assert launch.call_args.kwargs["isolated_config_dir"] == selected
    assert poll.call_args.kwargs["isolated_config_dir"] == selected
    assert launch.call_args.kwargs["config_path"] == Path("/synthetic-config")
    assert poll.call_args.kwargs["config_path"] == Path("/synthetic-config")
    assert launch.call_args.kwargs["robotwin_submit_context"] is None
    launch.assert_called_once()
    effects["mkdir"].assert_not_called()
    effects["_bootstrap_robotwin_sky"].assert_not_called()
    module.subprocess.run.assert_not_called()


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
        module._submit_and_wait(
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


def test_render_workflow_injects_solution_smoke_metadata(monkeypatch) -> None:
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
    assert envs["NPA_S3_BUCKET"] == "bucket"
    assert envs["AWS_ENDPOINT_URL"] == "https://storage.example"
    assert "AWS_ACCESS_KEY_ID" not in envs
    assert "AWS_SECRET_ACCESS_KEY" not in envs
    assert "AWS_SESSION_TOKEN" not in envs
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in envs
    assert task["resources"]["image_id"] == "docker:registry.example/npa-byof:demo"


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


def test_wait_timeout_less_than_negative_one_is_rejected() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="must be -1"):
        module._wait_for_terminal(
            "run", sky_bin="sky", wait_timeout=-2, poll_interval=1
        )


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

    def fake_run(cmd, **kwargs):
        del kwargs
        seen.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"default": {"Kubernetes": ["compute"]}}', stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._ensure_infra_enabled(
        sky_bin="/opt/sky",
        infra="k8s/customer-mk8s",
        config_path="/tmp/skypilot.yaml",
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

    def _leak_kubeconfig(tmp: Path) -> None:
        os.environ["KUBECONFIG"] = str(tmp / "leaked")

    monkeypatch.setattr(
        module, "_normalize_kubeconfig_current_context", _leak_kubeconfig
    )
    monkeypatch.setattr(module, "_default_infra", lambda: "k8s/demo")
    monkeypatch.setattr(
        module, "_write_default_k8s_config", lambda *_a, **_k: "/tmp/skypilot.yaml"
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
    monkeypatch.setattr(module, "sky_environment", lambda *_a, **_k: os.environ.copy())

    args = module._parse_args(
        [
            "--yaml",
            str(YAML_PATH),
            "--direct-launch",
            "--output-root",
            "s3://bucket/prefix",
        ]
    )
    assert module._submit_and_wait(args) == 0
    assert os.environ.get("KUBECONFIG") == original
    # Restoring this wrapper's environment does not grant ownership of the API.
    assert seen_cmds == []
