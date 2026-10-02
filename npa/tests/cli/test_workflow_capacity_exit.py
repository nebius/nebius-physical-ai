"""Only proven pre-launch capacity errors may request an automatic retry."""

import pytest

from npa.cli.workbench.workflow import _submit_failure_code
from npa.execution_preflight import ExecutionPreflightError
from npa.orchestration.skypilot.k8s_gpu_catalog import (
    KubernetesGpuCatalogError,
    PendingGpuPlacementError,
    PermanentlyUnsatisfiableAcceleratorError,
    TemporarilyUnavailableAcceleratorError,
)
from npa.orchestration.skypilot.workflow import SkyPilotSubmitError


def temporary_preflight(error_type=TemporarilyUnavailableAcceleratorError):
    cause = error_type("One node must become free")
    wrapper = ExecutionPreflightError("gpu", "capacity unavailable", status="unknown")
    wrapper.__cause__ = cause
    return wrapper


@pytest.mark.parametrize(
    "error_type", [TemporarilyUnavailableAcceleratorError, PendingGpuPlacementError]
)
def test_typed_preflight_cause_survives_execution_wrapper(error_type):
    assert _submit_failure_code(temporary_preflight(error_type)) == 75


@pytest.mark.parametrize("attempted,expected", [(False, 75), (True, 1), (None, 1)])
@pytest.mark.parametrize(
    "error_type", [TemporarilyUnavailableAcceleratorError, PendingGpuPlacementError]
)
def test_final_preflight_race_retries_only_with_no_provider_launch(
    attempted, expected, error_type
):
    error = SkyPilotSubmitError("submission failed", launch_attempted=attempted)
    error.__cause__ = temporary_preflight(error_type)
    assert _submit_failure_code(error) == expected


@pytest.mark.parametrize(
    "error",
    [
        KubernetesGpuCatalogError("Forbidden"),
        PermanentlyUnsatisfiableAcceleratorError("Shape cannot fit"),
        RuntimeError("free shared GPU capacity is indeterminate"),
        ValueError("Invalid configuration"),
    ],
)
def test_message_text_and_non_capacity_errors_cannot_trigger_retry(error):
    assert _submit_failure_code(error) == 1


def test_cause_cycle_fails_closed():
    error = RuntimeError("cycle")
    error.__cause__ = error
    assert _submit_failure_code(error) == 1


def test_transaction_evidence_overrides_no_launch_hint():
    from types import SimpleNamespace

    error = SkyPilotSubmitError(
        "ambiguous",
        launch_attempted=False,
        transaction=SimpleNamespace(launch_sequence=1),
    )
    error.__cause__ = temporary_preflight()
    assert _submit_failure_code(error) == 1


@pytest.mark.parametrize("temporary,expected", [(True, 75), (False, 1)])
def test_real_submit_cli_preserves_capacity_code_before_launch(
    monkeypatch, tmp_path, temporary, expected
):
    from shutil import which
    from typer.testing import CliRunner
    from npa.cli.main import app
    from npa.cli.workbench import workflow

    spec = tmp_path / "work.yaml"
    spec.write_text(
        "apiVersion: npa.workflow/v0.0.1\nkind: Workflow\n"
        "metadata: {name: unit}\nconfig: {bucket: unit-output}\n"
        "resources: {cpu: {cloud: kubernetes, cpus: 1, memory: 1Gi}}\n"
        "initial: execute\nstates:\n  execute:\n    resources: cpu\n"
        '    run: {shell: "true"}\n    terminal: true\n'
    )

    def check(*args, **kwargs):
        if temporary:
            raise temporary_preflight()
        raise ExecutionPreflightError("scope", "identity unavailable")

    monkeypatch.setattr(workflow, "_execution_target_preflight", check)
    monkeypatch.setattr(workflow, "_verify_submit_controller_owner", lambda **kw: None)
    monkeypatch.setattr(workflow, "_available_kube_contexts", lambda: ["unit-context"])
    monkeypatch.setattr(workflow, "_adopt_npa_kubeconfig", lambda _: True)
    monkeypatch.setattr(
        workflow,
        "_preflight_submit_images",
        lambda *a, **kw: pytest.fail("No image staging"),
    )
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(spec),
            "--run-id",
            "capacity-unit",
            "--project",
            "unit",
            "--infra",
            "k8s/unit-context",
            "--sky-bin",
            which("true"),
            "--image",
            "registry.example/unit:dev",
            "--no-deploy-if-absent",
            "--no-stage-src",
            "--skip-preflight",
        ],
    )
    assert result.exit_code == expected, result.output
