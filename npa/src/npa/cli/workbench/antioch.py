"""CLI for the Antioch CPU control-plane Workbench adapter."""

from __future__ import annotations

import json
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

import typer

from npa.lifecycle_intent import OperationIntent, intent_boundary
from npa.sdk.workbench import antioch as sdk
from npa.workbench.antioch.manager import AntiochManager
from npa.workbench.antioch.project import package_project
from npa.workbench.antioch.runtime import (
    ensure_runtime,
    runtime_has_proprietary_distribution,
    terms_preflight,
)
from npa.workbench.antioch.schemas import CollectRequest, ResumeRequest, SubmitRequest
from npa.workbench.antioch.vendor_cli import AntiochCli

app = typer.Typer(
    name="antioch",
    help="Run Antioch simulations and collect policy-compatible data.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def _emit(value: Any, output: OutputFormat) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    typer.echo(
        json.dumps(payload, sort_keys=True)
        if output == OutputFormat.json
        else "\n".join(f"{key}: {item}" for key, item in payload.items())
    )


def _fail(exc: Exception) -> None:
    retryable = bool(getattr(exc, "retryable", False))
    error_type = str(getattr(exc, "error_type", type(exc).__name__))
    typer.echo(
        json.dumps(
            {
                "error": {
                    "type": error_type,
                    "message": str(exc),
                    "retryable": retryable,
                    "terminal": not retryable,
                }
            }
        ),
        err=True,
    )
    raise typer.Exit(1)


def _request(
    input_path: str,
    output_path: str,
    workflow_run: str,
    state_id: str,
    robot_type: str,
    task: str,
    suite: str,
    scenario: str,
    scenario_case: str,
    parameters_json: str,
) -> SubmitRequest:
    parameters = json.loads(parameters_json)
    if not isinstance(parameters, dict):
        raise ValueError("--parameters-json must be a JSON object")
    return SubmitRequest(
        input_path=input_path,
        output_path=output_path,
        workflow_run=workflow_run,
        state_id=state_id,
        robot_type=robot_type,
        task=task,
        suite=suite,
        scenario=scenario,
        scenario_case=scenario_case,
        parameters=parameters,
    )


@app.command("health")
def health(output: OutputFormat = typer.Option(OutputFormat.text, "--output")) -> None:
    try:
        _emit({"status": "ok", **AntiochCli(ensure_runtime()).health()}, output)
    except Exception as exc:
        _fail(exc)


@app.command("terms-preflight")
def terms_preflight_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Verify explicit, scoped Antioch terms acceptance before runtime use."""
    try:
        _emit({"status": "accepted", **terms_preflight()}, output)
    except Exception as exc:
        _fail(exc)


@app.command("package-project")
def package_project_cmd(
    project_dir: Path = typer.Option(
        ..., "--project-dir", exists=True, file_okay=False
    ),
    package_dir: Path = typer.Option(..., "--package-dir"),
    source_name: str = typer.Option(..., "--source-name"),
    source_revision: str = typer.Option(..., "--source-revision"),
    source_license: str = typer.Option(..., "--source-license"),
    source_sha256: str = typer.Option(..., "--source-sha256"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Build a deterministic, credential-free immutable project package."""
    try:
        manifest = package_project(
            project_dir,
            package_dir,
            source_name=source_name,
            source_revision=source_revision,
            source_license=source_license,
            source_sha256=source_sha256,
        )
        _emit(
            {
                "status": "packaged",
                "package_dir": str(package_dir),
                "manifest": manifest.model_dump(mode="json"),
            },
            output,
        )
    except Exception as exc:
        _fail(exc)


@app.command("system-info")
def system_info(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    _emit(
        {
            "status": "ok",
            "cpu_only": True,
            "proprietary_payload_baked": runtime_has_proprietary_distribution(),
        },
        output,
    )


@app.command("live-start")
def live_start_cmd(
    source: Path = typer.Option(..., "--source", exists=True, file_okay=False),
    project_id: str = typer.Option(..., "--project-id"),
    client_bundle: Path = typer.Option(
        ..., "--client-bundle", exists=True, file_okay=False
    ),
    scenario_timeout_seconds: int = typer.Option(
        14_400, "--scenario-timeout-seconds", min=60
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Start a continuing streamed OpenPI scenario under tmux supervision."""
    try:
        _emit(
            sdk.live_start(
                source=source,
                project_id=project_id,
                client_bundle=client_bundle,
                scenario_timeout_seconds=scenario_timeout_seconds,
            ),
            output,
        )
    except Exception as exc:
        _fail(exc)


@app.command("live-status")
def live_status_cmd(
    project_id: str = typer.Option(..., "--project-id"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Inspect exact local tmux supervisor state without reading auth storage."""
    try:
        _emit(sdk.live_status(project_id=project_id), output)
    except Exception as exc:
        _fail(exc)


@app.command("live-stop")
def live_stop_cmd(
    project_id: str = typer.Option(..., "--project-id"),
    timeout_seconds: float = typer.Option(120.0, "--timeout-seconds", min=1.0),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Cancel the exact scenario, then stop its exact sim service."""
    try:
        _emit(
            sdk.live_stop(project_id=project_id, timeout_seconds=timeout_seconds),
            output,
        )
    except Exception as exc:
        _fail(exc)


@app.command("live-k8s-deploy")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
def live_k8s_deploy_cmd(
    runtime_config: Path = typer.Option(
        ..., "--runtime-config", exists=True, dir_okay=False
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Reconcile the same-pod Antioch tunnel and cluster-local policy path."""
    try:
        _emit(sdk.live_k8s_deploy(runtime_config=runtime_config), output)
    except Exception as exc:
        _fail(exc)


@app.command("live-k8s-status")
@intent_boundary(OperationIntent.OBSERVE)
def live_k8s_status_cmd(
    runtime_config: Path = typer.Option(
        ..., "--runtime-config", exists=True, dir_okay=False
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Return sanitized adapter and retained-policy readiness."""
    try:
        _emit(sdk.live_k8s_status(runtime_config=runtime_config), output)
    except Exception as exc:
        _fail(exc)


@app.command("live-k8s-stop")
@intent_boundary(OperationIntent.DESTROY)
def live_k8s_stop_cmd(
    runtime_config: Path = typer.Option(
        ..., "--runtime-config", exists=True, dir_okay=False
    ),
    timeout_seconds: float = typer.Option(1_200.0, "--timeout-seconds", min=1.0),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Stop the exact scenario before its supported service tunnel."""
    try:
        _emit(
            sdk.live_k8s_stop(
                runtime_config=runtime_config, timeout_seconds=timeout_seconds
            ),
            output,
        )
    except Exception as exc:
        _fail(exc)


@app.command("live-k8s-finalize-cutover")
@intent_boundary(OperationIntent.DESTROY)
def live_k8s_finalize_cutover_cmd(
    runtime_config: Path = typer.Option(
        ..., "--runtime-config", exists=True, dir_okay=False
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    """Disable the exact owned public rollback Service after acceptance."""
    try:
        _emit(sdk.live_k8s_finalize_cutover(runtime_config=runtime_config), output)
    except Exception as exc:
        _fail(exc)


def _submit_action(
    action: str,
    input_path: str,
    output_path: str,
    workflow_run: str,
    state_id: str,
    robot_type: str,
    task: str,
    suite: str,
    scenario: str,
    scenario_case: str,
    parameters_json: str,
    endpoint: str,
    output: OutputFormat,
) -> None:
    try:
        _emit(
            getattr(sdk, action)(
                _request(
                    input_path,
                    output_path,
                    workflow_run,
                    state_id,
                    robot_type,
                    task,
                    suite,
                    scenario,
                    scenario_case,
                    parameters_json,
                ),
                endpoint=endpoint,
            ),
            output,
        )
    except Exception as exc:
        _fail(exc)


@app.command("submit")
def submit(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    workflow_run: str = typer.Option(..., "--workflow-run"),
    state_id: str = typer.Option(..., "--state-id"),
    robot_type: str = typer.Option(..., "--robot-type"),
    task: str = typer.Option(..., "--task"),
    suite: str = typer.Option("", "--suite"),
    scenario: str = typer.Option("", "--scenario"),
    scenario_case: str = typer.Option("", "--case"),
    parameters_json: str = typer.Option("{}", "--parameters-json"),
    endpoint: str = typer.Option("", "--endpoint"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    _submit_action(
        "submit",
        input_path,
        output_path,
        workflow_run,
        state_id,
        robot_type,
        task,
        suite,
        scenario,
        scenario_case,
        parameters_json,
        endpoint,
        output,
    )


@app.command("run")
def run(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    workflow_run: str = typer.Option(..., "--workflow-run"),
    state_id: str = typer.Option(..., "--state-id"),
    robot_type: str = typer.Option(..., "--robot-type"),
    task: str = typer.Option(..., "--task"),
    suite: str = typer.Option("", "--suite"),
    scenario: str = typer.Option("", "--scenario"),
    scenario_case: str = typer.Option("", "--case"),
    parameters_json: str = typer.Option("{}", "--parameters-json"),
    endpoint: str = typer.Option("", "--endpoint"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    _submit_action(
        "run",
        input_path,
        output_path,
        workflow_run,
        state_id,
        robot_type,
        task,
        suite,
        scenario,
        scenario_case,
        parameters_json,
        endpoint,
        output,
    )


def _resume(
    action: str,
    output_path: str,
    workflow_run: str,
    state_id: str,
    endpoint: str,
    output: OutputFormat,
    rerun_terminal: bool = False,
) -> None:
    try:
        request = ResumeRequest(
            output_path=output_path,
            workflow_run=workflow_run,
            state_id=state_id,
            rerun_terminal=rerun_terminal,
        )
        _emit(getattr(sdk, action)(request, endpoint=endpoint), output)
    except Exception as exc:
        _fail(exc)


@app.command("status")
def status(
    output_path: str = typer.Option(..., "--output-path"),
    workflow_run: str = typer.Option(..., "--workflow-run"),
    state_id: str = typer.Option(..., "--state-id"),
    endpoint: str = typer.Option("", "--endpoint"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    _resume("status", output_path, workflow_run, state_id, endpoint, output)


@app.command("reconcile")
def reconcile(
    output_path: str = typer.Option(..., "--output-path"),
    workflow_run: str = typer.Option(..., "--workflow-run"),
    state_id: str = typer.Option(..., "--state-id"),
    endpoint: str = typer.Option("", "--endpoint"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    _resume("reconcile", output_path, workflow_run, state_id, endpoint, output)


@app.command("cancel")
def cancel(
    output_path: str = typer.Option(..., "--output-path"),
    workflow_run: str = typer.Option(..., "--workflow-run"),
    state_id: str = typer.Option(..., "--state-id"),
    endpoint: str = typer.Option("", "--endpoint"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    _resume("cancel", output_path, workflow_run, state_id, endpoint, output)


@app.command("resume")
def resume(
    output_path: str = typer.Option(..., "--output-path"),
    workflow_run: str = typer.Option(..., "--workflow-run"),
    state_id: str = typer.Option(..., "--state-id"),
    rerun_terminal: bool = typer.Option(False, "--rerun-terminal"),
    endpoint: str = typer.Option("", "--endpoint"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    _resume(
        "resume", output_path, workflow_run, state_id, endpoint, output, rerun_terminal
    )


@app.command("collect")
def collect(
    output_path: str = typer.Option(..., "--output-path"),
    workflow_run: str = typer.Option(..., "--workflow-run"),
    state_id: str = typer.Option(..., "--state-id"),
    require_policy_dataset: bool = typer.Option(
        True, "--require-policy-dataset/--allow-artifacts-only"
    ),
    endpoint: str = typer.Option("", "--endpoint"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    try:
        request = CollectRequest(
            output_path=output_path,
            workflow_run=workflow_run,
            state_id=state_id,
            require_policy_dataset=require_policy_dataset,
        )
        _emit(sdk.collect(request, endpoint=endpoint), output)
    except Exception as exc:
        _fail(exc)


@app.command("list")
def list_operations(
    output_path: str = typer.Option(..., "--output-path"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    try:
        _emit(
            {
                "operations": [
                    item.model_dump(mode="json")
                    for item in AntiochManager().states.list(output_path)
                ]
            },
            output,
        )
    except Exception as exc:
        _fail(exc)


@app.command("deploy")
def deploy(
    image: str = typer.Option(..., "--image"),
    name: str = typer.Option("npa-antioch", "--name"),
    namespace: str = typer.Option("default", "--namespace"),
    antioch_config_secret: str = typer.Option(..., "--antioch-config-secret"),
    service_token_secret: str = typer.Option(..., "--service-token-secret"),
    terms_acceptance_secret: str = typer.Option(..., "--terms-acceptance-secret"),
    s3_credentials_secret: str = typer.Option("", "--s3-credentials-secret"),
    storage_endpoint: str = typer.Option(
        "https://storage.eu-north1.nebius.cloud", "--storage-endpoint"
    ),
    apply: bool = typer.Option(False, "--apply"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    manifest = _deployment(
        image,
        name,
        namespace,
        antioch_config_secret,
        service_token_secret,
        terms_acceptance_secret,
        s3_credentials_secret,
        storage_endpoint,
    )
    if apply:
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=json.dumps(manifest),
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if result.returncode:
            _fail(RuntimeError("kubectl failed to apply the Antioch deployment"))
    _emit({"status": "applied" if apply else "rendered", "manifest": manifest}, output)


def _deployment(
    image: str,
    name: str,
    namespace: str,
    config_secret: str,
    token_secret: str,
    terms_secret: str,
    s3_secret: str = "",
    storage_endpoint: str = "https://storage.eu-north1.nebius.cloud",
) -> dict[str, Any]:
    labels = {"app.kubernetes.io/name": name}
    container = {
        "name": "antioch",
        "image": image,
        "ports": [{"containerPort": 8789}],
        "env": [
            {"name": "ANTIOCH_CONFIG_DIR", "value": "/etc/antioch"},
            {
                "name": "NPA_ANTIOCH_RUNTIME_CACHE",
                "value": "/workspace/.cache/npa/antioch",
            },
            {
                "name": "ANTIOCH_WORKBENCH_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": token_secret, "key": "token"}},
            },
            {
                "name": "NPA_ANTIOCH_ACCEPT_TERMS",
                "valueFrom": {
                    "secretKeyRef": {"name": terms_secret, "key": "accepted"}
                },
            },
            {"name": "AWS_ENDPOINT_URL", "value": storage_endpoint},
        ],
        "volumeMounts": [
            {"name": "antioch-config", "mountPath": "/etc/antioch", "readOnly": True},
            {"name": "runtime-cache", "mountPath": "/workspace/.cache/npa/antioch"},
        ],
        "resources": {
            "requests": {"cpu": "250m", "memory": "512Mi"},
            "limits": {"cpu": "2", "memory": "2Gi"},
        },
    }
    if s3_secret:
        container["envFrom"] = [{"secretRef": {"name": s3_secret}}]
    pod = {
        "metadata": {"labels": labels},
        "spec": {
            "securityContext": {"runAsNonRoot": True, "fsGroup": 10001},
            "containers": [container],
            "volumes": [
                {"name": "antioch-config", "secret": {"secretName": config_secret}},
                {"name": "runtime-cache", "emptyDir": {}},
            ],
        },
    }
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": name, "namespace": namespace},
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": labels},
                    "template": pod,
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": name, "namespace": namespace},
                "spec": {
                    "selector": labels,
                    "ports": [{"port": 8789, "targetPort": 8789}],
                },
            },
        ],
    }
