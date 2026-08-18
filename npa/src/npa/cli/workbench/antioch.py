"""CLI for the Antioch CPU control-plane Workbench adapter."""

from __future__ import annotations

import json
import subprocess
from enum import Enum
from typing import Any

import typer

from npa.sdk.workbench import antioch as sdk
from npa.workbench.antioch.manager import AntiochManager
from npa.workbench.antioch.runtime import (
    ensure_runtime,
    runtime_has_proprietary_distribution,
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
    typer.echo(
        json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}),
        err=True,
    )
    raise typer.Exit(1)


def _request(
    input_path: str,
    output_path: str,
    workflow_run: str,
    state_id: str,
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


def _submit_action(
    action: str,
    input_path: str,
    output_path: str,
    workflow_run: str,
    state_id: str,
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
    robot_type: str = typer.Option("cartpole", "--robot-type"),
    task: str = typer.Option("Balance a cartpole", "--task"),
    endpoint: str = typer.Option("", "--endpoint"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    try:
        request = CollectRequest(
            output_path=output_path,
            workflow_run=workflow_run,
            state_id=state_id,
            require_policy_dataset=require_policy_dataset,
            robot_type=robot_type,
            task=task,
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
    s3_credentials_secret: str = typer.Option("", "--s3-credentials-secret"),
    apply: bool = typer.Option(False, "--apply"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output"),
) -> None:
    manifest = _deployment(
        image,
        name,
        namespace,
        antioch_config_secret,
        service_token_secret,
        s3_credentials_secret,
    )
    if apply:
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=json.dumps(manifest),
            text=True,
            capture_output=True,
            check=False,
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
    s3_secret: str = "",
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
