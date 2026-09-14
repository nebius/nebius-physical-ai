"""CLI for Enactic OpenArm MuJoCo and Isaac Lab workloads."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from enum import Enum
from typing import Any

import typer

from npa.clients.credentials import apply_shared_credential_env, load_credentials
from npa.clients.project_credentials import storage_env_for_project
from npa.deploy.images import container_image_for_tool
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.sdk.workbench import openarm as sdk
from npa.workbench.openarm.schemas import (
    DEFAULT_ISAAC_TASK,
    DEFAULT_PORT,
    DEFAULT_STEPS,
    DEFAULT_TOKEN_ENV,
)

app = typer.Typer(
    name="openarm",
    help="Enactic OpenArm simulation with real MuJoCo and Isaac Sim/Isaac Lab.",
    no_args_is_help=True,
)

GPU_NODE_SELECTORS = {
    "l40s": "gpu-l40s-d",
    "rtx6000": "gpu-rtx6000",
    "rtxpro6000": "gpu-rtx6000",
}


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def _emit(payload: dict[str, Any], output_format: OutputFormat) -> None:
    if output_format == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo("\n".join(f"{key}: {value}" for key, value in payload.items()))


@app.command("run")
@intent_boundary(OperationIntent.MUTATE)
@json_stdout_contract
def run_cmd(
    simulator: str = typer.Option(..., "--simulator", help="mujoco or isaac-lab."),
    output_path: str = typer.Option(..., "--output-path", help="S3 artifact prefix."),
    steps: int = typer.Option(DEFAULT_STEPS, "--steps", help="Physics/control steps."),
    seed: int = typer.Option(17, "--seed"),
    render: bool = typer.Option(
        False, "--render/--no-render", help="Encode a MuJoCo MP4."
    ),
    task: str = typer.Option(
        DEFAULT_ISAAC_TASK, "--task", help="Upstream OpenArm Isaac environment id."
    ),
    num_envs: int = typer.Option(
        64, "--num-envs", help="Isaac vectorized environments."
    ),
    isaac_mode: str = typer.Option("rollout", "--isaac-mode", help="rollout or train."),
    max_iterations: int = typer.Option(
        1, "--max-iterations", help="RSL-RL iterations in train mode."
    ),
    service: bool = typer.Option(
        False, "--service", help="Submit to a deployed service."
    ),
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env"),
    wait: bool = typer.Option(False, "--wait", help="Wait for a service run."),
    poll_seconds: float = typer.Option(30.0, "--poll-seconds"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Run a real OpenArm control rollout or upstream Isaac Lab training."""
    try:
        response = sdk.run(
            simulator=simulator,
            output_path=output_path,
            steps=steps,
            seed=seed,
            render=render,
            task=task,
            num_envs=num_envs,
            isaac_mode=isaac_mode,
            max_iterations=max_iterations,
            mode="service" if service else "local",
            endpoint=endpoint,
            token_env=token_env,
        )
        payload = response.model_dump(mode="json")
        if service and wait:
            payload = _wait(response.run_id, endpoint, token_env, poll_seconds)
    except (
        sdk.OpenArmValidationError,
        sdk.OpenArmServiceError,
        ValueError,
        RuntimeError,
    ) as exc:
        _fail(str(exc))
    _emit(payload, output_format)


def _wait(
    run_id: str, endpoint: str, token_env: str, poll_seconds: float
) -> dict[str, Any]:
    while True:
        status = sdk.status(run_id=run_id, endpoint=endpoint, token_env=token_env)
        if status.status == "completed":
            return status.model_dump(mode="json")
        if status.status == "failed":
            raise sdk.OpenArmServiceError(status.error or "OpenArm run failed")
        time.sleep(max(0.0, poll_seconds))


@app.command("status")
@intent_boundary(OperationIntent.OBSERVE)
@json_stdout_contract
def status_cmd(
    run_id: str = typer.Argument(...),
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Read one service-run state."""
    try:
        payload = sdk.status(
            run_id=run_id, endpoint=endpoint, token_env=token_env
        ).model_dump(mode="json")
    except (sdk.OpenArmValidationError, sdk.OpenArmServiceError) as exc:
        _fail(str(exc))
    _emit(payload, output_format)


@app.command("list")
@intent_boundary(OperationIntent.OBSERVE)
@json_stdout_contract
def list_cmd(
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """List service runs."""
    try:
        payload = sdk.list_runs(endpoint=endpoint, token_env=token_env).model_dump(
            mode="json"
        )
    except (sdk.OpenArmValidationError, sdk.OpenArmServiceError) as exc:
        _fail(str(exc))
    _emit(payload, output_format)


@app.command("system-info")
@intent_boundary(OperationIntent.OBSERVE)
@json_stdout_contract
def system_info_cmd(
    service: bool = typer.Option(False, "--service"),
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Show packaged upstream and simulator identities without fetching Isaac."""
    try:
        payload = sdk.system_info(
            service=service, endpoint=endpoint, token_env=token_env
        ).model_dump(mode="json")
    except (sdk.OpenArmValidationError, sdk.OpenArmServiceError) as exc:
        _fail(str(exc))
    _emit(payload, output_format)


@app.command("serve")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(DEFAULT_PORT, "--port"),
) -> None:
    """Start the OpenArm service in the current container."""
    import uvicorn

    uvicorn.run("npa.workbench.openarm.service:app", host=host, port=port)


def _service_env(project: str, token_env: str) -> dict[str, str]:
    credentials = load_credentials()
    env = {"OPENARM_AUTH_MODE": "token", "OPENARM_PORT": str(DEFAULT_PORT)}
    apply_shared_credential_env(env, credentials)
    if project:
        env.update(storage_env_for_project(project))
    token = os.environ.get(token_env, "")
    if not token:
        _fail(f"{token_env} is required to deploy the authenticated service")
    env["OPENARM_TOKEN"] = token
    return {key: value for key, value in env.items() if value}


def _manifest(
    project: str,
    image: str,
    name: str,
    namespace: str,
    cache_pvc: str,
    token_env: str,
    node_selector_key: str,
    node_selector_value: str,
) -> dict[str, Any]:
    env = _service_env(project, token_env)
    volume = (
        {"name": "isaac-cache", "persistentVolumeClaim": {"claimName": cache_pvc}}
        if cache_pvc
        else {"name": "isaac-cache", "emptyDir": {"sizeLimit": "50Gi"}}
    )
    labels = {
        "app.kubernetes.io/name": "npa-openarm",
        "app.kubernetes.io/instance": name,
    }
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": f"{name}-env", "namespace": namespace},
                "type": "Opaque",
                "data": {
                    key: base64.b64encode(value.encode()).decode()
                    for key, value in env.items()
                },
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": name, "namespace": namespace, "labels": labels},
                "spec": {
                    "replicas": 1,
                    "strategy": {"type": "Recreate"},
                    "selector": {"matchLabels": {"app.kubernetes.io/instance": name}},
                    "template": {
                        "metadata": {"labels": labels},
                        "spec": {
                            "nodeSelector": {node_selector_key: node_selector_value},
                            "tolerations": [
                                {
                                    "key": "nvidia.com/gpu",
                                    "operator": "Exists",
                                    "effect": "NoSchedule",
                                }
                            ],
                            "securityContext": {"fsGroup": 1000},
                            "volumes": [volume],
                            "containers": [
                                {
                                    "name": "service",
                                    "image": image,
                                    "args": ["npa", "workbench", "openarm", "serve"],
                                    "envFrom": [{"secretRef": {"name": f"{name}-env"}}],
                                    "ports": [
                                        {"name": "http", "containerPort": DEFAULT_PORT}
                                    ],
                                    "volumeMounts": [
                                        {
                                            "name": "isaac-cache",
                                            "mountPath": "/opt/isaac-cache",
                                        }
                                    ],
                                    "resources": {
                                        "requests": {"nvidia.com/gpu": "1"},
                                        "limits": {"nvidia.com/gpu": "1"},
                                    },
                                    "readinessProbe": {
                                        "httpGet": {"path": "/health", "port": "http"}
                                    },
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "capabilities": {"drop": ["ALL"]},
                                        "seccompProfile": {"type": "RuntimeDefault"},
                                    },
                                }
                            ],
                        },
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": name, "namespace": namespace},
                "spec": {
                    "selector": {"app.kubernetes.io/instance": name},
                    "ports": [
                        {"name": "http", "port": DEFAULT_PORT, "targetPort": "http"}
                    ],
                },
            },
        ],
    }


def _kubectl(
    arguments: list[str], *, payload: str | None = None, kubeconfig: str = ""
) -> None:
    command = (
        ["kubectl"] + (["--kubeconfig", kubeconfig] if kubeconfig else []) + arguments
    )
    try:
        subprocess.run(command, input=payload, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"kubectl failed: {exc}")


@app.command("deploy")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def deploy_cmd(
    project: str = typer.Option("", "--project", "-p"),
    image: str = typer.Option("", "--image"),
    name: str = typer.Option("npa-openarm", "--name"),
    namespace: str = typer.Option("workbench", "--namespace"),
    kubeconfig: str = typer.Option("", "--kubeconfig"),
    isaac_cache_pvc: str = typer.Option("", "--isaac-cache-pvc"),
    gpu_type: str = typer.Option(
        "rtxpro6000", "--gpu-type", help="GPU type: l40s, rtx6000, or rtxpro6000."
    ),
    node_selector_key: str = typer.Option(
        "node.kubernetes.io/instance-type", "--node-selector-key"
    ),
    node_selector_value: str = typer.Option("", "--node-selector-value"),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Ensure an authenticated single-RTX OpenArm service is deployed."""
    selector_value = node_selector_value.strip() or GPU_NODE_SELECTORS.get(
        gpu_type.strip().lower()
    )
    if not selector_value:
        _fail(
            "--gpu-type must be l40s, rtx6000, or rtxpro6000 unless "
            "--node-selector-value is provided"
        )
    resolved_image = image or container_image_for_tool("openarm")
    manifest = _manifest(
        project,
        resolved_image,
        name,
        namespace,
        isaac_cache_pvc,
        token_env,
        node_selector_key,
        selector_value,
    )
    if dry_run:
        redacted = json.loads(json.dumps(manifest))
        redacted["items"][0]["data"] = {
            key: "<redacted>" for key in redacted["items"][0]["data"]
        }
        _emit(redacted, output_format)
        return
    _kubectl(["apply", "-f", "-"], payload=json.dumps(manifest), kubeconfig=kubeconfig)
    _kubectl(
        ["rollout", "status", f"deployment/{name}", "-n", namespace],
        kubeconfig=kubeconfig,
    )
    _emit(
        {
            "status": "deployed",
            "name": name,
            "namespace": namespace,
            "image": resolved_image,
            "node_selector": {node_selector_key: selector_value},
        },
        output_format,
    )


@app.command("delete")
@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def delete_cmd(
    name: str = typer.Option("npa-openarm", "--name"),
    namespace: str = typer.Option("workbench", "--namespace"),
    kubeconfig: str = typer.Option("", "--kubeconfig"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Delete only the named OpenArm service resources; retain cache PVCs."""
    for kind in ("service", "deployment", "secret"):
        target = f"{name}-env" if kind == "secret" else name
        _kubectl(
            ["delete", kind, target, "-n", namespace, "--ignore-not-found=true"],
            kubeconfig=kubeconfig,
        )
    _emit(
        {
            "status": "deleted",
            "name": name,
            "namespace": namespace,
            "cache_retained": True,
        },
        output_format,
    )
