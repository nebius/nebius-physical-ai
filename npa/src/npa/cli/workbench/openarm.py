"""CLI for Enactic OpenArm MuJoCo and Isaac Lab workloads."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from enum import Enum
from typing import Annotated, Any

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


_RENDER_HELP = "Encode a MuJoCo MP4."
_TASK_HELP = "Upstream OpenArm Isaac environment id."
_NUM_ENVS_HELP = "Isaac vectorized environments."
_ITERATIONS_HELP = "RSL-RL iterations in train mode."
ServiceOption = Annotated[
    bool, typer.Option("--service", help="Submit to a deployed service.")
]
GpuTypeOption = Annotated[
    str,
    typer.Option("--gpu-type", help="GPU type: l40s, rtx6000, or rtxpro6000."),
]
NodeSelectorKeyOption = Annotated[str, typer.Option("--node-selector-key")]


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def _emit(payload: dict[str, Any], output_format: OutputFormat) -> None:
    if output_format == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo("\n".join(f"{key}: {value}" for key, value in payload.items()))


def _run_payload(
    simulator: str,
    output_path: str,
    steps: int,
    seed: int,
    render: bool,
    task: str,
    num_envs: int,
    isaac_mode: str,
    max_iterations: int,
    service: bool,
    endpoint: str,
    token_env: str,
    wait: bool,
    poll_seconds: float,
) -> dict[str, Any]:
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
    if service and wait:
        return _wait(response.run_id, endpoint, token_env, poll_seconds)
    return response.model_dump(mode="json")


def _safe_run_payload(*args: Any) -> dict[str, Any]:
    try:
        return _run_payload(*args)
    except (
        sdk.OpenArmValidationError,
        sdk.OpenArmServiceError,
        ValueError,
        RuntimeError,
    ) as exc:
        _fail(str(exc))
    raise AssertionError("_fail always exits")


@app.command("run")
@intent_boundary(OperationIntent.MUTATE)
@json_stdout_contract
def run_cmd(
    simulator: str = typer.Option(..., "--simulator", help="mujoco or isaac-lab."),
    output_path: str = typer.Option(..., "--output-path", help="S3 artifact prefix."),
    steps: int = typer.Option(DEFAULT_STEPS, "--steps", help="Physics/control steps."),
    seed: int = typer.Option(17, "--seed"),
    render: bool = typer.Option(False, "--render/--no-render", help=_RENDER_HELP),
    task: str = typer.Option(DEFAULT_ISAAC_TASK, "--task", help=_TASK_HELP),
    num_envs: int = typer.Option(64, "--num-envs", help=_NUM_ENVS_HELP),
    isaac_mode: str = typer.Option("rollout", "--isaac-mode", help="rollout or train."),
    max_iterations: int = typer.Option(1, "--max-iterations", help=_ITERATIONS_HELP),
    service: ServiceOption = False,
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env"),
    wait: bool = typer.Option(False, "--wait", help="Wait for a service run."),
    poll_seconds: float = typer.Option(30.0, "--poll-seconds"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Run a real OpenArm control rollout or upstream Isaac Lab training."""
    payload = _safe_run_payload(
        simulator,
        output_path,
        steps,
        seed,
        render,
        task,
        num_envs,
        isaac_mode,
        max_iterations,
        service,
        endpoint,
        token_env,
        wait,
        poll_seconds,
    )
    _emit(payload, output_format)


@app.command("qualify")
@intent_boundary(OperationIntent.MUTATE)
@json_stdout_contract
def qualify_cmd(
    input_path: str = typer.Option(..., "--input-path", help="S3 workflow root."),
    output_path: str = typer.Option(..., "--output-path", help="S3 report prefix."),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Validate every MuJoCo and Isaac artifact and write a qualification report."""
    try:
        response = sdk.qualify(input_path=input_path, output_path=output_path)
    except (sdk.OpenArmValidationError, ValueError, RuntimeError) as exc:
        _fail(str(exc))
    _emit(response.model_dump(mode="json", by_alias=True), output_format)


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
    host: str = typer.Option("127.0.0.1", "--host"),
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


def _cache_volume(cache_pvc: str) -> dict[str, Any]:
    return {
        "name": "isaac-cache",
        "persistentVolumeClaim": {"claimName": cache_pvc},
    }


def _service_container(image: str, secret_name: str) -> dict[str, Any]:
    return {
        "name": "service",
        "image": image,
        "envFrom": [{"secretRef": {"name": secret_name}}],
        "ports": [{"name": "http", "containerPort": DEFAULT_PORT}],
        "volumeMounts": [{"name": "isaac-cache", "mountPath": "/opt/isaac-cache"}],
        "resources": {
            "requests": {"nvidia.com/gpu": "1"},
            "limits": {"nvidia.com/gpu": "1"},
        },
        "readinessProbe": {"httpGet": {"path": "/health", "port": "http"}},
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
            "seccompProfile": {"type": "RuntimeDefault"},
        },
    }


def _pod_spec(
    image: str,
    secret_name: str,
    cache_pvc: str,
    node_selector_key: str,
    node_selector_value: str,
) -> dict[str, Any]:
    return {
        "nodeSelector": {node_selector_key: node_selector_value},
        "tolerations": [
            {
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        ],
        "securityContext": {"fsGroup": 1000},
        "volumes": [_cache_volume(cache_pvc)],
        "containers": [_service_container(image, secret_name)],
    }


def _deployment(
    image: str,
    name: str,
    namespace: str,
    labels: dict[str, str],
    cache_pvc: str,
    node_selector_key: str,
    node_selector_value: str,
) -> dict[str, Any]:
    pod_spec = _pod_spec(
        image, f"{name}-env", cache_pvc, node_selector_key, node_selector_value
    )
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "replicas": 1,
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": {"app.kubernetes.io/instance": name}},
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }


def _secret(env: dict[str, str], name: str, namespace: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": f"{name}-env", "namespace": namespace},
        "type": "Opaque",
        "data": {
            key: base64.b64encode(value.encode()).decode() for key, value in env.items()
        },
    }


def _service(name: str, namespace: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "selector": {"app.kubernetes.io/instance": name},
            "ports": [{"name": "http", "port": DEFAULT_PORT, "targetPort": "http"}],
        },
    }


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
    labels = {
        "app.kubernetes.io/name": "npa-openarm",
        "app.kubernetes.io/instance": name,
    }
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            _secret(env, name, namespace),
            _deployment(
                image,
                name,
                namespace,
                labels,
                cache_pvc,
                node_selector_key,
                node_selector_value,
            ),
            _service(name, namespace),
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


def _node_selector(gpu_type: str, explicit_value: str) -> str:
    value = explicit_value.strip() or GPU_NODE_SELECTORS.get(gpu_type.strip().lower())
    if value:
        return value
    _fail(
        "--gpu-type must be l40s, rtx6000, or rtxpro6000 unless "
        "--node-selector-value is provided"
    )
    raise AssertionError("_fail always exits")


def _redacted_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(manifest))
    redacted["items"][0]["data"] = {
        key: "<redacted>" for key in redacted["items"][0]["data"]
    }
    return redacted


def _apply_deployment(
    manifest: dict[str, Any], name: str, namespace: str, kubeconfig: str
) -> None:
    _kubectl(["apply", "-f", "-"], payload=json.dumps(manifest), kubeconfig=kubeconfig)
    _kubectl(
        ["rollout", "status", f"deployment/{name}", "-n", namespace],
        kubeconfig=kubeconfig,
    )


def _deployment_result(
    project: str,
    image: str,
    name: str,
    namespace: str,
    kubeconfig: str,
    cache_pvc: str,
    gpu_type: str,
    selector_key: str,
    selector_value: str,
    token_env: str,
    dry_run: bool,
) -> dict[str, Any]:
    resolved_selector = _node_selector(gpu_type, selector_value)
    resolved_image = image or container_image_for_tool("openarm")
    manifest = _manifest(
        project,
        resolved_image,
        name,
        namespace,
        cache_pvc,
        token_env,
        selector_key,
        resolved_selector,
    )
    if dry_run:
        return _redacted_manifest(manifest)
    _apply_deployment(manifest, name, namespace, kubeconfig)
    return {
        "status": "deployed",
        "name": name,
        "namespace": namespace,
        "image": resolved_image,
        "node_selector": {selector_key: resolved_selector},
    }


@app.command("deploy")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def deploy_cmd(
    project: str = typer.Option("", "--project", "-p"),
    image: str = typer.Option("", "--image"),
    name: str = typer.Option("npa-openarm", "--name"),
    namespace: str = typer.Option("workbench", "--namespace"),
    kubeconfig: str = typer.Option("", "--kubeconfig"),
    isaac_cache_pvc: str = typer.Option("npa-openarm-isaac-cache", "--isaac-cache-pvc"),
    gpu_type: GpuTypeOption = "rtxpro6000",
    node_selector_key: NodeSelectorKeyOption = "node.kubernetes.io/instance-type",
    node_selector_value: str = typer.Option("", "--node-selector-value"),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format"),
) -> None:
    """Ensure an authenticated single-RTX OpenArm service is deployed."""
    payload = _deployment_result(
        project,
        image,
        name,
        namespace,
        kubeconfig,
        isaac_cache_pvc,
        gpu_type,
        node_selector_key,
        node_selector_value,
        token_env,
        dry_run,
    )
    _emit(
        payload,
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
