"""Typer CLI for `npa workbench detection-training`."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import typer

from npa.clients.config import resolve_container_registry
from npa.clients.credentials import load_credentials
from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY, container_image_for_tool
from npa.workbench.detection_training.schemas import (
    DEFAULT_LANCE_URI,
    DEFAULT_PORT,
    DEFAULT_TOKEN_ENV,
    EvalRequest,
    TrainRequest,
)

app = typer.Typer(
    name="detection-training",
    help="Train Faster R-CNN detectors from LanceDB materialized views.",
    no_args_is_help=True,
)

DEFAULT_IMAGE = container_image_for_tool("detection-training", registry=DEFAULT_CONTAINER_REGISTRY)
DEFAULT_NAME = "npa-detection-training"
DEFAULT_NAMESPACE = "default"
GPU_NODE_SELECTORS = {
    "h100": "gpu-h100-sxm",
    "l40s": "gpu-l40s-d",
}


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def emit(payload: dict[str, Any], *, output: OutputFormat, text: str | None = None) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text if text is not None else "\n".join(f"{key}: {value}" for key, value in payload.items()))


def deploy_cmd(
    project: str = typer.Option("", "--project", "-p", help="Project alias used to resolve container_registry."),
    cluster_name: str = typer.Option("npa-workbench-eu-north1", "--cluster-name", help="NPA cluster profile name for cached kubeconfig."),
    kubeconfig: str = typer.Option("", "--kubeconfig", help="Kubeconfig path override."),
    image: str = typer.Option("", "--image", help=f"Container image to deploy. Defaults to {DEFAULT_IMAGE}."),
    name: str = typer.Option(DEFAULT_NAME, "--name", help="Kubernetes deployment/service name."),
    namespace: str = typer.Option(DEFAULT_NAMESPACE, "--namespace", help="Kubernetes namespace."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Service port."),
    input_path: str = typer.Option(DEFAULT_LANCE_URI, "--input-path", help="Default LanceDB input URI."),
    output_path: str = typer.Option("", "--output-path", help="Default S3 output URI."),
    gpu_type: str = typer.Option("h100", "--gpu-type", help="GPU type: h100 or l40s."),
    node_selector_key: str = typer.Option("node.kubernetes.io/instance-type", "--node-selector-key", help="GPU node selector label key."),
    node_selector_value: str = typer.Option("", "--node-selector-value", help="GPU node selector label value override."),
    image_pull_secret: str = typer.Option("npa-nebius-registry", "--image-pull-secret", help="Kubernetes imagePullSecret name for private registries."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    auth_mode: str = typer.Option("none", "--auth-mode", help="Auth mode: none or token."),
    destroy: bool = typer.Option(False, "--destroy", help="Delete the Kubernetes service, deployment, and secret."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print Kubernetes manifest without applying it."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Deploy the detection-training service to an NPA Workbench Kubernetes cluster."""
    if port < 1024 or port > 65535:
        fail("--port must be between 1024 and 65535")
    if auth_mode not in {"none", "token"}:
        fail("--auth-mode must be none or token")
    if not output_path and not destroy:
        fail("--output-path is required")
    resolved_kubeconfig = _resolve_kubeconfig(cluster_name=cluster_name, kubeconfig=kubeconfig)
    if destroy:
        _kubectl(["delete", "service", name, "-n", namespace, "--ignore-not-found=true"], dry_run=dry_run, kubeconfig=resolved_kubeconfig)
        _kubectl(["delete", "deployment", name, "-n", namespace, "--ignore-not-found=true"], dry_run=dry_run, kubeconfig=resolved_kubeconfig)
        _kubectl(["delete", "secret", f"{name}-env", "-n", namespace, "--ignore-not-found=true"], dry_run=dry_run, kubeconfig=resolved_kubeconfig)
        emit({"status": "deleted", "name": name, "namespace": namespace}, output=output)
        return

    selector_value = node_selector_value.strip() or GPU_NODE_SELECTORS.get(gpu_type.strip().lower())
    if not selector_value:
        fail("--gpu-type must be h100 or l40s unless --node-selector-value is provided")
    resolved_image = image.strip() or container_image_for_tool(
        "detection-training",
        registry=resolve_container_registry(project or None),
    )
    manifest = _kubernetes_manifest(
        image=resolved_image,
        name=name,
        namespace=namespace,
        port=port,
        input_path=input_path,
        output_path=output_path,
        node_selector_key=node_selector_key,
        node_selector_value=selector_value,
        image_pull_secret=image_pull_secret,
        auth_mode=auth_mode,
        token_env=token_env,
    )
    if dry_run:
        typer.echo(json.dumps(_redact_manifest(manifest), indent=2, sort_keys=True))
        return
    if auth_mode == "none":
        typer.echo(
            "Warning: --auth-mode none deploys detection-training without token auth. The service "
            "drives GPU training and carries S3 credentials, and any pod in the cluster can reach it. "
            "Use --auth-mode token with DETECTION_TRAINING_TOKEN set.",
            err=True,
        )
    if image_pull_secret:
        _ensure_image_pull_secret(
            image=resolved_image,
            secret_name=image_pull_secret,
            namespace=namespace,
            kubeconfig=resolved_kubeconfig,
        )
    _kubectl(["apply", "-f", "-"], stdin=json.dumps(manifest), kubeconfig=resolved_kubeconfig)
    _kubectl(["rollout", "status", f"deployment/{name}", "-n", namespace, "--timeout=900s"], kubeconfig=resolved_kubeconfig)
    endpoint = f"http://{name}.{namespace}.svc.cluster.local:{port}"
    emit(
        {
            "status": "deployed",
            "name": name,
            "namespace": namespace,
            "image": resolved_image,
            "endpoint": endpoint,
            "node_selector": {node_selector_key: selector_value},
        },
        output=output,
        text=f"Detection-training service deployed: {endpoint}",
    )


def train_cmd(
    view: str = typer.Option(..., "--view", help="Lance materialized view name."),
    output_uri: str = typer.Option(..., "--output-uri", "--output-path", help="S3/local output URI."),
    lance_uri: str = typer.Option(DEFAULT_LANCE_URI, "--lance-uri", "--input-path", help="LanceDB URI."),
    num_classes: int = typer.Option(10, "--num-classes", help="Detector class count."),
    epochs: int = typer.Option(10, "--epochs", help="Training epochs."),
    batch_size: int = typer.Option(8, "--batch-size", help="Training batch size."),
    learning_rate: float = typer.Option(0.005, "--learning-rate", help="SGD learning rate."),
    validation_filter_sql: str = typer.Option("", "--validation-filter-sql", help="Optional validation filter SQL."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Detection-training service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Start a detection-training run."""
    payload = TrainRequest(
        view=view,
        lance_uri=lance_uri,
        output_uri=output_uri,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_filter_sql=validation_filter_sql or None,
    ).model_dump(mode="json")
    if service:
        result = request_json("POST", resolve_endpoint(endpoint), "/train", payload=payload, token_env=token_env, timeout=60.0)
    else:
        from npa.sdk.workbench.detection_training import train

        result = train(**payload).model_dump(mode="json")
    emit(result, output=output, text=f"run_id: {result.get('run_id')}\nstatus: {result.get('status')}")


def eval_cmd(
    checkpoint_uri: str = typer.Option(..., "--checkpoint-uri", help="Checkpoint S3/local URI."),
    eval_view: str = typer.Option(..., "--eval-view", help="Lance materialized view to evaluate."),
    output_uri: str = typer.Option(..., "--output-uri", "--output-path", help="S3/local output URI."),
    lance_uri: str = typer.Option(DEFAULT_LANCE_URI, "--lance-uri", "--input-path", help="LanceDB URI."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Detection-training service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Evaluate a detection-training checkpoint."""
    payload = EvalRequest(
        checkpoint_uri=checkpoint_uri,
        eval_view=eval_view,
        lance_uri=lance_uri,
        output_uri=output_uri,
    ).model_dump(mode="json")
    if service:
        result = request_json("POST", resolve_endpoint(endpoint), "/eval", payload=payload, token_env=token_env, timeout=900.0)
    else:
        from npa.sdk.workbench.detection_training import eval as sdk_eval

        result = sdk_eval(**payload).model_dump(mode="json")
    emit(result, output=output, text=f"mAP: {result.get('mAP')}\neval_run_id: {result.get('eval_run_id')}")


def status_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Training run ID."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Detection-training service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Fetch training run status."""
    if service:
        result = request_json(
            "GET",
            resolve_endpoint(endpoint),
            "/status",
            params={"run_id": run_id},
            token_env=token_env,
            timeout=30.0,
        )
    else:
        from npa.sdk.workbench.detection_training import status

        result = status(run_id=run_id).model_dump(mode="json")
    emit(result, output=output, text=f"status: {result.get('status')}\nepochs_completed: {result.get('epochs_completed')}")


def system_info_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Detection-training service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show detection-training runtime information."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/system-info", token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.detection_training.service import system_info_payload

        result = system_info_payload()
    emit(result, output=output)


def list_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Detection-training service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    cluster_name: str = typer.Option("npa-workbench-eu-north1", "--cluster-name", help="NPA cluster profile name for cached kubeconfig."),
    kubeconfig: str = typer.Option("", "--kubeconfig", help="Kubeconfig path override."),
    namespace: str = typer.Option(DEFAULT_NAMESPACE, "--namespace", help="Kubernetes namespace for local listing."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List service-managed runs or Kubernetes resources."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/runs", token_env=token_env, timeout=30.0)
        emit(result, output=output, text="\n".join(run["run_id"] for run in result.get("runs", [])) or "No runs found.")
        return
    stdout = _kubectl(
        [
            "get",
            "deploy,svc",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/name=npa-detection-training",
            "-o",
            "json",
        ],
        capture=True,
        kubeconfig=_resolve_kubeconfig(cluster_name=cluster_name, kubeconfig=kubeconfig),
    )
    data = json.loads(stdout or "{}")
    names = [item.get("metadata", {}).get("name", "") for item in data.get("items", [])]
    result = {"namespace": namespace, "resources": names, "count": len(names)}
    emit(result, output=output, text="\n".join(names) or "No detection-training resources found.")


def resolve_endpoint(endpoint: str) -> str:
    resolved = endpoint.strip() or os.environ.get("NPA_DETECTION_TRAINING_ENDPOINT", "")
    if not resolved:
        fail("--endpoint is required")
    if not resolved.startswith(("http://", "https://")):
        fail("--endpoint must be an http:// or https:// URL")
    return resolved.rstrip("/")


def request_json(
    method: str,
    endpoint: str,
    path: str,
    *,
    token_env: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(
            method,
            f"{endpoint}{path}",
            headers=headers,
            json=payload,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        fail(f"Detection-training request failed ({exc.response.status_code}): {exc.response.text.strip()}")
    except httpx.HTTPError as exc:
        fail(f"Cannot reach detection-training endpoint {endpoint}: {exc}")
    try:
        data = response.json()
    except ValueError:
        fail("Detection-training endpoint returned non-JSON response")
    if not isinstance(data, dict):
        fail("Detection-training endpoint returned an unexpected response")
    return data


def _kubernetes_manifest(
    *,
    image: str,
    name: str,
    namespace: str,
    port: int,
    input_path: str,
    output_path: str,
    node_selector_key: str,
    node_selector_value: str,
    image_pull_secret: str,
    auth_mode: str,
    token_env: str,
) -> dict[str, Any]:
    env = _service_env(input_path=input_path, output_path=output_path, auth_mode=auth_mode, token_env=token_env, port=port)
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": f"{name}-env", "namespace": namespace},
                "type": "Opaque",
                "data": {key: base64.b64encode(value.encode("utf-8")).decode("ascii") for key, value in env.items()},
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": name,
                    "namespace": namespace,
                    "labels": {"app.kubernetes.io/name": "npa-detection-training", "app.kubernetes.io/instance": name},
                },
                "spec": {
                    "replicas": 1,
                    "strategy": {"type": "Recreate"},
                    "selector": {"matchLabels": {"app.kubernetes.io/instance": name}},
                    "template": {
                        "metadata": {"labels": {"app.kubernetes.io/name": "npa-detection-training", "app.kubernetes.io/instance": name}},
                        "spec": {
                            "nodeSelector": {node_selector_key: node_selector_value},
                            **({"imagePullSecrets": [{"name": image_pull_secret}]} if image_pull_secret else {}),
                            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                            "containers": [
                                {
                                    "name": "service",
                                    "image": image,
                                    "imagePullPolicy": "Always",
                                    "ports": [{"containerPort": port, "name": "http"}],
                                    "envFrom": [{"secretRef": {"name": f"{name}-env"}}],
                                    "resources": {
                                        "limits": {"nvidia.com/gpu": "1"},
                                        "requests": {"nvidia.com/gpu": "1"},
                                    },
                                    "readinessProbe": {"httpGet": {"path": "/health", "port": "http"}, "initialDelaySeconds": 10, "periodSeconds": 10},
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
                    "ports": [{"name": "http", "port": port, "targetPort": "http"}],
                },
            },
        ],
    }


def _service_env(*, input_path: str, output_path: str, auth_mode: str, token_env: str, port: int) -> dict[str, str]:
    creds = load_credentials()
    env = {
        "DETECTION_TRAINING_AUTH_MODE": auth_mode,
        "DETECTION_TRAINING_PORT": str(port),
        "NPA_INPUT_PATH": input_path,
        "NPA_OUTPUT_PATH": output_path,
        "AWS_REGION": os.environ.get("AWS_REGION", "auto"),
    }
    if auth_mode == "token":
        token = os.environ.get(token_env, "")
        if not token:
            fail(f"{token_env} is required when --auth-mode token")
        env["DETECTION_TRAINING_TOKEN"] = token
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or creds.s3_access_key_id
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or creds.s3_secret_access_key
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or creds.s3_endpoint
    if access_key:
        env["AWS_ACCESS_KEY_ID"] = access_key
    if secret_key:
        env["AWS_SECRET_ACCESS_KEY"] = secret_key
    if endpoint:
        env["AWS_ENDPOINT_URL"] = endpoint
        env["AWS_ENDPOINT_URL_S3"] = endpoint
        env["NEBIUS_S3_ENDPOINT"] = endpoint
    return {key: value for key, value in env.items() if value}


def _ensure_image_pull_secret(*, image: str, secret_name: str, namespace: str, kubeconfig: str) -> None:
    registry = _image_registry(image)
    if not registry:
        return
    docker_config = _docker_auth_config(registry)
    payload = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "type": "kubernetes.io/dockerconfigjson",
        "data": {
            ".dockerconfigjson": base64.b64encode(json.dumps(docker_config).encode("utf-8")).decode("ascii"),
        },
    }
    _kubectl(["apply", "-f", "-"], stdin=json.dumps(payload), kubeconfig=kubeconfig)


def _image_registry(image: str) -> str:
    if "/" not in image:
        return ""
    first = image.split("/", 1)[0]
    if "." in first or ":" in first or first == "localhost":
        return first
    return ""


def _docker_auth_config(registry: str) -> dict[str, Any]:
    config_path = Path.home() / ".docker" / "config.json"
    if not config_path.exists():
        fail(f"Cannot create image pull secret for {registry}: {config_path} does not exist")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"Cannot parse {config_path}: {exc}")
    entry = _docker_auth_entry(config, registry)
    if not entry:
        helper = _docker_credential_helper(config, registry)
        if not helper:
            fail(f"Cannot find Docker auth or credential helper for {registry}")
        entry = _docker_auth_from_helper(helper, registry)
    return {"auths": {registry: entry}}


def _docker_auth_entry(config: dict[str, Any], registry: str) -> dict[str, str] | None:
    auths = config.get("auths", {})
    if not isinstance(auths, dict):
        return None
    for candidate in (registry, f"https://{registry}", f"http://{registry}"):
        raw = auths.get(candidate)
        if isinstance(raw, dict) and (raw.get("auth") or raw.get("identitytoken")):
            return {key: value for key, value in raw.items() if isinstance(value, str)}
    return None


def _docker_credential_helper(config: dict[str, Any], registry: str) -> str:
    helpers = config.get("credHelpers", {})
    if isinstance(helpers, dict):
        for candidate in (registry, f"https://{registry}", f"http://{registry}"):
            helper = helpers.get(candidate)
            if isinstance(helper, str) and helper:
                return helper
    store = config.get("credsStore")
    return store if isinstance(store, str) else ""


def _docker_auth_from_helper(helper: str, registry: str) -> dict[str, str]:
    executable = f"docker-credential-{helper}"
    if shutil.which(executable) is None:
        fail(f"Docker credential helper {executable} is not installed")
    try:
        result = subprocess.run(
            [executable, "get"],
            input=registry,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        fail(f"Docker credential helper {executable} cannot read credentials for {registry}: {detail}")
    except json.JSONDecodeError as exc:
        fail(f"Docker credential helper {executable} returned invalid JSON: {exc}")
    username = str(payload.get("Username") or payload.get("username") or "")
    secret = str(payload.get("Secret") or payload.get("secret") or "")
    if not username or not secret:
        fail(f"Docker credential helper {executable} returned incomplete credentials for {registry}")
    auth = base64.b64encode(f"{username}:{secret}".encode("utf-8")).decode("ascii")
    return {"username": username, "password": secret, "auth": auth}


def _kubectl(
    args: list[str],
    *,
    stdin: str | None = None,
    dry_run: bool = False,
    capture: bool = False,
    kubeconfig: str = "",
) -> str:
    cmd = ["kubectl"]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    cmd.extend(args)
    if dry_run:
        typer.echo(" ".join(cmd))
        return ""
    try:
        result = subprocess.run(cmd, input=stdin, text=True, capture_output=True, check=True)
    except FileNotFoundError:
        fail("kubectl is not installed or not on PATH")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        fail(f"kubectl command failed: {detail}")
    if not capture and result.stdout.strip():
        typer.echo(result.stdout.strip())
    return result.stdout


def _resolve_kubeconfig(*, cluster_name: str, kubeconfig: str) -> str:
    if kubeconfig.strip():
        return kubeconfig.strip()
    if not cluster_name.strip():
        return ""
    path = Path.home() / ".npa" / "clusters" / cluster_name.strip() / "kubeconfig"
    return str(path) if path.exists() else ""


def _redact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(manifest))
    for item in redacted.get("items", []):
        if item.get("kind") == "Secret":
            item["data"] = {key: "<redacted>" for key in item.get("data", {})}
    return redacted


app.command("deploy")(deploy_cmd)
app.command("train")(train_cmd)
app.command("eval")(eval_cmd)
app.command("status")(status_cmd)
app.command("system-info")(system_info_cmd)
app.command("list")(list_cmd)
