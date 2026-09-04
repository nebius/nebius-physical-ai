"""Deploy the RoboCasa service to Kubernetes."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import typer

from npa.clients.config import resolve_container_registry
from npa.clients.credentials import apply_shared_credential_env, load_credentials
from npa.clients.project_credentials import storage_env_for_project
from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY, container_image_for_tool
from npa.workbench.robocasa.schemas import DEFAULT_PORT, DEFAULT_TOKEN_ENV

from npa.cli.workbench.robocasa.helpers import OutputFormat, emit, fail

DEFAULT_IMAGE = container_image_for_tool("robocasa", registry=DEFAULT_CONTAINER_REGISTRY)
DEFAULT_NAME = "npa-robocasa"
DEFAULT_NAMESPACE = "default"

GPU_NODE_SELECTORS = {
    "h100": "gpu-h100-sxm",
    "l40s": "gpu-l40s-d",
    "rtx6000": "gpu-rtx6000",
    "rtxpro6000": "gpu-rtx6000",
}


def deploy_cmd(
    project: str = typer.Option("", "--project", "-p", help="Project alias used to resolve container_registry."),
    cluster_name: str = typer.Option(
        "",
        "--cluster-name",
        help="NPA cluster profile whose cached kubeconfig to use. Empty (the default) uses the ambient kubeconfig.",
    ),
    kubeconfig: str = typer.Option("", "--kubeconfig", help="Kubeconfig path override."),
    image: str = typer.Option("", "--image", help=f"Container image to deploy. Defaults to {DEFAULT_IMAGE}."),
    name: str = typer.Option(DEFAULT_NAME, "--name", help="Kubernetes deployment/service name."),
    namespace: str = typer.Option(DEFAULT_NAMESPACE, "--namespace", help="Kubernetes namespace."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Service port."),
    output_path: str = typer.Option("", "--output-path", help="Default S3 output URI."),
    gpu_type: str = typer.Option("rtxpro6000", "--gpu-type", help="GPU type: h100, l40s, rtx6000, or rtxpro6000."),
    node_selector_key: str = typer.Option("node.kubernetes.io/instance-type", "--node-selector-key", help="GPU node selector label key."),
    node_selector_value: str = typer.Option("", "--node-selector-value", help="GPU node selector label value override."),
    image_pull_secret: str = typer.Option("", "--image-pull-secret", help="Existing operator-managed Kubernetes imagePullSecret for a private registry."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    auth_mode: str = typer.Option("token", "--auth-mode", help="Auth mode: none or token. Defaults to token (secure)."),
    insecure_no_auth: bool = typer.Option(
        False,
        "--insecure-no-auth",
        help="Explicitly deploy without token auth (overrides --auth-mode to none). Not recommended.",
    ),
    destroy: bool = typer.Option(False, "--destroy", help="Delete the Kubernetes service, deployment, and secret."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print Kubernetes manifest without applying it."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Deploy the RoboCasa service to an NPA Workbench Kubernetes cluster."""
    if port < 1024 or port > 65535:
        fail("--port must be between 1024 and 65535")
    if insecure_no_auth:
        auth_mode = "none"
    if auth_mode not in {"none", "token"}:
        fail("--auth-mode must be none or token")
    resolved_kubeconfig = _resolve_kubeconfig(cluster_name=cluster_name, kubeconfig=kubeconfig)
    if destroy:
        _kubectl(["delete", "service", name, "-n", namespace, "--ignore-not-found=true"], dry_run=dry_run, kubeconfig=resolved_kubeconfig)
        _kubectl(["delete", "deployment", name, "-n", namespace, "--ignore-not-found=true"], dry_run=dry_run, kubeconfig=resolved_kubeconfig)
        _kubectl(["delete", "secret", f"{name}-env", "-n", namespace, "--ignore-not-found=true"], dry_run=dry_run, kubeconfig=resolved_kubeconfig)

        emit({"status": "deleted", "name": name, "namespace": namespace}, output=output)
        return

    selector_value = node_selector_value.strip() or GPU_NODE_SELECTORS.get(gpu_type.strip().lower())
    if not selector_value:
        fail(
            "--gpu-type must be one of "
            + ", ".join(sorted(GPU_NODE_SELECTORS))
            + " unless --node-selector-value is provided"
        )
    resolved_image = image.strip() or container_image_for_tool(
        "robocasa",
        registry=resolve_container_registry(project or None),
    )
    manifest = _kubernetes_manifest(
        project=project,
        image=resolved_image,
        name=name,
        namespace=namespace,
        port=port,
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
            "Warning: --auth-mode none deploys robocasa without token auth. The service "
            "drives GPU simulation and carries S3 credentials, and any pod in the cluster can reach it. "
            "Use --auth-mode token with ROBOCASA_TOKEN set.",
            err=True,
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
        text=f"RoboCasa service deployed: {endpoint}",
    )


def _kubernetes_manifest(
    *,
    project: str,
    image: str,
    name: str,
    namespace: str,
    port: int,
    output_path: str,
    node_selector_key: str,
    node_selector_value: str,
    image_pull_secret: str,
    auth_mode: str,
    token_env: str,
) -> dict[str, Any]:
    env = _service_env(
        project=project,
        output_path=output_path,
        auth_mode=auth_mode,
        token_env=token_env,
        port=port,
    )
    env_checksum = hashlib.sha256(
        json.dumps(env, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
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
                    "labels": {"app.kubernetes.io/name": "npa-robocasa", "app.kubernetes.io/instance": name},
                },
                "spec": {
                    "replicas": 1,
                    "strategy": {"type": "Recreate"},
                    "selector": {"matchLabels": {"app.kubernetes.io/instance": name}},
                    "template": {
                        "metadata": {
                            "labels": {
                                "app.kubernetes.io/name": "npa-robocasa",
                                "app.kubernetes.io/instance": name,
                            },
                            "annotations": {
                                "npa.nebius.ai/env-checksum": env_checksum,
                            },
                        },
                        "spec": {
                            "nodeSelector": {node_selector_key: node_selector_value},
                            **({"imagePullSecrets": [{"name": image_pull_secret}]} if image_pull_secret else {}),
                            "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                            "securityContext": {"fsGroup": 1000, "fsGroupChangePolicy": "OnRootMismatch"},
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
                    "ports": [{"name": "http", "port": port, "targetPort": "http"}],
                },
            },
        ],
    }


def _service_env(
    *,
    project: str,
    output_path: str,
    auth_mode: str,
    token_env: str,
    port: int,
) -> dict[str, str]:
    creds = load_credentials()
    env = {
        "ROBOCASA_AUTH_MODE": auth_mode,
        "ROBOCASA_PORT": str(port),
        "NPA_OUTPUT_PATH": output_path,
        "AWS_REGION": os.environ.get("AWS_REGION", "auto"),
        "NUMBA_CACHE_DIR": "/tmp/numba_cache",
    }
    apply_shared_credential_env(env, creds)
    if project.strip():
        project_storage = storage_env_for_project(project.strip())
        env.update(project_storage)
        endpoint = project_storage.get("AWS_ENDPOINT_URL", "")
        if endpoint:
            env["AWS_ENDPOINT_URL_S3"] = endpoint
            env["NEBIUS_S3_ENDPOINT"] = endpoint
    if auth_mode == "token":
        token = os.environ.get(token_env, "")
        if not token:
            fail(f"{token_env} is required when --auth-mode token")
        env["ROBOCASA_TOKEN"] = token
    if not project.strip():
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
