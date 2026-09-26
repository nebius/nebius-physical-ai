"""Render and apply a private Kubernetes MJLab service using existing Secrets."""

from __future__ import annotations

import json
import shlex
import subprocess

from pydantic import BaseModel, Field

from npa.workbench.storage_scope import StorageScope

from .schemas import MjlabError


class DeployRequest(BaseModel):
    """Describe a service on an existing GPU Kubernetes cluster.

    Args:
        image: Complete operator-built MJLab image reference.
        token_secret: Existing Secret with an MJLAB_TOKEN key.
        storage_secret: Existing Secret containing the storage environment.
        allowed_s3_roots: Comma-separated authorized S3 prefixes.
        accelerator: Exact nvidia.com/gpu.product node label.
        name: Kubernetes resource name.
        namespace: Kubernetes namespace, default workbench.
        gpu_count: Visible GPUs on one node.
    Returns:
        Validated deployment settings.
    Raises:
        ValueError: Invalid configuration.
    """

    image: str = Field(min_length=1)
    token_secret: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    storage_secret: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    allowed_s3_roots: str
    accelerator: str = Field(min_length=1)
    name: str = Field(default="mjlab", pattern=r"^[a-z0-9][a-z0-9-]*$")
    namespace: str = Field(default="workbench", pattern=r"^[a-z0-9][a-z0-9-]*$")
    gpu_count: int = Field(default=1, gt=0)


def deploy(
    request: DeployRequest, *, kubeconfig: str = "", dry_run: bool = False
) -> dict:
    """Deploy the service without provisioning nodes or embedding credentials.

    Args:
        request: Existing cluster, Secret references and image configuration.
        kubeconfig: Optional kubeconfig file.
        dry_run: Return the concrete manifest without applying it.
    Returns:
        Manifest and deployment status.
    Raises:
        MjlabError: kubectl rejected the deployment.
        ValueError: No authorized S3 roots were supplied.
    """
    manifest = _manifest(request)
    command = ["kubectl"] + (["--kubeconfig", kubeconfig] if kubeconfig else [])
    if not dry_run:
        result = subprocess.run(
            command + ["apply", "-f", "-"],
            input=json.dumps(manifest),
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise MjlabError(
                "kubectl apply failed; verify cluster access and namespace permissions"
            )
    return {
        "status": "planned" if dry_run else "applied",
        "manifest": manifest,
        "port_forward": shlex.join(
            command
            + [
                "-n",
                request.namespace,
                "port-forward",
                f"service/{request.name}",
                "8080:8080",
            ]
        ),
    }


def _manifest(request):
    roots = [
        root.strip() for root in request.allowed_s3_roots.split(",") if root.strip()
    ]
    if not StorageScope.from_config(s3_roots=roots).s3_roots:
        raise ValueError("At least one allowed S3 root is required")
    labels = {
        "app.kubernetes.io/name": request.name,
        "app.kubernetes.io/managed-by": "npa",
    }
    metadata = {"name": request.name, "namespace": request.namespace, "labels": labels}
    pod = {
        "automountServiceAccountToken": False,
        "nodeSelector": {"nvidia.com/gpu.product": request.accelerator},
        "containers": [_container(request, roots)],
    }
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": metadata,
        "spec": {
            "replicas": 1,
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": labels},
            "template": {"metadata": {"labels": labels}, "spec": pod},
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": metadata,
        "spec": {
            "type": "ClusterIP",
            "selector": labels,
            "ports": [{"port": 8080, "targetPort": 8080}],
        },
    }
    return {"apiVersion": "v1", "kind": "List", "items": [deployment, service]}


def _container(request, roots):
    return {
        "name": "mjlab",
        "image": request.image,
        "ports": [{"containerPort": 8080}],
        "envFrom": [{"secretRef": {"name": request.storage_secret}}],
        "env": [
            {
                "name": "MJLAB_TOKEN",
                "valueFrom": {
                    "secretKeyRef": {"name": request.token_secret, "key": "MJLAB_TOKEN"}
                },
            },
            {"name": "MJLAB_ALLOWED_S3_ROOTS", "value": ",".join(roots)},
        ],
        "resources": {"limits": {"nvidia.com/gpu": request.gpu_count}},
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "allowPrivilegeEscalation": False,
        },
        "readinessProbe": {"httpGet": {"path": "/health", "port": 8080}},
    }
