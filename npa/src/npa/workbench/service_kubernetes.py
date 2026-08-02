"""Deploy a workbench service into Kubernetes, where workflow stages can reach it.

Workbench services could be put on a local docker daemon, on a managed VM, or registered as a
hosted endpoint. None of those helps a workflow stage: a stage runs in a pod, and a pod cannot
reach ``http://localhost:8686`` on an operator's laptop. That gap is why templates could not
retire — `dataset-ingest-curate` writes to LanceDB and `bdd100k-pipeline` calls
detection-training, and both failed live with ``[Errno -2] Name or service not known``
(EVIDENCE §R16, §R41).

A Deployment plus a ClusterIP Service gives the cluster a stable DNS name, which is what a spec
can put in its config and what every pod in the namespace can resolve. Written once and shared,
because the second service needed exactly the same five fixes as the first.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

#: Everything this module creates carries it, so `--destroy` can find its own objects and only
#: its own objects.
MANAGED_BY_LABEL = {"app.kubernetes.io/managed-by": "npa", "app.kubernetes.io/part-of": "npa-workbench"}

DEFAULT_NAMESPACE = "default"

#: Secret env names copied from the operator's environment into the pod. LanceDB needs them to
#: read and write an s3:// storage path.
STORAGE_SECRET_ENVS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


class ServiceKubernetesError(RuntimeError):
    """Raised when an in-cluster workbench service cannot be deployed or inspected."""


@dataclass(frozen=True)
class KubernetesDeployment:
    """What a deploy produced, and how to reach it."""

    name: str
    namespace: str
    port: int
    image: str
    endpoint: str
    storage_path: str
    status: str
    manifests: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime": "kubernetes",
            "name": self.name,
            "namespace": self.namespace,
            "port": self.port,
            "image": self.image,
            "endpoint": self.endpoint,
            "storage_path": self.storage_path,
            "status": self.status,
        }


def service_endpoint(name: str, namespace: str, port: int) -> str:
    """The in-cluster DNS name a pod in any namespace can resolve."""

    return f"http://{name}.{namespace}.svc.cluster.local:{port}"


def build_manifests(
    *,
    name: str,
    namespace: str = DEFAULT_NAMESPACE,
    port: int,
    image: str,
    service_env: dict[str, str],
    storage_path: str = "",
    storage_endpoint_url: str = "",
    secret_name: str = "",
    image_pull_secrets: tuple[str, ...] = (),
    replicas: int = 1,
) -> list[dict[str, Any]]:
    """Return the Deployment and Service documents, as data rather than a template string.

    Data because it is testable: a rendered YAML string can only be checked by parsing it back,
    and the interesting assertions here are about probe wiring and where credentials come from.
    """

    if not image.strip():
        raise ServiceKubernetesError("an image reference is required")


    labels = {"app": name, **MANAGED_BY_LABEL}
    env: list[dict[str, Any]] = [
        {"name": key, "value": value} for key, value in sorted(service_env.items())
    ]
    if storage_endpoint_url:
        env.append({"name": "AWS_ENDPOINT_URL", "value": storage_endpoint_url})
        env.append({"name": "NEBIUS_S3_ENDPOINT", "value": storage_endpoint_url})
    if secret_name:
        # From a Secret, never from the manifest: a deploy should not leave keys in
        # `kubectl get deploy -o yaml` for everyone with read access on the namespace.
        for key in STORAGE_SECRET_ENVS:
            env.append(
                {
                    "name": key,
                    "valueFrom": {"secretKeyRef": {"name": secret_name, "key": key}},
                }
            )

    container: dict[str, Any] = {
        "name": name,
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "ports": [{"containerPort": port, "name": "http"}],
        "env": env,
        # Readiness gates the Service's endpoints, so a stage never resolves the DNS name to a
        # pod that is still opening its storage.
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": port},
            "initialDelaySeconds": 5,
            "periodSeconds": 5,
            "failureThreshold": 12,
        },
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": port},
            "initialDelaySeconds": 30,
            "periodSeconds": 30,
            "failureThreshold": 6,
        },
    }

    pod_spec: dict[str, Any] = {"containers": [container]}
    if image_pull_secrets:
        pod_spec["imagePullSecrets"] = [{"name": ref} for ref in image_pull_secrets]

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": name}},
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "type": "ClusterIP",
            "selector": {"app": name},
            "ports": [{"name": "http", "port": port, "targetPort": port}],
        },
    }
    return [deployment, service]


#: Name of the pull secret this module maintains when the image lives in a private registry.
MANAGED_PULL_SECRET = "npa-registry"


def ensure_registry_secret(
    secret_name: str,
    namespace: str,
    registry: str,
    *,
    runner: Any = None,
) -> None:
    """Create or refresh the image-pull secret from a freshly minted registry token.

    A long-lived Deployment cannot borrow SkyPilot's trick of passing credentials per submit:
    the kubelet pulls whenever it restarts a pod, using whatever the namespace holds. Reusing a
    shared secret means the deploy silently depends on somebody else's refresh cron — live, the
    first attempt sat in ImagePullBackOff with `401 Unauthorized` against a tag that exists.
    """

    from npa.workflows.sim2real.registry_auth import mint_nebius_registry_token

    try:
        token = mint_nebius_registry_token()
    except Exception as exc:  # pragma: no cover - depends on the operator's IAM setup
        raise ServiceKubernetesError(
            f"could not mint a registry token for {registry}: {exc}"
        ) from exc

    run = runner or _kubectl
    built = run(
        [
            "create",
            "secret",
            "docker-registry",
            secret_name,
            f"--namespace={namespace}",
            f"--docker-server={registry}",
            "--docker-username=iam",
            f"--docker-password={token}",
            "--dry-run=client",
            "-o",
            "json",
        ]
    )
    if built.returncode != 0:
        raise ServiceKubernetesError(
            f"could not build the registry secret: {(built.stderr or built.stdout).strip()}"
        )
    applied = run(["apply", "-f", "-"], stdin=built.stdout)
    if applied.returncode != 0:
        raise ServiceKubernetesError(
            f"could not apply the registry secret: {(applied.stderr or applied.stdout).strip()}"
        )


def registry_host(image: str) -> str:
    """Return the registry host of an image reference, or "" for a bare/Docker Hub name."""

    if "/" not in image:
        # `npa-lancedb:1` is a Docker Hub name whose tag contains the only colon; without this
        # the tag would be mistaken for a registry port.
        return ""
    head = image.split("/", 1)[0]
    return head if ("." in head or ":" in head) else ""


def _kubectl(args: list[str], *, stdin: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    binary = os.environ.get("NPA_KUBECTL_BIN") or "kubectl"
    return subprocess.run(
        [binary, *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def apply(
    manifests: list[dict[str, Any]],
    *,
    runner: Any = None,
) -> str:
    """Apply the manifests, returning kubectl's output."""

    run = runner or _kubectl
    payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": manifests})
    result = run(["apply", "-f", "-"], stdin=payload)
    if result.returncode != 0:
        raise ServiceKubernetesError(
            f"kubectl apply failed ({result.returncode}): {(result.stderr or result.stdout).strip()}"
        )
    return (result.stdout or "").strip()


def wait_available(
    name: str,
    namespace: str,
    *,
    timeout_seconds: int = 300,
    runner: Any = None,
) -> None:
    """Block until THIS rollout's pods are serving, or say why they are not.

    `kubectl wait --for=condition=Available` is the obvious choice and the wrong one: during a
    rolling update it is satisfied by the OLD ReplicaSet's healthy pod. A deploy that shipped a
    broken image reported "running" while the new pod sat in CrashLoopBackOff, and the next live
    run failed against the old code with no sign anything was wrong (EVIDENCE.md §R41).
    `rollout status` waits for the new ReplicaSet specifically.
    """

    run = runner or _kubectl
    result = run(
        [
            "rollout",
            "status",
            f"--namespace={namespace}",
            f"--timeout={timeout_seconds}s",
            f"deployment/{name}",
        ],
        timeout=timeout_seconds + 30,
    )
    if result.returncode != 0:
        raise ServiceKubernetesError(
            f"LanceDB deployment {name} did not roll out within {timeout_seconds}s: "
            f"{(result.stderr or result.stdout).strip()}"
        )


def destroy(name: str, namespace: str, *, runner: Any = None) -> str:
    """Remove the Deployment and Service this module created."""

    run = runner or _kubectl
    result = run(
        [
            "delete",
            f"--namespace={namespace}",
            "--ignore-not-found",
            "deployment,service",
            "-l",
            f"app={name},app.kubernetes.io/managed-by=npa",
        ]
    )
    if result.returncode != 0:
        raise ServiceKubernetesError(
            f"kubectl delete failed ({result.returncode}): {(result.stderr or result.stdout).strip()}"
        )
    return (result.stdout or "").strip()


def ensure_storage_secret(
    secret_name: str,
    namespace: str,
    credentials: dict[str, str],
    *,
    runner: Any = None,
) -> None:
    """Create or update the Secret holding the object-store keys."""

    missing = [key for key in STORAGE_SECRET_ENVS if not credentials.get(key)]
    if missing:
        raise ServiceKubernetesError(
            "LanceDB needs object-store credentials to serve an s3:// storage path; missing "
            + ", ".join(missing)
        )
    run = runner or _kubectl
    literals = [f"--from-literal={key}={credentials[key]}" for key in STORAGE_SECRET_ENVS]
    result = run(
        [
            "create",
            "secret",
            "generic",
            secret_name,
            f"--namespace={namespace}",
            *literals,
            "--dry-run=client",
            "-o",
            "json",
        ]
    )
    if result.returncode != 0:
        raise ServiceKubernetesError(
            f"could not build the LanceDB storage secret: {(result.stderr or result.stdout).strip()}"
        )
    applied = run(["apply", "-f", "-"], stdin=result.stdout)
    if applied.returncode != 0:
        raise ServiceKubernetesError(
            f"could not apply the LanceDB storage secret: "
            f"{(applied.stderr or applied.stdout).strip()}"
        )
