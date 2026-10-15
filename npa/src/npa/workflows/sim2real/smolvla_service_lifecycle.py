"""Owned start/cleanup lifecycle for a private typed SmolVLA service."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.workflow_io import (
    POLICY_INTERCHANGE_COMPONENT_SCHEMA,
    publish_component_record,
    write_json,
)


IMAGE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
HF_SOURCE_RE = re.compile(r"^hf://(?P<repo>[^@\s]+)@(?P<revision>[0-9a-f]{40})$")
PORT = 8080


class SmolVLAServiceError(RuntimeError):
    """Raised when lifecycle ownership, readiness, or checkpoint binding fails."""


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    if not result:
        raise SmolVLAServiceError("run id must contain a DNS-safe character")
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{result[:31]}-{digest}"


def resource_names(run_id: str, role: str) -> dict[str, str]:
    if role not in {"base", "adapted"}:
        raise SmolVLAServiceError("checkpoint role must be base or adapted")
    stem = f"npa-smolvla-{_slug(run_id)}-{role}"
    return {
        "deployment": stem,
        "service": stem,
        "credentials": f"{stem}-storage",
    }


def _labels(run_id: str, role: str, deployment: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/managed-by": "npa",
        "app.kubernetes.io/part-of": "sim2real-policy-interchange",
        "app": deployment,
        "npa.nebius.ai/cleanup-owner": deployment,
        "npa.nebius.ai/run-hash": hashlib.sha256(run_id.encode()).hexdigest()[:16],
        "npa.nebius.ai/checkpoint-role": role,
    }


def _download_program(checkpoint_uri: str, target: str) -> str:
    if checkpoint_uri.startswith("s3://"):
        from urllib.parse import urlparse

        parsed = urlparse(checkpoint_uri)
        if not parsed.netloc or not parsed.path.strip("/"):
            raise SmolVLAServiceError("private checkpoint S3 prefix is invalid")
        return f"""import boto3,pathlib,os
bucket={parsed.netloc!r}; prefix={parsed.path.lstrip("/")!r}.rstrip('/')+'/'
root=pathlib.Path({target!r}); root.mkdir(parents=True,exist_ok=False)
client=boto3.client('s3',endpoint_url=os.environ['AWS_ENDPOINT_URL'])
items=[item for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket,Prefix=prefix) for item in page.get('Contents',[])]
assert items,'checkpoint prefix is empty'
for item in items:
 relative=pathlib.PurePosixPath(item['Key'][len(prefix):])
 assert relative.parts and '..' not in relative.parts and not relative.is_absolute(),'unsafe checkpoint key'
 path=root.joinpath(*relative.parts); path.parent.mkdir(parents=True,exist_ok=True)
 client.download_file(bucket,item['Key'],str(path))
"""
    match = HF_SOURCE_RE.fullmatch(checkpoint_uri)
    if not match:
        raise SmolVLAServiceError(
            "checkpoint must be a private s3:// prefix or hf://repo@40-hex-revision"
        )
    return f"""import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id={match.group("repo")!r},revision={match.group("revision")!r},local_dir={target!r},token=os.environ.get('HF_TOKEN') or None)
"""


def _adapter_download_program(adapter_uri: str, target: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(adapter_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise SmolVLAServiceError("adapter URI must name one s3:// object")
    return f"""import boto3,os,pathlib
target=pathlib.Path({target!r})
client=boto3.client('s3',endpoint_url=os.environ['AWS_ENDPOINT_URL'])
client.download_file({parsed.netloc!r},{parsed.path.lstrip("/")!r},str(target))
"""


def build_manifests(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if not IMAGE_RE.fullmatch(args.runtime_image):
        raise SmolVLAServiceError("SmolVLA runtime image must be digest-pinned")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise SmolVLAServiceError("baked source commit must be 40 lowercase hex")
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", args.auth_secret):
        raise SmolVLAServiceError("auth secret must be a Kubernetes DNS label")
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?", args.policy_cache_pvc):
        raise SmolVLAServiceError(
            "policy cache PVC must be an explicit Kubernetes DNS subdomain"
        )
    names = resource_names(args.run_id, args.checkpoint_role)
    labels = _labels(args.run_id, args.checkpoint_role, names["deployment"])
    selector = {"app": names["deployment"]}
    python = "/opt/lerobot/venv/bin/python"
    policy_root = f"/workspace/{names['deployment']}"
    policy_path = f"{policy_root}/policy"
    adapter_path = f"{policy_root}/adapter.json"
    checkpoint_identity = hashlib.sha256(args.checkpoint_uri.encode()).hexdigest()
    checkpoint_marker = f"{policy_root}/checkpoint-{checkpoint_identity}.complete"
    shell = (
        "set -euo pipefail; "
        + "mkdir -p "
        + shlex.quote(policy_root)
        + "; if test -f "
        + shlex.quote(checkpoint_marker)
        + "; then test -d "
        + shlex.quote(policy_path)
        + "; else test ! -e "
        + shlex.quote(policy_path)
        + "; "
        + python
        + " -c "
        + shlex.quote(_download_program(args.checkpoint_uri, policy_path))
        + "; touch "
        + shlex.quote(checkpoint_marker)
        + "; fi; "
        + (
            python
            + " -c "
            + shlex.quote(
                _adapter_download_program(args.bind_adapter_uri, adapter_path)
            )
            + "; "
            + python
            + " -m npa.workbench.lerobot.policy_container bind-interchange "
            + f"--checkpoint-path {shlex.quote(policy_path)} "
            + f"--adapter-spec {shlex.quote(adapter_path)} "
            + f"--output-json {shlex.quote(policy_root + '/base-binding.json')}; "
            if args.bind_adapter_uri
            else ""
        )
        + f"export NPA_POLICY_CHECKPOINT={shlex.quote(policy_path)}; "
        + "exec "
        + python
        + " -m npa.workbench.lerobot.policy_container serve --host 0.0.0.0 --port 8080"
    )
    env: list[dict[str, Any]] = [
        {"name": "NPA_POLICY_HARDWARE_TOPOLOGY", "value": args.hardware_topology},
        {
            "name": "NPA_POLICY_AUTH_TOKEN",
            "valueFrom": {
                "secretKeyRef": {"name": args.auth_secret, "key": "auth_token"}
            },
        },
    ]
    manifests: dict[str, dict[str, Any]] = {}
    if args.checkpoint_uri.startswith("s3://") or args.bind_adapter_uri:
        required = {
            key: os.environ.get(key, "")
            for key in (
                "AWS_ENDPOINT_URL",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
            )
        }
        if args.command == "start" and not all(required.values()):
            raise SmolVLAServiceError(
                "private checkpoint or adapter requires exact task-scoped S3 credentials"
            )
        manifests["secret"] = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": names["credentials"],
                "namespace": args.namespace,
                "labels": labels,
            },
            "type": "Opaque",
            "data": {
                key: base64.b64encode(value.encode()).decode()
                for key, value in required.items()
            }
            if args.command == "start"
            else {},
        }
        env.extend(
            {
                "name": key,
                "valueFrom": {
                    "secretKeyRef": {"name": names["credentials"], "key": key}
                },
            }
            for key in required
        )
    elif args.hf_token_secret:
        env.append(
            {
                "name": "HF_TOKEN",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": args.hf_token_secret,
                        "key": "token",
                        "optional": True,
                    }
                },
            }
        )
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Always",
        "serviceAccountName": args.service_account,
        "containers": [
            {
                "name": "smolvla-policy",
                "image": args.runtime_image,
                "imagePullPolicy": "IfNotPresent",
                "command": ["/bin/bash", "-lc", shell],
                "ports": [{"name": "http", "containerPort": PORT}],
                "env": env,
                "resources": {
                    "requests": {
                        "cpu": args.server_cpu,
                        "memory": args.server_memory,
                        "nvidia.com/gpu": str(args.gpu_count),
                    },
                    "limits": {"nvidia.com/gpu": str(args.gpu_count)},
                },
                "volumeMounts": [{"name": "policy", "mountPath": "/workspace"}],
                "readinessProbe": {
                    "httpGet": {"path": "/health", "port": PORT},
                    "periodSeconds": 5,
                    "failureThreshold": 240,
                },
            }
        ],
        "volumes": [
            {
                "name": "policy",
                "persistentVolumeClaim": {"claimName": args.policy_cache_pvc},
            }
        ],
    }
    if args.gpu_node_selector_key:
        pod_spec["nodeSelector"] = {
            args.gpu_node_selector_key: args.gpu_node_selector_value
        }
    manifests["deployment"] = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": names["deployment"],
            "namespace": args.namespace,
            "labels": labels,
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": selector},
            "template": {
                "metadata": {"labels": labels},
                "spec": pod_spec,
            },
        },
    }
    manifests["service"] = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": names["service"],
            "namespace": args.namespace,
            "labels": labels,
        },
        "spec": {
            "type": "ClusterIP",
            "selector": selector,
            "ports": [{"name": "http", "port": PORT, "targetPort": PORT}],
        },
    }
    return manifests


def _clients() -> tuple[Any, Any]:
    from kubernetes import client, config

    config.load_incluster_config()
    return client.CoreV1Api(), client.AppsV1Api()


def _status(exc: Exception) -> int | None:
    return getattr(exc, "status", None)


def _read_or_none(reader: Any, name: str, namespace: str) -> Any | None:
    try:
        return reader(name, namespace, _request_timeout=30)
    except Exception as exc:
        if _status(exc) == 404:
            return None
        raise


def _assert_absent(api: Any, apps: Any, manifests: dict[str, dict[str, Any]]) -> None:
    namespace = manifests["deployment"]["metadata"]["namespace"]
    checks = [
        (apps.read_namespaced_deployment, manifests["deployment"]),
        (api.read_namespaced_service, manifests["service"]),
    ]
    if "secret" in manifests:
        checks.append((api.read_namespaced_secret, manifests["secret"]))
    for reader, manifest in checks:
        name = manifest["metadata"]["name"]
        if _read_or_none(reader, name, namespace) is not None:
            raise SmolVLAServiceError(
                f"refusing to replace pre-existing owned target {name!r}"
            )


def _create(
    api: Any,
    apps: Any,
    manifests: dict[str, dict[str, Any]],
    *,
    created: set[str],
) -> None:
    namespace = manifests["deployment"]["metadata"]["namespace"]
    if "secret" in manifests:
        api.create_namespaced_secret(
            namespace, manifests["secret"], _request_timeout=30
        )
        created.add("secret")
    apps.create_namespaced_deployment(
        namespace, manifests["deployment"], _request_timeout=30
    )
    created.add("deployment")
    api.create_namespaced_service(namespace, manifests["service"], _request_timeout=30)
    created.add("service")


def _ready(
    api: Any, apps: Any, manifests: dict[str, dict[str, Any]], timeout: float
) -> tuple[str, dict[str, Any]]:
    namespace = manifests["deployment"]["metadata"]["namespace"]
    name = manifests["deployment"]["metadata"]["name"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        deployment = apps.read_namespaced_deployment(
            name, namespace, _request_timeout=30
        )
        if int(getattr(deployment.status, "ready_replicas", 0) or 0) == 1:
            host = f"{name}.{namespace}.svc"
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(
                    f"http://{host}:{PORT}/health", timeout=10
                ) as response:
                    payload = json.loads(response.read())
                if (
                    payload.get("status") == "ok"
                    and isinstance(payload.get("checkpoint"), dict)
                    and isinstance(payload.get("interchange"), dict)
                ):
                    return host, payload
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
        time.sleep(5)
    raise SmolVLAServiceError("SmolVLA service readiness timed out")


def _owned(actual: Any, expected: dict[str, Any]) -> bool:
    labels = getattr(getattr(actual, "metadata", None), "labels", None) or {}
    expected_labels = expected["metadata"]["labels"]
    return all(labels.get(key) == value for key, value in expected_labels.items())


def _cleanup(
    api: Any,
    apps: Any,
    manifests: dict[str, dict[str, Any]],
    *,
    created: set[str] | None = None,
) -> dict[str, bool]:
    namespace = manifests["deployment"]["metadata"]["namespace"]
    targets = [
        (
            "deployment",
            apps.read_namespaced_deployment,
            apps.delete_namespaced_deployment,
            manifests["deployment"],
        ),
        (
            "service",
            api.read_namespaced_service,
            api.delete_namespaced_service,
            manifests["service"],
        ),
    ]
    if "secret" in manifests:
        targets.append(
            (
                "secret",
                api.read_namespaced_secret,
                api.delete_namespaced_secret,
                manifests["secret"],
            )
        )
    result: dict[str, bool] = {}
    for kind, reader, deleter, manifest in targets:
        if created is not None and kind not in created:
            result[kind] = True
            continue
        name = manifest["metadata"]["name"]
        actual = _read_or_none(reader, name, namespace)
        if actual is None:
            result[kind] = True
            continue
        if not _owned(actual, manifest):
            raise SmolVLAServiceError(
                f"refusing cleanup of {kind} {name!r} without exact ownership"
            )
        deleter(name, namespace, _request_timeout=30)
        result[kind] = True
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    manifests = build_manifests(args)
    api, apps = _clients()
    if args.command == "start":
        _assert_absent(api, apps, manifests)
        created: set[str] = set()
        try:
            _create(api, apps, manifests, created=created)
            host, health = _ready(api, apps, manifests, args.ready_timeout_seconds)
        except Exception:
            _cleanup(api, apps, manifests, created=created)
            raise
        result = {
            "schema": "npa.sim2real.smolvla_service_start.v1",
            "run_id": args.run_id,
            "checkpoint_role": args.checkpoint_role,
            "checkpoint_uri_sha256": hashlib.sha256(
                args.checkpoint_uri.encode()
            ).hexdigest(),
            "runtime_image": args.runtime_image,
            "source_commit": args.source_commit,
            "policy_cache_pvc_sha256": hashlib.sha256(
                args.policy_cache_pvc.encode()
            ).hexdigest(),
            "service_host": host,
            "port": PORT,
            "public_ingress": False,
            "application_authentication": "kubernetes_secret_bearer",
            "checkpoint": health["checkpoint"],
            "interchange": health["interchange"],
            "status": "ready",
        }
        stage = 5 if args.checkpoint_role == "base" else 10
    else:
        result = {
            "schema": "npa.sim2real.smolvla_service_cleanup.v1",
            "run_id": args.run_id,
            "checkpoint_role": args.checkpoint_role,
            "deleted": _cleanup(api, apps, manifests),
            "status": "absent",
        }
        stage = 7 if args.checkpoint_role == "base" else 13
    with tempfile.TemporaryDirectory(prefix="npa-smolvla-lifecycle-") as directory:
        write_json(args.output_uri, result, directory=Path(directory))
    publish_component_record(
        component_schema=POLICY_INTERCHANGE_COMPONENT_SCHEMA,
        root_uri=args.component_root_uri,
        stage=stage,
        name=f"smolvla_{args.checkpoint_role}_service_{args.command}",
        tier="WORKS",
        evidence="Owned digest-pinned private SmolVLA service lifecycle.",
        artifacts={"receipt": args.output_uri, "status": result["status"]},
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "cleanup"):
        item = commands.add_parser(command)
        item.add_argument("--run-id", required=True)
        item.add_argument(
            "--checkpoint-role", choices=("base", "adapted"), required=True
        )
        item.add_argument("--checkpoint-uri", required=True)
        item.add_argument("--bind-adapter-uri", default="")
        item.add_argument("--runtime-image", required=True)
        item.add_argument("--source-commit", required=True)
        item.add_argument("--namespace", required=True)
        item.add_argument("--service-account", required=True)
        item.add_argument("--auth-secret", required=True)
        item.add_argument("--hf-token-secret", default="")
        item.add_argument("--gpu-count", type=int, default=1)
        item.add_argument("--server-cpu", default="8")
        item.add_argument("--server-memory", default="64Gi")
        item.add_argument("--policy-cache-pvc", required=True)
        item.add_argument("--gpu-node-selector-key", default="")
        item.add_argument("--gpu-node-selector-value", default="")
        item.add_argument("--hardware-topology", default="single-gpu")
        item.add_argument("--ready-timeout-seconds", type=float, default=1200.0)
        item.add_argument("--output-uri", required=True)
        item.add_argument("--component-root-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    result = _run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
