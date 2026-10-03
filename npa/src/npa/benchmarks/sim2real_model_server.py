"""Render a pinned Kubernetes model server for the Sim2Real benchmark."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml


def _server_argv(model: dict[str, Any], *, service_name: str) -> list[str]:
    common = [
        "--revision",
        str(model["revision"]),
        "--served-model-name",
        "benchmark-model",
    ]
    if model["server"] == "sglang":
        argv = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model["repository"]),
            *common,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--context-length",
            str(model["context_limit"]),
            "--tp",
            str(model["tensor_parallel_size"]),
            "--tool-call-parser",
            str(model["tool_call_parser"]),
        ]
        if model.get("reasoning_parser"):
            argv.extend(["--reasoning-parser", str(model["reasoning_parser"])])
        if int(model["tensor_parallel_size"]) > 8:
            argv.extend(
                [
                    "--dist-init-addr",
                    f"{service_name}-0.{service_name}-headless:5000",
                    "--nnodes",
                    "2",
                    "--node-rank",
                    "${ORDINAL}",
                ]
            )
    elif model["server"] == "vllm":
        argv = [
            "vllm",
            "serve",
            str(model["repository"]),
            *common,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max-model-len",
            str(model["context_limit"]),
            "--tensor-parallel-size",
            str(model["tensor_parallel_size"]),
            "--tool-call-parser",
            str(model["tool_call_parser"]),
        ]
        if model.get("reasoning_parser"):
            argv.extend(["--reasoning-parser", str(model["reasoning_parser"])])
    else:
        raise ValueError(f"unsupported server engine: {model['server']}")
    argv.extend(str(value) for value in model.get("server_arguments") or [])
    return argv


def render_server_resources(
    model: dict[str, Any], *, namespace: str, service_name: str
) -> list[dict[str, Any]]:
    """Return namespace, services, and StatefulSet for one pinned model."""

    image = str(model["server_image"])
    if "@sha256:" not in image:
        raise ValueError("model server image must be pinned by sha256 digest")
    tp = int(model["tensor_parallel_size"])
    if tp < 1 or tp > 16 or (tp > 8 and tp != 16):
        raise ValueError("this benchmark supports TP 1-8 on one node or TP 16 on two")
    if model["server"] == "vllm" and tp > 8:
        raise ValueError(
            "vLLM tensor parallelism above 8 requires a provider-neutral "
            "multi-node rendezvous contract, which this benchmark does not implement"
        )
    replicas = 2 if tp == 16 else 1
    gpus_per_pod = 8 if tp >= 8 else tp
    argv = _server_argv(model, service_name=service_name)
    command = "ORDINAL=${POD_NAME##*-}; exec " + " ".join(
        "${ORDINAL}" if value == "${ORDINAL}" else shlex.quote(value) for value in argv
    )
    labels = {"app.kubernetes.io/name": service_name}
    pod_spec: dict[str, Any] = {
        "terminationGracePeriodSeconds": 120,
        "hostNetwork": True,
        "hostIPC": True,
        "dnsPolicy": "ClusterFirstWithHostNet",
        "affinity": {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {"matchLabels": labels},
                        "topologyKey": "kubernetes.io/hostname",
                    }
                ]
            }
        },
        "containers": [
            {
                "name": "server",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "command": ["bash", "-lc", command],
                "ports": [
                    {"name": "openai", "containerPort": 8000},
                    {"name": "distributed", "containerPort": 5000},
                ],
                "env": [
                    {
                        "name": "POD_NAME",
                        "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                    },
                    {
                        "name": "HF_TOKEN",
                        "valueFrom": {
                            "secretKeyRef": {"name": "model-access", "key": "HF_TOKEN"}
                        },
                    },
                    {
                        "name": "HF_HOME",
                        "value": f"/mnt/data/model-cache/{service_name}",
                    },
                    {
                        "name": "HF_HUB_CACHE",
                        "value": f"/mnt/data/model-cache/{service_name}/hub",
                    },
                    {
                        "name": "HUGGINGFACE_HUB_CACHE",
                        "value": f"/mnt/data/model-cache/{service_name}/hub",
                    },
                    {
                        "name": "TRANSFORMERS_CACHE",
                        "value": f"/mnt/data/model-cache/{service_name}/hub",
                    },
                    {
                        "name": "HF_XET_CACHE",
                        "value": f"/mnt/data/model-cache/{service_name}/xet",
                    },
                    {
                        "name": "TORCH_HOME",
                        "value": f"/mnt/data/model-cache/{service_name}/torch",
                    },
                    {"name": "NCCL_DEBUG", "value": "INFO"},
                    {"name": "NCCL_IB_DISABLE", "value": "0"},
                    # With host networking these nodes resolve their hostname
                    # to 127.0.1.1.  Pin the bootstrap/control sockets to the
                    # routable interface so multi-node Gloo/NCCL ranks do not
                    # advertise loopback to one another.
                    {"name": "GLOO_SOCKET_IFNAME", "value": "eth0"},
                    {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
                ],
                "resources": {
                    "requests": {"nvidia.com/gpu": gpus_per_pod},
                    "limits": {"nvidia.com/gpu": gpus_per_pod},
                },
                "securityContext": {"capabilities": {"add": ["IPC_LOCK"]}},
                "volumeMounts": [
                    {"name": "model-cache", "mountPath": "/mnt/data"},
                    {"name": "shm", "mountPath": "/dev/shm"},
                    {"name": "infiniband", "mountPath": "/dev/infiniband"},
                ],
            }
        ],
        "volumes": [
            {
                "name": "model-cache",
                "hostPath": {"path": "/mnt/data", "type": "Directory"},
            },
            {
                "name": "shm",
                "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
            },
            {
                "name": "infiniband",
                "hostPath": {"path": "/dev/infiniband", "type": "Directory"},
            },
        ],
    }
    return [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace}},
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": f"{service_name}-headless", "namespace": namespace},
            "spec": {
                "clusterIP": "None",
                "publishNotReadyAddresses": True,
                "selector": labels,
                "ports": [{"name": "distributed", "port": 5000}],
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": service_name, "namespace": namespace},
            "spec": {
                "selector": {"statefulset.kubernetes.io/pod-name": f"{service_name}-0"},
                "ports": [{"name": "openai", "port": 8000, "targetPort": "openai"}],
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {"name": service_name, "namespace": namespace},
            "spec": {
                "serviceName": f"{service_name}-headless",
                "replicas": replicas,
                "podManagementPolicy": "Parallel",
                "selector": {"matchLabels": labels},
                "template": {"metadata": {"labels": labels}, "spec": pod_spec},
            },
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--service-name", required=True)
    args = parser.parse_args()
    catalog = json.loads(args.models.read_text(encoding="utf-8"))
    resources = render_server_resources(
        catalog["models"][args.model_index],
        namespace=args.namespace,
        service_name=args.service_name,
    )
    yaml.safe_dump_all(resources, sys.stdout, sort_keys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
