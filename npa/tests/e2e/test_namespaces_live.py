"""Exercise native namespace access and CPU placement on an owned live cluster."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import pytest
import yaml

from npa.workbench.namespaces import (
    apply_namespace,
    namespace_manifests,
    write_namespace_context,
)


pytestmark = [pytest.mark.e2e, pytest.mark.timeout(0)]


def _run(args, kubeconfig, context, *, payload=None):
    return subprocess.run(
        ["kubectl", "--context", context, *args],
        env={**os.environ, "KUBECONFIG": str(kubeconfig)},
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )


def _account_config(admin, context, namespace, directory):
    source = _run(
        ["config", "view", "--raw", "--minify", "--flatten", "-o", "json"],
        admin,
        context,
    )
    token = _run(["create", "token", "npa-workbench", "-n", namespace], admin, context)
    assert source.returncode == token.returncode == 0, "Could not prepare test identity"
    config = json.loads(source.stdout)
    config["users"] = [
        {"name": "namespace-test", "user": {"token": token.stdout.strip()}}
    ]
    config["contexts"][0]["context"]["user"] = "namespace-test"
    path = directory / f"{namespace}-source"
    path.write_text(yaml.safe_dump(config))
    path.chmod(0o600)
    settings = write_namespace_context(
        namespace,
        context=context,
        kubeconfig=str(path),
        output_dir=directory / namespace,
    )
    return Path(settings["kubeconfig"])


@pytest.fixture(scope="module")
def live_namespaces(tmp_path_factory):
    if (
        os.environ.get("NPA_NAMESPACE_LIVE_E2E") != "1"
        or os.environ.get("NPA_INTEGRATION_E2E") != "1"
    ):
        pytest.skip("Explicit namespace live validation is disabled")
    context = os.environ.get("NPA_NAMESPACE_LIVE_CONTEXT", "")
    admin = os.environ.get("KUBECONFIG", "")
    assert context and admin, "Select the exact disposable cluster and kubeconfig"
    directory = tmp_path_factory.mktemp("namespace-clients")
    directory.chmod(0o700)
    names = [f"npa-test-{uuid.uuid4().hex[:12]}" for _ in range(2)]
    created = []
    try:
        for name in names:
            absent = _run(
                ["get", "namespace", name, "--ignore-not-found", "-o", "name"],
                admin,
                context,
            )
            assert absent.returncode == 0 and not absent.stdout.strip(), (
                "Test namespace must be new"
            )
            created.append(name)
            apply_namespace(name, context=context, kubeconfig=admin)
        clients = [_account_config(admin, context, name, directory) for name in names]
        yield context, admin, names, clients
    finally:
        for name in created:
            result = _run(
                ["delete", "namespace", name, "--ignore-not-found", "--wait=true"],
                admin,
                context,
            )
            assert result.returncode == 0, "Owned test namespace cleanup failed"
            for document in namespace_manifests(name):
                if document["kind"] in {"ClusterRole", "ClusterRoleBinding"}:
                    result = _run(
                        [
                            "delete",
                            document["kind"],
                            document["metadata"]["name"],
                            "--ignore-not-found",
                        ],
                        admin,
                        context,
                    )
                    assert result.returncode == 0, (
                        "Owned discovery binding cleanup failed"
                    )


def test_same_resource_name_resolves_in_each_private_context(live_namespaces):
    context, _admin, names, clients = live_namespaces
    for namespace, client in zip(names, clients, strict=True):
        result = _run(
            ["create", "configmap", "shared-name", f"--from-literal=team={namespace}"],
            client,
            context,
        )
        assert result.returncode == 0, "Namespaced create failed"
        result = _run(
            ["get", "configmap", "shared-name", "-o", "json"], client, context
        )
        assert result.returncode == 0, "Namespaced read failed"
        document = json.loads(result.stdout)
        assert document["metadata"]["namespace"] == namespace
        assert document["data"]["team"] == namespace


def test_actual_cross_namespace_and_cluster_mutations_are_denied(live_namespaces):
    context, _admin, names, clients = live_namespaces
    for index, client in enumerate(clients):
        other = names[1 - index]
        for args in (
            ["get", "secrets", "-n", other],
            ["get", "pods", "-n", other],
            ["create", "configmap", "forbidden", "-n", other],
            [
                "create",
                "rolebinding",
                "escalation",
                "--clusterrole=admin",
                "--serviceaccount=default:default",
            ],
            [
                "create",
                "clusterrolebinding",
                "namespace-escalation-test",
                "--dry-run=server",
                "--clusterrole=cluster-admin",
                "--serviceaccount=default:default",
            ],
        ):
            result = _run(args, client, context)
            assert result.returncode != 0 and "forbidden" in result.stderr.lower(), (
                "Expected real Kubernetes RBAC denial"
            )


def _completed_pod(client, context):
    while True:
        result = _run(["get", "pod", "same-job", "-o", "json"], client, context)
        assert result.returncode == 0, "Could not observe owned CPU pod"
        pod = json.loads(result.stdout)
        phase = pod["status"]["phase"]
        assert phase != "Failed", "Owned CPU pod failed"
        if phase == "Succeeded":
            return pod
        states = [
            item.get("state", {}) for item in pod["status"].get("containerStatuses", [])
        ]
        blocked = {"ErrImagePull", "ImagePullBackOff", "CreateContainerConfigError"}
        assert not any(
            item.get("waiting", {}).get("reason") in blocked for item in states
        ), "Owned CPU pod has an actionable startup failure"
        time.sleep(1)


def test_cpu_pods_execute_in_each_selected_namespace(live_namespaces):
    context, _admin, names, clients = live_namespaces
    image = os.environ.get(
        "NPA_NAMESPACE_LIVE_IMAGE", "docker.io/library/busybox:1.37.0"
    )
    for namespace, client in zip(names, clients, strict=True):
        pod = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": "same-job"},
            "spec": {
                "restartPolicy": "Never",
                "serviceAccountName": "npa-workbench",
                "containers": [
                    {
                        "name": "proof",
                        "image": image,
                        "command": ["sh", "-c", 'printf "%s" "$POD_NAMESPACE"'],
                        "env": [
                            {
                                "name": "POD_NAMESPACE",
                                "valueFrom": {
                                    "fieldRef": {"fieldPath": "metadata.namespace"}
                                },
                            }
                        ],
                    }
                ],
            },
        }
        result = _run(["create", "-f", "-"], client, context, payload=pod)
        assert result.returncode == 0, "CPU pod creation failed"
        finished = _completed_pod(client, context)
        assert finished["metadata"]["namespace"] == namespace
        result = _run(["logs", "same-job"], client, context)
        assert result.returncode == 0 and result.stdout == namespace


def test_reapply_removes_researcher_membership(live_namespaces):
    context, admin, names, _clients = live_namespaces
    namespace = names[0]
    user = f"npa-test-user-{uuid.uuid4().hex[:12]}"
    apply_namespace(namespace, context=context, kubeconfig=admin, users=(user,))
    args = ["auth", "can-i", "create", "pods", "-n", namespace, "--as", user]
    assert _run(args, admin, context).returncode == 0, (
        "Researcher grant was ineffective"
    )
    apply_namespace(namespace, context=context, kubeconfig=admin)
    result = _run(args, admin, context)
    assert result.returncode != 0 and result.stdout.strip() == "no", (
        "Removed researcher retained this grant"
    )
