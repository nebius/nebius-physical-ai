"""Exercise namespace creation, existing access, default selection, and CPU placement."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import pytest
import yaml

from npa.clients.kubernetes_namespace import context_namespace
from npa.workbench.namespaces import apply_namespace, write_namespace_context


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
    original = Path(admin).read_bytes()
    created, clients = [], []
    try:
        for _ in range(2):
            name = f"npa-test-{uuid.uuid4().hex[:12]}"
            result = apply_namespace(name, context=context, kubeconfig=admin)
            assert result["status"] == "created", "Test namespace must be new"
            created.append(name)
            settings = write_namespace_context(
                name, context=context, kubeconfig=admin, output_dir=directory / name
            )
            clients.append(Path(settings["kubeconfig"]))
        assert Path(admin).read_bytes() == original, "Source kubeconfig was modified"
        yield context, admin, created, clients, directory
    finally:
        for name in created:
            result = _run(
                ["delete", "namespace", name, "--ignore-not-found", "--wait=true"],
                admin,
                context,
            )
            assert result.returncode == 0, "Owned test namespace cleanup failed"


def test_same_resource_name_resolves_in_each_private_context(live_namespaces):
    context, _admin, names, clients, _directory = live_namespaces
    for namespace, client in zip(names, clients, strict=True):
        result = _run(
            [
                "create",
                "configmap",
                "shared-name",
                f"--from-literal=namespace={namespace}",
            ],
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
        assert document["data"]["namespace"] == namespace


def test_reusing_an_existing_namespace_preserves_its_access(live_namespaces):
    context, admin, names, _clients, _directory = live_namespaces
    namespace = names[0]
    resources = [
        "get",
        "roles,rolebindings,serviceaccounts",
        "-n",
        namespace,
        "-o",
        "json",
    ]
    before = _run(resources, admin, context)
    assert before.returncode == 0
    result = apply_namespace(namespace, context=context, kubeconfig=admin)
    assert result["status"] == "existing"
    after = _run(resources, admin, context)
    assert after.returncode == 0
    assert json.loads(before.stdout) == json.loads(after.stdout)


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
    context, _admin, names, clients, _directory = live_namespaces
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
        assert _completed_pod(client, context)["metadata"]["namespace"] == namespace
        result = _run(["logs", "same-job"], client, context)
        assert result.returncode == 0 and result.stdout == namespace


def test_unset_namespace_uses_kubernetes_default(live_namespaces):
    context, admin, _names, _clients, directory = live_namespaces
    result = _run(
        ["config", "view", "--minify", "--flatten", "--raw", "-o", "json"],
        admin,
        context,
    )
    assert result.returncode == 0
    document = json.loads(result.stdout)
    document["contexts"][0]["context"].pop("namespace", None)
    client = directory / "unset-namespace"
    client.write_text(yaml.safe_dump(document))
    client.chmod(0o600)
    assert context_namespace(context=context, kubeconfig=str(client)) == "default"
    name = f"npa-default-{uuid.uuid4().hex[:12]}"
    created = _run(
        ["create", "configmap", name, "--from-literal=proof=default"], client, context
    )
    assert created.returncode == 0, "Default namespace creation failed"
    try:
        observed = _run(["get", "configmap", name, "-o", "json"], client, context)
        assert observed.returncode == 0
        assert json.loads(observed.stdout)["metadata"]["namespace"] == "default"
    finally:
        deleted = _run(["delete", "configmap", name, "--wait=true"], client, context)
        assert deleted.returncode == 0, (
            "Owned default-namespace resource cleanup failed"
        )
