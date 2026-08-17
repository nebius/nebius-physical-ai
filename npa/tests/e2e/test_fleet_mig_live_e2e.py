"""Destructive live qualification for the RTX PRO 6000 hardware MIG fleet.

Run only against the operator-authorized disposable/qualification cluster. The
workload matrix occupies all six physical MIG slices with real ``vectorAdd``
CUDA kernels; it does not use sleeps, MPS, or time slicing as GPU evidence.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from npa.fleet.mig import _kubectl_env, verify_mig_cluster

pytestmark = pytest.mark.e2e_pipeline


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


if not _enabled("NPA_INTEGRATION_E2E"):
    pytest.skip(
        "set NPA_INTEGRATION_E2E=1 for live infrastructure", allow_module_level=True
    )

KUBECTL = os.environ.get("NPA_KUBECTL_BIN", "kubectl")
KUBECONFIG = Path(os.environ.get("NPA_FLEET_MIG_KUBECONFIG", ""))
CUDA_IMAGE = os.environ.get(
    "NPA_FLEET_MIG_CUDA_IMAGE", "nvcr.io/nvidia/gpu-operator:v26.3.3"
)
NAMESPACE = os.environ.get("NPA_FLEET_MIG_NAMESPACE", "npa-mig-live-e2e")


def _run(args: list[str], *, stdin: str | None = None, check: bool = True):
    result = subprocess.run(
        [KUBECTL, "--kubeconfig", str(KUBECONFIG), *args],
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
        env=_kubectl_env(),
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"kubectl {' '.join(args)} failed ({result.returncode}): {result.stderr}"
        )
    return result


def _json(args: list[str]) -> dict:
    return json.loads(_run([*args, "-o", "json"]).stdout)


def _wait_for(predicate, description: str):
    while True:
        value = predicate()
        if value:
            return value
        time.sleep(3)


def _pod_manifest(name: str, node: str, resource: str, *, hold: bool) -> dict:
    identity = "nvidia-smi -L; nvidia-smi; "
    command = (
        identity + "i=0; while :; do /usr/bin/vectorAdd; "
        "i=$((i+1)); if [ $((i % 20)) -eq 0 ]; then echo CUDA_LOOPS=$i; fi; done"
        if hold
        else identity + "/usr/bin/vectorAdd"
    )
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {"app": "npa-mig-live-e2e"},
        },
        "spec": {
            "restartPolicy": "Never",
            "runtimeClassName": "nvidia",
            "nodeSelector": {"kubernetes.io/hostname": node},
            "containers": [
                {
                    "name": "cuda",
                    "image": CUDA_IMAGE,
                    "command": ["sh", "-c", command],
                    "resources": {
                        "requests": {resource: 1},
                        "limits": {resource: 1},
                    },
                }
            ],
        },
    }


def _apply(objects: list[dict]) -> None:
    payload = "\n---\n".join(json.dumps(obj) for obj in objects)
    _run(["apply", "-f", "-"], stdin=payload)


def _phase(name: str) -> str:
    result = _run(["get", "pod", name, "-n", NAMESPACE, "-o", "json"], check=False)
    if result.returncode != 0:
        return ""
    status = json.loads(result.stdout).get("status", {})
    phase = str(status.get("phase", ""))
    terminal_reasons = {
        "CrashLoopBackOff",
        "CreateContainerConfigError",
        "CreateContainerError",
        "ErrImagePull",
        "ImagePullBackOff",
        "InvalidImageName",
        "RunContainerError",
    }
    containers = [
        *status.get("initContainerStatuses", []),
        *status.get("containerStatuses", []),
    ]
    reasons = {
        str(item.get("state", {}).get("waiting", {}).get("reason", ""))
        for item in containers
    }
    terminal = sorted(reasons & terminal_reasons)
    if phase == "Failed" or terminal:
        raise AssertionError(
            f"pod {name} entered terminal state phase={phase!r}, reasons={terminal!r}"
        )
    return phase


def _assert_profile_identity(logs: str, resource: str) -> None:
    expected_profile = resource.removeprefix("nvidia.com/mig-")
    assert f"MIG {expected_profile}" in logs, logs
    # The non-privileged MIG workload cannot use --query-gpu=memory.total on
    # Blackwell (it returns Insufficient Permissions), while the ordinary
    # nvidia-smi MIG-device table exposes the CUDA-visible framebuffer total.
    match = re.search(r"\d+MiB\s*/\s*(\d+)MiB", logs)
    assert match, logs
    memory_mib = int(match.group(1))
    if expected_profile == "1g.24gb":
        assert 23_000 <= memory_mib <= 25_000, memory_mib
    elif expected_profile == "2g.48gb":
        assert 47_000 <= memory_mib <= 50_000, memory_mib
    else:  # pragma: no cover - the supported profile contract is deliberately closed
        raise AssertionError(f"unexpected MIG profile {expected_profile!r}")


@pytest.fixture(autouse=True)
def _namespace_cleanup():
    _run(["create", "namespace", NAMESPACE], check=False)
    _run(
        [
            "delete",
            "pod",
            "-n",
            NAMESPACE,
            "-l",
            "app=npa-mig-live-e2e",
            "--ignore-not-found",
        ]
    )
    try:
        yield
    finally:
        # Cleanup must not mask the qualification result. Deleting the run-owned
        # namespace also removes Pending exhaustion probes and event history.
        _run(
            ["delete", "namespace", NAMESPACE, "--ignore-not-found", "--wait=true"],
            check=False,
        )


def test_live_baseline_is_exact_on_both_nodes() -> None:
    report = verify_mig_cluster(
        kubectl_bin=KUBECTL, kubeconfig=KUBECONFIG, expected_nodes=2
    )
    assert report.ready, report.errors
    assert len(report.nodes) == 2
    assert all(node.schedulable for node in report.nodes)


def test_all_profiles_concurrency_exhaustion_and_reuse() -> None:
    if not _enabled("NPA_FLEET_MIG_RUN_WORKLOAD_MATRIX"):
        pytest.skip("set NPA_FLEET_MIG_RUN_WORKLOAD_MATRIX=1 for destructive occupancy")

    baseline = verify_mig_cluster(
        kubectl_bin=KUBECTL, kubeconfig=KUBECONFIG, expected_nodes=2
    )
    assert baseline.ready, baseline.errors
    nodes = [node.name for node in baseline.nodes]
    holders: list[dict] = []
    for node_index, node in enumerate(nodes):
        holders.extend(
            [
                _pod_manifest(
                    f"hold-{node_index}-1g-a", node, "nvidia.com/mig-1g.24gb", hold=True
                ),
                _pod_manifest(
                    f"hold-{node_index}-1g-b", node, "nvidia.com/mig-1g.24gb", hold=True
                ),
                _pod_manifest(
                    f"hold-{node_index}-2g", node, "nvidia.com/mig-2g.48gb", hold=True
                ),
            ]
        )
    _apply(holders)
    holder_names = [pod["metadata"]["name"] for pod in holders]
    _wait_for(
        lambda: all(_phase(name) == "Running" for name in holder_names),
        "all six MIG holders Running",
    )

    uuids: set[str] = set()
    holder_resources = {
        pod["metadata"]["name"]: next(
            iter(pod["spec"]["containers"][0]["resources"]["limits"])
        )
        for pod in holders
    }
    for name in holder_names:
        logs = _run(["logs", name, "-n", NAMESPACE]).stdout
        _wait_for(
            lambda name=name: (
                "Test PASSED" in _run(["logs", name, "-n", NAMESPACE]).stdout
            ),
            f"{name} CUDA kernel",
        )
        logs = _run(["logs", name, "-n", NAMESPACE]).stdout
        _assert_profile_identity(logs, holder_resources[name])
        match = re.search(r"MIG-[0-9a-fA-F-]+", logs)
        assert match, logs
        uuids.add(match.group(0))
    assert len(uuids) == 6, uuids

    extras: list[dict] = []
    for node_index, node in enumerate(nodes):
        for suffix, resource in (
            ("1g", "nvidia.com/mig-1g.24gb"),
            ("2g", "nvidia.com/mig-2g.48gb"),
        ):
            extras.append(
                _pod_manifest(f"extra-{node_index}-{suffix}", node, resource, hold=True)
            )
    _apply(extras)
    for pod in extras:
        name = pod["metadata"]["name"]
        resource = next(iter(pod["spec"]["containers"][0]["resources"]["limits"]))
        _wait_for(lambda name=name: _phase(name) == "Pending", f"{name} Pending")
        event_text = _wait_for(
            lambda name=name: (
                _run(
                    ["events", "-n", NAMESPACE, "--for", f"pod/{name}"], check=False
                ).stdout
                or False
            ),
            f"{name} scheduling event",
        )
        assert f"Insufficient {resource}" in event_text, event_text
        assert (
            "nvidia.com/gpu" not in pod["spec"]["containers"][0]["resources"]["limits"]
        )

    _run(
        [
            "delete",
            "pod",
            "-n",
            NAMESPACE,
            *holder_names,
            *[p["metadata"]["name"] for p in extras],
        ]
    )
    _wait_for(
        lambda: not _json(["get", "pods", "-n", NAMESPACE]).get("items"),
        "holder cleanup",
    )

    reuse: list[dict] = []
    for node_index, node in enumerate(nodes):
        reuse.extend(
            [
                _pod_manifest(
                    f"reuse-{node_index}-1g", node, "nvidia.com/mig-1g.24gb", hold=False
                ),
                _pod_manifest(
                    f"reuse-{node_index}-2g", node, "nvidia.com/mig-2g.48gb", hold=False
                ),
            ]
        )
    _apply(reuse)
    for pod in reuse:
        name = pod["metadata"]["name"]
        _wait_for(lambda name=name: _phase(name) == "Succeeded", f"{name} reuse")
        logs = _run(["logs", name, "-n", NAMESPACE]).stdout
        assert "Test PASSED" in logs
        resource = next(iter(pod["spec"]["containers"][0]["resources"]["limits"]))
        _assert_profile_identity(logs, resource)

    final = verify_mig_cluster(
        kubectl_bin=KUBECTL, kubeconfig=KUBECONFIG, expected_nodes=2
    )
    assert final.ready, final.errors
