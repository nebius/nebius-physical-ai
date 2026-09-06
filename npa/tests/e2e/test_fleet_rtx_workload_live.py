"""Operator-invoked CUDA and graphics execution on every selected Fleet target."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

import pytest

from npa.cluster.gpu_health import DEFAULT_GRAPHICS_SMOKE_IMAGE
from npa.fleet.spec import load_spec

pytestmark = pytest.mark.e2e

# Use the same immutable, payload-clean image as the provisioning graphics gate.
# Calling its baked interpreter directly does not fetch Isaac or model weights.
COMMAND = r"""
set -euo pipefail
"${NPA_IMAGE_PYTHON:?image must declare its baked interpreter}" - <<'PY'
import json
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)
assert 'RTX PRO 6000' in torch.cuda.get_device_name(0)
x = torch.arange(1024, device='cuda', dtype=torch.float32)
y = x.square() + 3 * x
torch.cuda.synchronize()
assert torch.equal(y.cpu(), torch.arange(1024, dtype=torch.float32).square() + 3 * torch.arange(1024))
a = torch.ones((256, 256), device='cuda')
b = a @ a
torch.cuda.synchronize()
assert torch.equal(b, torch.full_like(b, 256))
print('NPA_CUDA_EXECUTED ' + json.dumps({'elements': x.numel(), 'matrix_size': 256, 'sum': y.sum().item()}), flush=True)
PY
python3 - <<'PY'
import ctypes
import os
ctypes.CDLL('libGLX_nvidia.so.0')
print('NPA_GLX_LOADED', flush=True)
os._exit(0)
PY
python3 - <<'PY'
import ctypes
import os
ctypes.CDLL('libEGL_nvidia.so.0')
print('NPA_EGL_LOADED', flush=True)
os._exit(0)
PY
vulkaninfo --summary
""".strip()


def test_every_fleet_target_executes_cuda_and_graphics() -> None:
    if os.environ.get("NPA_FLEET_RTX_RUN_WORKLOADS") != "1":
        pytest.skip("explicitly enable real workloads on the owner-selected Fleet")
    from kubernetes import client, config

    spec = load_spec(Path(os.environ["NPA_FLEET_RTX_VERIFY_SPEC"]))
    configs = json.loads(Path(os.environ["NPA_FLEET_RTX_KUBECONFIGS"]).read_text())
    evidence = Path(os.environ["NPA_FLEET_RTX_WORKLOAD_EVIDENCE_DIR"])
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert evidence.stat().st_mode & 0o077 == 0, "evidence must be owner-private"
    assert spec.profile
    for key in ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN", "TF_VAR_iam_token"):
        os.environ.pop(key, None)
    os.environ.update(NEBIUS_PROFILE=spec.profile, NPA_NEBIUS_PROFILE=spec.profile)
    targets = list(spec.cluster_targets())
    assert targets
    results = []
    for index, (project, cluster) in enumerate(targets):
        assert cluster.gpu_workload_profile == "rtx-rendering"
        api = config.new_client_from_config(
            config_file=configs[project.key()][cluster.name], persist_config=False,
        )
        with api:
            core = client.CoreV1Api(api)
            batch = client.BatchV1Api(api)
            # A missing RuntimeClass rejects pod creation before any Pod status
            # exists. Check it before creating a Job that could otherwise wait.
            assert client.NodeV1Api(api).read_runtime_class("nvidia").handler == "nvidia"
            selector = {
                "node.kubernetes.io/instance-type": cluster.gpu_nodes.platform,
                "nebius.com/resource-preset": cluster.gpu_nodes.preset,
            }
            nodes = core.list_node(label_selector=",".join(
                f"{key}={value}" for key, value in selector.items()
            )).items
            assert len(nodes) == cluster.gpu_nodes.count
            assert all(
                any(c.type == "Ready" and c.status == "True" for c in n.status.conditions)
                and not n.spec.unschedulable
                and int(n.status.allocatable.get("nvidia.com/gpu", 0)) == 8
                for n in nodes
            ), "every requested eight-GPU worker must be Ready before qualification"
            name = "npa-rtx-qualify-" + uuid.uuid4().hex[:12]
            manifest = {
                "apiVersion": "batch/v1", "kind": "Job",
                "metadata": {"name": name},
                "spec": {
                    "backoffLimit": 0,
                    "template": {"spec": {
                        "restartPolicy": "Never", "runtimeClassName": "nvidia",
                        "nodeSelector": selector,
                        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists",
                                         "effect": "NoSchedule"}],
                        "containers": [{
                            "name": "qualify", "image": DEFAULT_GRAPHICS_SMOKE_IMAGE,
                            "command": ["/bin/bash", "-c", COMMAND],
                            "env": [{"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "all"}],
                            "resources": {"limits": {"nvidia.com/gpu": 1}},
                            "securityContext": {"allowPrivilegeEscalation": False,
                                                "capabilities": {"drop": ["ALL"]}},
                        }],
                    }},
                },
            }
            job = batch.create_namespaced_job("default", manifest)
            uid = job.metadata.uid
            try:
                while True:
                    current = batch.read_namespaced_job(name, "default")
                    assert current.metadata.uid == uid, "qualification Job identity changed"
                    if any(c.status == "True" and c.type in {"Complete", "Failed"}
                           for c in current.status.conditions or []):
                        break
                    pending = core.list_namespaced_pod(
                        "default", label_selector=f"batch.kubernetes.io/controller-uid={uid}",
                    ).items
                    fatal = {
                        "ImagePullBackOff", "ErrImagePull", "InvalidImageName",
                        "CreateContainerConfigError", "CreateContainerError",
                    }
                    assert not any(
                        status.state.waiting and status.state.waiting.reason in fatal
                        for pod in pending for status in pod.status.container_statuses or []
                    ), "qualification pod cannot start; inspect its image and configuration"
                    time.sleep(2)
                pods = core.list_namespaced_pod(
                    "default", label_selector=f"batch.kubernetes.io/controller-uid={uid}",
                ).items
                assert len(pods) == 1
                pod = pods[0]
                output = core.read_namespaced_pod_log(pod.metadata.name, "default")
                private = evidence / f"target-{index}-execution.json"
                private.write_text(json.dumps({
                    "job": api.sanitize_for_serialization(current),
                    "pod": api.sanitize_for_serialization(pod), "logs": output,
                }))
                private.chmod(0o600)
                assert pod.status.phase == "Succeeded", "driver workload failed; inspect private evidence"
                assert current.status.succeeded == 1
                assert all(marker in output for marker in (
                    "NPA_CUDA_EXECUTED", "NPA_GLX_LOADED", "NPA_EGL_LOADED",
                    "Vulkan Instance Version",
                ))
                assert re.search(r"(?m)^GPU[0-9]+:", output)
                assert re.search(r"deviceName\s*=.*NVIDIA.*RTX PRO 6000", output)
                statuses = pod.status.container_statuses
                assert len(statuses) == 1 and statuses[0].state.terminated.exit_code == 0
                digest = DEFAULT_GRAPHICS_SMOKE_IMAGE.split("@", 1)[1]
                assert digest in statuses[0].image_id, "runtime digest differs from the pinned image"
                results.append({
                    "target_index": index, "gpu_workers": len(nodes),
                    "cuda": "executed", "glx": "loaded", "egl": "loaded",
                    "vulkan": "instance-created-and-nvidia-device-enumerated",
                    "image_digest": digest,
                    "execution_sha256": hashlib.sha256(private.read_bytes()).hexdigest(),
                })
            finally:
                batch.delete_namespaced_job(name, "default", body=client.V1DeleteOptions(
                    preconditions=client.V1Preconditions(uid=uid), propagation_policy="Foreground",
                ))
                while True:
                    jobs = batch.list_namespaced_job(
                        "default", field_selector=f"metadata.name={name}",
                    ).items
                    pods = core.list_namespaced_pod(
                        "default", label_selector=f"batch.kubernetes.io/controller-uid={uid}",
                    ).items
                    if not jobs and not pods:
                        break
                    time.sleep(2)
                if results and results[-1]["target_index"] == index:
                    results[-1]["job_and_pods_removed"] = True
                summary = evidence / "sanitized-results.json"
                summary.write_text(json.dumps(results, indent=2))
                summary.chmod(0o600)
    assert len(results) == len(targets)
