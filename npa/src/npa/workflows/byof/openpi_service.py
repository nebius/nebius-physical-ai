"""Cross-pod Kubernetes serving gate for the pinned upstream OpenPI server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import textwrap
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request

from npa.workflows.byof.openpi_pipeline import (
    ACTION_DIM,
    DEFAULT_CHECKPOINT_URI,
    DEFAULT_CONFIG_NAME,
    RUNTIME_IMAGE_RE,
    SOURCE_LICENSE,
    SOURCE_REF,
    _checkpoint_provenance,
    _redistribution_evidence,
    _write_json_uri,
)
from npa.workflows.byof.openpi_service_diagnostics import standalone_program

MANAGED_BY = "npa-openpi-four-mode"
SERVER_PORT = 8000
SERVER_DIAGNOSTICS_PORT = 8001
DEFAULT_SERVER_READY_TIMEOUT_SECONDS = 1_200.0
DEFAULT_CLIENT_TIMEOUT_SECONDS = 600.0
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 180.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_API_TIMEOUT_SECONDS = 30.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
_TERMINAL_WAITING_REASONS = frozenset(
    {
        "CrashLoopBackOff",
        "CreateContainerConfigError",
        "CreateContainerError",
        "ErrImagePull",
        "ImagePullBackOff",
        "InvalidImageName",
        "RunContainerError",
    }
)
LABELS = {
    "app.kubernetes.io/managed-by": "npa",
    "app.kubernetes.io/part-of": "openpi-four-mode",
}
CONTROLLER_MANAGED_BY = "npa-openpi-service-controller"


class OpenPIServiceError(RuntimeError):
    """Raised when the cross-pod OpenPI service contract fails."""


_PI05_NOISE_SERVER_PROGRAM = r"""
        import argparse, hashlib, hmac, os, re
        import numpy as np
        from openpi.policies import policy_config
        from openpi.serving import websocket_policy_server
        from openpi.training import config as training_config

        class ExplicitNoisePolicy:
            def __init__(self, policy, model_config):
                self.policy = policy
                self.shape = (int(model_config.action_horizon), int(model_config.action_dim))
                actual = (int(policy._model.action_horizon), int(policy._model.action_dim))
                if actual != self.shape:
                    raise RuntimeError(f"loaded policy/model noise shape mismatch: {actual} != {self.shape}")
                self.metadata = dict(policy.metadata)
                self.metadata["npa_explicit_noise_protocol"] = {
                    "schema": "npa.openpi.pi05-explicit-noise.v1",
                    "shape": list(self.shape), "dtype": "float32",
                }

            def infer(self, observation):
                observation = dict(observation)
                noise = observation.pop("npa/sampling_noise", None)
                sequence = observation.pop("npa/sampling_sequence", None)
                source_wrist = np.ascontiguousarray(observation["observation/wrist_image_left"])
                transformed = self.policy._input_transform(dict(observation))
                images, masks = transformed["image"], transformed["image_mask"]
                left = np.ascontiguousarray(images["left_wrist_0_rgb"])
                right = np.ascontiguousarray(images["right_wrist_0_rgb"])
                left_mask = bool(np.asarray(masks["left_wrist_0_rgb"]).all())
                right_mask = bool(np.asarray(masks["right_wrist_0_rgb"]).any())
                if not left_mask or right_mask or not left.any() or right.any():
                    raise RuntimeError("pinned policy transformed wrist mask/content contract failed")
                transform_receipt = {
                    "schema": "npa.openpi.pi05-live-wrist-transform.v1",
                    "source_wrist_sha256": hashlib.sha256(source_wrist.tobytes()).hexdigest(),
                    "left_wrist_0_rgb_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
                    "right_wrist_0_rgb_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
                    "left_wrist_mask": left_mask,
                    "right_wrist_mask": right_mask,
                }
                if noise is None and sequence is None:
                    result = self.policy.infer(observation)
                    result["npa_wrist_transform_receipt"] = transform_receipt
                    return result
                if noise is None or not isinstance(sequence, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,128}", sequence):
                    raise ValueError("explicit noise and a valid sequence identity are required together")
                noise = np.asarray(noise)
                if noise.dtype != np.float32 or noise.shape != self.shape or not np.isfinite(noise).all():
                    raise ValueError(f"noise must be finite float32{self.shape}, got {noise.dtype}{noise.shape}")
                digest = hashlib.sha256(np.ascontiguousarray(noise).tobytes()).hexdigest()
                result = self.policy.infer(observation, noise=noise)
                result["npa_sampler_noise_sha256"] = digest
                result["npa_sampler_noise_shape"] = list(noise.shape)
                result["npa_sampling_sequence"] = sequence
                result["npa_wrist_transform_receipt"] = transform_receipt
                return result

        class AuthenticatedNoiseServer(websocket_policy_server.WebsocketPolicyServer):
            def __init__(self, *args, auth_token, **kwargs):
                if not auth_token:
                    raise RuntimeError("OpenPI service auth token is empty")
                self._auth_token = auth_token
                super().__init__(*args, **kwargs)

            async def _handler(self, websocket):
                supplied = websocket.request.headers.get("Authorization", "")
                accepted = (
                    f"Bearer {self._auth_token}",
                    f"Api-Key {self._auth_token}",
                )
                if not any(hmac.compare_digest(supplied, value) for value in accepted):
                    await websocket.close(code=1008, reason="authentication required")
                    return
                await super()._handler(websocket)

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--port", type=int, required=True)
        args = parser.parse_args()
        auth_token = os.environ.get("NPA_OPENPI_SERVICE_AUTH_TOKEN", "")
        if not auth_token:
            raise RuntimeError("OpenPI service auth token is empty")
        selected = training_config.get_config(args.config)
        real_policy = policy_config.create_trained_policy(selected, args.checkpoint)
        wrapped = ExplicitNoisePolicy(real_policy, selected.model)
        AuthenticatedNoiseServer(
            policy=wrapped, host="0.0.0.0", port=args.port,
            metadata=wrapped.metadata, auth_token=auth_token,
        ).serve_forever()
        """

_GENERIC_SERVER_PROGRAM = r"""
        import argparse
        from openpi.policies import policy_config
        from openpi.serving import websocket_policy_server
        from openpi.training import config as training_config

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--port", type=int, required=True)
        args = parser.parse_args()
        selected = training_config.get_config(args.config)
        policy = policy_config.create_trained_policy(selected, args.checkpoint)
        websocket_policy_server.WebsocketPolicyServer(
            policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata,
        ).serve_forever()
        """


def _pi05_noise_server_program(*, adapted: bool = False) -> str:
    """Return the scoped wrapper that adds explicit-noise inference receipts."""

    program = textwrap.dedent(_PI05_NOISE_SERVER_PROGRAM).strip()
    if adapted:
        original = (
            "selected = training_config.get_config(args.config)\n"
            "real_policy = policy_config.create_trained_policy(selected, args.checkpoint)"
        )
        replacement = (
            "from npa.workflows.byof.openpi_selected_policy import load_selected_policy\n"
            "real_policy, selected = load_selected_policy(args.checkpoint)"
        )
        if program.count(original) != 1:
            raise OpenPIServiceError("adapted policy loader insertion contract changed")
        program = program.replace(original, replacement)
    return program


def _generic_server_program() -> str:
    """Return the unchanged unauthenticated upstream-compatible server path."""

    return textwrap.dedent(_GENERIC_SERVER_PROGRAM).strip()


def _public_checkpoint_verify_program(provenance: Mapping[str, object]) -> str:
    """Build an in-pod full-byte verifier for a generation-bound GCS manifest."""

    records = provenance.get("objects")
    if not isinstance(records, list) or not records:
        raise OpenPIServiceError("public checkpoint provenance lacks object records")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return f"""import base64,hashlib,json,os,pathlib
root=pathlib.Path(os.environ['NPA_RESOLVED_CHECKPOINT_DIR'])
expected=json.loads({encoded!r})
actual=[]
for path in sorted(item for item in root.rglob('*') if item.is_file()):
    md5=hashlib.md5(usedforsecurity=False)
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): md5.update(chunk)
    actual.append({{'path':path.relative_to(root).as_posix(),'size':path.stat().st_size,
                   'md5Hash':base64.b64encode(md5.digest()).decode('ascii')}})
comparable=[{{'path':row['path'],'size':row['size'],'md5Hash':row['md5Hash']}} for row in expected]
assert actual==comparable,'downloaded checkpoint differs from generation-bound GCS manifest'
"""


def _inject_public_checkpoint_binding(
    manifests: dict[str, dict[str, Any]], provenance: Mapping[str, object]
) -> None:
    container = manifests["deployment"]["spec"]["template"]["spec"]["containers"][0]
    manifest_sha256 = provenance.get("sha256")
    if not isinstance(manifest_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", manifest_sha256
    ):
        raise OpenPIServiceError(
            "public checkpoint provenance lacks a generation-manifest digest"
        )
    container["env"].append(
        {
            "name": "NPA_OPENPI_PUBLIC_CHECKPOINT_MANIFEST_SHA256",
            "value": manifest_sha256,
        }
    )
    shell = container["command"][2]
    marker = "exec /opt/venv/bin/python -c "
    if marker not in shell:
        raise OpenPIServiceError("unexpected OpenPI server exec contract")
    verifier = "/opt/venv/bin/python -c " + shlex.quote(
        _public_checkpoint_verify_program(provenance)
    )
    prefix = 'export NPA_RESOLVED_CHECKPOINT_DIR="$checkpoint_dir"; '
    container["command"][2] = shell.replace(marker, f"{prefix}{verifier}; {marker}", 1)


def _safe_name(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return f"npa-openpi-{digest}"


def controller_service_account_name(run_id: str) -> str:
    """Return the run-scoped ServiceAccount used by the Sky control task."""

    return f"{_safe_name(run_id)}-ctl"


def service_resource_names(run_id: str) -> dict[str, str]:
    """Return every deterministic Kubernetes identity owned by one service run."""

    base_name = _safe_name(run_id)
    return {
        "secret": f"{base_name}-runtime",
        "legacy_terms_secret": f"{base_name}-terms",
        "service": f"{base_name}-policy",
        "deployment": base_name,
        "client_job": f"{base_name}-client",
    }


def build_controller_rbac_manifests(
    *, run_id: str, namespace: str, service_account: str
) -> dict[str, dict[str, Any]]:
    """Build name-scoped RBAC with only unavoidable Kubernetes residual scope."""

    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", service_account):
        raise OpenPIServiceError(
            f"service account must be a Kubernetes DNS label: {service_account!r}"
        )
    if len(service_account) > 63:
        raise OpenPIServiceError(
            "service account exceeds the 63-character DNS-label limit"
        )
    owner_hash = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    metadata: dict[str, Any] = {
        "name": service_account,
        "namespace": namespace,
        "labels": {
            "app.kubernetes.io/managed-by": CONTROLLER_MANAGED_BY,
            "npa.nebius.ai/run-hash": owner_hash,
        },
        "annotations": {"npa.nebius.ai/cleanup-owner": run_id},
    }

    def fresh_metadata() -> dict[str, Any]:
        return {
            **metadata,
            "labels": dict(metadata["labels"]),
            "annotations": dict(metadata["annotations"]),
        }

    resource_names = service_resource_names(run_id)
    role_rules = [
        {
            "apiGroups": [""],
            "resources": ["pods"],
            # Deployment/Job pod names are controller-generated, and Kubernetes
            # RBAC cannot constrain list by label selector. This is the only
            # residual namespace-wide read; pod logs and pod deletion are not
            # granted.
            "verbs": ["list"],
        },
        {
            "apiGroups": [""],
            "resources": ["services", "secrets"],
            "verbs": ["create"],
        },
        {
            "apiGroups": [""],
            "resources": ["services"],
            "resourceNames": [resource_names["service"]],
            "verbs": ["get", "delete"],
        },
        {
            "apiGroups": [""],
            "resources": ["secrets"],
            "resourceNames": [
                resource_names["secret"],
                resource_names["legacy_terms_secret"],
            ],
            "verbs": ["get", "delete"],
        },
        {
            "apiGroups": ["apps"],
            "resources": ["deployments"],
            "verbs": ["create"],
        },
        {
            "apiGroups": ["apps"],
            "resources": ["deployments"],
            "resourceNames": [resource_names["deployment"]],
            "verbs": ["get", "delete"],
        },
        {
            "apiGroups": ["batch"],
            "resources": ["jobs"],
            "verbs": ["create"],
        },
        {
            "apiGroups": ["batch"],
            "resources": ["jobs"],
            "resourceNames": [resource_names["client_job"]],
            "verbs": ["get", "delete"],
        },
    ]
    return {
        "service_account": {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": fresh_metadata(),
        },
        "role": {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": fresh_metadata(),
            "rules": role_rules,
        },
        "role_binding": {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": fresh_metadata(),
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": service_account,
                    "namespace": namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": service_account,
            },
        },
    }


def _run_labels(run_id: str, name: str) -> dict[str, str]:
    return {
        **LABELS,
        "app.kubernetes.io/name": "openpi-policy",
        "npa.nebius.ai/run-hash": hashlib.sha256(run_id.encode("utf-8")).hexdigest()[
            :16
        ],
        "npa.nebius.ai/cleanup-owner": name,
    }


def _client_program() -> str:
    """Program executed in a separate CPU-only client pod."""

    return r"""
import json
import os
import time
import urllib.request

import numpy as np
from openpi_client import websocket_client_policy

host = os.environ["OPENPI_SERVICE_HOST"]
port = int(os.environ.get("OPENPI_SERVICE_PORT", "8000"))
frame = np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3)
observation = {
    "observation/exterior_image_1_left": np.asarray(frame % 251, dtype=np.uint8),
    "observation/wrist_image_left": np.asarray(np.flip(frame, axis=1) % 253, dtype=np.uint8),
    "observation/joint_position": np.asarray(
        [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
        dtype=np.float32,
    ),
    "observation/gripper_position": np.asarray([0.04], dtype=np.float32),
    "prompt": "pick up the fork",
}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(f"http://{host}:{port}/healthz") as response:
    health = response.read().decode("utf-8").strip()
token = os.environ.get("NPA_OPENPI_SERVICE_AUTH_TOKEN", "")
client = websocket_client_policy.WebsocketClientPolicy(
    host, port, api_key=token or None
)
requests = []
for index in range(2):
    started = time.perf_counter()
    response = client.infer(observation)
    actions = np.asarray(response["actions"])
    if actions.ndim != 2 or actions.shape[0] < 5 or actions.shape[1] != 8:
        raise RuntimeError(f"served trajectory {index} has invalid shape {actions.shape}")
    if actions.dtype != np.dtype("float64") or not np.isfinite(actions).all():
        raise RuntimeError(f"served trajectory {index} violates dtype/finiteness")
    requests.append({
        "request_index": index,
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "finite": True,
        "round_trip_ms": round((time.perf_counter() - started) * 1000, 3),
        "server_infer_ms": round(float(response["server_timing"]["infer_ms"]), 3),
        "first_five_targets_sha256": __import__("hashlib").sha256(
            actions[:5].tobytes(order="C")
        ).hexdigest(),
    })
result = {
    "schema": "npa.workbench.openpi.cross-pod-client.v1",
    "status": "passed",
    "healthz": health,
    "request_count": len(requests),
    "requests": requests,
}
encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
if len(encoded.encode("utf-8")) > 3900:
    raise RuntimeError("client evidence exceeds Kubernetes termination-message limit")
with open("/dev/termination-log", "w", encoding="utf-8") as stream:
    stream.write(encoded)
print("NPA_OPENPI_CLIENT_RESULT=" + encoded, flush=True)
""".strip()


def _server_hardware_program() -> str:
    """Self-contained GPU proof for a server image that does not embed NPA."""

    return (
        standalone_program()
        + "\n"
        + r"""
import importlib.metadata
import json
import os
import re
import subprocess

import jax
from jax.extend import backend as jax_backend

expected_type = os.environ["OPENPI_EXPECTED_GPU_TYPE"]
expected_count = int(os.environ["OPENPI_EXPECTED_GPU_COUNT"])
expected_capability = os.environ["OPENPI_EXPECTED_COMPUTE_CAPABILITY"]
devices = jax.devices()
if len(devices) != expected_count:
    raise RuntimeError(f"expected {expected_count} visible GPU(s), got {len(devices)}")
kinds = [str(getattr(device, "device_kind", "")) for device in devices]
if expected_type and any(expected_type.upper() not in item.upper() for item in kinds):
    raise RuntimeError(f"expected GPU type {expected_type!r}, got {kinds!r}")
capabilities = []
for device in devices:
    raw = getattr(device, "compute_capability", "")
    raw = raw() if callable(raw) else raw
    major = getattr(raw, "major", None)
    minor = getattr(raw, "minor", None)
    if major is not None and minor is not None:
        value = f"{int(major)}.{int(minor)}"
    else:
        match = re.search(r"(\d{1,2})[.,](\d)", str(raw))
        if match:
            value = f"{int(match.group(1))}.{int(match.group(2))}"
        else:
            compact = re.search(r"(?:sm_)?(\d{2,3})", str(raw), re.IGNORECASE)
            if not compact:
                raise RuntimeError(f"cannot parse compute capability {raw!r}")
            number = int(compact.group(1))
            value = f"{number // 10}.{number % 10}"
    capabilities.append(value)
if expected_capability and any(value != expected_capability for value in capabilities):
    raise RuntimeError(
        f"expected compute capability {expected_capability}, got {capabilities}"
    )
sm100_probe = {"required": expected_capability == "10.0"}
if sm100_probe["required"]:
    probe = "/usr/local/bin/npa-openpi-sm100-probe"
    output = subprocess.run(
        [probe], check=True, capture_output=True, text=True
    ).stdout.strip()
    elf = subprocess.run(
        ["cuobjdump", "--list-elf", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    if "sm_100" not in (elf.stdout + elf.stderr).lower():
        raise RuntimeError("CUDA probe has no sm_100 ELF")
    sm100_probe.update({"passed": True, "output": output, "elf_contains_sm100": True})
nvidia_smi = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip().splitlines()
if len(nvidia_smi) != expected_count:
    raise RuntimeError("nvidia-smi visible GPU count disagrees with JAX")
result = {
    "gpu_count_allocated": len(devices),
    "device_kinds": kinds,
    "compute_capabilities": capabilities,
    "nvidia_smi": nvidia_smi,
    "jax": jax.__version__,
    "jaxlib": importlib.metadata.version("jaxlib"),
    "xla_platform_version": str(jax_backend.get_backend().platform_version),
    "sm100_probe": sm100_probe,
}
encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
with diagnostic_path("npa-openpi-server-hardware.json").open("w", encoding="utf-8") as stream:
    stream.write(encoded)
print("NPA_OPENPI_SERVER_HARDWARE=" + encoded, flush=True)
""".strip()
    )


def _redirect_policy_cache_to_durable_storage(deployment: dict[str, Any]) -> None:
    """Send the policy checkpoint to the operator's weight cache when there is one.

    `OPENPI_DATA_HOME` is where the runtime-fetched checkpoint lands, and it was
    backed by an ``emptyDir``: a Deployment replaces its pod on every rollout, image
    change and node drain, so the download was paid again each time on a GPU that is
    already running. With a claim configured it moves to the shared cache like every
    other runtime-fetched weight; without one the ephemeral volume stays exactly as
    it was.
    """

    from npa.workbench.model_cache import (
        RUNTIME_KUBERNETES,
        model_cache_env,
        model_cache_host_path,
        model_cache_pvc,
        pod_config_with_model_cache,
        resolve_model_cache_root,
    )

    root = resolve_model_cache_root(runtime=RUNTIME_KUBERNETES)
    claim = model_cache_pvc()
    host_path = model_cache_host_path()
    if not root or not (claim or host_path):
        return

    pod_spec = deployment["spec"]["template"]["spec"]
    patched = pod_config_with_model_cache(
        {"spec": pod_spec},
        root=root,
        pvc=claim,
        host_path=host_path,
        container_names=("openpi-policy",),
    )
    pod_spec.update(patched["spec"])
    cache_env = model_cache_env(root)
    for container in pod_spec["containers"]:
        if container.get("name") != "openpi-policy":
            continue
        env = [
            item
            for item in container.get("env", [])
            if item.get("name") not in cache_env
        ]
        env.extend(
            {"name": key, "value": value} for key, value in sorted(cache_env.items())
        )
        container["env"] = env


def _validate_manifest_inputs(
    runtime_image: str, auth_secret: str, gpu_count: int
) -> None:
    if not RUNTIME_IMAGE_RE.fullmatch(runtime_image):
        raise OpenPIServiceError("service runtime image must be digest-pinned")
    valid_secret = re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", auth_secret)
    if auth_secret and (len(auth_secret) > 63 or not valid_secret):
        raise OpenPIServiceError(
            f"auth secret must be a Kubernetes DNS label: {auth_secret!r}"
        )
    if gpu_count < 1:
        raise OpenPIServiceError("OpenPI policy server requires at least one GPU")


def _policy_server_shell(config_name: str, auth_secret: str) -> str:
    hardware_probe = "/opt/venv/bin/python -c " + shlex.quote(
        _server_hardware_program()
    )
    program = _pi05_noise_server_program() if auth_secret else _generic_server_program()
    diagnostics_probe = "/opt/venv/bin/python -c " + shlex.quote(
        standalone_program() + "\nprint(diagnostics_directory())"
    )
    return (
        "set -euo pipefail; "
        f"diagnostics_dir=$({diagnostics_probe}); "
        f"{hardware_probe}; "
        f"/opt/venv/bin/python -m http.server {SERVER_DIAGNOSTICS_PORT} "
        '--bind 0.0.0.0 --directory "$diagnostics_dir" >/dev/null 2>&1 & '
        "checkpoint_dir=$(/opt/venv/bin/python -c 'import os; "
        "from openpi.shared import download; "
        'print(download.maybe_download(os.environ["OPENPI_CHECKPOINT_URI"], token="anon"))\'); '
        "exec /opt/venv/bin/python -c "
        f"{shlex.quote(program)} --port={SERVER_PORT} "
        f'--config={shlex.quote(config_name)} --checkpoint="$checkpoint_dir"'
    )


def _policy_container(options: Mapping[str, Any], server_shell: str) -> dict[str, Any]:
    return {
        "name": "openpi-policy",
        "image": options["runtime_image"],
        "imagePullPolicy": "IfNotPresent",
        "command": ["/bin/bash", "-lc", server_shell],
        "ports": [
            {"name": "websocket", "containerPort": SERVER_PORT},
            {"name": "diagnostics", "containerPort": SERVER_DIAGNOSTICS_PORT},
        ],
        "env": _policy_environment(options),
        "resources": _policy_resources(options),
        "volumeMounts": [
            {"name": "openpi-cache", "mountPath": "/workspace/openpi-server-cache"}
        ],
        "readinessProbe": {
            "httpGet": {"path": "/healthz", "port": SERVER_PORT},
            "periodSeconds": 5,
            "failureThreshold": 240,
        },
        "livenessProbe": {
            "httpGet": {"path": "/healthz", "port": SERVER_PORT},
            "initialDelaySeconds": options["liveness_delay"],
            "periodSeconds": 30,
            "failureThreshold": 6,
        },
    }


def _policy_environment(options: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"name": "OPENPI_DATA_HOME", "value": "/workspace/openpi-server-cache"},
        {"name": "OPENPI_CHECKPOINT_URI", "value": options["checkpoint_uri"]},
        {"name": "OPENPI_EXPECTED_GPU_TYPE", "value": options["expected_gpu_type"]},
        {"name": "OPENPI_EXPECTED_GPU_COUNT", "value": str(options["gpu_count"])},
        {
            "name": "OPENPI_EXPECTED_COMPUTE_CAPABILITY",
            "value": options["expected_compute_capability"],
        },
    ]


def _policy_resources(options: Mapping[str, Any]) -> dict[str, Any]:
    gpu_count = str(options["gpu_count"])
    return {
        "requests": {
            "cpu": options["server_cpu"],
            "memory": options["server_memory"],
            "nvidia.com/gpu": gpu_count,
        },
        "limits": {"nvidia.com/gpu": gpu_count},
    }


def _deployment_manifest(options: Mapping[str, Any]) -> dict[str, Any]:
    labels, selector = options["labels"], options["selector"]
    pod_spec = {
        "restartPolicy": "Always",
        "imagePullSecrets": options["pull_secrets"],
        "nodeSelector": options["node_selector"],
        "containers": [_policy_container(options, options["server_shell"])],
        "volumes": [
            {"name": "openpi-cache", "emptyDir": {"sizeLimit": options["cache_size"]}}
        ],
    }
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": options["name"],
            "namespace": options["namespace"],
            "labels": labels,
        },
        "spec": {
            "replicas": 1,
            "progressDeadlineSeconds": math.ceil(options["server_timeout"]),
            "selector": {"matchLabels": selector},
            "template": {
                "metadata": {"labels": {**labels, **selector}},
                "spec": pod_spec,
            },
        },
    }


def _service_manifest(options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": options["service_name"],
            "namespace": options["namespace"],
            "labels": options["labels"],
        },
        "spec": {
            "type": "ClusterIP",
            "selector": options["selector"],
            "ports": [
                {"name": "websocket", "port": SERVER_PORT, "targetPort": SERVER_PORT},
                {
                    "name": "diagnostics",
                    "port": SERVER_DIAGNOSTICS_PORT,
                    "targetPort": SERVER_DIAGNOSTICS_PORT,
                },
            ],
        },
    }


def _client_manifest(options: Mapping[str, Any]) -> dict[str, Any]:
    container = _client_container(options)
    pod_spec = {
        "restartPolicy": "Never",
        "imagePullSecrets": options["pull_secrets"],
        "containers": [container],
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": options["client_name"],
            "namespace": options["namespace"],
            "labels": options["labels"],
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": math.ceil(options["client_timeout"]),
            "template": {
                "metadata": {
                    "labels": {**options["labels"], "app": options["client_name"]}
                },
                "spec": pod_spec,
            },
        },
    }


def _client_container(options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": "openpi-client",
        "image": options["runtime_image"],
        "imagePullPolicy": "IfNotPresent",
        "command": ["/opt/venv/bin/python", "-c", _client_program()],
        "env": [
            {"name": "OPENPI_SERVICE_HOST", "value": options["service_name"]},
            {"name": "OPENPI_SERVICE_PORT", "value": str(SERVER_PORT)},
        ],
        "resources": {
            "requests": {
                "cpu": options["client_cpu"],
                "memory": options["client_memory"],
            }
        },
    }


def _manifest_options(
    values: Mapping[str, Any], names: Mapping[str, str]
) -> dict[str, Any]:
    name = names["deployment"]
    labels = _run_labels(str(values["run_id"]), name)
    key, value = values["gpu_node_selector_key"], values["gpu_node_selector_value"]
    return dict(values) | {
        "name": name,
        "service_name": names["service"],
        "client_name": names["client_job"],
        "labels": labels,
        "selector": {"npa.nebius.ai/cleanup-owner": name, "app": name},
        "pull_secrets": [{"name": values["pull_secret"]}]
        if values["pull_secret"]
        else [],
        "node_selector": {key: value} if key and value else {},
        "liveness_delay": values["liveness_initial_delay_seconds"],
        "server_shell": _policy_server_shell(
            str(values["config_name"]), str(values["auth_secret"])
        ),
    }


def _assembled_manifests(
    options: Mapping[str, Any], auth_secret: str
) -> dict[str, dict[str, Any]]:
    deployment = _deployment_manifest(options)
    service = _service_manifest(options)
    client_job = _client_manifest(options)
    _inject_auth_reference(deployment, auth_secret)
    _inject_auth_reference(client_job, auth_secret)
    _redirect_policy_cache_to_durable_storage(deployment)
    return {"deployment": deployment, "service": service, "client_job": client_job}


def _inject_auth_reference(manifest: dict[str, Any], auth_secret: str) -> None:
    if not auth_secret:
        return
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    container["env"].append(
        {
            "name": "NPA_OPENPI_SERVICE_AUTH_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": auth_secret, "key": "auth_token"}},
        }
    )


def build_manifests(
    *,
    run_id: str,
    namespace: str,
    runtime_image: str,
    checkpoint_uri: str,
    config_name: str,
    gpu_count: int,
    expected_gpu_type: str,
    expected_compute_capability: str,
    server_cpu: str,
    server_memory: str,
    client_cpu: str,
    client_memory: str,
    pull_secret: str,
    liveness_initial_delay_seconds: int,
    gpu_node_selector_key: str,
    gpu_node_selector_value: str,
    cache_size: str,
    auth_secret: str = "",
    server_ready_timeout_seconds: float = DEFAULT_SERVER_READY_TIMEOUT_SECONDS,
    client_timeout_seconds: float = DEFAULT_CLIENT_TIMEOUT_SECONDS,
) -> dict[str, dict[str, Any]]:
    """Build the exact Deployment/ClusterIP/Job objects as testable data."""
    _validate_manifest_inputs(runtime_image, auth_secret, gpu_count)
    server_timeout = _validated_positive_seconds(
        server_ready_timeout_seconds, "server ready timeout"
    )
    client_timeout = _validated_positive_seconds(
        client_timeout_seconds, "client timeout"
    )
    names = service_resource_names(run_id)
    options = _manifest_options(
        locals() | {"server_timeout": server_timeout, "client_timeout": client_timeout},
        names,
    )
    return _assembled_manifests(options, auth_secret)


def _api_exception_status(exc: Exception) -> int | None:
    return getattr(exc, "status", None)


def _validated_positive_seconds(value: float, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OpenPIServiceError(f"{label} must be a positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise OpenPIServiceError(f"{label} must be positive, got {value!r}")
    return parsed


def _api_call(
    function: Callable[..., Any],
    *args: object,
    request_timeout: float,
    **kwargs: object,
) -> Any:
    """Bound Kubernetes API calls; tolerate minimal unit-test doubles."""

    try:
        return function(
            *args,
            **kwargs,
            _request_timeout=_validated_positive_seconds(
                request_timeout, "Kubernetes API timeout"
            ),
        )
    except TypeError as exc:
        if "_request_timeout" not in str(exc):
            raise
        return function(*args, **kwargs)


def _assert_created_object_owned(
    obj: object, manifest: Mapping[str, Any], *, kind: str
) -> None:
    metadata = getattr(obj, "metadata", None)
    actual_labels = getattr(metadata, "labels", None) or {}
    expected_labels = manifest["metadata"].get("labels", {})
    mismatches = {
        key: {"expected": value, "actual": actual_labels.get(key)}
        for key, value in expected_labels.items()
        if actual_labels.get(key) != value
    }
    if mismatches:
        name = manifest["metadata"]["name"]
        raise OpenPIServiceError(
            f"refusing to adopt uncertain {kind} creation for {name!r}: "
            f"ownership labels differ: {mismatches}"
        )


def _create_exact_object(
    creator: Callable[..., Any],
    reader: Callable[..., Any],
    *,
    namespace: str,
    manifest: Mapping[str, Any],
    kind: str,
    tracker: set[str],
    request_timeout: float,
) -> None:
    name = str(manifest["metadata"]["name"])
    try:
        _api_call(
            creator,
            namespace,
            manifest,
            request_timeout=request_timeout,
        )
    except Exception as create_exc:
        # The request may have reached the API server even when its response was
        # lost. Mark the deterministic identity for ownership-checked cleanup.
        tracker.add(kind)
        try:
            existing = _api_call(
                reader,
                name,
                namespace,
                request_timeout=request_timeout,
            )
        except Exception as read_exc:
            if _api_exception_status(read_exc) == 404:
                raise create_exc
            raise OpenPIServiceError(
                f"{kind} {name!r} creation is uncertain and exact identity "
                f"could not be read for cleanup: {read_exc}"
            ) from create_exc
        _assert_created_object_owned(existing, manifest, kind=kind)
        raise create_exc
    tracker.add(kind)


def _create_server_objects(
    api: Any,
    apps: Any,
    manifests: Mapping[str, dict[str, Any]],
    *,
    created: set[str] | None = None,
    request_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
) -> None:
    tracker = created if created is not None else set()
    namespace = manifests["deployment"]["metadata"]["namespace"]
    if "secret" in manifests:
        _create_exact_object(
            api.create_namespaced_secret,
            api.read_namespaced_secret,
            namespace=namespace,
            manifest=manifests["secret"],
            kind="secret",
            tracker=tracker,
            request_timeout=request_timeout,
        )
    _create_exact_object(
        apps.create_namespaced_deployment,
        apps.read_namespaced_deployment,
        namespace=namespace,
        manifest=manifests["deployment"],
        kind="deployment",
        tracker=tracker,
        request_timeout=request_timeout,
    )
    _create_exact_object(
        api.create_namespaced_service,
        api.read_namespaced_service,
        namespace=namespace,
        manifest=manifests["service"],
        kind="service",
        tracker=tracker,
        request_timeout=request_timeout,
    )


def _create_client_job(
    batch: Any,
    manifests: Mapping[str, dict[str, Any]],
    *,
    created: set[str] | None = None,
    request_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
) -> None:
    tracker = created if created is not None else set()
    namespace = manifests["client_job"]["metadata"]["namespace"]
    _create_exact_object(
        batch.create_namespaced_job,
        batch.read_namespaced_job,
        namespace=namespace,
        manifest=manifests["client_job"],
        kind="client_job",
        tracker=tracker,
        request_timeout=request_timeout,
    )


def _assert_targets_absent(
    api: Any,
    apps: Any,
    batch: Any,
    *,
    namespace: str,
    names: Mapping[str, str],
    request_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
) -> None:
    """Fail closed before create if any deterministic target name already exists."""

    readers = (
        ("client_job", batch.read_namespaced_job),
        ("deployment", apps.read_namespaced_deployment),
        ("service", api.read_namespaced_service),
        ("secret", api.read_namespaced_secret),
        ("legacy_terms_secret", api.read_namespaced_secret),
    )
    for key, reader in readers:
        if key not in names:
            continue
        try:
            _api_call(
                reader,
                names[key],
                namespace,
                request_timeout=request_timeout,
            )
        except Exception as exc:
            if _api_exception_status(exc) == 404:
                continue
            raise
        raise OpenPIServiceError(
            f"refusing to create or clean pre-existing {key} {names[key]!r}; "
            "run ownership is not proven"
        )
    pod_selector = f"npa.nebius.ai/cleanup-owner={names['deployment']}"
    pods = _api_call(
        api.list_namespaced_pod,
        namespace,
        label_selector=pod_selector,
        request_timeout=request_timeout,
    ).items
    if pods:
        raise OpenPIServiceError(
            "refusing to create or clean pre-existing pods with the deterministic "
            f"run selector: {[pod.metadata.name for pod in pods]}"
        )


def _pod_state(pod: Any) -> dict[str, object]:
    metadata = getattr(pod, "metadata", None)
    status = getattr(pod, "status", None)
    conditions = []
    for condition in getattr(status, "conditions", None) or []:
        conditions.append(
            {
                "type": str(getattr(condition, "type", "")),
                "status": str(getattr(condition, "status", "")),
                "reason": str(getattr(condition, "reason", "") or ""),
                "message": str(getattr(condition, "message", "") or "")[-2000:],
            }
        )
    containers = []
    for container in getattr(status, "container_statuses", None) or []:
        state = getattr(container, "state", None)
        waiting = getattr(state, "waiting", None)
        terminated = getattr(state, "terminated", None)
        containers.append(
            {
                "name": str(getattr(container, "name", "")),
                "ready": bool(getattr(container, "ready", False)),
                "restart_count": int(getattr(container, "restart_count", 0) or 0),
                "waiting_reason": str(getattr(waiting, "reason", "") or ""),
                "waiting_message": str(getattr(waiting, "message", "") or "")[-2000:],
                "terminated_reason": str(getattr(terminated, "reason", "") or ""),
                "terminated_message": str(getattr(terminated, "message", "") or "")[
                    -2000:
                ],
                "exit_code": getattr(terminated, "exit_code", None),
            }
        )
    return {
        "name": str(getattr(metadata, "name", "")),
        "phase": str(getattr(status, "phase", "") or ""),
        "conditions": conditions,
        "containers": containers,
    }


def _terminal_pod_problem(pod: Any) -> str:
    state = _pod_state(pod)
    if state["phase"] == "Failed":
        return f"pod entered Failed: {json.dumps(state, sort_keys=True)}"
    for condition in state["conditions"]:  # type: ignore[union-attr]
        if (
            condition["type"] == "PodScheduled"
            and condition["status"] == "False"
            and condition["reason"] == "Unschedulable"
        ):
            return f"pod is Unschedulable: {json.dumps(state, sort_keys=True)}"
    for container in state["containers"]:  # type: ignore[union-attr]
        if container["waiting_reason"] in _TERMINAL_WAITING_REASONS:
            return (
                f"pod container is {container['waiting_reason']}: "
                f"{json.dumps(state, sort_keys=True)}"
            )
        exit_code = container["exit_code"]
        if exit_code not in (None, 0):
            return (
                f"pod container exited {exit_code}: {json.dumps(state, sort_keys=True)}"
            )
    return ""


def _deployment_state(deployment: Any, pods: Sequence[Any]) -> dict[str, object]:
    status = getattr(deployment, "status", None)
    return {
        "available_replicas": int(getattr(status, "available_replicas", 0) or 0),
        "ready_replicas": int(getattr(status, "ready_replicas", 0) or 0),
        "unavailable_replicas": int(getattr(status, "unavailable_replicas", 0) or 0),
        "conditions": [
            {
                "type": str(getattr(condition, "type", "")),
                "status": str(getattr(condition, "status", "")),
                "reason": str(getattr(condition, "reason", "") or ""),
                "message": str(getattr(condition, "message", "") or "")[-2000:],
            }
            for condition in getattr(status, "conditions", None) or []
        ],
        "pods": [_pod_state(pod) for pod in pods],
    }


def _read_server_state(
    api: Any, apps: Any, namespace: str, name: str, request_timeout: float
) -> tuple[Any, Sequence[Any], dict[str, object]]:
    deployment = _api_call(
        apps.read_namespaced_deployment,
        name,
        namespace,
        request_timeout=request_timeout,
    )
    pods = _api_call(
        api.list_namespaced_pod,
        namespace,
        label_selector=f"npa.nebius.ai/cleanup-owner={name},app={name}",
        request_timeout=request_timeout,
    ).items
    return deployment, pods, _deployment_state(deployment, pods)


def _ready_server_pod(
    deployment: Any, pods: Sequence[Any], state: Mapping[str, object]
) -> Any | None:
    if int(deployment.status.available_replicas or 0) < 1:
        return None
    ready = [
        pod
        for pod in pods
        if pod.status.phase == "Running"
        and any(
            condition.type == "Ready" and condition.status == "True"
            for condition in pod.status.conditions or []
        )
    ]
    if len(ready) > 1:
        raise OpenPIServiceError(f"server Deployment has multiple ready pods: {state}")
    return ready[0] if ready else None


def _validate_server_progress(state: Mapping[str, object], pods: Sequence[Any]) -> None:
    for condition in state.get("conditions", []):
        if (
            condition.get("reason") == "ProgressDeadlineExceeded"
            and condition.get("status") == "False"
        ):
            raise OpenPIServiceError(
                "server Deployment exceeded its progress deadline: "
                f"{json.dumps(state, sort_keys=True)}"
            )
    for pod in pods:
        problem = _terminal_pod_problem(pod)
        if problem:
            raise OpenPIServiceError(f"server {problem}")


def _service_endpoint_ready(service_dns: str, http_timeout: float) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(
            f"http://{service_dns}:{SERVER_PORT}/healthz", timeout=http_timeout
        ) as response:
            health = response.read().decode("utf-8").strip()
    except urllib.error.URLError as error:
        # Pod readiness can precede the Service backend's routing update.
        if isinstance(error.reason, ConnectionRefusedError):
            return False
        raise
    if health != "OK":
        raise OpenPIServiceError(f"unexpected service health response {health!r}")
    return True


def _wait_server_ready(
    api: Any,
    apps: Any,
    namespace: str,
    name: str,
    *,
    timeout: float = DEFAULT_SERVER_READY_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    request_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
    health_check: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    timeout = _validated_positive_seconds(timeout, "server ready timeout")
    poll_interval = _validated_positive_seconds(poll_interval, "poll interval")
    deadline = clock() + timeout
    last_state: dict[str, object] = {}
    while clock() < deadline:
        deployment, pods, last_state = _read_server_state(
            api, apps, namespace, name, request_timeout
        )
        server = _ready_server_pod(deployment, pods, last_state)
        if server is not None:
            endpoint_ready = health_check is None or health_check()
            if endpoint_ready:
                return server
            last_state["service_endpoint_ready"] = False
        _validate_server_progress(last_state, pods)
        sleep(min(poll_interval, max(0.0, deadline - clock())))
    raise OpenPIServiceError(
        f"server readiness timed out after {timeout:g}s; last_state="
        f"{json.dumps(last_state, sort_keys=True)}"
    )


def _wait_client(
    api: Any,
    batch: Any,
    namespace: str,
    name: str,
    *,
    timeout: float = DEFAULT_CLIENT_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    request_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    timeout = _validated_positive_seconds(timeout, "client timeout")
    poll_interval = _validated_positive_seconds(poll_interval, "poll interval")
    deadline = clock() + timeout
    last_state: dict[str, object] = {}
    while clock() < deadline:
        job = _api_call(
            batch.read_namespaced_job,
            name,
            namespace,
            request_timeout=request_timeout,
        )
        pods = _api_call(
            api.list_namespaced_pod,
            namespace,
            label_selector=f"job-name={name}",
            request_timeout=request_timeout,
        ).items
        status = getattr(job, "status", None)
        last_state = {
            "active": int(getattr(status, "active", 0) or 0),
            "succeeded": int(getattr(status, "succeeded", 0) or 0),
            "failed": int(getattr(status, "failed", 0) or 0),
            "conditions": [
                {
                    "type": str(getattr(condition, "type", "")),
                    "status": str(getattr(condition, "status", "")),
                    "reason": str(getattr(condition, "reason", "") or ""),
                    "message": str(getattr(condition, "message", "") or "")[-2000:],
                }
                for condition in getattr(status, "conditions", None) or []
            ],
            "pods": [_pod_state(pod) for pod in pods],
        }
        if int(job.status.succeeded or 0) == 1 and len(pods) == 1:
            return pods[0]
        if int(job.status.failed or 0) > 0:
            if pods:
                raise OpenPIServiceError(
                    f"client job failed: {json.dumps(last_state, sort_keys=True)}"
                )
            raise OpenPIServiceError(f"client job {name} failed without a pod")
        for pod in pods:
            problem = _terminal_pod_problem(pod)
            if problem:
                raise OpenPIServiceError(f"client {problem}")
        sleep(min(poll_interval, max(0.0, deadline - clock())))
    raise OpenPIServiceError(
        f"client completion timed out after {timeout:g}s; last_state="
        f"{json.dumps(last_state, sort_keys=True)}"
    )


def _client_result_from_termination(pod: Any) -> dict[str, object]:
    status = getattr(pod, "status", None)
    for container in getattr(status, "container_statuses", None) or []:
        if str(getattr(container, "name", "")) != "openpi-client":
            continue
        terminated = getattr(getattr(container, "state", None), "terminated", None)
        message = str(getattr(terminated, "message", "") or "")
        if not message:
            break
        try:
            value = json.loads(message)
        except json.JSONDecodeError as exc:
            raise OpenPIServiceError(
                f"client termination evidence is invalid JSON: {message[-2000:]}"
            ) from exc
        if isinstance(value, dict):
            return value
    raise OpenPIServiceError("client pod has no termination-message evidence")


def _prefixed_json(text: str, prefix: str) -> dict[str, object]:
    for line in text.splitlines():
        if prefix not in line:
            continue
        payload = line.split(prefix, 1)[1].strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OpenPIServiceError(
                f"invalid {prefix.rstrip('=')} evidence: {payload[-2000:]}"
            ) from exc
        if isinstance(value, dict):
            return value
    raise OpenPIServiceError(f"server logs contain no {prefix.rstrip('=')} evidence")


def _delete_and_verify(
    api: Any,
    apps: Any,
    batch: Any,
    *,
    namespace: str,
    names: Mapping[str, str],
    created: set[str] | None = None,
    manifests: Mapping[str, Mapping[str, Any]] | None = None,
    timeout: float = DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    request_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, bool]:
    timeout = _validated_positive_seconds(timeout, "cleanup timeout")
    poll_interval = _validated_positive_seconds(poll_interval, "poll interval")
    deadline = clock() + timeout
    created_keys = set(names) if created is None else set(created)
    pod_selector = f"npa.nebius.ai/cleanup-owner={names['deployment']}"
    initial_pods = _api_call(
        api.list_namespaced_pod,
        namespace,
        label_selector=pod_selector,
        request_timeout=request_timeout,
    ).items
    for pod in initial_pods:
        labels = pod.metadata.labels or {}
        if (
            labels.get("app.kubernetes.io/managed-by") != "npa"
            or labels.get("app.kubernetes.io/part-of") != "openpi-four-mode"
            or labels.get("npa.nebius.ai/cleanup-owner") != names["deployment"]
        ):
            raise OpenPIServiceError(
                f"refusing cleanup because pod {pod.metadata.name!r} does not "
                "prove exact OpenPI service ownership"
            )
        pod_app = labels.get("app")
        if pod_app == names["deployment"] and "deployment" not in created_keys:
            raise OpenPIServiceError(
                f"refusing cleanup because server pod {pod.metadata.name!r} has no "
                "controller-created Deployment"
            )
        if pod_app == names["client_job"] and "client_job" not in created_keys:
            raise OpenPIServiceError(
                f"refusing cleanup because client pod {pod.metadata.name!r} has no "
                "controller-created Job"
            )
        if pod_app not in {names["deployment"], names["client_job"]}:
            raise OpenPIServiceError(
                f"refusing cleanup because pod {pod.metadata.name!r} has an "
                f"unexpected app identity {pod_app!r}"
            )
    delete_options = {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "propagationPolicy": "Foreground",
    }
    operation_types = (
        ("client_job", batch.delete_namespaced_job),
        ("deployment", apps.delete_namespaced_deployment),
        ("service", api.delete_namespaced_service),
        ("secret", api.delete_namespaced_secret),
        ("legacy_terms_secret", api.delete_namespaced_secret),
    )
    operations = tuple(
        (key, function, names[key]) for key, function in operation_types if key in names
    )
    issues: list[str] = []
    readers_by_key = {
        "client_job": batch.read_namespaced_job,
        "deployment": apps.read_namespaced_deployment,
        "service": api.read_namespaced_service,
        "secret": api.read_namespaced_secret,
        "legacy_terms_secret": api.read_namespaced_secret,
    }
    deletable_keys = set(created_keys)
    if manifests is not None:
        for key in created_keys:
            if key not in readers_by_key or key not in manifests:
                continue
            try:
                obj = _api_call(
                    readers_by_key[key],
                    names[key],
                    namespace,
                    request_timeout=request_timeout,
                )
            except Exception as exc:
                if _api_exception_status(exc) == 404:
                    continue
                issues.append(f"{key} ownership verification is uncertain: {exc}")
                deletable_keys.discard(key)
                continue
            try:
                _assert_created_object_owned(obj, manifests[key], kind=key)
            except OpenPIServiceError as exc:
                issues.append(str(exc))
                deletable_keys.discard(key)
    for key, function, name in operations:
        if key not in deletable_keys:
            continue
        try:
            _api_call(
                function,
                name,
                namespace,
                body=delete_options,
                request_timeout=request_timeout,
            )
        except Exception as exc:
            if _api_exception_status(exc) != 404:
                issues.append(f"{key} delete request failed: {exc}")
    readers = (
        ("client_job", batch.read_namespaced_job),
        ("deployment", apps.read_namespaced_deployment),
        ("service", api.read_namespaced_service),
        ("secret", api.read_namespaced_secret),
        ("legacy_terms_secret", api.read_namespaced_secret),
    )
    verified: dict[str, bool] = {}
    for key, reader in readers:
        if key not in names:
            continue
        if key not in created_keys:
            try:
                _api_call(
                    reader,
                    names[key],
                    namespace,
                    request_timeout=request_timeout,
                )
            except Exception as exc:
                if _api_exception_status(exc) == 404:
                    verified[key] = True
                    continue
                issues.append(f"{key} absence is uncertain: {exc}")
                continue
            issues.append(f"unowned {key} {names[key]!r} appeared during cleanup")
            continue
        last_state: dict[str, object] = {}
        while clock() < deadline:
            try:
                obj = _api_call(
                    reader,
                    names[key],
                    namespace,
                    request_timeout=request_timeout,
                )
            except Exception as exc:
                if _api_exception_status(exc) == 404:
                    verified[key] = True
                    break
                issues.append(f"{key} absence is uncertain: {exc}")
                break
            metadata = getattr(obj, "metadata", None)
            last_state = {
                "deletion_timestamp": str(
                    getattr(metadata, "deletion_timestamp", "") or ""
                ),
                "finalizers": list(getattr(metadata, "finalizers", None) or []),
            }
            sleep(min(poll_interval, max(0.0, deadline - clock())))
        if key not in verified and not any(
            issue.startswith(f"{key} absence is uncertain") for issue in issues
        ):
            issues.append(
                f"{key} deletion timed out after {timeout:g}s; "
                f"last_state={json.dumps(last_state, sort_keys=True)}"
            )

    pods: Sequence[Any] = []
    while clock() < deadline:
        try:
            pods = _api_call(
                api.list_namespaced_pod,
                namespace,
                label_selector=pod_selector,
                request_timeout=request_timeout,
            ).items
        except Exception as exc:
            issues.append(f"pod absence is uncertain: {exc}")
            break
        if not pods:
            verified["pods"] = True
            break
        for pod in pods:
            labels = pod.metadata.labels or {}
            if (
                labels.get("app.kubernetes.io/managed-by") != "npa"
                or labels.get("app.kubernetes.io/part-of") != "openpi-four-mode"
                or labels.get("npa.nebius.ai/cleanup-owner") != names["deployment"]
                or labels.get("app") not in {names["deployment"], names["client_job"]}
            ):
                issues.append(
                    f"foreign pod {pod.metadata.name!r} matched the cleanup selector"
                )
                break
        if issues and issues[-1].startswith("foreign pod"):
            break
        sleep(min(poll_interval, max(0.0, deadline - clock())))
    if "pods" not in verified and not any(
        issue.startswith("pod absence is uncertain") or issue.startswith("foreign pod")
        for issue in issues
    ):
        issues.append(
            f"pod deletion timed out after {timeout:g}s; last_state="
            f"{json.dumps([_pod_state(pod) for pod in pods], sort_keys=True)}"
        )
    if issues:
        raise OpenPIServiceError("exact cleanup incomplete: " + "; ".join(issues))
    return verified


def _service_timeouts(args: argparse.Namespace) -> dict[str, float]:
    fields = {
        "server": (args.server_ready_timeout_seconds, "server ready timeout"),
        "client": (args.client_timeout_seconds, "client timeout"),
        "cleanup": (args.cleanup_timeout_seconds, "cleanup timeout"),
        "poll": (args.poll_interval_seconds, "poll interval"),
        "api": (args.api_timeout_seconds, "Kubernetes API timeout"),
        "http": (args.http_timeout_seconds, "HTTP timeout"),
    }
    return {
        name: _validated_positive_seconds(value, label)
        for name, (value, label) in fields.items()
    }


def _service_manifests(
    args: argparse.Namespace, timeouts: Mapping[str, float]
) -> dict[str, dict[str, Any]]:
    return build_manifests(
        run_id=args.run_id,
        namespace=args.namespace,
        runtime_image=args.runtime_image,
        checkpoint_uri=args.checkpoint_uri,
        config_name=args.config_name,
        gpu_count=args.gpu_count,
        expected_gpu_type=args.expected_gpu_type,
        expected_compute_capability=args.expected_compute_capability,
        server_cpu=args.server_cpu,
        server_memory=args.server_memory,
        client_cpu=args.client_cpu,
        client_memory=args.client_memory,
        pull_secret=args.pull_secret,
        liveness_initial_delay_seconds=args.liveness_initial_delay_seconds,
        gpu_node_selector_key=args.gpu_node_selector_key,
        gpu_node_selector_value=args.gpu_node_selector_value,
        cache_size=args.service_cache_size,
        server_ready_timeout_seconds=timeouts["server"],
        client_timeout_seconds=timeouts["client"],
    )


def _validate_served_requests(client_result: Mapping[str, Any]) -> None:
    if (
        client_result.get("status") != "passed"
        or client_result.get("request_count") != 2
    ):
        raise OpenPIServiceError(
            f"cross-pod client evidence is invalid: {client_result}"
        )
    requests = client_result.get("requests")
    if not isinstance(requests, list) or len(requests) != 2:
        raise OpenPIServiceError("cross-pod client did not return exactly two requests")
    for request in requests:
        shape = list(request.get("shape", []))
        valid = (
            request.get("dtype") == "float64"
            and request.get("finite") is True
            and shape[1:] == [ACTION_DIM]
            and bool(shape)
            and int(shape[0]) >= 5
        )
        if not valid:
            raise OpenPIServiceError(f"invalid served request evidence: {request}")


def _service_health(
    service_dns: str, http_timeout: float
) -> tuple[str, dict[str, Any]]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(
        f"http://{service_dns}:{SERVER_PORT}/healthz", timeout=http_timeout
    ) as response:
        health = response.read().decode("utf-8").strip()
    if health != "OK":
        raise OpenPIServiceError(f"unexpected service health response {health!r}")
    url = f"http://{service_dns}:{SERVER_DIAGNOSTICS_PORT}/npa-openpi-server-hardware.json"
    with opener.open(url, timeout=http_timeout) as response:
        hardware = json.loads(response.read().decode("utf-8"))
    if not isinstance(hardware, dict):
        raise OpenPIServiceError("server hardware endpoint returned a non-object")
    return health, hardware


def _service_result(
    args: argparse.Namespace, evidence: Mapping[str, Any]
) -> dict[str, object]:
    names, timeouts = evidence["names"], evidence["timeouts"]
    return {
        "schema": "npa.workbench.openpi.pi05-cross-pod-service.v1",
        "status": "passed",
        "mode": "serve",
        "source": _service_source_evidence(),
        "redistribution": _redistribution_evidence(),
        "runtime_image": args.runtime_image,
        "hardware": evidence["hardware"],
        "checkpoint": {
            "uri": args.checkpoint_uri,
            "provenance": evidence["provenance"],
            "weights_baked": False,
        },
        "topology": _service_topology_evidence(args, names, evidence),
        "probes": {
            "readiness": {"path": "/healthz", "passed": True},
            "liveness": {"path": "/healthz", "configured": True},
            "clusterip_health": evidence["health"],
        },
        "failure_recovery_deadlines_seconds": _service_deadlines(timeouts),
        "client": evidence["client_result"],
        "request_count": 2,
        "all_trajectories_float64_finite_t_ge_5_x_8": True,
        "service_seconds_before_cleanup": round(
            time.perf_counter() - evidence["started"], 3
        ),
        "artifact_access": _public_service_access(),
        "cleanup_identity": _cleanup_identity(args, names),
        "limitations": [
            "clusterip_only_no_external_ingress_tested",
            "no_physical_franka_task_success_claim",
        ],
    }


def _public_service_access() -> dict[str, object]:
    return {
        "checkpoint_source": "public_anonymous_gcs",
        "npa_terms_attestation_required": False,
        "credential_persisted_after_cleanup": False,
    }


def _service_deadlines(timeouts: Mapping[str, float]) -> dict[str, float | int]:
    return {
        "server_ready": timeouts["server"],
        "client": timeouts["client"],
        "client_job_active_deadline": math.ceil(timeouts["client"]),
        "cleanup": timeouts["cleanup"],
        "kubernetes_api_request": timeouts["api"],
        "http_request": timeouts["http"],
        "poll_interval": timeouts["poll"],
    }


def _service_source_evidence() -> dict[str, str]:
    return {
        "repository": "https://github.com/Physical-Intelligence/openpi",
        "ref": SOURCE_REF,
        "license": SOURCE_LICENSE,
        "server_entrypoint": "upstream_openpi_websocket_policy_server",
        "transport": "upstream_openpi_websocket",
    }


def _service_topology_evidence(
    args: argparse.Namespace, names: Mapping[str, str], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "deployment": names["deployment"],
        "service": names["service"],
        "service_type": "ClusterIP",
        "public_ingress": False,
        "server_pod_uid": evidence["server_uid"],
        "client_pod_uid": evidence["client_uid"],
        "separate_pods": True,
        "server_gpu_request": args.gpu_count,
        "client_gpu_request": 0,
        "client_created_after_clusterip_health": True,
        "controller_service_account": args.controller_service_account,
    }


def _cleanup_identity(
    args: argparse.Namespace, names: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "managed_by": MANAGED_BY,
        "exact_names": names,
        "pod_selector": f"npa.nebius.ai/cleanup-owner={names['deployment']}",
        "controller_service_account": args.controller_service_account,
        "controller_rbac_lifecycle": "external_exact_owner_apply_delete",
    }


def _operate_service(
    args: argparse.Namespace,
    clients: tuple[Any, Any, Any],
    manifests: Mapping[str, Any],
    names: Mapping[str, str],
    timeouts: Mapping[str, float],
    provenance: Mapping[str, Any],
    created: set[str],
    started: float,
) -> dict[str, object]:
    server, client, health, hardware = _run_service_pods(
        args, clients, manifests, names, timeouts, created
    )
    client_result = _client_result_from_termination(client)
    _validate_served_requests(client_result)
    server_uid, client_uid = str(server.metadata.uid), str(client.metadata.uid)
    if not server_uid or not client_uid or server_uid == client_uid:
        raise OpenPIServiceError("server and client were not distinct Kubernetes pods")
    evidence = {
        "names": names,
        "timeouts": timeouts,
        "hardware": hardware,
        "provenance": provenance,
        "health": health,
        "client_result": client_result,
        "server_uid": server_uid,
        "client_uid": client_uid,
        "started": started,
    }
    return _service_result(args, evidence)


def _run_service_pods(
    args: argparse.Namespace,
    clients: tuple[Any, Any, Any],
    manifests: Mapping[str, Any],
    names: Mapping[str, str],
    timeouts: Mapping[str, float],
    created: set[str],
) -> tuple[Any, Any, str, dict[str, Any]]:
    api, apps, batch = clients
    _create_server_objects(
        api, apps, manifests, created=created, request_timeout=timeouts["api"]
    )
    dns = f"{names['service']}.{args.namespace}.svc.cluster.local"
    server = _wait_server_ready(
        api,
        apps,
        args.namespace,
        names["deployment"],
        timeout=timeouts["server"],
        poll_interval=timeouts["poll"],
        request_timeout=timeouts["api"],
        health_check=lambda: _service_endpoint_ready(dns, timeouts["http"]),
    )
    health, hardware = _service_health(dns, timeouts["http"])
    _create_client_job(
        batch, manifests, created=created, request_timeout=timeouts["api"]
    )
    client = _wait_client(
        api,
        batch,
        args.namespace,
        names["client_job"],
        timeout=timeouts["client"],
        poll_interval=timeouts["poll"],
        request_timeout=timeouts["api"],
    )
    return server, client, health, hardware


def _cleanup_service_run(
    clients: tuple[Any, Any, Any],
    args: argparse.Namespace,
    names: Mapping[str, str],
    created: set[str],
    manifests: Mapping[str, Any],
    timeouts: Mapping[str, float],
) -> dict[str, bool]:
    if not created:
        return {}
    api, apps, batch = clients
    return _delete_and_verify(
        api,
        apps,
        batch,
        namespace=args.namespace,
        names=names,
        created=created,
        manifests=manifests,
        timeout=timeouts["cleanup"],
        poll_interval=timeouts["poll"],
        request_timeout=timeouts["api"],
    )


def _publish_service_result(
    args: argparse.Namespace, result: dict[str, Any], cleanup: Mapping[str, bool]
) -> int:
    result["cleanup"] = {"all_exact_resources_absent": True, "verified": cleanup}
    cleanup_artifact = {
        "schema": "npa.workbench.openpi.service-cleanup.v1",
        "service_artifact_uri": args.output_uri,
        "all_exact_resources_absent": True,
        "verified": cleanup,
    }
    _write_json_uri(args.cleanup_output_uri, cleanup_artifact)
    result["cleanup_artifact_uri"] = args.cleanup_output_uri
    _write_json_uri(args.output_uri, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _service_execution_context(args: argparse.Namespace) -> dict[str, Any]:
    from kubernetes import client, config

    config.load_incluster_config()
    timeouts = _service_timeouts(args)
    clients = (client.CoreV1Api(), client.AppsV1Api(), client.BatchV1Api())
    manifests = _service_manifests(args, timeouts)
    provenance = _checkpoint_provenance(args.checkpoint_uri)
    if args.checkpoint_uri.startswith("gs://"):
        _inject_public_checkpoint_binding(manifests, provenance)
    return {
        "timeouts": timeouts,
        "clients": clients,
        "manifests": manifests,
        "provenance": provenance,
        "names": service_resource_names(args.run_id),
    }


def _execute_owned_service(
    args: argparse.Namespace, context: Mapping[str, Any]
) -> tuple[dict[str, object], dict[str, bool]]:
    clients, names = context["clients"], context["names"]
    api, apps, batch = clients
    created: set[str] = set()
    result: dict[str, object] | None = None
    run_error: Exception | None = None
    try:
        _assert_targets_absent(
            api,
            apps,
            batch,
            namespace=args.namespace,
            names=names,
            request_timeout=context["timeouts"]["api"],
        )
        result = _operate_service(
            args,
            clients,
            context["manifests"],
            names,
            context["timeouts"],
            context["provenance"],
            created,
            time.perf_counter(),
        )
    except Exception as error:
        run_error = error
    cleanup = _finish_owned_service(args, context, created, run_error)
    if run_error is not None:
        raise run_error
    if result is None:
        raise OpenPIServiceError("service run did not produce evidence")
    return result, cleanup


def _finish_owned_service(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    created: set[str],
    run_error: Exception | None,
) -> dict[str, bool]:
    try:
        return _cleanup_service_run(
            context["clients"],
            args,
            context["names"],
            created,
            context["manifests"],
            context["timeouts"],
        )
    except Exception as cleanup_error:
        if run_error is not None:
            raise OpenPIServiceError(
                f"service failed ({run_error}); cleanup also failed ({cleanup_error})"
            ) from run_error
        raise


def _run(args: argparse.Namespace) -> int:
    if not RUNTIME_IMAGE_RE.fullmatch(args.runtime_image):
        raise OpenPIServiceError("service runtime image must be digest-pinned")
    if args.cleanup_output_uri == args.output_uri:
        raise OpenPIServiceError(
            "service success and cleanup evidence must use distinct output URIs"
        )
    result, cleanup_verified = _execute_owned_service(
        args, _service_execution_context(args)
    )
    if not cleanup_verified or not all(cleanup_verified.values()):
        raise OpenPIServiceError(
            "service resources were not independently verified absent"
        )
    return _publish_service_result(args, result, cleanup_verified)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--cleanup-output-uri", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--checkpoint-uri", default=DEFAULT_CHECKPOINT_URI)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--expected-gpu-type", required=True)
    parser.add_argument("--expected-compute-capability", required=True)
    parser.add_argument("--server-cpu", default="16")
    parser.add_argument("--server-memory", default="96Gi")
    parser.add_argument("--client-cpu", default="2")
    parser.add_argument("--client-memory", default="8Gi")
    parser.add_argument(
        "--pull-secret",
        default="",
        help="Optional operator-managed Kubernetes pull secret for a BYOF registry.",
    )
    parser.add_argument("--liveness-initial-delay-seconds", type=int, default=600)
    parser.add_argument("--gpu-node-selector-key", default="nebius.com/gpu-name")
    parser.add_argument("--gpu-node-selector-value", default="B200")
    parser.add_argument("--service-cache-size", default="40Gi")
    parser.add_argument("--controller-service-account", required=True)
    parser.add_argument(
        "--server-ready-timeout-seconds",
        type=float,
        default=DEFAULT_SERVER_READY_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--client-timeout-seconds", type=float, default=DEFAULT_CLIENT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--cleanup-timeout-seconds",
        type=float,
        default=DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--api-timeout-seconds", type=float, default=DEFAULT_API_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--http-timeout-seconds", type=float, default=DEFAULT_HTTP_TIMEOUT_SECONDS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return _run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
