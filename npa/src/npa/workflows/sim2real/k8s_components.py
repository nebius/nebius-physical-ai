"""Kubernetes manifests for immutable Sim2Real sibling Jobs.

The application is part of each digest-pinned image. These manifests never
download source, clone a repository, or install packages after scheduling.
"""

from __future__ import annotations

import os
from typing import Any

from npa.workbench.cosmos.reason import (
    apply_cosmos_reason_kubernetes_env,
    cosmos_reason_k8s_shell_preamble,
    vlm_k8s_component,
)
from npa.workflows.sim2real.constants import DEFAULT_SIM_BACKEND, SIM_BACKEND_ISAAC
from npa.workflows.sim2real.models import Sim2RealLoopConfig, Sim2RealLoopError
from npa.workflows.sim2real.utils import _split_csv


def _safe_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in str(value)]
    return "-".join(part for part in "".join(chars).split("-") if part)


def _label_value(value: str) -> str:
    return (_safe_slug(value)[:63] or "run").rstrip("-")


def _image_pull_policy(image: str) -> str:
    """Choose the image pull policy without weakening immutable provenance."""

    override = os.environ.get("NPA_SIM2REAL_IMAGE_PULL_POLICY", "").strip()
    if override:
        return override
    if "@sha256:" in image:
        return "IfNotPresent"
    tag = image.rsplit(":", 1)[-1] if ":" in image.rsplit("/", 1)[-1] else ""
    if "genuine" in tag:
        return "Always"
    return "IfNotPresent"


def _indexed_component_job_manifest(
    image: str,
    *,
    component: str,
    env: dict[str, str],
    config: Sim2RealLoopConfig,
    namespace: str,
    job_name: str,
    completions: int,
    parallelism: int,
    timeout_s: int,
    gpu_product: str | None = None,
) -> dict[str, Any]:
    manifest = _component_job_manifest(
        image,
        component=component,
        env=env,
        config=config,
        namespace=namespace,
        job_name=job_name,
        timeout_s=timeout_s,
        gpu_product=gpu_product,
    )
    manifest["spec"]["completions"] = completions
    manifest["spec"]["parallelism"] = parallelism
    manifest["spec"]["completionMode"] = "Indexed"
    return manifest


def _component_job_manifest(
    image: str,
    *,
    component: str,
    env: dict[str, str],
    config: Sim2RealLoopConfig,
    namespace: str,
    job_name: str,
    timeout_s: int,
    gpu_product: str | None = None,
) -> dict[str, Any]:
    selected_gpu_product = gpu_product or config.k8s_gpu_product
    env_values = _kubernetes_component_env(
        env,
        config,
        isaac_backed=(component == "heldout_eval" and config.sim_backend == SIM_BACKEND_ISAAC),
    )
    # Each sibling attests its own immutable image, not the controller image.
    env_values["NPA_SIM2REAL_RUNTIME_IMAGE"] = image.removeprefix("docker:")
    pull_secrets = [
        {"name": name} for name in _split_csv(config.k8s_image_pull_secrets)
    ]
    env_from = [
        {"secretRef": {"name": name}}
        for name in _split_csv(config.k8s_env_secret_names)
    ]
    template_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "serviceAccountName": config.k8s_service_account,
        "containers": [
            {
                "name": "component",
                "image": image,
                "imagePullPolicy": _image_pull_policy(image),
                "command": ["bash", "-lc"],
                "args": [
                    _component_job_script(component, sim_backend=config.sim_backend)
                ],
                "env": [
                    {"name": key, "value": value}
                    for key, value in sorted(env_values.items())
                    if value != "" or key == "ACCEPT_EULA"
                ],
                "envFrom": env_from,
                "resources": {
                    "requests": {
                        "cpu": "4",
                        "memory": "16Gi",
                        config.k8s_gpu_resource: 1,
                    },
                    "limits": {config.k8s_gpu_resource: 1},
                },
            }
        ],
        "nodeSelector": {"nvidia.com/gpu.product": selected_gpu_product},
    }
    if pull_secrets:
        template_spec["imagePullSecrets"] = pull_secrets
    labels = {
        "app.kubernetes.io/name": "sim2real-sibling-component",
        "app.kubernetes.io/component": component.replace("_", "-"),
        "sim2real.local/run-id": _label_value(config.run_id),
    }
    job_spec: dict[str, Any] = {
        # Replaced by ``configure_gpu_job`` with the selected native retry
        # policy before this manifest reaches the Kubernetes API.
        "backoffLimit": 1,
        "template": {"metadata": {"labels": labels}, "spec": template_spec},
    }
    if timeout_s > 0:
        job_spec["activeDeadlineSeconds"] = timeout_s
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": labels,
            "annotations": {"sim2real.local/gpu-request": f"{selected_gpu_product}:1"},
        },
        "spec": job_spec,
    }


def _component_job_script(
    component: str, *, sim_backend: str = DEFAULT_SIM_BACKEND
) -> str:
    if component in {"vlm_eval", "vlm_eval_reason2", "vlm_eval_reason3"}:
        subcommand = (
            "component-vlm-eval "
            '--input-uri "${NPA_SIM2REAL_ROLLOUT_URI}" '
            '--output-uri "${NPA_SIM2REAL_OUTPUT_URI}" '
            '--rollout-id "${NPA_SIM2REAL_ROLLOUT_ID}" '
            '--model "${NPA_SIM2REAL_VLM_MODEL}" '
            '--threshold "${NPA_SIM2REAL_THRESHOLD}"'
        )
    elif component == "heldout_eval":
        subcommand = (
            "component-heldout-eval "
            '--heldout-envs-uri "${NPA_SIM2REAL_HELDOUT_ENVS_URI}" '
            '--inner-evidence-uri "${NPA_SIM2REAL_INNER_EVIDENCE_URI}" '
            '--output-uri "${NPA_SIM2REAL_OUTPUT_URI}" '
            '--threshold "${NPA_SIM2REAL_THRESHOLD}" '
            '--limit "${NPA_SIM2REAL_HELDOUT_EVAL_LIMIT:-0}" '
            '--sim-backend "${NPA_SIM2REAL_SIM_BACKEND:-isaac}" '
            '--isaac-task "${NPA_SIM2REAL_ISAAC_TASK:-}" '
            '--scene-spec-uri "${NPA_SIM2REAL_SCENE_SPEC_URI:-}" '
            '--assets-uri "${NPA_SIM2REAL_ASSETS_URI:-}" '
            '--cameras-uri "${NPA_SIM2REAL_CAMERAS_URI:-}" '
            '--robot-spec-uri "${NPA_SIM2REAL_ROBOT_SPEC_URI:-}" '
            '--robot-source "${NPA_SIM2REAL_ROBOT_SOURCE:-}" '
            '--robot-preset "${NPA_SIM2REAL_ROBOT_PRESET:-}"'
        )
    elif component == "cosmos2_transfer":
        subcommand = (
            "component-cosmos2-transfer "
            '--input-uri "${NPA_SIM2REAL_INPUT_URI}" '
            '--output-uri "${NPA_SIM2REAL_OUTPUT_URI}" '
            '--augmented-frames-uri "${NPA_SIM2REAL_AUGMENTED_FRAMES_URI}" '
            '--assets-uri "${NPA_SIM2REAL_ASSETS_URI:-}" '
            '--scene-spec-uri "${NPA_SIM2REAL_SCENE_SPEC_URI:-}" '
            '--image "${NPA_SIM2REAL_AUGMENT_IMAGE:-}"'
        )
    elif component == "policy_actions":
        subcommand = (
            "component-policy-actions "
            '--train-envs-uri "${NPA_SIM2REAL_TRAIN_ENVS_URI}" '
            '--output-uri "${NPA_SIM2REAL_OUTPUT_URI}" '
            '--policy-image "${NPA_SIM2REAL_POLICY_IMAGE}" '
            '--limit "${NPA_SIM2REAL_ACTION_LIMIT:-256}" '
            '--seed "${NPA_SIM2REAL_SEED:-42}" '
            '--rollout-count "${NPA_SIM2REAL_ROLLOUT_COUNT:-3}" '
            '--steps-per-rollout "${NPA_SIM2REAL_STEPS_PER_ROLLOUT:-4}"'
        )
    elif component == "envgen_raw_shard":
        subcommand = (
            "python -m npa.workflows.sim2real_envgen raw-shard "
            '--run-id "${NPA_SIM2REAL_RUN_ID}" '
            '--output-uri "${NPA_SIM2REAL_OUTPUT_URI}" '
            '--env-count "${NPA_SIM2REAL_ENV_COUNT}" '
            '--shard-index "${JOB_COMPLETION_INDEX:-0}" '
            '--shard-count "${NPA_SIM2REAL_SHARD_COUNT}" '
            '--train-fraction "${NPA_SIM2REAL_TRAIN_FRACTION}" '
            '--seed "${NPA_SIM2REAL_SEED}" '
            '--augmented-frames-uri "${NPA_SIM2REAL_AUGMENTED_FRAMES_URI:-}" '
            '--scene-spec-uri "${NPA_SIM2REAL_SCENE_SPEC_URI:-}" '
            "--output-dir /tmp/npa-envgen-shard"
        )
    else:
        raise Sim2RealLoopError(f"unsupported image component: {component}")
    vlm_preamble = ""
    if vlm_k8s_component(component):
        vlm_preamble = cosmos_reason_k8s_shell_preamble()
    if component == "heldout_eval" and sim_backend == SIM_BACKEND_ISAAC:
        heldout_entry_cmd = (
            '"$PYBIN" -m npa.workflows.sim2real.heldout_entry '
            '--heldout-envs-uri "${NPA_SIM2REAL_HELDOUT_ENVS_URI}" '
            '--inner-evidence-uri "${NPA_SIM2REAL_INNER_EVIDENCE_URI}" '
            '--output-uri "${NPA_SIM2REAL_OUTPUT_URI}" '
            '--threshold "${NPA_SIM2REAL_THRESHOLD}" '
            '--limit "${NPA_SIM2REAL_HELDOUT_EVAL_LIMIT:-0}" '
            '--sim-backend "${NPA_SIM2REAL_SIM_BACKEND:-isaac}" '
            '--isaac-task "${NPA_SIM2REAL_ISAAC_TASK:-}" '
            '--scene-spec-uri "${NPA_SIM2REAL_SCENE_SPEC_URI:-}" '
            '--assets-uri "${NPA_SIM2REAL_ASSETS_URI:-}" '
            '--cameras-uri "${NPA_SIM2REAL_CAMERAS_URI:-}" '
            '--robot-spec-uri "${NPA_SIM2REAL_ROBOT_SPEC_URI:-}" '
            '--robot-source "${NPA_SIM2REAL_ROBOT_SOURCE:-}" '
            '--robot-preset "${NPA_SIM2REAL_ROBOT_PRESET:-}"'
        )
        return f"""set -euo pipefail
{vlm_preamble}export NPA_SKIP_EAGER_IMPORTS=1
export PYTHONUNBUFFERED=1
PYBIN=/isaac-sim/python.sh
if [ ! -x "$PYBIN" ]; then PYBIN=python; fi
"$PYBIN" -m npa.workflows.sim2real.runtime_attestation
exec {heldout_entry_cmd}
"""
    exec_cmd = (
        subcommand
        if component == "envgen_raw_shard"
        else f"python -m npa.workflows.sim2real {subcommand}"
    )
    return f"""set -euo pipefail
export NPA_SKIP_EAGER_IMPORTS=1
export PYTHONUNBUFFERED=1
{vlm_preamble}python -m npa.workflows.sim2real.runtime_attestation
exec {exec_cmd}
"""


def _kubernetes_component_env(
    env: dict[str, str],
    config: Sim2RealLoopConfig,
    *,
    isaac_backed: bool = False,
) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in env.items():
        if (
            key.startswith("NPA_SIM2REAL")
            or key.startswith("NPA_COSMOS_")
            or key
            in {
                "HF_HOME",
                "HF_XET_CACHE",
                "UV_CACHE_DIR",
                "XDG_CACHE_HOME",
            }
        ):
            safe[key] = value
    endpoint = (
        config.s3_endpoint
        or env.get("AWS_ENDPOINT_URL", "")
        or os.environ.get("AWS_ENDPOINT_URL", "")
    )
    safe["AWS_ENDPOINT_URL"] = endpoint
    safe["S3_ENDPOINT_URL"] = endpoint
    apply_cosmos_reason_kubernetes_env(safe)
    safe.pop("ACCEPT_EULA", None)
    if isaac_backed:
        from npa.serverless_common.env import resolved_isaac_eula_env

        source = dict(os.environ)
        if "ACCEPT_EULA" in env:
            source["ACCEPT_EULA"] = str(env["ACCEPT_EULA"])
        safe.update(resolved_isaac_eula_env(source))
    safe["NPA_SIM2REAL_SOURCE_SHA"] = str(
        env.get("NPA_SIM2REAL_SOURCE_SHA")
        or os.environ.get("NPA_SIM2REAL_SOURCE_SHA")
        or ""
    ).strip()
    return safe
