"""Render the sim2real SkyPilot runbook into a plain Kubernetes Job manifest.

Raw ``sky jobs launch`` against the runbook is blocked by the SkyPilot 0.12.2
pre-setup ``getcwd()`` bug, and the previous direct-Kubernetes route required a
private operator pack. This module closes that gap in-repo: it reads the
committed runbook (a single SkyPilot task document), applies operator
overrides, and emits a Job manifest that ``kubectl apply -f`` can run on any
cluster with GPUs — no SkyPilot controller and no operator pack required.

The runbook stays the single source of truth: envs, resources, and the
setup/run scripts are taken from the YAML, never duplicated here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class Sim2RealMaterializeError(RuntimeError):
    """Raised when the runbook cannot be rendered into a runnable Job."""


_PLACEHOLDER_IMAGE_MARKERS = ("example.invalid", "<your-registry-id>")
_DNS1123_SANITIZE = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class MaterializedJob:
    manifest: dict[str, Any]
    job_name: str
    namespace: str
    image: str
    warnings: list[str] = field(default_factory=list)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.manifest, sort_keys=False, default_flow_style=False)


def default_runbook_path() -> Path:
    """Locate the committed runbook relative to this source tree."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "npa" / "workflows" / "workbench" / "sim2real" / "runbook.yaml"
        if candidate.is_file():
            return candidate
        candidate = parent / "workflows" / "workbench" / "sim2real" / "runbook.yaml"
        if candidate.is_file():
            return candidate
    raise Sim2RealMaterializeError(
        "could not locate the committed sim2real runbook; pass an explicit path"
    )


def load_runbook_task(path: Path) -> dict[str, Any]:
    """Load the SkyPilot task document (name/resources/envs/setup/run) from the runbook."""
    documents = [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]
    tasks = [doc for doc in documents if isinstance(doc, dict) and "run" in doc]
    if not tasks:
        raise Sim2RealMaterializeError(f"no SkyPilot task document with a run block in {path}")
    if len(tasks) > 1:
        raise Sim2RealMaterializeError(
            f"{path} contains {len(tasks)} task documents; the materializer supports exactly one"
        )
    return tasks[0]


def _job_name(run_id: str, task_name: str) -> str:
    base = run_id or task_name or "sim2real"
    slug = _DNS1123_SANITIZE.sub("-", base.lower()).strip("-") or "sim2real"
    name = slug if slug.startswith("sim2real") else f"sim2real-{slug}"
    return name[:63].rstrip("-")


def _gpu_count(accelerators: Any) -> int:
    if not accelerators:
        return 0
    text = str(accelerators)
    if ":" in text:
        try:
            return int(text.rsplit(":", 1)[1])
        except ValueError as exc:
            raise Sim2RealMaterializeError(f"unparseable accelerators value: {text!r}") from exc
    return 1


def _numeric_prefix(value: Any) -> str:
    match = re.match(r"\d+", str(value or ""))
    return match.group(0) if match else ""


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def materialize_k8s_job(
    runbook_path: Path | None = None,
    *,
    run_id: str = "",
    image: str = "",
    env_overrides: dict[str, str] | None = None,
    namespace: str = "",
    include_setup: bool = True,
) -> MaterializedJob:
    """Render the runbook into a Kubernetes Job manifest dict.

    ``env_overrides`` wins over the runbook's materialized literals, mirroring
    the ``--env KEY=VALUE`` contract of ``sky jobs launch``. The runbook ships
    a placeholder ``image_id``, so ``image`` (or an override of the relevant
    env) is required unless the runbook was edited to a concrete registry.
    """
    path = runbook_path or default_runbook_path()
    task = load_runbook_task(path)
    resources = task.get("resources") or {}
    warnings: list[str] = []

    envs: dict[str, str] = {
        str(key): str(value) for key, value in (task.get("envs") or {}).items()
    }
    for key, value in (env_overrides or {}).items():
        envs[str(key)] = str(value)
    if run_id:
        envs["NPA_SIM2REAL_RUN_ID"] = run_id

    unresolved = sorted(key for key, value in envs.items() if "${" in value)
    if unresolved:
        raise Sim2RealMaterializeError(
            "runbook envs contain unexpanded ${VAR} references (SkyPilot does not "
            f"interpolate them and neither does Kubernetes): {', '.join(unresolved)}"
        )

    resolved_image = image.strip()
    if not resolved_image:
        image_id = str(resources.get("image_id") or "")
        resolved_image = image_id.removeprefix("docker:").strip()
    if not resolved_image or any(marker in resolved_image for marker in _PLACEHOLDER_IMAGE_MARKERS):
        raise Sim2RealMaterializeError(
            "the runbook ships a placeholder image; pass a registry-qualified trainer "
            f"image (got {resolved_image!r}). Example: --image cr.<region>.nebius.cloud/"
            "<your-registry-id>/npa-lerobot-vlm-rl:<tag>"
        )

    scripts = []
    if include_setup and task.get("setup"):
        scripts.append(str(task["setup"]))
    scripts.append(str(task["run"]))
    command_script = "\n".join(scripts)

    gpu_resource = envs.get("NPA_SIM2REAL_K8S_GPU_RESOURCE", "nvidia.com/gpu")
    gpu_count = _gpu_count(resources.get("accelerators"))
    limits: dict[str, Any] = {}
    if gpu_count:
        limits[gpu_resource] = gpu_count
    cpu = _numeric_prefix(resources.get("cpus"))
    if cpu:
        limits["cpu"] = cpu
    memory = _numeric_prefix(resources.get("memory"))
    if memory:
        limits["memory"] = f"{memory}Gi"

    container: dict[str, Any] = {
        "name": "sim2real",
        "image": resolved_image,
        "command": ["/bin/bash", "-c", command_script],
        "env": [{"name": key, "value": value} for key, value in sorted(envs.items())],
    }
    if limits:
        container["resources"] = {"limits": limits}
    secret_names = _split_csv(envs.get("NPA_SIM2REAL_K8S_ENV_SECRET_NAMES", ""))
    if secret_names:
        container["envFrom"] = [{"secretRef": {"name": name}} for name in secret_names]
        warnings.append(
            "the Job references secrets "
            f"({', '.join(secret_names)}); create them in the namespace before applying"
        )

    pod_spec: dict[str, Any] = {"restartPolicy": "Never", "containers": [container]}
    service_account = envs.get("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "").strip()
    if service_account:
        pod_spec["serviceAccountName"] = service_account
    pull_secrets = _split_csv(envs.get("NPA_SIM2REAL_K8S_IMAGE_PULL_SECRETS", ""))
    if pull_secrets:
        pod_spec["imagePullSecrets"] = [{"name": name} for name in pull_secrets]
    gpu_product = envs.get("NPA_SIM2REAL_K8S_GPU_PRODUCT", "").strip()
    if gpu_count and gpu_product:
        pod_spec["nodeSelector"] = {"nvidia.com/gpu.product": gpu_product}

    job_spec: dict[str, Any] = {"backoffLimit": 0, "template": {"spec": pod_spec}}
    timeout = _numeric_prefix(envs.get("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S"))
    if timeout:
        job_spec["activeDeadlineSeconds"] = int(timeout)

    resolved_namespace = (
        namespace.strip() or envs.get("NPA_SIM2REAL_K8S_NAMESPACE", "").strip() or "default"
    )
    job_name = _job_name(run_id, str(task.get("name") or ""))
    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": resolved_namespace,
            "labels": {"app.kubernetes.io/part-of": "npa-sim2real"},
        },
        "spec": job_spec,
    }
    return MaterializedJob(
        manifest=manifest,
        job_name=job_name,
        namespace=resolved_namespace,
        image=resolved_image,
        warnings=warnings,
    )
