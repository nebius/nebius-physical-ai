"""Resolve Sim2Real health settings without constructing an execution image plan."""

from __future__ import annotations

import os
from dataclasses import dataclass

from npa.workflows.sim2real.artifact_config import _artifact_bucket, _artifact_endpoint
from npa.workflows.sim2real.models import new_run_id
from npa.workflows.sim2real.utils import _serviceaccount_namespace, _split_csv


@dataclass(frozen=True)
class Sim2RealDiagnosticConfig:
    """Resolved storage and Kubernetes settings consumed by health checks.

    Args:
        run_id: Identifier included in the health report.
        s3_bucket: Bucket whose reachability is checked.
        s3_endpoint: Endpoint used by the storage probe.
        k8s_namespace: Namespace used by permission and cache checks.
        k8s_context: Kubernetes context selected for the probes.
        k8s_kubeconfig: Explicit kubeconfig used by the probes.
        k8s_isaac_cache_pvc: Runtime-cache claim checked for Bound RWX status.
        k8s_gpu_resource: Resource counted on schedulable nodes.
        k8s_gpu_product: Requested GPU product checked against node labels.
        k8s_gpu_candidates: Accepted alternative GPU products.
    Returns:
        None.
    Raises:
        None.
    """

    run_id: str
    s3_bucket: str
    s3_endpoint: str
    k8s_namespace: str
    k8s_context: str
    k8s_kubeconfig: str
    k8s_isaac_cache_pvc: str
    k8s_gpu_resource: str
    k8s_gpu_product: str
    k8s_gpu_candidates: tuple[str, ...]


def build_diagnostic_config_from_env(
    *,
    run_id: str,
    s3_bucket: str = "",
    s3_endpoint: str = "",
    k8s_namespace: str = "",
    k8s_context: str = "",
    k8s_kubeconfig: str = "",
) -> Sim2RealDiagnosticConfig:
    """Resolve image-independent health settings with the existing env precedence.

    Args:
        run_id: Identifier included in the report.
        s3_bucket: Explicit bucket, otherwise the storage environment fallback.
        s3_endpoint: Explicit endpoint, otherwise the storage environment fallback.
        k8s_namespace: Explicit namespace, otherwise the configured or pod namespace.
        k8s_context: Explicit context, otherwise the configured environment value.
        k8s_kubeconfig: Explicit path, otherwise KUBECONFIG then the NPA environment.
    Returns:
        Storage and cluster settings with no execution image fields.
    Raises:
        OSError: Reading the current pod's namespace fails.
    """

    return _diagnostic_config(
        run_id or os.environ.get("NPA_SIM2REAL_RUN_ID") or new_run_id(),
        _artifact_bucket(s3_bucket),
        _artifact_endpoint(s3_endpoint),
        k8s_namespace,
        k8s_context,
        k8s_kubeconfig,
    )


def _diagnostic_config(
    run_id: str,
    bucket: str,
    endpoint: str,
    namespace: str,
    context: str,
    kubeconfig: str,
) -> Sim2RealDiagnosticConfig:
    return Sim2RealDiagnosticConfig(
        run_id=run_id,
        s3_bucket=bucket,
        s3_endpoint=endpoint,
        k8s_namespace=namespace
        or os.environ.get("NPA_SIM2REAL_K8S_NAMESPACE")
        or _serviceaccount_namespace()
        or "default",
        k8s_context=context or os.environ.get("NPA_SIM2REAL_K8S_CONTEXT", ""),
        k8s_kubeconfig=kubeconfig
        or os.environ.get("KUBECONFIG")
        or os.environ.get("NPA_SIM2REAL_KUBECONFIG", ""),
        k8s_isaac_cache_pvc=os.environ.get("NPA_SIM2REAL_ISAAC_CACHE_PVC")
        or "npa-sim2real-isaac-cache",
        k8s_gpu_resource=os.environ.get("NPA_SIM2REAL_K8S_GPU_RESOURCE")
        or "nvidia.com/gpu",
        k8s_gpu_product=os.environ.get("NPA_SIM2REAL_K8S_GPU_PRODUCT")
        or "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        k8s_gpu_candidates=tuple(
            _split_csv(os.environ.get("NPA_SIM2REAL_K8S_GPU_CANDIDATES", ""))
        ),
    )
