"""Qualify CUDA and graphics readiness on every selected Fleet RTX worker."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import uuid

from npa.cluster.gpu_driver import gpus_per_node
from npa.cluster.gpu_health import GpuHealthConfig, GpuHealthError, validate_gpu_health
from npa.cluster_backends.process import BackendCommandError, require_bin, run_capture
from npa.fleet.storage_identity import (
    StorageIdentityError,
    resolve_fleet_identity,
    resolve_fleet_targets,
)
from npa.lifecycle_intent import OperationIntent, intent_boundary


class GraphicsVerificationError(RuntimeError):
    """A Fleet graphics target or private evidence destination is invalid."""


_FAILURES = (
    BackendCommandError,
    GpuHealthError,
    GraphicsVerificationError,
    StorageIdentityError,
    OSError,
    ValueError,
)


@intent_boundary(OperationIntent.MUTATE)
def verify_graphics(
    spec, *, only_projects=None, only_clusters=None, project_prefix=None,
    profile=None, evidence_dir: Path | None = None, concurrency: int = 1,
    stabilization_seconds: int | None = None, timeout_minutes: int | None = None,
) -> dict:
    """Qualify CUDA, GLX, EGL, and Vulkan on every selected RTX worker.

    Args:
        spec: Loaded Fleet declaration.
        only_projects: Existing Fleet project selectors.
        only_clusters: Existing Fleet cluster selectors within project scope.
        project_prefix: Optional Fleet project display-name override.
        profile: Optional provider authentication profile override.
        evidence_dir: Owner-only directory for exact private receipts.
        concurrency: Maximum clusters to qualify in parallel.
        stabilization_seconds: Optional healthy-state observation override.
        timeout_minutes: Optional per-cluster qualification timeout override.
    Returns:
        Publication-safe cluster, worker, capability, and evidence counts.
    Raises:
        GraphicsVerificationError: Options or evidence storage are unsafe.
        StorageIdentityError: Target selection is invalid.
    """
    _validate_options(concurrency, stabilization_seconds, timeout_minutes)
    targets = resolve_fleet_targets(
        spec, only_projects=only_projects, only_clusters=only_clusters,
        project_prefix=project_prefix, profile=profile,
    )
    directory = _evidence_directory(evidence_dir)
    reports = _run_targets(spec, targets, directory, concurrency, profile,
                           project_prefix, stabilization_seconds, timeout_minutes)
    return _aggregate(reports)


def _validate_options(concurrency, stabilization_seconds, timeout_minutes) -> None:
    if concurrency < 1:
        raise GraphicsVerificationError(
            "graphics verification concurrency must be positive"
        )
    if stabilization_seconds is not None and stabilization_seconds < 0:
        raise GraphicsVerificationError("stabilization seconds cannot be negative")
    if timeout_minutes is not None and timeout_minutes <= 0:
        raise GraphicsVerificationError("timeout minutes must be positive")


def _evidence_directory(directory: Path | None) -> Path:
    parent = Path(
        directory or Path.home() / ".npa" / "graphics-verification"
    ).expanduser()
    resolved = parent.resolve()
    if any(
        (candidate / ".git").exists() for candidate in (resolved, *resolved.parents)
    ):
        raise GraphicsVerificationError(
            "exact evidence must remain outside Git checkouts"
        )
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    state = parent.stat()
    if parent.is_symlink() or state.st_mode & 0o077 or state.st_uid != os.getuid():
        raise GraphicsVerificationError("evidence directory must be owner-private")
    run_directory = parent / uuid.uuid4().hex
    run_directory.mkdir(mode=0o700)
    return run_directory


def _run_targets(
    spec,
    targets,
    directory,
    concurrency,
    profile,
    project_prefix,
    stabilization_seconds,
    timeout_minutes,
) -> list[dict]:
    reports: list[dict | None] = [None] * len(targets)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _verify_target,
                spec,
                project,
                cluster,
                index,
                directory,
                profile,
                project_prefix,
                stabilization_seconds,
                timeout_minutes,
            ): index
            for index, (project, cluster) in enumerate(targets)
        }
        for future in as_completed(futures):
            reports[futures[future]] = future.result()
    return [report for report in reports if report is not None]


def _verify_target(
    spec,
    project,
    cluster,
    index,
    directory,
    profile,
    project_prefix,
    stabilization_seconds,
    timeout_minutes,
) -> dict:
    public = _target_report(index, cluster)
    private = {"target_index": index, "run_id": uuid.uuid4().hex}
    try:
        config = _health_config(cluster, stabilization_seconds, timeout_minutes)
        identity = resolve_fleet_identity(spec, project, cluster, profile=profile,
                                          project_prefix=project_prefix)
        private["identity_sha256"] = identity.evidence_sha256
        private["identity"] = json.loads(identity.evidence_json)
        health = validate_gpu_health(run_capture, kubectl_bin=require_bin("kubectl"),
                                     kubeconfig_path=identity.kubeconfig, config=config)
        private["health"] = health
        _record_pass(public, health, config)
    except _FAILURES as error:
        _record_failure(public, private, error)
    except BaseException:
        public["failures"].append("verification_interrupted")
        raise
    finally:
        public["passed"] = not public["failures"]
        private["report"] = dict(public)
        public["evidence_sha256"] = _write_receipt(directory, index, private)
    return public


def _target_report(index, cluster) -> dict:
    return {
        "target_index": index,
        "passed": False,
        "gpu_workers": cluster.gpu_count(),
        "gpus": 0,
        "cuda_workers": 0,
        "glx_workers": 0,
        "egl_workers": 0,
        "vulkan_workers": 0,
        "failures": [],
        "evidence_sha256": "",
    }


def _health_config(cluster, stabilization_seconds, timeout_minutes) -> GpuHealthConfig:
    if cluster.gpu_workload_profile != "rtx-rendering":
        raise GraphicsVerificationError(
            "target does not declare the RTX rendering profile"
        )
    if cluster.gpu_count() <= 0 or not cluster.gpu_nodes:
        raise GraphicsVerificationError("RTX rendering target has no GPU workers")
    if gpus_per_node(cluster.gpu_nodes.preset) != 8:
        raise GraphicsVerificationError(
            "RTX rendering qualification requires 8-GPU workers"
        )
    if cluster.resolved_gpu_driver_mode() != "operator":
        raise GraphicsVerificationError(
            "RTX graphics qualification requires operator drivers"
        )
    return GpuHealthConfig(
        expected_nodes=cluster.cpu_count() + cluster.gpu_count(),
        expected_gpu_nodes=cluster.gpu_count(),
        gpu_preset=cluster.gpu_nodes.preset,
        gpu_platform=cluster.gpu_nodes.platform,
        driver_mode="operator",
        nvswitch=cluster.resolved_enable_gpu_cluster(),
        stabilization_seconds=(
            cluster.gpu_health_stabilization_seconds
            if stabilization_seconds is None
            else stabilization_seconds
        ),
        timeout_seconds=60
        * (
            cluster.gpu_health_timeout_minutes
            if timeout_minutes is None
            else timeout_minutes
        ),
        cuda_smoke=True,
        cuda_smoke_image=cluster.gpu_cuda_smoke_image,
        graphics_smoke=True,
        graphics_smoke_image=cluster.gpu_graphics_smoke_image,
    )


def _record_pass(public, health, config) -> None:
    cuda = health.get("cuda_smokes") or []
    graphics = health.get("graphics_smokes") or []
    public["gpus"] = config.expected_gpus
    public["cuda_workers"] = sum(item.get("vectoradd") == "passed" for item in cuda)
    public["glx_workers"] = sum(item.get("glx") == "loaded" for item in graphics)
    public["egl_workers"] = sum(item.get("egl") == "loaded" for item in graphics)
    public["vulkan_workers"] = sum(
        item.get("nvidia_device") == "enumerated" for item in graphics
    )
    if min(public[name] for name in _CAPABILITY_COUNTS) != config.expected_gpu_nodes:
        public["failures"].append("partial_graphics_evidence")


_CAPABILITY_COUNTS = ("cuda_workers", "glx_workers", "egl_workers", "vulkan_workers")


def _record_failure(public, private, error) -> None:
    private["failure_type"] = type(error).__name__
    private["failure_detail"] = str(error)
    if isinstance(error, GpuHealthError):
        private["health"] = error.evidence
    if isinstance(error, GraphicsVerificationError):
        category = "invalid_graphics_target"
    elif isinstance(error, StorageIdentityError):
        category = "target_identity_failed"
    elif isinstance(error, GpuHealthError) and "graphics" in str(error).lower():
        category = "graphics_validation_failed"
    elif isinstance(error, GpuHealthError):
        category = "gpu_health_failed"
    else:
        category = "verification_operation_failed"
    public["failures"].append(category)


def _write_receipt(directory: Path, index: int, receipt: dict) -> str:
    content = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(content).hexdigest()
    path = directory / f"target-{index}-{digest}.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
    return digest


def _aggregate(reports: list[dict]) -> dict:
    successful = [report for report in reports if report["passed"]]
    result = {
        "passed": len(successful) == len(reports),
        "selected_clusters": len(reports),
        "verified_clusters": len(successful),
        "clusters": reports,
    }
    for name in ("gpu_workers", "gpus", *_CAPABILITY_COUNTS):
        result[name] = sum(report[name] for report in successful)
    result["failures"] = sorted(
        {item for report in reports for item in report["failures"]}
    )
    serialized = json.dumps(reports, sort_keys=True, separators=(",", ":")).encode()
    result["evidence_sha256"] = hashlib.sha256(serialized).hexdigest()
    return result
