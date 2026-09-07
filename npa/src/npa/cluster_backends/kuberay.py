"""Validated opt-in CPU RayCluster policy for the existing mk8s backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

RAY_VERSION = "2.58.0"
RAY_IMAGE = (
    "docker.io/rayproject/ray@sha256:"
    "507464fe56b3d24cec2e812a25850db91b97d52752d63119fecf6914f7b0a37a"
)


@dataclass(frozen=True)
class KubeRaySpec:
    """A fixed CPU worker group; native Ray APIs own application execution."""

    enabled: bool = False
    worker_replicas: int = 1
    worker_cpus: int = 2
    worker_memory_gib: int = 4

    def validate(self, *, cpu_nodes: Any | None) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("kuberay.enabled must be a boolean")
        for name in ("worker_replicas", "worker_cpus", "worker_memory_gib"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"kuberay.{name} must be a positive integer")
        if not self.enabled:
            if self != KubeRaySpec():
                raise ValueError("kuberay worker settings require enabled: true")
            return
        if cpu_nodes is None or cpu_nodes.count <= 0 or cpu_nodes.is_gpu():
            raise ValueError("kuberay requires a nonempty CPU node pool")
        if not re.fullmatch(r"cpu-[a-z0-9-]+", cpu_nodes.platform) or not re.fullmatch(
            r"[1-9][0-9]*vcpu-[1-9][0-9]*gb", cpu_nodes.preset
        ):
            raise ValueError("kuberay requires an explicit CPU platform and preset")

    def plan(self) -> dict[str, Any]:
        return {**asdict(self), "ray_version": RAY_VERSION, "image": RAY_IMAGE}


def kuberay_spec_from_mapping(value: Any) -> KubeRaySpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("kuberay must be a mapping")
    unknown = set(value) - set(KubeRaySpec.__dataclass_fields__)
    if unknown:
        raise ValueError(
            "kuberay has unsupported fields; only enabled, worker_replicas, "
            "worker_cpus and worker_memory_gib are supported (CPU RayCluster only)"
        )
    if value.get("enabled") is not True and set(value) - {"enabled"}:
        raise ValueError("kuberay worker settings require enabled: true")
    return KubeRaySpec(**value)


def validate_kuberay(cluster: Any) -> None:
    policy = cluster.kuberay
    if policy is None:
        return
    if not isinstance(policy, KubeRaySpec):
        raise ValueError("kuberay must be a KubeRaySpec")
    if cluster.backend_name() != "mk8s":
        raise ValueError("kuberay is supported only by the mk8s backend")
    policy.validate(cpu_nodes=cluster.cpu_nodes)


def validate_recipe_kuberay_compatibility(cluster: Any, recipe_dir: Path) -> None:
    """Require reviewed module wiring and safety bytes before any cloud mutation.

    Alternate recipe revisions are deliberately unsupported unless their KubeRay
    contract matches the reviewed vendored files. Declarations alone cannot prove
    that an application actually consumes the requested values.
    """

    validate_kuberay(cluster)
    if cluster.kuberay is None or not cluster.kuberay.enabled:
        return
    contract = json.loads(
        Path(__file__).with_name("kuberay_recipe_contract.json").read_text()
    )
    for relative, expected in contract.items():
        path = recipe_dir.parent / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(
                "Selected k8s-training recipe cannot honor the reviewed CPU "
                "KubeRay contract. Use the vendored recipe; unsupported pinned, "
                "upstream or local recipes are rejected before provisioning."
            )
