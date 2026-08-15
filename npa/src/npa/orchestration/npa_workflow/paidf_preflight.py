"""Fail-fast prerequisites for the Physical AI Data Factory workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from npa.orchestration.npa_workflow.kubernetes_prerequisites import (
    format_cpu_memory_requirement,
    ready_schedulable_cpu_nodes,
)
from npa.orchestration.skypilot.controller import (
    DEFAULT_K8S_CONTROLLER_CPUS,
    DEFAULT_K8S_CONTROLLER_MEMORY_GB,
)


Issue = tuple[str, str]
_REQUIRED_SECRET_ENVS = (
    "NEBIUS_TOKEN_FACTORY_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
)
_TRANSFER_REPO = "nvidia/Cosmos-Transfer2.5-2B"
_PAIDF_CPU_MILLICORES = (4 + DEFAULT_K8S_CONTROLLER_CPUS) * 1000
_PAIDF_MEMORY_BYTES = (16 + DEFAULT_K8S_CONTROLLER_MEMORY_GB) * 1024**3


def cpu_placement_requirement() -> str:
    return format_cpu_memory_requirement(
        _PAIDF_CPU_MILLICORES, _PAIDF_MEMORY_BYTES
    )


def static_prerequisites(
    *,
    requested_secret_envs: Sequence[str],
    secret_values: Mapping[str, str],
    hf_validator: Callable[[str, str], Any],
) -> list[Issue]:
    """Validate runtime secret forwarding and gated Transfer access."""

    issues: list[Issue] = []
    requested = {str(name).strip() for name in requested_secret_envs}
    not_forwarded = [name for name in _REQUIRED_SECRET_ENVS if name not in requested]
    if not_forwarded:
        issues.append(
            (
                "PAIDF runtime credentials are not forwarded with --secret-env: "
                + ", ".join(not_forwarded),
                "add `--secret-env NEBIUS_TOKEN_FACTORY_KEY --secret-env "
                "AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY --secret-env "
                "HF_TOKEN`; values resolve from the environment or the selected "
                "project's NPA credential store",
            )
        )

    hf_token = str(secret_values.get("HF_TOKEN") or "").strip()
    if hf_token:
        result = hf_validator(hf_token, _TRANSFER_REPO)
        if not getattr(result, "ok", False):
            detail = str(getattr(result, "error", "") or "access not verified")
            issues.append(
                (
                    f"Hugging Face access failed for {_TRANSFER_REPO} ({detail})",
                    f"accept the NVIDIA model terms at https://huggingface.co/{_TRANSFER_REPO}, "
                    "then run `npa workbench health access --capability paidf`",
                )
            )
    return issues


def _ready_schedulable_cpu_nodes(nodes_json: str) -> list[str]:
    return ready_schedulable_cpu_nodes(
        nodes_json,
        minimum_cpu_millicores=_PAIDF_CPU_MILLICORES,
        minimum_memory_bytes=_PAIDF_MEMORY_BYTES,
    )


def kubernetes_prerequisites(*, runner: Callable[[list[str]], Any]) -> list[Issue]:
    """Validate the CPU placement shared by PAIDF stages and its controller."""

    nodes = runner(["get", "nodes", "-o", "json"])
    if getattr(nodes, "returncode", 1) != 0:
        return [
            (
                "Kubernetes nodes cannot be listed for PAIDF CPU/GPU placement",
                "refresh the selected kube context and grant read access to nodes; "
                "run `kubectl get nodes -o wide`",
            )
        ]
    if _ready_schedulable_cpu_nodes(str(getattr(nodes, "stdout", ""))):
        return []
    return [
        (
            "no Ready, schedulable, appropriately untainted node can fit the "
            f"PAIDF CPU stage plus SkyPilot controller ({cpu_placement_requirement()})",
            "provide a node with sufficient allocatable CPU and memory (the default "
            "cpu-d3/8vcpu-32gb preset is sufficient), wait for Ready, then rerun "
            "`kubectl get nodes -o json` on the selected context",
        )
    ]
