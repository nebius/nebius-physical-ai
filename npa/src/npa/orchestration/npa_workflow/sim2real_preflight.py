"""Fail-fast prerequisites for the canonical compositional Sim2Real workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import re
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
_IMAGE_KEYS = (
    "controller_image",
    "transfer_image",
    "envgen_image",
    "isaac_image",
    "viewer_image",
)
_REQUIRED_SECRET_ENVS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "NEBIUS_TOKEN_FACTORY_KEY",
)
_DIGEST_IMAGE = re.compile(r"^[^/\s]+/.+@sha256:[0-9a-fA-F]{64}$")
_SIM2REAL_CPU_MILLICORES = (8 + DEFAULT_K8S_CONTROLLER_CPUS) * 1000
_SIM2REAL_MEMORY_BYTES = (32 + DEFAULT_K8S_CONTROLLER_MEMORY_GB) * 1024**3


def cpu_placement_requirement() -> str:
    return format_cpu_memory_requirement(
        _SIM2REAL_CPU_MILLICORES, _SIM2REAL_MEMORY_BYTES
    )


def static_prerequisites(
    config: Mapping[str, Any],
    *,
    requested_secret_envs: Sequence[str],
    secret_values: Mapping[str, str],
    hf_validator: Callable[[str, str], Any],
    token_factory_validator: Callable[[str, str], Any],
) -> list[Issue]:
    """Validate immutable inputs, consent, secret forwarding, and gated access."""

    issues: list[Issue] = []
    invalid_images = [
        key
        for key in _IMAGE_KEYS
        if not _DIGEST_IMAGE.fullmatch(str(config.get(key) or "").strip())
    ]
    if invalid_images:
        issues.append(
            (
                "Sim2Real image inputs are missing or not registry-qualified immutable "
                f"digests: {', '.join(invalid_images)}",
                "set every listed --var to <registry>/<repository>@sha256:<64-hex>; "
                "run `npa workbench workflow preflight-images <spec> --var ...` to "
                "verify the exact bytes",
            )
        )

    pvc = str(config.get("isaac_cache_pvc") or "").strip()
    if not pvc:
        issues.append(
            (
                "config.isaac_cache_pvc is empty",
                "create and warm the shared Isaac cache from "
                "npa/docker/workbench/common/warm-isaac-cache.yaml, then pass "
                "--var isaac_cache_pvc=<bound-rwx-pvc>",
            )
        )

    requested = {str(name).strip() for name in requested_secret_envs}
    not_forwarded = [name for name in _REQUIRED_SECRET_ENVS if name not in requested]
    if not_forwarded:
        issues.append(
            (
                "required runtime credentials are not forwarded with --secret-env: "
                + ", ".join(not_forwarded),
                "add `--secret-env AWS_ACCESS_KEY_ID --secret-env "
                "AWS_SECRET_ACCESS_KEY --secret-env HF_TOKEN --secret-env "
                "NEBIUS_TOKEN_FACTORY_KEY`; values resolve from the "
                "environment or the selected project's NPA credential store",
            )
        )

    hf_token = str(secret_values.get("HF_TOKEN") or "").strip()
    if hf_token:
        repos = ["nvidia/Cosmos-Transfer2.5-2B"]
        denied: list[str] = []
        for repo in (item for item in repos if item):
            result = hf_validator(hf_token, repo)
            if not getattr(result, "ok", False):
                detail = str(getattr(result, "error", "") or "access not verified")
                denied.append(f"{repo} ({detail})")
        if denied:
            issues.append(
                (
                    "Hugging Face access failed for required runtime model(s): "
                    + "; ".join(denied),
                    "accept each model's terms while signed in to the account that owns "
                    "HF_TOKEN, then run `npa workbench health access --capability "
                    "sim2real` until the repository passes",
                )
            )
    token_factory_key = str(secret_values.get("NEBIUS_TOKEN_FACTORY_KEY") or "").strip()
    if token_factory_key:
        model = str(config.get("cosmos3_model") or "nvidia/Cosmos3-Super-Reasoner").strip()
        result = token_factory_validator(token_factory_key, model)
        if not getattr(result, "ok", False):
            detail = str(getattr(result, "error", "") or "access not verified")
            issues.append(
                (
                    f"Token Factory access failed for {model}: {detail}",
                    "verify the selected key, project-specific model availability, and "
                    "balance with `npa workbench token-factory models` and a minimal "
                    "inference before workflow submission",
                )
            )
    elif "NEBIUS_TOKEN_FACTORY_KEY" in requested:
        issues.append(
            (
                "NEBIUS_TOKEN_FACTORY_KEY could not be resolved for Sim2Real",
                "store the key in the selected project's private NPA credential store "
                "or runtime environment, then rerun Token Factory model preflight",
            )
        )
    return issues


def _ready_schedulable_cpu_nodes(nodes_json: str) -> list[str]:
    """Return nodes able to host one CPU stage and the SkyPilot controller."""
    return ready_schedulable_cpu_nodes(
        nodes_json,
        minimum_cpu_millicores=_SIM2REAL_CPU_MILLICORES,
        minimum_memory_bytes=_SIM2REAL_MEMORY_BYTES,
    )


def kubernetes_prerequisites(
    config: Mapping[str, Any],
    *,
    runner: Callable[[list[str]], Any],
    namespace: str = "default",
) -> list[Issue]:
    """Validate cluster objects the real Sim2Real/SkyPilot path consumes."""

    issues: list[Issue] = []
    nodes = runner(["get", "nodes", "-o", "json"])
    if getattr(nodes, "returncode", 1) != 0:
        issues.append(
            (
                "Kubernetes nodes cannot be listed for Sim2Real CPU/GPU placement",
                "refresh the selected kube context and grant read access to nodes; run "
                "`kubectl get nodes -o wide`",
            )
        )
    elif not _ready_schedulable_cpu_nodes(str(getattr(nodes, "stdout", ""))):
        issues.append(
            (
                "no Ready, schedulable, appropriately untainted node can fit the "
                "Sim2Real CPU stage plus SkyPilot controller "
                f"({cpu_placement_requirement()})",
                "provide a node with sufficient allocatable CPU and memory, such as "
                "cpu-e2/16vcpu-64gb (the "
                "8vcpu-32gb nominal preset loses capacity to Kubernetes reserve), wait for "
                "Ready, then rerun `kubectl get nodes -o json` on the selected context",
            )
        )

    pvc_name = str(config.get("isaac_cache_pvc") or "").strip()
    if pvc_name:
        pvc_result = runner(["get", "pvc", pvc_name, "-n", namespace, "-o", "json"])
        pvc: dict[str, Any] = {}
        if getattr(pvc_result, "returncode", 1) == 0:
            try:
                pvc = json.loads(str(getattr(pvc_result, "stdout", ""))) or {}
            except json.JSONDecodeError:
                pvc = {}
        modes = set((pvc.get("spec") or {}).get("accessModes") or [])
        if (
            (pvc.get("status") or {}).get("phase") != "Bound"
            or "ReadWriteMany" not in modes
        ):
            issues.append(
                (
                    f"Isaac cache PVC {pvc_name!r} is missing or is not Bound ReadWriteMany "
                    f"in namespace {namespace!r}",
                    "apply npa/docker/workbench/common/warm-isaac-cache.yaml with the "
                    "selected immutable Isaac image, wait for the warm Job to complete, "
                    "and pass that PVC name",
                )
            )

    return issues
