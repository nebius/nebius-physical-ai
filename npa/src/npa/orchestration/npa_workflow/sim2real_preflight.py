"""Fail-fast prerequisites for the canonical compositional Sim2Real workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import re
from typing import Any


Issue = tuple[str, str]
_ACCEPTED_EULA_VALUES = frozenset({"1", "TRUE", "Y", "YES"})
_IMAGE_KEYS = (
    "controller_image",
    "transfer_image",
    "envgen_image",
    "reason_image",
    "isaac_image",
    "viewer_image",
)
_REQUIRED_SECRET_ENVS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
)
_DIGEST_IMAGE = re.compile(r"^[^/\s]+/.+@sha256:[0-9a-fA-F]{64}$")


def static_prerequisites(
    config: Mapping[str, Any],
    *,
    requested_secret_envs: Sequence[str],
    secret_values: Mapping[str, str],
    hf_validator: Callable[[str, str], Any],
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

    invalid_eulas = [
        key
        for key in ("omni_kit_accept_eula", "isaacsim_accept_eula")
        if str(config.get(key) or "").strip().upper() not in _ACCEPTED_EULA_VALUES
    ]
    if invalid_eulas:
        issues.append(
            (
                "explicit NVIDIA Isaac/Omniverse acceptance is missing: "
                + ", ".join(invalid_eulas),
                "after reviewing the linked NVIDIA terms, pass "
                "--var omni_kit_accept_eula=YES --var isaacsim_accept_eula=YES",
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
                "AWS_SECRET_ACCESS_KEY --secret-env HF_TOKEN`; values resolve from the "
                "environment or the selected project's NPA credential store",
            )
        )

    hf_token = str(secret_values.get("HF_TOKEN") or "").strip()
    if hf_token:
        repos = list(
            dict.fromkeys(
                [
                    "nvidia/Cosmos-Transfer2.5-2B",
                    str(config.get("reason2_model") or "").strip(),
                    str(config.get("reason3_model") or "").strip(),
                ]
            )
        )
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
                    "sim2real` until all three repositories PASS",
                )
            )
    return issues


def _cpu_millicores(value: object) -> int:
    raw = str(value or "").strip()
    try:
        return int(raw[:-1]) if raw.endswith("m") else int(float(raw) * 1000)
    except ValueError:
        return 0


def _memory_bytes(value: object) -> int:
    raw = str(value or "").strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]i?|)", raw)
    if not match:
        return 0
    number = float(match.group(1))
    suffix = match.group(2)
    if not suffix:
        return int(number)
    powers = {letter: index for index, letter in enumerate("KMGTPE", 1)}
    return int(number * ((1024 if suffix.endswith("i") else 1000) ** powers[suffix[0]]))


def _ready_schedulable_cpu_nodes(nodes_json: str) -> list[str]:
    """Return non-GPU nodes able to host the 8 CPU / 32 GiB Sim2Real profile."""

    try:
        items = (json.loads(nodes_json) or {}).get("items") or []
    except (TypeError, json.JSONDecodeError):
        return []
    ready: list[str] = []
    for node in items:
        spec = node.get("spec") or {}
        status = node.get("status") or {}
        allocatable = status.get("allocatable") or {}
        conditions = status.get("conditions") or []
        is_ready = any(
            item.get("type") == "Ready" and str(item.get("status")).lower() == "true"
            for item in conditions
        )
        blocking_taint = any(
            str(item.get("effect") or "") in {"NoSchedule", "NoExecute"}
            for item in (spec.get("taints") or [])
        )
        if (
            is_ready
            and not spec.get("unschedulable", False)
            and not blocking_taint
            and _cpu_millicores(allocatable.get("cpu")) >= 8000
            and _memory_bytes(allocatable.get("memory")) >= 32 * 1024**3
            and _cpu_millicores(allocatable.get("nvidia.com/gpu")) == 0
        ):
            ready.append(str((node.get("metadata") or {}).get("name") or "<unnamed>"))
    return ready


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
                "no Ready, untainted, non-GPU node can fit the Sim2Real 8 vCPU / "
                "32 GiB CPU profile (and the 4 vCPU / 16 GiB SkyPilot controller)",
                "add a dedicated CPU node pool such as cpu-e2/16vcpu-64gb (the "
                "8vcpu-32gb nominal preset loses capacity to Kubernetes reserve), wait for "
                "Ready, then rerun `kubectl get nodes -o json`",
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

    queue = str(config.get("gpu_queue") or "").strip()
    if queue:
        result = runner(["get", "localqueue.kueue.x-k8s.io", queue, "-n", namespace, "-o", "json"])
        if getattr(result, "returncode", 1) != 0:
            issues.append(
                (
                    f"Kueue LocalQueue {queue!r} is not readable in namespace {namespace!r}",
                    "install/configure Kueue and apply the Sim2Real ResourceFlavor, "
                    "ClusterQueue, and LocalQueue before submission",
                )
            )

    priority = str(config.get("gpu_priority_class") or "").strip()
    if priority:
        result = runner(["get", "priorityclass.scheduling.k8s.io", priority, "-o", "json"])
        if getattr(result, "returncode", 1) != 0:
            issues.append(
                (
                    f"PriorityClass {priority!r} is not readable",
                    "create the Sim2Real PriorityClass before submission (the canonical "
                    "default is sim2real-production)",
                )
            )
    return issues
