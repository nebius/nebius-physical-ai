"""Capacity-aware GPU placement for direct-Kubernetes Sim2Real Jobs.

The retry boundary in this module is deliberately narrow: a Job may move to
the next compatible product only when Kubernetes supplies concrete scheduling
or GPU-capacity evidence.  Image pulls, credentials, container exits, model
weights, and application errors are never treated as placement failures.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


Kubectl = Callable[..., Any]


class GpuCapacityExhausted(RuntimeError):
    """Raised after every compatible cluster GPU product is exhausted."""

    def __init__(self, message: str, *, provenance: dict[str, Any]) -> None:
        super().__init__(message)
        self.provenance = provenance


class GpuJobFailure(RuntimeError):
    """Raised for a non-capacity Job failure, which must not change products."""

    def __init__(self, message: str, *, provenance: dict[str, Any]) -> None:
        super().__init__(message)
        self.provenance = provenance


@dataclass(frozen=True)
class CandidatePlan:
    products: tuple[str, ...]
    discovered_products: tuple[str, ...]
    skipped: tuple[dict[str, str], ...]


_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("rtx-pro-6000", re.compile(r"(?:RTX.*6000|6000.*RTX)", re.I)),
    ("l40s", re.compile(r"(?:^|[^A-Z0-9])L40S(?:[^A-Z0-9]|$)", re.I)),
    ("h100", re.compile(r"(?:^|[^A-Z0-9])H100(?:[^A-Z0-9]|$)", re.I)),
    ("h200", re.compile(r"(?:^|[^A-Z0-9])H200(?:[^A-Z0-9]|$)", re.I)),
    ("b200", re.compile(r"(?:^|[^A-Z0-9])B200(?:[^A-Z0-9]|$)", re.I)),
    ("b300", re.compile(r"(?:^|[^A-Z0-9])B300(?:[^A-Z0-9]|$)", re.I)),
)

_ISAAC_FAMILIES = frozenset({"rtx-pro-6000", "l40s"})
_TRANSFER_FAMILIES = frozenset(
    {"rtx-pro-6000", "l40s", "h100", "h200", "b200"}
)
_NON_ISAAC_FAMILIES = frozenset(
    {"rtx-pro-6000", "l40s", "h100", "h200", "b200", "b300"}
)
_ARCH_MARKERS = {
    "rtx-pro-6000": ("sm120",),
    # Repository CUDA images that support L40S commonly advertise their
    # Ampere/Ada-compatible fatbin as sm80/sm90 rather than spelling out sm89.
    "l40s": ("sm80", "sm89", "sm90"),
    "h100": ("sm90",),
    "h200": ("sm90",),
    "b200": ("sm100",),
    "b300": ("sm103",),
}


def split_candidate_products(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return a de-duplicated ordered candidate surface."""

    if value is None:
        return ()
    raw = value if not isinstance(value, str) else re.split(r"[,;\n]", value)
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        product = str(item).strip()
        if product and product not in seen:
            result.append(product)
            seen.add(product)
    return tuple(result)


def normalize_gpu_family(product: str) -> str:
    """Normalize Kubernetes product-label variants to a repository GPU family."""

    cleaned = str(product or "").strip()
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(cleaned):
            return family
    return "unknown"


def products_from_node_payload(payload: str | dict[str, Any]) -> tuple[str, ...]:
    """Discover unique product labels from the actual cluster node inventory."""

    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return ()
    else:
        decoded = payload
    products: list[str] = []
    seen: set[str] = set()
    for node in decoded.get("items", []) or []:
        labels = (node.get("metadata") or {}).get("labels") or {}
        product = str(labels.get("nvidia.com/gpu.product") or "").strip()
        if product and product not in seen:
            products.append(product)
            seen.add(product)
    return tuple(products)


def workload_kind(component: str, *, sim_backend: str = "") -> str:
    name = str(component or "").lower()
    if "isaac" in name or (name == "heldout_eval" and sim_backend == "isaac"):
        return "isaac"
    if "cosmos2_transfer" in name or "transfer" in name:
        return "cosmos_transfer"
    if "vlm_eval" in name or "reason" in name:
        return "cosmos_reason"
    return "non_isaac"


def product_is_compatible(product: str, *, workload: str, image: str = "") -> bool:
    """Apply RT-core, model, and advertised image-architecture constraints."""

    family = normalize_gpu_family(product)
    allowed = (
        _ISAAC_FAMILIES
        if workload == "isaac"
        else _TRANSFER_FAMILIES
        if workload == "cosmos_transfer"
        else _NON_ISAAC_FAMILIES
    )
    if family not in allowed:
        return False
    lowered_image = str(image or "").lower().replace("_", "").replace("-", "")
    advertised = set(re.findall(r"sm(?:80|89|90|100|103|120)", lowered_image))
    if not advertised:
        return True
    return bool(advertised.intersection(_ARCH_MARKERS[family]))


def ordered_compatible_products(
    *,
    preferred: str,
    explicit: str | Sequence[str] | None,
    discovered: Iterable[str],
    workload: str,
    image: str = "",
) -> CandidatePlan:
    """Resolve an ordered plan using real node labels whenever discovery works."""

    inventory = split_candidate_products(tuple(discovered))
    requested = split_candidate_products((preferred, *split_candidate_products(explicit)))
    discovery_available = bool(inventory)
    by_family: dict[str, list[str]] = {}
    for product in inventory:
        by_family.setdefault(normalize_gpu_family(product), []).append(product)

    ordered: list[str] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(product: str) -> None:
        if product not in seen:
            ordered.append(product)
            seen.add(product)

    for requested_product in requested:
        family = normalize_gpu_family(requested_product)
        if not product_is_compatible(requested_product, workload=workload, image=image):
            skipped.append(
                {
                    "product": requested_product,
                    "status": "filtered",
                    "scheduling_reason": f"incompatible with {workload} workload/image",
                }
            )
            continue
        if discovery_available:
            matches = (
                [item for item in inventory if item == requested_product]
                or by_family.get(family, [])
            )
            if not matches:
                skipped.append(
                    {
                        "product": requested_product,
                        "status": "unavailable",
                        "scheduling_reason": (
                            "no nodes matching selected GPU product in discovered cluster labels"
                        ),
                    }
                )
                continue
            for match in matches:
                add(match)
        else:
            add(requested_product)

    for product in inventory:
        if product_is_compatible(product, workload=workload, image=image):
            add(product)

    return CandidatePlan(tuple(ordered), inventory, tuple(skipped))


def capacity_scheduling_reason(
    *,
    pod_payload: str | dict[str, Any] = "",
    event_payload: str | dict[str, Any] = "",
    gpu_resource: str = "nvidia.com/gpu",
    product: str = "",
) -> str:
    """Return concrete GPU capacity/selector evidence, never runtime evidence."""

    def decode(value: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(value or "{}")
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    messages: list[str] = []
    pods = decode(pod_payload)
    for pod in pods.get("items", []) or []:
        for condition in (pod.get("status") or {}).get("conditions", []) or []:
            if str(condition.get("reason") or "").lower() == "unschedulable":
                messages.append(str(condition.get("message") or ""))
    events = decode(event_payload)
    for event in events.get("items", []) or []:
        reason = str(event.get("reason") or "")
        message = str(event.get("message") or "")
        if reason.lower() in {"failedscheduling", "unschedulable"}:
            messages.append(message)

    resource = re.escape(gpu_resource)
    for message in messages:
        lowered = message.lower()
        if re.search(rf"insufficient\s+{resource}", message, re.I):
            return f"Unschedulable: {message.strip()}"
        selector_evidence = (
            "didn't match pod's node affinity/selector" in lowered
            or "did not match pod's node affinity/selector" in lowered
            or "node selector" in lowered
            or "node affinity" in lowered
            or "no nodes match" in lowered
        )
        if selector_evidence and product:
            return f"Unschedulable for GPU product {product}: {message.strip()}"
    return ""


def _attempt_job_name(base: str, index: int) -> str:
    if index == 0:
        return base
    suffix = f"-gpu{index + 1}"
    return f"{base[: 63 - len(suffix)].rstrip('-')}{suffix}"


def _pod_proof(payload: str) -> dict[str, Any]:
    try:
        items = (json.loads(payload).get("items") or [])
    except (json.JSONDecodeError, AttributeError):
        items = []
    if not items:
        return {}
    pod = items[0]
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    return {
        "pod_name": (pod.get("metadata") or {}).get("name", ""),
        "node_name": (pod.get("spec") or {}).get("nodeName", ""),
        "image_digests": [
            str(status.get("imageID") or "")
            for status in statuses
            if status.get("imageID")
        ],
    }


def run_gpu_job_with_fallback(
    *,
    kubectl: Kubectl,
    manifest_factory: Callable[[str, str], dict[str, Any]],
    base_job_name: str,
    namespace: str,
    image: str,
    preferred_product: str,
    explicit_candidates: str | Sequence[str] | None,
    workload: str,
    gpu_resource: str,
    gpu_count: int,
    timeout_s: int,
    wait_for_completion: bool = True,
) -> dict[str, Any]:
    """Apply/wait a Job, retrying only concrete GPU scheduling failures."""

    nodes = kubectl(["get", "nodes", "-o", "json"], timeout_s=120)
    discovered = products_from_node_payload(nodes.stdout) if nodes.returncode == 0 else ()
    plan = ordered_compatible_products(
        preferred=preferred_product,
        explicit=explicit_candidates,
        discovered=discovered,
        workload=workload,
        image=image,
    )
    attempts: list[dict[str, Any]] = [dict(item) for item in plan.skipped]
    provenance: dict[str, Any] = {
        "candidate_order": list(plan.products),
        "discovered_products": list(plan.discovered_products),
        "attempts": attempts,
        "selected_product": "",
        "selected_node": "",
        "allocated_gpu": {"resource": gpu_resource, "count": gpu_count},
        "image": image,
        "image_digests": [],
    }
    if not plan.products:
        raise GpuCapacityExhausted(
            f"no compatible GPU products are available for {workload}; "
            f"discovered={list(plan.discovered_products)} skipped={attempts}",
            provenance=provenance,
        )

    probe_s = max(0, int(os.environ.get("NPA_SIM2REAL_GPU_SCHEDULING_PROBE_SECONDS", "30")))
    poll_s = max(1, int(os.environ.get("NPA_SIM2REAL_GPU_SCHEDULING_POLL_SECONDS", "2")))
    for index, product in enumerate(plan.products):
        attempt_started = time.monotonic()
        job_name = _attempt_job_name(base_job_name, index)
        manifest = manifest_factory(product, job_name)
        kubectl(
            ["delete", "job", job_name, "-n", namespace, "--ignore-not-found=true"],
            timeout_s=120,
        )
        apply = kubectl(
            ["apply", "-f", "-"], stdin=json.dumps(manifest), timeout_s=120
        )
        attempt: dict[str, Any] = {
            "product": product,
            "job_name": job_name,
            "image": image,
            "status": "applied",
            "scheduling_reason": "",
        }
        attempts.append(attempt)
        if apply.returncode != 0:
            attempt.update(
                status="apply_failed",
                scheduling_reason=str(apply.stderr or apply.stdout),
                duration_s=round(time.monotonic() - attempt_started, 3),
            )
            raise GpuJobFailure(
                f"Kubernetes apply failed for {job_name}; refusing GPU product fallback",
                provenance=provenance,
            )

        deadline = time.monotonic() + probe_s
        capacity_reason = ""
        pod_result = None
        while True:
            pod_result = kubectl(
                ["get", "pods", "-n", namespace, "-l", f"job-name={job_name}", "-o", "json"],
                timeout_s=120,
            )
            proof = _pod_proof(pod_result.stdout if pod_result.returncode == 0 else "")
            if proof.get("node_name"):
                break
            events = kubectl(
                [
                    "get",
                    "events",
                    "-n",
                    namespace,
                    "--field-selector",
                    f"involvedObject.name={job_name}",
                    "-o",
                    "json",
                ],
                timeout_s=120,
            )
            capacity_reason = capacity_scheduling_reason(
                pod_payload=pod_result.stdout if pod_result.returncode == 0 else "",
                event_payload=events.stdout if events.returncode == 0 else "",
                gpu_resource=gpu_resource,
                product=product,
            )
            if capacity_reason or time.monotonic() >= deadline:
                break
            time.sleep(poll_s)

        if capacity_reason:
            attempt.update(
                status="unschedulable",
                scheduling_reason=capacity_reason,
                duration_s=round(time.monotonic() - attempt_started, 3),
            )
            kubectl(
                ["delete", "job", job_name, "-n", namespace, "--ignore-not-found=true", "--wait=true"],
                timeout_s=300,
            )
            continue

        if not wait_for_completion:
            proof = _pod_proof(pod_result.stdout if pod_result and pod_result.returncode == 0 else "")
            attempt.update(
                status="scheduled" if proof.get("node_name") else "submitted",
                scheduling_reason="scheduled" if proof.get("node_name") else "no capacity failure observed",
                node_name=str(proof.get("node_name") or ""),
                image_digests=list(proof.get("image_digests") or []),
                duration_s=round(time.monotonic() - attempt_started, 3),
            )
            provenance.update(
                selected_product=product,
                selected_node=str(proof.get("node_name") or ""),
                image_digests=list(proof.get("image_digests") or []),
                job_name=job_name,
                duration_s=round(time.monotonic() - attempt_started, 3),
            )
            return provenance

        wait = kubectl(
            [
                "wait",
                f"job/{job_name}",
                "-n",
                namespace,
                "--for=condition=complete",
                f"--timeout={max(1, timeout_s)}s",
            ],
            timeout_s=timeout_s + 60,
        )
        pod_result = kubectl(
            ["get", "pods", "-n", namespace, "-l", f"job-name={job_name}", "-o", "json"],
            timeout_s=120,
        )
        proof = _pod_proof(pod_result.stdout if pod_result.returncode == 0 else "")
        if wait.returncode == 0:
            attempt.update(
                status="complete",
                scheduling_reason="scheduled",
                node_name=str(proof.get("node_name") or ""),
                image_digests=list(proof.get("image_digests") or []),
                duration_s=round(time.monotonic() - attempt_started, 3),
            )
            provenance.update(
                selected_product=product,
                selected_node=str(proof.get("node_name") or ""),
                image_digests=list(proof.get("image_digests") or []),
                job_name=job_name,
                duration_s=round(time.monotonic() - attempt_started, 3),
            )
            return provenance

        events = kubectl(
            [
                "get",
                "events",
                "-n",
                namespace,
                "--field-selector",
                f"involvedObject.name={job_name}",
                "-o",
                "json",
            ],
            timeout_s=120,
        )
        capacity_reason = capacity_scheduling_reason(
            pod_payload=pod_result.stdout if pod_result.returncode == 0 else "",
            event_payload=events.stdout if events.returncode == 0 else "",
            gpu_resource=gpu_resource,
            product=product,
        )
        if capacity_reason:
            attempt.update(
                status="unschedulable",
                scheduling_reason=capacity_reason,
                duration_s=round(time.monotonic() - attempt_started, 3),
            )
            kubectl(
                ["delete", "job", job_name, "-n", namespace, "--ignore-not-found=true", "--wait=true"],
                timeout_s=300,
            )
            continue
        attempt.update(
            status="failed",
            scheduling_reason=str(wait.stderr or wait.stdout or "Job did not complete"),
            duration_s=round(time.monotonic() - attempt_started, 3),
        )
        raise GpuJobFailure(
            f"Kubernetes Job {job_name} failed without GPU capacity evidence; "
            "refusing to change workload product",
            provenance=provenance,
        )

    exact = "; ".join(
        f"{item.get('product')}: {item.get('scheduling_reason')}"
        for item in attempts
    )
    raise GpuCapacityExhausted(
        f"all compatible GPU products exhausted for {workload}: {exact}",
        provenance=provenance,
    )
