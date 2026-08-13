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

from npa.workflows.sim2real.models import Sim2RealLoopConfig


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


def gpu_fallback_report_contract(
    config: Sim2RealLoopConfig, components: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return the public fail-closed placement contract and component evidence."""

    configured = list(
        dict.fromkeys(
            [config.k8s_gpu_product, *getattr(config, "k8s_gpu_candidates", ())]
        )
    )
    evidence: list[dict[str, Any]] = []
    for component in components:
        artifacts = dict(component.get("artifacts") or {})
        if not artifacts.get("gpu_request"):
            continue
        evidence.append(
            {
                "component": component.get("name", ""),
                "candidate_order": artifacts.get("gpu_candidate_order", []),
                "attempts": artifacts.get("gpu_attempts", []),
                "selected_product": artifacts.get("selected_gpu_product", ""),
                "selected_node": artifacts.get("selected_gpu_node", ""),
                "allocated_gpu": artifacts.get("allocated_gpu", {}),
                "minimum_vram_gb": artifacts.get("minimum_vram_gb", 0),
                "model_requirement": artifacts.get("model_requirement", ""),
                "job_name": artifacts.get("job_name", ""),
                "image_digests": artifacts.get("image_digests", []),
                "status": component.get("tier", ""),
                "duration_s": artifacts.get("duration_s", ""),
                "artifact": next(
                    (
                        artifacts[key]
                        for key in (
                            "remote",
                            "report",
                            "checkpoint",
                            "prefix",
                            "raw_envs",
                        )
                        if artifacts.get(key)
                    ),
                    "",
                ),
            }
        )
    return {
        "configured_order": configured,
        "discovery_source": (
            "actual nvidia.com/gpu.product node labels plus ordered configuration"
        ),
        "retry_evidence": (
            "only Kubernetes Unschedulable GPU capacity/product selector evidence"
        ),
        "compatibility_filters": [
            "workload GPU family",
            "image-advertised CUDA SM",
            "model/operator minimum VRAM",
        ],
        "never_retry": [
            "runtime",
            "image_pull",
            "credential",
            "checkpoint_or_weight",
            "container_exit",
            "application_failure",
        ],
        "isaac_compatible_families": ["RTX PRO 6000", "L40S"],
        "isaac_excluded_families": ["H100", "H200", "B200", "B300"],
        "preserves_real_tier": True,
        "exhaustion": "blocking failure with exact scheduler evidence",
        "component_provenance": evidence,
    }


_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("rtx-pro-6000", re.compile(r"(?:RTX.*6000|6000.*RTX)", re.I)),
    ("l40s", re.compile(r"(?:^|[^A-Z0-9])L40S(?:[^A-Z0-9]|$)", re.I)),
    ("h100", re.compile(r"(?:^|[^A-Z0-9])H100(?:[^A-Z0-9]|$)", re.I)),
    ("h200", re.compile(r"(?:^|[^A-Z0-9])H200(?:[^A-Z0-9]|$)", re.I)),
    ("b200", re.compile(r"(?:^|[^A-Z0-9])B200(?:[^A-Z0-9]|$)", re.I)),
    ("b300", re.compile(r"(?:^|[^A-Z0-9])B300(?:[^A-Z0-9]|$)", re.I)),
)

_ISAAC_FAMILIES = frozenset({"rtx-pro-6000", "l40s"})
_TRANSFER_FAMILIES = frozenset({"rtx-pro-6000", "l40s", "h100", "h200", "b200"})
_NON_ISAAC_FAMILIES = frozenset(
    {"rtx-pro-6000", "l40s", "h100", "h200", "b200", "b300"}
)
# Image tags are only architecture evidence when they spell out a SASS (``sm``)
# or PTX (``compute``) target.  Keep these sets aligned with the measured
# fatbin/PTX inventory in ``npa/docker/workbench/blackwell-dc-images.json``.
# In particular, RTX PRO 6000 is CUDA major 12: an image that advertises any
# architecture must explicitly carry sm_120 SASS or compute_120 PTX.  sm_100 /
# sm_103 SASS is a different CUDA major and is never treated as portable to RTX.
_ARCH_MARKERS = {
    "rtx-pro-6000": frozenset({"sm120", "compute120"}),
    # L40S (sm_89) may execute older same-major sm_80 SASS; sm_90 is a different
    # CUDA major and must not qualify an otherwise L40S-incompatible image.
    "l40s": frozenset({"sm80", "sm89", "compute80", "compute89"}),
    "h100": frozenset({"sm90", "compute90"}),
    "h200": frozenset({"sm90", "compute90"}),
    "b200": frozenset({"sm100", "compute100"}),
    # Repository packaging metadata proves sm_100 -> sm_103 same-major forward
    # compatibility; the reverse is intentionally not claimed for B200 above.
    "b300": frozenset({"sm100", "sm103", "compute100", "compute103"}),
}
_ARCH_MARKER_RE = re.compile(
    r"(?<![a-z0-9])(?:sm|compute)[_-]?(?:80|89|90|100|103|120)(?![a-z0-9])"
)
_FAMILY_VRAM_GB = {
    "rtx-pro-6000": 96,
    "l40s": 48,
    "h100": 80,
    "h200": 141,
    "b200": 180,
    "b300": 270,
}
_WORKLOAD_MIN_VRAM_GB = {
    "isaac": 48,
    "cosmos_transfer": 48,
    "cosmos_reason": 24,
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


def minimum_vram_for_workload(
    workload: str, *, model: str = "", explicit: str | int | None = None
) -> int:
    """Resolve a conservative VRAM floor from operator or model requirements."""

    if str(explicit or "").strip():
        try:
            required = int(str(explicit).strip())
        except ValueError:
            return 10**9
        return required if required >= 0 else 10**9
    baseline = _WORKLOAD_MIN_VRAM_GB.get(workload, 0)
    if workload != "cosmos_reason":
        return baseline
    parameter_counts = [
        int(match)
        for match in re.findall(r"(?:^|[^0-9])(\d{1,3})\s*[bB](?:[^a-zA-Z]|$)", model)
    ]
    largest = max(parameter_counts, default=0)
    if largest >= 70:
        return max(baseline, 141)
    if largest >= 32:
        return max(baseline, 80)
    if largest >= 14:
        return max(baseline, 48)
    return baseline


def product_is_compatible(
    product: str,
    *,
    workload: str,
    image: str = "",
    minimum_vram_gb: int = 0,
) -> bool:
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
    if _FAMILY_VRAM_GB.get(family, 0) < minimum_vram_gb:
        return False
    lowered_image = str(image or "").lower()
    advertised = {
        marker.replace("_", "").replace("-", "")
        for marker in _ARCH_MARKER_RE.findall(lowered_image)
    }
    if family in {"h100", "h200", "b200", "b300"} and not advertised:
        return False
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
    minimum_vram_gb: int = 0,
) -> CandidatePlan:
    """Resolve an ordered plan using real node labels whenever discovery works."""

    inventory = split_candidate_products(tuple(discovered))
    requested = split_candidate_products(
        (preferred, *split_candidate_products(explicit))
    )
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

    def valid_selector_value(product: str) -> bool:
        return bool(
            len(product) <= 63
            and re.fullmatch(r"[A-Za-z0-9](?:[-A-Za-z0-9_.]*[A-Za-z0-9])?", product)
        )

    for requested_product in requested:
        family = normalize_gpu_family(requested_product)
        if not product_is_compatible(
            requested_product,
            workload=workload,
            image=image,
            minimum_vram_gb=minimum_vram_gb,
        ):
            skipped.append(
                {
                    "product": requested_product,
                    "status": "filtered",
                    "scheduling_reason": (
                        f"incompatible with {workload} workload/image or "
                        f"minimum {minimum_vram_gb} GiB VRAM"
                    ),
                }
            )
            continue
        if discovery_available:
            matches = [
                item for item in inventory if item == requested_product
            ] or by_family.get(family, [])
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
        elif not valid_selector_value(requested_product):
            skipped.append(
                {
                    "product": requested_product,
                    "status": "unresolved_alias",
                    "scheduling_reason": (
                        "node discovery was unavailable and this alias is not a valid "
                        "Kubernetes node-label value"
                    ),
                }
            )
        else:
            add(requested_product)

    for product in inventory:
        if product_is_compatible(
            product,
            workload=workload,
            image=image,
            minimum_vram_gb=minimum_vram_gb,
        ):
            add(product)

    return CandidatePlan(tuple(ordered), inventory, tuple(skipped))


def _attempt_job_name(base: str, index: int) -> str:
    if index == 0:
        return base
    suffix = f"-gpu{index + 1}"
    return f"{base[: 63 - len(suffix)].rstrip('-')}{suffix}"


def _verify_runtime_digest(
    snapshot: Any, expected_image: str, *, container_name: str
) -> list[str]:
    """Require every completed application container to report the chosen digest."""

    expected_digest = expected_image.rsplit("@", 1)[-1]
    image_ids = snapshot.image_digests
    statuses = [
        status
        for pod in snapshot.pods
        for status in pod.containers
        if status.name == container_name
    ]
    if not statuses or any(
        expected_digest not in status.image_id for status in statuses
    ):
        raise GpuJobFailure(
            "completed Kubernetes Job did not attest the requested image digest",
            provenance={
                "expected_image": expected_image,
                "observed_image_ids": image_ids,
                "job_snapshot": snapshot.to_dict(),
            },
        )
    return image_ids


def run_gpu_job_with_fallback(
    *,
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
    minimum_vram_gb: int | None = None,
    model: str = "",
    client: Any | None = None,
    heartbeat: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Create/adopt a queued GPU Job and reconcile only structured API state.

    Kueue owns ordinary contention, so a pending or quota-waiting Workload never
    causes delete/recreate churn or product fallback. A product is skipped only
    when the typed Node inventory proves that it has no ready allocatable
    capacity. Application/data/image failures remain terminal on the selected
    product.
    """

    from npa.workflows.sim2real.job_scheduling import configure_gpu_job
    from npa.workflows.sim2real.k8s_client import (
        KubernetesJobClient,
        KubernetesJobFailed,
    )

    job_client = client or KubernetesJobClient.from_environment(namespace=namespace)
    inventory = tuple(job_client.list_gpu_products(resource=gpu_resource))
    discovered = tuple(item.product for item in inventory)
    inventory_by_product = {item.product: item for item in inventory}
    required_vram_gb = (
        minimum_vram_for_workload(
            workload,
            model=model,
            explicit=os.environ.get("NPA_SIM2REAL_MIN_GPU_VRAM_GB"),
        )
        if minimum_vram_gb is None
        else minimum_vram_gb
    )
    plan = ordered_compatible_products(
        preferred=preferred_product,
        explicit=explicit_candidates,
        discovered=discovered,
        workload=workload,
        image=image,
        minimum_vram_gb=required_vram_gb,
    )
    attempts: list[dict[str, Any]] = [dict(item) for item in plan.skipped]
    provenance: dict[str, Any] = {
        "scheduler": "kueue",
        "classification_source": "official_python_kubernetes_client",
        "candidate_order": list(plan.products),
        "discovered_products": list(plan.discovered_products),
        "inventory": [
            {
                "product": item.product,
                "ready_nodes": item.ready_nodes,
                "allocatable": item.allocatable,
            }
            for item in inventory
        ],
        "attempts": attempts,
        "selected_product": "",
        "selected_node": "",
        "allocated_gpu": {"resource": gpu_resource, "count": gpu_count},
        "image": image,
        "image_digests": [],
        "minimum_vram_gb": required_vram_gb,
        "model_requirement": model,
    }
    if not plan.products:
        raise GpuCapacityExhausted(
            f"no compatible GPU products are available for {workload}; "
            f"discovered={list(plan.discovered_products)} skipped={attempts}",
            provenance=provenance,
        )

    source_sha = os.environ.get("NPA_SIM2REAL_SOURCE_SHA", "").strip()
    if not source_sha:
        if os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip() == "1":
            raise GpuJobFailure(
                "real GPU Jobs require an exact source SHA",
                provenance=provenance,
            )
        source_sha = "unit-test-source"
    hang_timeout_s = int(os.environ.get("NPA_SIM2REAL_K8S_HANG_TIMEOUT_S", "0") or 0)
    for index, product in enumerate(plan.products):
        item = inventory_by_product.get(product)
        if item is None or item.ready_nodes == 0 or item.allocatable < gpu_count:
            attempts.append(
                {
                    "product": product,
                    "status": "unavailable",
                    "scheduling_reason": "structured_node_inventory_has_no_ready_capacity",
                    "ready_nodes": int(item.ready_nodes if item else 0),
                    "allocatable": int(item.allocatable if item else 0),
                }
            )
            continue

        started = time.monotonic()
        job_name = _attempt_job_name(base_job_name, index)
        manifest = configure_gpu_job(
            manifest_factory(product, job_name),
            image=image,
            product=product,
            gpu_resource=gpu_resource,
            gpu_count=gpu_count,
        )
        manifest_annotations = dict(
            manifest.get("metadata", {}).get("annotations") or {}
        )
        runtime_dependencies = {
            "mode": manifest_annotations.get(
                "sim2real.npa.dev/runtime-dependencies", "image-only"
            ),
            "isaac_cache_pvc": manifest_annotations.get(
                "sim2real.npa.dev/isaac-cache-pvc", ""
            ),
        }
        container_name = str(
            manifest["spec"]["template"]["spec"]["containers"][0].get("name") or ""
        )
        attempt: dict[str, Any] = {
            "product": product,
            "job_name": job_name,
            "image": image,
            "status": "reconciling",
            "scheduling_reason": "",
            "runtime_dependencies": runtime_dependencies,
        }
        attempts.append(attempt)
        uid, adopted = job_client.create_or_adopt(
            manifest,
            run_id=os.environ.get("NPA_SIM2REAL_RUN_ID", base_job_name),
            source_sha=source_sha,
            runtime_image=image,
        )
        attempt["job_uid"] = uid
        attempt["adopted"] = adopted

        if wait_for_completion:
            try:
                snapshot = job_client.watch_until_terminal(
                    job_name,
                    namespace=namespace,
                    timeout_s=timeout_s,
                    hang_timeout_s=hang_timeout_s,
                    heartbeat=heartbeat,
                )
            except KubernetesJobFailed as exc:
                snapshot = exc.snapshot
                attempt.update(
                    status="failed",
                    condition_type=snapshot.condition_type,
                    condition_reason=snapshot.condition_reason,
                    pod_states=[pod.__dict__ for pod in snapshot.pods],
                    duration_s=round(time.monotonic() - started, 3),
                )
                raise GpuJobFailure(
                    f"Kubernetes Job {job_name} failed on {product}; "
                    "structured application/runtime failure is not GPU fallback evidence",
                    provenance=provenance,
                ) from exc
        else:
            snapshot = job_client.snapshot(job_name, namespace=namespace)

        if (
            os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip() == "1"
            and not snapshot.kueue.workload_name
        ):
            raise GpuJobFailure(
                f"Kubernetes Job {job_name} has no observable Kueue Workload",
                provenance={**provenance, "job_snapshot": snapshot.to_dict()},
            )

        # A non-blocking submission is allowed to expose queued/admission state
        # before a Pod exists. Runtime attestation is mandatory at completion.
        image_digests = (
            _verify_runtime_digest(snapshot, image, container_name=container_name)
            if snapshot.state == "complete"
            else snapshot.image_digests
        )
        selected_nodes = snapshot.selected_nodes
        attempt.update(
            status="complete" if snapshot.state == "complete" else snapshot.state,
            scheduling_reason=snapshot.structured_scheduling_reason,
            node_name=selected_nodes[0] if selected_nodes else "",
            image_digests=image_digests,
            kueue=snapshot.kueue.__dict__,
            duration_s=round(time.monotonic() - started, 3),
        )
        provenance.update(
            selected_product=product,
            selected_node=selected_nodes[0] if selected_nodes else "",
            image_digests=image_digests,
            job_name=job_name,
            job_uid=uid,
            adopted=adopted,
            kueue=snapshot.kueue.__dict__,
            job_snapshot=snapshot.to_dict(),
            runtime_dependencies=runtime_dependencies,
            duration_s=round(time.monotonic() - started, 3),
        )
        return provenance

    raise GpuCapacityExhausted(
        f"all compatible GPU products lack ready structured capacity for {workload}",
        provenance=provenance,
    )
