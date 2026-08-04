"""Focused scheduler simulations for Sim2Real GPU-capacity fallback."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from npa.workflows.sim2real.gpu_fallback import (
    GpuCapacityExhausted,
    GpuJobFailure,
    capacity_scheduling_reason,
    minimum_vram_for_workload,
    normalize_gpu_family,
    ordered_compatible_products,
    product_is_compatible,
    products_from_node_payload,
    run_gpu_job_with_fallback,
)


RTX = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
L40S = "NVIDIA-L40S"
H100 = "NVIDIA-H100-80GB-HBM3"
H200 = "NVIDIA-H200"
B300 = "NVIDIA-B300"


def _nodes(*products: str) -> str:
    return json.dumps(
        {
            "items": [
                {"metadata": {"labels": {"nvidia.com/gpu.product": product}}}
                for product in products
            ]
        }
    )


def test_product_normalization_handles_cluster_label_variants() -> None:
    assert normalize_gpu_family(RTX) == "rtx-pro-6000"
    assert normalize_gpu_family("NVIDIA RTX 6000 Ada PRO") == "rtx-pro-6000"
    assert normalize_gpu_family("NVIDIA-L40S-SHARED") == "l40s"
    assert normalize_gpu_family(H100) == "h100"
    assert normalize_gpu_family(H200) == "h200"
    assert normalize_gpu_family(B300) == "b300"
    assert normalize_gpu_family("tenant-special") == "unknown"


def test_node_inventory_and_explicit_order_resolve_to_actual_labels() -> None:
    discovered = products_from_node_payload(_nodes(L40S, RTX, H100))
    plan = ordered_compatible_products(
        preferred="RTX PRO 6000",
        explicit="L40S,H100",
        discovered=discovered,
        workload="isaac",
        image="registry/npa-isaac-lab:2.3.2",
    )
    assert plan.products == (RTX, L40S)
    assert any(normalize_gpu_family(item["product"]) == "h100" for item in plan.skipped)


def test_isaac_excludes_h100_h200_and_datacenter_blackwell() -> None:
    for product in (H100, H200, "NVIDIA-B200", B300):
        assert not product_is_compatible(
            product, workload="isaac", image="registry/npa-isaac-lab:2.3.2"
        )
    assert product_is_compatible(
        RTX, workload="isaac", image="registry/npa-isaac-lab:2.3.2"
    )
    assert product_is_compatible(
        L40S, workload="isaac", image="registry/npa-isaac-lab:2.3.2"
    )


def test_non_isaac_filter_honors_transfer_and_image_architecture() -> None:
    assert not product_is_compatible(
        B300, workload="cosmos_transfer", image="registry/transfer:2.5.1"
    )
    assert product_is_compatible(
        H100,
        workload="cosmos_reason",
        image="registry/reason:cuda-sm90-sm120",
    )
    assert not product_is_compatible(
        H100,
        workload="cosmos_reason",
        image="registry/reason:architecture-unproven",
    )
    assert not product_is_compatible(
        H100,
        workload="cosmos_reason",
        image="registry/reason:cuda-sm120",
    )


def test_non_isaac_filter_honors_model_and_operator_vram_requirements() -> None:
    assert minimum_vram_for_workload("cosmos_reason", model="Reason2-8B") == 24
    assert minimum_vram_for_workload("cosmos_reason", model="Reason2-70B") == 141
    plan = ordered_compatible_products(
        preferred=L40S,
        explicit=(H100, H200),
        discovered=(L40S, H100, H200),
        workload="cosmos_reason",
        image="registry/reason:cuda-sm90",
        minimum_vram_gb=80,
    )
    assert plan.products == (H100, H200)
    assert any(item["product"] == L40S for item in plan.skipped)
    assert (
        minimum_vram_for_workload(
            "cosmos_reason", model="Reason2-2B", explicit="invalid"
        )
        == 10**9
    )


def test_unschedulable_detection_is_gpu_specific() -> None:
    pods = {
        "items": [
            {
                "status": {
                    "conditions": [
                        {
                            "reason": "Unschedulable",
                            "message": "0/2 nodes are available: 2 Insufficient nvidia.com/gpu.",
                        }
                    ]
                }
            }
        ]
    }
    assert "Insufficient" in capacity_scheduling_reason(
        pod_payload=pods, gpu_resource="nvidia.com/gpu", product=RTX
    )
    unrelated = {
        "items": [
            {
                "reason": "FailedScheduling",
                "message": "0/2 nodes are available: 2 node(s) had untolerated taint.",
            }
        ]
    }
    assert not capacity_scheduling_reason(
        event_payload=unrelated, gpu_resource="nvidia.com/gpu", product=RTX
    )


class _Scheduler:
    def __init__(self, outcomes: dict[str, str]) -> None:
        self.outcomes = outcomes
        self.current_product = ""
        self.current_job = ""
        self.applied_products: list[str] = []

    def __call__(
        self,
        args: list[str],
        *,
        stdin: str | None = None,
        timeout_s: int = 300,
    ) -> subprocess.CompletedProcess[str]:
        del timeout_s
        if args[:2] == ["get", "nodes"]:
            return subprocess.CompletedProcess(args, 0, _nodes(RTX, L40S, H100), "")
        if args[:2] == ["apply", "-f"]:
            manifest = json.loads(stdin or "{}")
            self.current_product = manifest["spec"]["template"]["spec"]["nodeSelector"][
                "nvidia.com/gpu.product"
            ]
            self.current_job = manifest["metadata"]["name"]
            self.applied_products.append(self.current_product)
            return subprocess.CompletedProcess(args, 0, "created", "")
        if args[:2] == ["get", "pods"]:
            outcome = self.outcomes.get(self.current_product, "success")
            if outcome == "capacity":
                payload = {
                    "items": [
                        {
                            "status": {
                                "conditions": [
                                    {
                                        "reason": "Unschedulable",
                                        "message": "0/1 nodes are available: 1 Insufficient nvidia.com/gpu.",
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                payload = {
                    "items": [
                        {
                            "metadata": {"name": f"{self.current_job}-pod"},
                            "spec": {
                                "nodeName": f"node-{normalize_gpu_family(self.current_product)}"
                            },
                            "status": {
                                "containerStatuses": [
                                    {
                                        "imageID": "docker-pullable://registry/image@sha256:abc123"
                                    }
                                ]
                            },
                        }
                    ]
                }
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")
        if args[:2] == ["get", "events"]:
            return subprocess.CompletedProcess(args, 0, '{"items":[]}', "")
        if args and args[0] == "wait":
            outcome = self.outcomes.get(self.current_product, "success")
            if outcome == "runtime":
                return subprocess.CompletedProcess(args, 1, "", "ImagePullBackOff")
            return subprocess.CompletedProcess(args, 0, "complete", "")
        return subprocess.CompletedProcess(args, 0, "", "")


def _manifest(product: str, job_name: str) -> dict[str, Any]:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name},
        "spec": {
            "template": {
                "spec": {
                    "nodeSelector": {"nvidia.com/gpu.product": product},
                    "containers": [{"image": "registry/image@sha256:abc123"}],
                }
            }
        },
    }


def test_capacity_retry_order_and_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_GPU_SCHEDULING_PROBE_SECONDS", "0")
    scheduler = _Scheduler({RTX: "capacity", L40S: "success"})
    result = run_gpu_job_with_fallback(
        kubectl=scheduler,
        manifest_factory=_manifest,
        base_job_name="s2r-job",
        namespace="default",
        image="registry/image@sha256:abc123",
        preferred_product=RTX,
        explicit_candidates=(L40S,),
        workload="isaac",
        gpu_resource="nvidia.com/gpu",
        gpu_count=1,
        timeout_s=10,
    )
    assert scheduler.applied_products == [RTX, L40S]
    assert result["candidate_order"] == [RTX, L40S]
    assert result["selected_product"] == L40S
    assert result["selected_node"] == "node-l40s"
    assert result["allocated_gpu"] == {"resource": "nvidia.com/gpu", "count": 1}
    assert result["job_name"].endswith("-gpu2")
    assert result["image_digests"] == ["docker-pullable://registry/image@sha256:abc123"]
    assert [item["status"] for item in result["attempts"]] == [
        "unschedulable",
        "complete",
    ]


def test_zero_timeout_waits_without_imposing_job_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_GPU_SCHEDULING_PROBE_SECONDS", "0")
    scheduler = _Scheduler({RTX: "success"})
    result = run_gpu_job_with_fallback(
        kubectl=scheduler,
        manifest_factory=_manifest,
        base_job_name="s2r-unbounded",
        namespace="default",
        image="registry/image@sha256:abc123",
        preferred_product=RTX,
        explicit_candidates=(),
        workload="isaac",
        gpu_resource="nvidia.com/gpu",
        gpu_count=1,
        timeout_s=0,
    )
    assert result["selected_product"] == RTX
    assert result["attempts"][-1]["status"] == "complete"


def test_unrelated_runtime_failure_never_switches_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_GPU_SCHEDULING_PROBE_SECONDS", "0")
    scheduler = _Scheduler({RTX: "runtime", L40S: "success"})
    with pytest.raises(GpuJobFailure, match="refusing to change workload product"):
        run_gpu_job_with_fallback(
            kubectl=scheduler,
            manifest_factory=_manifest,
            base_job_name="s2r-job",
            namespace="default",
            image="registry/image@sha256:abc123",
            preferred_product=RTX,
            explicit_candidates=(L40S,),
            workload="isaac",
            gpu_resource="nvidia.com/gpu",
            gpu_count=1,
            timeout_s=10,
        )
    assert scheduler.applied_products == [RTX]


def test_all_compatible_products_exhausted_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_GPU_SCHEDULING_PROBE_SECONDS", "0")
    scheduler = _Scheduler({RTX: "capacity", L40S: "capacity"})
    with pytest.raises(
        GpuCapacityExhausted, match="all compatible GPU products exhausted"
    ) as exc:
        run_gpu_job_with_fallback(
            kubectl=scheduler,
            manifest_factory=_manifest,
            base_job_name="s2r-job",
            namespace="default",
            image="registry/image@sha256:abc123",
            preferred_product=RTX,
            explicit_candidates=(L40S, H100, H200),
            workload="isaac",
            gpu_resource="nvidia.com/gpu",
            gpu_count=1,
            timeout_s=10,
        )
    assert scheduler.applied_products == [RTX, L40S]
    assert all(product not in scheduler.applied_products for product in (H100, H200))
    assert len(exc.value.provenance["attempts"]) == 4  # two filtered + two attempted
