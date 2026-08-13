"""Structured scheduling tests for Sim2Real GPU Jobs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from npa.workflows.sim2real.gpu_fallback import (
    GpuCapacityExhausted,
    GpuJobFailure,
    minimum_vram_for_workload,
    normalize_gpu_family,
    ordered_compatible_products,
    product_is_compatible,
    products_from_node_payload,
    run_gpu_job_with_fallback,
)
from npa.workflows.sim2real.job_scheduling import (
    KUEUE_VERSION,
    configure_gpu_job,
    kueue_queue_manifests,
)
from npa.workflows.sim2real.k8s_client import (
    ContainerSnapshot,
    GpuProductInventory,
    JobSnapshot,
    KubernetesJobFailed,
    KueueAdmission,
    PodSnapshot,
)
from npa.workflows.sim2real.models import Sim2RealLoopError


RTX = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
L40S = "NVIDIA-L40S"
H100 = "NVIDIA-H100-80GB-HBM3"
H200 = "NVIDIA-H200"
B300 = "NVIDIA-B300"
DIGEST = "a" * 64
IMAGE = f"registry.example/npa-isaac@sha256:{DIGEST}"


def test_product_normalization_handles_cluster_label_variants() -> None:
    assert normalize_gpu_family(RTX) == "rtx-pro-6000"
    assert normalize_gpu_family("NVIDIA RTX 6000 Ada PRO") == "rtx-pro-6000"
    assert normalize_gpu_family("NVIDIA-L40S-SHARED") == "l40s"
    assert normalize_gpu_family(H100) == "h100"
    assert normalize_gpu_family(H200) == "h200"
    assert normalize_gpu_family(B300) == "b300"
    assert normalize_gpu_family("tenant-special") == "unknown"


def test_inventory_and_explicit_order_resolve_to_actual_labels() -> None:
    nodes = {
        "items": [
            {"metadata": {"labels": {"nvidia.com/gpu.product": product}}}
            for product in (L40S, RTX, H100)
        ]
    }
    discovered = products_from_node_payload(nodes)
    plan = ordered_compatible_products(
        preferred="RTX PRO 6000",
        explicit="L40S,H100",
        discovered=discovered,
        workload="isaac",
        image="registry/npa-isaac-lab:2.3.2",
    )
    assert plan.products == (RTX, L40S)
    assert any(normalize_gpu_family(item["product"]) == "h100" for item in plan.skipped)


def test_missing_inventory_never_submits_human_gpu_alias_as_selector() -> None:
    plan = ordered_compatible_products(
        preferred=RTX,
        explicit="RTX PRO 6000,L40S",
        discovered=(),
        workload="isaac",
        image="registry/npa-isaac-lab:2.3.2",
    )
    assert plan.products == (RTX, "L40S")
    assert any(item["status"] == "unresolved_alias" for item in plan.skipped)


def test_isaac_excludes_datacenter_gpu_families() -> None:
    for product in (H100, H200, "NVIDIA-B200", B300):
        assert not product_is_compatible(
            product, workload="isaac", image="registry/npa-isaac-lab:2.3.2"
        )
    assert product_is_compatible(RTX, workload="isaac", image="registry/isaac")
    assert product_is_compatible(L40S, workload="isaac", image="registry/isaac")


@pytest.mark.parametrize(
    ("product", "marker", "compatible"),
    [
        (RTX, "sm120", True),
        (RTX, "compute120", True),
        (RTX, "sm100", False),
        (L40S, "sm89", True),
        (L40S, "sm90", False),
    ],
)
def test_architecture_markers_are_fail_closed(
    product: str, marker: str, compatible: bool
) -> None:
    assert (
        product_is_compatible(
            product, workload="cosmos_reason", image=f"registry/reason:cuda-{marker}"
        )
        is compatible
    )


def test_model_vram_requirement_filters_products() -> None:
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


def _manifest(product: str, job_name: str) -> dict[str, Any]:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_name},
        "spec": {
            "template": {
                "spec": {
                    "nodeSelector": {"nvidia.com/gpu.product": product},
                    "containers": [{"name": "component", "image": IMAGE}],
                }
            }
        },
    }


def _snapshot(*, state: str = "complete", product: str = RTX) -> JobSnapshot:
    container = ContainerSnapshot(
        name="component",
        image=IMAGE,
        image_id=f"containerd://registry.example/npa-isaac@sha256:{DIGEST}",
        restart_count=0,
        exit_code=0 if state == "complete" else 2 if state == "failed" else None,
        terminated_reason="Completed"
        if state == "complete"
        else "Error"
        if state == "failed"
        else "",
    )
    pod = PodSnapshot(
        name="job-pod",
        uid="pod-uid",
        owner_uid="job-uid",
        phase="Succeeded"
        if state == "complete"
        else "Failed"
        if state == "failed"
        else "Pending",
        node_name=f"node-{normalize_gpu_family(product)}" if state != "queued" else "",
        deletion_timestamp="",
        scheduled_status="True" if state != "queued" else "False",
        scheduled_reason="" if state != "queued" else "Unschedulable",
        resource_requests={"nvidia.com/gpu": "1"},
        containers=(container,) if state != "queued" else (),
    )
    return JobSnapshot(
        name="job",
        namespace="default",
        uid="job-uid",
        resource_version="1",
        state="active" if state == "queued" else state,
        active=1 if state == "queued" else 0,
        succeeded=1 if state == "complete" else 0,
        failed=1 if state == "failed" else 0,
        deleting=False,
        condition_type="Complete"
        if state == "complete"
        else "Failed"
        if state == "failed"
        else "",
        condition_reason="BackoffLimitExceeded" if state == "failed" else "",
        condition_message="diagnostic prose is not classification",
        pods=(pod,),
        kueue=KueueAdmission(
            workload_name="job-workload",
            admitted=state != "queued",
            quota_reserved=state != "queued",
            assigned_flavors={"nvidia.com/gpu": "sim2real-rtx-pro-6000"}
            if state != "queued"
            else {},
        ),
    )


class _StructuredClient:
    def __init__(
        self,
        *,
        inventory: tuple[GpuProductInventory, ...] | None = None,
        snapshot: JobSnapshot | None = None,
    ) -> None:
        self.inventory = inventory or (
            GpuProductInventory(RTX, ready_nodes=1, allocatable=1),
        )
        self.result = snapshot or _snapshot()
        self.created: list[dict[str, Any]] = []

    def list_gpu_products(self, *, resource: str) -> tuple[GpuProductInventory, ...]:
        assert resource == "nvidia.com/gpu"
        return self.inventory

    def create_or_adopt(
        self, manifest: dict[str, Any], **identity: str
    ) -> tuple[str, bool]:
        assert identity["runtime_image"] == IMAGE
        self.created.append(manifest)
        return "job-uid", False

    def watch_until_terminal(self, name: str, **kwargs: Any) -> JobSnapshot:
        del name, kwargs
        if self.result.state == "failed":
            raise KubernetesJobFailed("typed failure", snapshot=self.result)
        return self.result

    def snapshot(self, name: str, **kwargs: Any) -> JobSnapshot:
        del name, kwargs
        return self.result


def test_job_configuration_requires_digest_and_native_failure_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_K8S_INFRA_RETRIES", "4")
    configured = configure_gpu_job(
        _manifest(RTX, "job"),
        image=IMAGE,
        product=RTX,
        gpu_resource="nvidia.com/gpu",
        gpu_count=1,
    )
    assert configured["spec"]["backoffLimit"] == 4
    assert configured["spec"]["suspend"] is True
    assert "activeDeadlineSeconds" not in configured["spec"]
    assert configured["metadata"]["labels"]["kueue.x-k8s.io/queue-name"]
    rules = configured["spec"]["podFailurePolicy"]["rules"]
    assert rules[0]["onPodConditions"][0]["type"] == "DisruptionTarget"
    assert rules[1]["action"] == "FailJob"
    with pytest.raises(Sim2RealLoopError, match="image@sha256"):
        configure_gpu_job(
            _manifest(RTX, "job"),
            image="registry/image:latest",
            product=RTX,
            gpu_resource="nvidia.com/gpu",
            gpu_count=1,
        )


def test_isaac_gpu_job_requires_and_mounts_offline_readonly_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_SIM2REAL_ISAAC_CACHE_PVC", raising=False)
    monkeypatch.setenv("NPA_SIM2REAL_ISAAC_IMAGE", IMAGE)
    with pytest.raises(Sim2RealLoopError, match="CACHE_PVC"):
        configure_gpu_job(
            _manifest(RTX, "isaac-missing-cache"),
            image=IMAGE,
            product=RTX,
            gpu_resource="nvidia.com/gpu",
            gpu_count=1,
        )

    monkeypatch.setenv("NPA_SIM2REAL_ISAAC_CACHE_PVC", "npa-isaac-cache")
    configured = configure_gpu_job(
        _manifest(RTX, "isaac-with-cache"),
        image=IMAGE,
        product=RTX,
        gpu_resource="nvidia.com/gpu",
        gpu_count=1,
    )
    pod = configured["spec"]["template"]["spec"]
    assert pod["volumes"] == [
        {
            "name": "isaac-runtime-cache",
            "persistentVolumeClaim": {
                "claimName": "npa-isaac-cache",
                "readOnly": True,
            },
        }
    ]
    container = pod["containers"][0]
    assert container["volumeMounts"] == [
        {
            "name": "isaac-runtime-cache",
            "mountPath": "/opt/isaac-cache",
            "readOnly": True,
        }
    ]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["NPA_SIM2REAL_ISAAC_CACHE_PVC"] == "npa-isaac-cache"
    assert env["NPA_ISAAC_CACHE_READONLY"] == "1"
    assert env["NPA_ISAAC_BOOTSTRAP_OFFLINE"] == "1"
    assert (
        configured["metadata"]["annotations"]["sim2real.npa.dev/runtime-dependencies"]
        == "content-addressed-readonly-pvc"
    )


def test_non_isaac_gpu_job_never_receives_isaac_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "NPA_SIM2REAL_ISAAC_IMAGE", f"registry.example/isaac@sha256:{'b' * 64}"
    )
    monkeypatch.setenv("NPA_SIM2REAL_ISAAC_CACHE_PVC", "npa-isaac-cache")
    configured = configure_gpu_job(
        _manifest(RTX, "cosmos"),
        image=IMAGE,
        product=RTX,
        gpu_resource="nvidia.com/gpu",
        gpu_count=1,
    )
    pod = configured["spec"]["template"]["spec"]
    assert "volumes" not in pod
    assert "sim2real.npa.dev/isaac-cache-pvc" not in configured["metadata"].get(
        "annotations", {}
    )


def test_kueue_queue_covers_every_production_job_request() -> None:
    manifests = kueue_queue_manifests(
        namespace="default",
        gpu_product=RTX,
        gpu_quota=7,
        cpu_quota=160,
        memory_quota="1300Gi",
    )
    assert KUEUE_VERSION == "0.17.3"
    cluster_queue = next(item for item in manifests if item["kind"] == "ClusterQueue")
    group = cluster_queue["spec"]["resourceGroups"][0]
    assert group["coveredResources"] == ["nvidia.com/gpu", "cpu", "memory"]
    assert group["flavors"][0]["resources"] == [
        {"name": "nvidia.com/gpu", "nominalQuota": 7},
        {"name": "cpu", "nominalQuota": 160},
        {"name": "memory", "nominalQuota": "1300Gi"},
    ]


@pytest.mark.parametrize(("cpu", "memory"), [(0, "1Gi"), ("1", "0")])
def test_kueue_queue_rejects_zero_compute_quota(cpu: int | str, memory: str) -> None:
    with pytest.raises(ValueError, match="quota"):
        kueue_queue_manifests(
            namespace="default",
            gpu_product=RTX,
            gpu_quota=7,
            cpu_quota=cpu,
            memory_quota=memory,
        )


def test_structured_capacity_skips_only_proven_unavailable_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", "1" * 40)
    client = _StructuredClient(
        inventory=(
            GpuProductInventory(RTX, ready_nodes=0, allocatable=0),
            GpuProductInventory(L40S, ready_nodes=1, allocatable=1),
        ),
        snapshot=_snapshot(product=L40S),
    )
    result = run_gpu_job_with_fallback(
        client=client,
        manifest_factory=_manifest,
        base_job_name="s2r-job",
        namespace="default",
        image=IMAGE,
        preferred_product=RTX,
        explicit_candidates=(L40S,),
        workload="isaac",
        gpu_resource="nvidia.com/gpu",
        gpu_count=1,
        timeout_s=0,
    )
    assert len(client.created) == 1
    assert result["selected_product"] == L40S
    assert [item["status"] for item in result["attempts"]] == [
        "unavailable",
        "complete",
    ]


def test_queue_contention_is_observed_without_delete_recreate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", "1" * 40)
    client = _StructuredClient(snapshot=_snapshot(state="queued"))
    result = run_gpu_job_with_fallback(
        client=client,
        manifest_factory=_manifest,
        base_job_name="queued-job",
        namespace="default",
        image=IMAGE,
        preferred_product=RTX,
        explicit_candidates=(L40S,),
        workload="isaac",
        gpu_resource="nvidia.com/gpu",
        gpu_count=1,
        timeout_s=0,
        wait_for_completion=False,
    )
    assert len(client.created) == 1
    assert result["selected_product"] == RTX
    assert result["attempts"][-1]["status"] == "active"
    assert result["kueue"]["admitted"] is False


def test_application_failure_never_falls_through_to_other_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", "1" * 40)
    client = _StructuredClient(
        inventory=(
            GpuProductInventory(RTX, ready_nodes=1, allocatable=1),
            GpuProductInventory(L40S, ready_nodes=1, allocatable=1),
        ),
        snapshot=_snapshot(state="failed"),
    )
    with pytest.raises(GpuJobFailure) as raised:
        run_gpu_job_with_fallback(
            client=client,
            manifest_factory=_manifest,
            base_job_name="failed-job",
            namespace="default",
            image=IMAGE,
            preferred_product=RTX,
            explicit_candidates=(L40S,),
            workload="isaac",
            gpu_resource="nvidia.com/gpu",
            gpu_count=1,
            timeout_s=0,
        )
    assert len(client.created) == 1
    assert (
        raised.value.provenance["attempts"][-1]["condition_reason"]
        == "BackoffLimitExceeded"
    )


def test_runtime_digest_attestation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", "1" * 40)
    bad_container = replace(
        _snapshot().pods[0].containers[0],
        image_id=f"containerd://image@sha256:{'b' * 64}",
    )
    pod = replace(_snapshot().pods[0], containers=(bad_container,))
    client = _StructuredClient(snapshot=replace(_snapshot(), pods=(pod,)))
    with pytest.raises(GpuJobFailure, match="attest"):
        run_gpu_job_with_fallback(
            client=client,
            manifest_factory=_manifest,
            base_job_name="wrong-image",
            namespace="default",
            image=IMAGE,
            preferred_product=RTX,
            explicit_candidates=(),
            workload="isaac",
            gpu_resource="nvidia.com/gpu",
            gpu_count=1,
            timeout_s=0,
        )


def test_no_ready_structured_capacity_is_blocking() -> None:
    client = _StructuredClient(
        inventory=(GpuProductInventory(RTX, ready_nodes=0, allocatable=0),)
    )
    with pytest.raises(GpuCapacityExhausted):
        run_gpu_job_with_fallback(
            client=client,
            manifest_factory=_manifest,
            base_job_name="no-capacity",
            namespace="default",
            image=IMAGE,
            preferred_product=RTX,
            explicit_candidates=(),
            workload="isaac",
            gpu_resource="nvidia.com/gpu",
            gpu_count=1,
            timeout_s=0,
        )
