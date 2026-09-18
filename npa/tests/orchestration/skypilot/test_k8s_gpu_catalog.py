from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

import npa.orchestration.skypilot.k8s_gpu_catalog as gpu_catalog
from npa.orchestration.skypilot.k8s_gpu_catalog import (
    KubernetesGpuCatalog,
    KubernetesGpuCatalogError,
    KubernetesGpuInventory,
    KubernetesGpuNode,
    UnsatisfiableAcceleratorError,
    context_from_infra,
    discover_kubernetes_gpu_catalog,
    discover_kubernetes_gpu_inventory,
    label_known_kubernetes_gpus_for_skypilot,
    parse_kubernetes_gpu_catalog,
    preflight_kubernetes_gpu_gang,
    _recover_idle_validation_scope,
    resolve_kubernetes_accelerator,
    spec_accelerators,
    wait_for_kubernetes_accelerators,
)


# Verbatim `sky show-gpus --infra k8s` output from a live Nebius managed-K8s
# cluster with two 8-GPU RTX PRO 6000 nodes plus a second H100 context.
LIVE_OUTPUT = """WARNING: `sky show-gpus` has been renamed to `sky gpus list`.

Kubernetes GPUs
GPU                                   UTILIZATION
RTXPRO-6000-BLACKWELL-SERVER-EDITION  14 of 16 free
H100                                  2 of 2 free

Context: npa-rtxpro-mk8s
GPU                                   REQUESTABLE_QTY_PER_NODE  UTILIZATION
RTXPRO-6000-BLACKWELL-SERVER-EDITION  1, 2, 4, 8                14 of 16 free

Context: npa-workbench-eu-north1
GPU   REQUESTABLE_QTY_PER_NODE  UTILIZATION
H100  1                         2 of 2 free

Kubernetes per-node GPU availability
CONTEXT          NODE       vCPU          GPU                                   NODE STATUS
npa-rtxpro-mk8s  node-a     183 of 192    RTXPRO-6000-BLACKWELL-SERVER-EDITION  Healthy
"""

# A fleet of single-GPU nodes: the shape that makes `NAME:2` impossible.
SINGLE_GPU_OUTPUT = """Kubernetes GPUs
GPU                                   UTILIZATION
RTXPRO-6000-BLACKWELL-SERVER-EDITION  2 of 2 free

Context: npa-cluster
GPU                                   REQUESTABLE_QTY_PER_NODE  UTILIZATION
RTXPRO-6000-BLACKWELL-SERVER-EDITION  1                         2 of 2 free
"""


def test_parse_reads_per_context_requestable_quantities() -> None:
    catalog = parse_kubernetes_gpu_catalog(LIVE_OUTPUT)

    assert catalog.quantities_by_accelerator == {
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION": frozenset({1, 2, 4, 8}),
        "H100": frozenset({1}),
    }


def test_parse_can_scope_to_one_context() -> None:
    catalog = parse_kubernetes_gpu_catalog(
        LIVE_OUTPUT, context="npa-workbench-eu-north1"
    )

    assert set(catalog.quantities_by_accelerator) == {"H100"}
    assert catalog.context == "npa-workbench-eu-north1"


@pytest.mark.parametrize("heading", ["", "Context:\n", "Context: different\n"])
def test_scoped_catalog_requires_the_exact_context_heading(heading: str) -> None:
    output = heading + "GPU  REQUESTABLE_QTY_PER_NODE\nB200  1, 2, 4\n"

    catalog = parse_kubernetes_gpu_catalog(output, context="selected-context")

    assert catalog.is_empty


def test_scoped_catalog_does_not_merge_an_unbound_table() -> None:
    output = (
        "GPU  REQUESTABLE_QTY_PER_NODE\nOTHER-GPU  1\n\n"
        "Context: selected-context\nGPU  REQUESTABLE_QTY_PER_NODE\nB200  1\n"
    )

    catalog = parse_kubernetes_gpu_catalog(output, context="selected-context")

    assert catalog.quantities_by_accelerator == {"B200": frozenset({1})}


def test_parse_ignores_the_trailing_per_node_table() -> None:
    catalog = parse_kubernetes_gpu_catalog(LIVE_OUTPUT)

    assert "node-a" not in catalog.quantities_by_accelerator
    assert "CONTEXT" not in catalog.quantities_by_accelerator


def test_spec_name_is_remapped_onto_the_advertised_product_string() -> None:
    catalog = parse_kubernetes_gpu_catalog(LIVE_OUTPUT, context="npa-rtxpro-mk8s")

    resolution = resolve_kubernetes_accelerator("RTXPRO6000:1", catalog=catalog)

    assert resolution.resolved == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    assert resolution.remapped is True
    assert "not advertised" in resolution.describe()


def test_the_nebius_node_label_name_also_reaches_the_gfd_name() -> None:
    # `sky gpus list` reported RTX6000 while the GPU operator was still labelling.
    catalog = parse_kubernetes_gpu_catalog(LIVE_OUTPUT, context="npa-rtxpro-mk8s")

    resolution = resolve_kubernetes_accelerator("RTX6000:1", catalog=catalog)

    assert resolution.resolved == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"


def test_an_exact_name_is_left_alone() -> None:
    catalog = parse_kubernetes_gpu_catalog(LIVE_OUTPUT, context="npa-rtxpro-mk8s")

    resolution = resolve_kubernetes_accelerator(
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:8", catalog=catalog
    )

    assert resolution.remapped is False
    assert "matches this cluster" in resolution.describe()


def test_two_gpus_per_task_is_rejected_on_single_gpu_nodes() -> None:
    catalog = parse_kubernetes_gpu_catalog(SINGLE_GPU_OUTPUT, context="npa-cluster")

    with pytest.raises(UnsatisfiableAcceleratorError) as excinfo:
        resolve_kubernetes_accelerator("RTXPRO6000:2", catalog=catalog)

    message = str(excinfo.value)
    assert "at most 1" in message
    assert "single node" in message
    assert "Adding nodes does not help" in message
    assert (
        "NPA_WORKFLOW_GPU_ACCELERATOR=RTXPRO-6000-BLACKWELL-SERVER-EDITION:1" in message
    )


@pytest.mark.parametrize("count", range(1, 9))
def test_integer_gpu_requests_fit_within_catalog_node_capacity(count) -> None:
    catalog = parse_kubernetes_gpu_catalog(LIVE_OUTPUT, context="npa-rtxpro-mk8s")
    resolution = resolve_kubernetes_accelerator(f"RTXPRO6000:{count}", catalog=catalog)
    assert resolution.resolved == f"RTXPRO-6000-BLACKWELL-SERVER-EDITION:{count}"


def test_an_unknown_accelerator_lists_the_available_ones() -> None:
    catalog = parse_kubernetes_gpu_catalog(LIVE_OUTPUT)

    with pytest.raises(UnsatisfiableAcceleratorError) as excinfo:
        resolve_kubernetes_accelerator("TPUv5:1", catalog=catalog)

    assert "H100" in str(excinfo.value)


def test_adjacent_products_are_not_treated_as_ambiguous_aliases() -> None:
    catalog = KubernetesGpuCatalog(
        quantities_by_accelerator={
            "H100-NVL": frozenset({1}),
            "H100-SXM": frozenset({1}),
        }
    )

    with pytest.raises(UnsatisfiableAcceleratorError) as excinfo:
        resolve_kubernetes_accelerator("H100:1", catalog=catalog)

    assert "does not auto-select prefix or fuzzy candidates" in str(excinfo.value)


@pytest.mark.parametrize(
    ("requested", "advertised"),
    [("A10:1", "A100"), ("L40:1", "L40S"), ("H100:1", "H100NVL")],
)
def test_unique_adjacent_product_never_silently_changes_cost_or_capacity(
    requested: str, advertised: str
) -> None:
    catalog = KubernetesGpuCatalog(
        quantities_by_accelerator={advertised: frozenset({1})}
    )

    with pytest.raises(UnsatisfiableAcceleratorError) as excinfo:
        resolve_kubernetes_accelerator(requested, catalog=catalog)

    assert advertised in str(excinfo.value)
    assert "does not auto-select" in str(excinfo.value)


def test_case_and_punctuation_normalization_is_exact() -> None:
    catalog = KubernetesGpuCatalog(
        quantities_by_accelerator={"H100-SXM": frozenset({1})}
    )

    result = resolve_kubernetes_accelerator("h100_sxm:1", catalog=catalog)

    assert result.resolved == "H100-SXM:1"


def test_an_empty_catalog_blames_the_gpu_operator() -> None:
    with pytest.raises(UnsatisfiableAcceleratorError) as excinfo:
        resolve_kubernetes_accelerator(
            "RTXPRO6000:1", catalog=KubernetesGpuCatalog(quantities_by_accelerator={})
        )

    assert "GPU operator" in str(excinfo.value)


@pytest.fixture()
def sky_bin(tmp_path):  # noqa: ANN001, ANN201 - pytest fixture
    path = tmp_path / "sky"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def test_discover_passes_exact_context_config_and_kubeconfig(
    sky_bin: str, tmp_path
) -> None:  # noqa: ANN001
    seen: dict[str, object] = {}
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        seen["cmd"] = cmd
        seen["env"] = kwargs["env"]
        return subprocess.CompletedProcess(cmd, 0, stdout=LIVE_OUTPUT, stderr="")

    catalog = discover_kubernetes_gpu_catalog(
        context="npa-rtxpro-mk8s",
        kubeconfig=kubeconfig,
        sky_bin=sky_bin,
        runner=fake_run,
    )

    assert seen["cmd"] == [
        sky_bin,
        "show-gpus",
        "--config",
        'kubernetes.allowed_contexts=["npa-rtxpro-mk8s"]',
        "--infra",
        "k8s/npa-rtxpro-mk8s",
    ]
    assert seen["env"]["KUBECONFIG"] == str(kubeconfig)
    assert set(catalog.quantities_by_accelerator) == {
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION"
    }


def test_discover_surfaces_a_failing_sky_invocation(sky_bin: str) -> None:
    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no kube context")

    with pytest.raises(KubernetesGpuCatalogError) as excinfo:
        discover_kubernetes_gpu_catalog(sky_bin=sky_bin, runner=fake_run)

    assert "no kube context" in str(excinfo.value)


def test_discover_reenables_kubernetes_after_api_server_restart(
    sky_bin: str,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        calls.append(cmd)
        if cmd[1] == "check":
            return subprocess.CompletedProcess(cmd, 0, stdout="enabled", stderr="")
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="Kubernetes is not enabled. To fix, run: sky check kubernetes",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout=LIVE_OUTPUT, stderr="")

    catalog = discover_kubernetes_gpu_catalog(
        context="npa-rtxpro-mk8s", sky_bin=sky_bin, runner=fake_run
    )

    assert [cmd[1:] for cmd in calls] == [
        [
            "show-gpus",
            "--config",
            'kubernetes.allowed_contexts=["npa-rtxpro-mk8s"]',
            "--infra",
            "k8s/npa-rtxpro-mk8s",
        ],
        [
            "check",
            "--config",
            'kubernetes.allowed_contexts=["npa-rtxpro-mk8s"]',
            "kubernetes",
        ],
        [
            "show-gpus",
            "--config",
            'kubernetes.allowed_contexts=["npa-rtxpro-mk8s"]',
            "--infra",
            "k8s/npa-rtxpro-mk8s",
        ],
    ]
    assert not catalog.is_empty


def test_discover_reports_failed_kubernetes_reenable(sky_bin: str) -> None:
    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        if cmd[1] == "check":
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="exact context unavailable"
            )
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="Kubernetes is not enabled. To fix, run: sky check kubernetes",
            stderr="",
        )

    with pytest.raises(KubernetesGpuCatalogError, match="exact context unavailable"):
        discover_kubernetes_gpu_catalog(sky_bin=sky_bin, runner=fake_run)


def test_empty_discovery_is_read_only_and_never_labels_nodes(
    sky_bin: str, monkeypatch
) -> None:  # noqa: ANN001
    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "npa.orchestration.skypilot.k8s_gpu_catalog."
        "label_known_kubernetes_gpus_for_skypilot",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("catalog discovery must not mutate Kubernetes nodes")
        ),
    )

    catalog = discover_kubernetes_gpu_catalog(
        context="npa-cluster", sky_bin=sky_bin, runner=fake_run
    )

    assert catalog.is_empty


def test_known_rtxpro_label_is_exact_context_scoped() -> None:
    inventory = KubernetesGpuInventory(
        context="ctx",
        ready_nodes=1,
        eligible_gpu_nodes=1,
        capacity=1,
        allocatable=1,
        products=("NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",),
        node_labels={
            "node-a": {
                "nvidia.com/gpu.product": (
                    "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
                )
            }
        },
    )
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        seen.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="node/node-a labelled", stderr=""
        )

    assert (
        label_known_kubernetes_gpus_for_skypilot(
            context="ctx", inventory=inventory, runner=fake_run
        )
        == 1
    )
    assert seen == [
        [
            "kubectl",
            "--context",
            "ctx",
            "label",
            "node",
            "node-a",
            "skypilot.co/accelerator=rtxpro6000",
        ]
    ]


def test_known_b200_label_is_exact_context_scoped() -> None:
    inventory = KubernetesGpuInventory(
        context="ctx",
        ready_nodes=1,
        eligible_gpu_nodes=1,
        capacity=1,
        allocatable=1,
        products=("NVIDIA-B200",),
        node_labels={"node-a": {"nvidia.com/gpu.product": "NVIDIA-B200"}},
    )
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        seen.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="node/node-a labelled", stderr=""
        )

    assert (
        label_known_kubernetes_gpus_for_skypilot(
            context="ctx", inventory=inventory, runner=fake_run
        )
        == 1
    )
    assert seen == [
        [
            "kubectl",
            "--context",
            "ctx",
            "label",
            "node",
            "node-a",
            "skypilot.co/accelerator=B200",
        ]
    ]


def test_known_gpu_label_rbac_failure_is_immediate_and_actionable() -> None:
    inventory = KubernetesGpuInventory(
        context="ctx",
        ready_nodes=1,
        eligible_gpu_nodes=1,
        capacity=1,
        allocatable=1,
        products=("RTX6000",),
        node_labels={"node-a": {"nebius.com/gpu-name": "RTX6000"}},
    )

    with pytest.raises(KubernetesGpuCatalogError, match="RBAC.*patch/update"):
        label_known_kubernetes_gpus_for_skypilot(
            context="ctx",
            inventory=inventory,
            runner=lambda cmd, **_kwargs: subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="Error from server (Forbidden)"
            ),
        )


def test_inventory_prefers_gfd_product_over_same_node_provider_alias() -> None:
    payload = {
        "items": [
            {
                "metadata": {
                    "name": "gpu-node",
                    "labels": {
                        "nvidia.com/gpu.product": "NVIDIA-RTX-PRO-6000",
                        "nebius.com/gpu-name": "RTX6000",
                    },
                },
                "spec": {},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "capacity": {"nvidia.com/gpu": "1"},
                    "allocatable": {"nvidia.com/gpu": "1"},
                },
            }
        ]
    }

    inventory = discover_kubernetes_gpu_inventory(
        context="ctx",
        runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    assert inventory.products == ("NVIDIA-RTX-PRO-6000",)
    assert inventory.to_dict()["accelerator_product"] == "NVIDIA-RTX-PRO-6000"


def test_unknown_gpu_is_never_fuzzy_labelled() -> None:
    inventory = KubernetesGpuInventory(
        context="ctx",
        ready_nodes=1,
        eligible_gpu_nodes=1,
        capacity=1,
        allocatable=1,
        products=("NVIDIA-RTX-PRO-5000-Blackwell",),
        node_labels={
            "node-a": {"nvidia.com/gpu.product": "NVIDIA-RTX-PRO-5000-Blackwell"}
        },
    )

    def unexpected(*args, **kwargs):  # noqa: ANN001, ANN202
        raise AssertionError("unknown products must not be labelled")

    assert (
        label_known_kubernetes_gpus_for_skypilot(
            context="ctx", inventory=inventory, runner=unexpected
        )
        == 0
    )


def test_spec_accelerators_reads_only_kubernetes_profiles() -> None:
    resources = {
        "gpu": {"cloud": "kubernetes", "accelerators": "RTXPRO6000:1"},
        "gpu_big": {"cloud": "k8s", "accelerators": "RTXPRO6000:8"},
        "vm": {"cloud": "nebius", "accelerators": "H100:1"},
        "cpu": {"cloud": "kubernetes", "cpus": 4},
    }

    assert spec_accelerators(resources) == ["RTXPRO6000:1", "RTXPRO6000:8"]


def test_spec_accelerators_tolerates_a_missing_block() -> None:
    assert spec_accelerators(None) == []
    assert spec_accelerators({}) == []


def test_readiness_waits_after_kubernetes_allocatable_until_skypilot_labels() -> None:
    catalogs = iter(
        [
            KubernetesGpuCatalog(quantities_by_accelerator={}),
            parse_kubernetes_gpu_catalog(SINGLE_GPU_OUTPUT, context="npa-cluster"),
        ]
    )
    messages: list[str] = []
    clock = iter([0.0, 0.0, 1.0, 1.0])

    result = wait_for_kubernetes_accelerators(
        ["RTXPRO6000:1"],
        context="npa-cluster",
        timeout=10,
        poll_interval=1,
        discover=lambda: next(catalogs),
        allocatable=lambda: 1,
        on_status=messages.append,
        monotonic=lambda: next(clock),
        sleeper=lambda _seconds: None,
    )

    assert result["RTXPRO6000:1"].resolved.endswith(":1")
    assert any(
        "Kubernetes allocatable=1; SkyPilot discovery=pending" in item
        for item in messages
    )
    assert messages[-1].startswith(
        "GPU readiness: Kubernetes allocatable=1; SkyPilot discovery=ready"
    )


def test_explicit_known_label_repair_precedes_catalog_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    inventory = KubernetesGpuInventory(
        context="ctx",
        ready_nodes=1,
        eligible_gpu_nodes=1,
        capacity=1,
        allocatable=1,
        products=("RTX6000",),
        node_labels={"node-a": {"nebius.com/gpu-name": "RTX6000"}},
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.k8s_gpu_catalog.discover_kubernetes_gpu_inventory",
        lambda **_kwargs: inventory,
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.k8s_gpu_catalog."
        "label_known_kubernetes_gpus_for_skypilot",
        lambda **_kwargs: (events.append("label"), 1)[1],
    )

    result = wait_for_kubernetes_accelerators(
        [],
        context="ctx",
        label_known_gpus=True,
        discover=lambda: (
            events.append("discover"),
            KubernetesGpuCatalog(
                quantities_by_accelerator={"rtxpro6000": frozenset({1})}
            ),
        )[1],
        monotonic=lambda: 0.0,
    )

    assert events == ["label", "discover"]
    assert result["rtxpro6000:1"].resolved == "rtxpro6000:1"


def test_readiness_timeout_is_clear_and_preserves_capacity() -> None:
    times = iter([0.0, 0.0, 2.0])

    with pytest.raises(KubernetesGpuCatalogError) as excinfo:
        wait_for_kubernetes_accelerators(
            ["RTXPRO6000:1"],
            context="npa-cluster",
            timeout=1,
            poll_interval=1,
            discover=lambda: KubernetesGpuCatalog(quantities_by_accelerator={}),
            allocatable=lambda: 1,
            monotonic=lambda: next(times),
            sleeper=lambda _seconds: None,
        )

    message = str(excinfo.value)
    assert "Kubernetes allocatable=1" in message
    assert "Capacity was left running" in message


@pytest.mark.parametrize(
    ("infra", "expected"),
    [
        ("k8s/npa-cluster", "npa-cluster"),
        ("kubernetes/npa-cluster", "npa-cluster"),
        ("nebius", ""),
        ("", ""),
    ],
)
def test_context_from_infra(infra: str, expected: str) -> None:
    assert context_from_infra(infra) == expected


def _node(
    name: str,
    *,
    product: str = "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
    free: int = 1,
    ready: bool = True,
    schedulable: bool = True,
) -> KubernetesGpuNode:
    return KubernetesGpuNode(
        name=name,
        ready=ready,
        schedulable=schedulable,
        products=(product,),
        capacity=1,
        allocatable=1,
        committed=1 - free,
        free=free,
        exclusion="" if ready and schedulable else "excluded",
        allocatable_pods=110,
        free_pod_slots=110,
    )


def test_gang_capacity_requires_distinct_compatible_free_nodes() -> None:
    inventory = KubernetesGpuInventory(
        context="exact-context",
        ready_nodes=3,
        eligible_gpu_nodes=3,
        capacity=3,
        allocatable=3,
        products=("RTXPRO-6000-BLACKWELL-SERVER-EDITION",),
        node_labels={},
        nodes=(_node("a"), _node("b"), _node("c", free=0)),
    )
    evidence = preflight_kubernetes_gpu_gang(
        inventory, accelerator="RTXPRO6000:1", node_count=2
    )
    assert evidence["compatible_free_nodes"] == 2
    assert evidence["selected_nodes"] == ["a", "b"]


def test_gang_capacity_matches_nvidia_product_label_to_skypilot_name() -> None:
    inventory = KubernetesGpuInventory(
        context="exact-context",
        ready_nodes=2,
        eligible_gpu_nodes=2,
        capacity=2,
        allocatable=2,
        products=("NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",),
        node_labels={},
        nodes=(
            _node("a", product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"),
            _node("b", product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"),
        ),
    )

    evidence = preflight_kubernetes_gpu_gang(
        inventory,
        accelerator="RTXPRO-6000-BLACKWELL-SERVER-EDITION:1",
        node_count=2,
    )

    assert evidence["compatible_free_nodes"] == 2
    assert evidence["selected_nodes"] == ["a", "b"]


@pytest.mark.parametrize(
    ("label", "product"),
    [
        ("nebius.com/gpu-name", "RTX6000"),
        ("skypilot.co/accelerator", "rtxpro6000"),
    ],
)
def test_discovered_provider_or_repaired_labels_support_gang_capacity(
    label: str, product: str
) -> None:
    nodes = {
        "items": [
            {
                "metadata": {
                    "name": f"gpu-{suffix}",
                    "labels": {label: product},
                },
                "spec": {},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "capacity": {"nvidia.com/gpu": "1"},
                    "allocatable": {
                        "nvidia.com/gpu": "1",
                        "cpu": "24",
                        "memory": "218Gi",
                        "pods": "110",
                    },
                },
            }
            for suffix in ("a", "b")
        ]
    }

    def runner(cmd, **_kwargs):  # noqa: ANN001 - test stub
        payload = {"items": []} if "pods" in cmd else nodes
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr=""
        )

    inventory = discover_kubernetes_gpu_inventory(
        context="exact-context", runner=runner
    )
    evidence = preflight_kubernetes_gpu_gang(
        inventory,
        accelerator="RTXPRO6000:1",
        node_count=2,
        cpus=16,
        memory="128Gi",
    )

    assert [node.products for node in inventory.nodes] == [
        (product,),
        (product,),
    ]
    assert evidence["compatible_free_nodes"] == 2
    assert evidence["selected_nodes"] == ["gpu-a", "gpu-b"]


def test_gang_capacity_subtracts_shared_and_incompatible_capacity() -> None:
    inventory = KubernetesGpuInventory(
        context="exact-context",
        ready_nodes=4,
        eligible_gpu_nodes=4,
        capacity=4,
        allocatable=4,
        products=("RTXPRO-6000-BLACKWELL-SERVER-EDITION", "H100"),
        node_labels={},
        nodes=(
            _node("occupied", free=0),
            _node("wrong-product", product="H100"),
            _node("cordoned", schedulable=False),
            _node("only-free"),
        ),
    )
    with pytest.raises(UnsatisfiableAcceleratorError, match="1 distinct compatible"):
        preflight_kubernetes_gpu_gang(
            inventory, accelerator="RTXPRO6000:1", node_count=2
        )


def test_gang_capacity_subtracts_cpu_and_memory_commitments() -> None:
    def resourced(name: str, *, cpu: int, memory: int) -> KubernetesGpuNode:
        return KubernetesGpuNode(
            **{
                **_node(name).to_dict(),
                "products": ("RTXPRO-6000-BLACKWELL-SERVER-EDITION",),
                "allocatable_cpu_millis": 32_000,
                "free_cpu_millis": cpu,
                "allocatable_memory_bytes": 256 * 1024**3,
                "free_memory_bytes": memory,
            }
        )

    inventory = KubernetesGpuInventory(
        context="exact-context",
        ready_nodes=3,
        eligible_gpu_nodes=3,
        capacity=3,
        allocatable=3,
        products=("RTXPRO-6000-BLACKWELL-SERVER-EDITION",),
        node_labels={},
        nodes=(
            resourced("cpu-busy", cpu=8_000, memory=256 * 1024**3),
            resourced("memory-busy", cpu=32_000, memory=64 * 1024**3),
            resourced("compatible", cpu=32_000, memory=256 * 1024**3),
        ),
    )
    with pytest.raises(UnsatisfiableAcceleratorError, match="1 distinct compatible"):
        preflight_kubernetes_gpu_gang(
            inventory,
            accelerator="RTXPRO6000:1",
            node_count=2,
            cpus=16,
            memory="128Gi",
        )


def test_gang_capacity_fails_closed_when_pod_inventory_is_unreadable() -> None:
    inventory = KubernetesGpuInventory(
        context="exact-context",
        ready_nodes=0,
        eligible_gpu_nodes=0,
        capacity=0,
        allocatable=0,
        products=(),
        node_labels={},
        error="kubectl pod inventory failed; free shared GPU capacity is unknown",
    )
    with pytest.raises(KubernetesGpuCatalogError, match="shared GPU capacity"):
        preflight_kubernetes_gpu_gang(
            inventory, accelerator="RTXPRO6000:1", node_count=2
        )


def test_live_inventory_uses_exact_context_and_subtracts_active_pods() -> None:
    nodes = {
        "items": [
            {
                "metadata": {
                    "name": "gpu-a",
                    "labels": {
                        "nvidia.com/gpu.product": (
                            "RTXPRO-6000-BLACKWELL-SERVER-EDITION"
                        )
                    },
                },
                "spec": {},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "capacity": {"nvidia.com/gpu": "2"},
                    "allocatable": {
                        "nvidia.com/gpu": "2",
                        "cpu": "32",
                        "memory": "64Gi",
                        "ephemeral-storage": "950G",
                        "pods": "110",
                    },
                },
            }
        ]
    }
    pods = {
        "items": [
            {
                "spec": {
                    "nodeName": "gpu-a",
                    "containers": [
                        {
                            "resources": {
                                "requests": {
                                    "nvidia.com/gpu": "1",
                                    "cpu": "8",
                                    "memory": "16Gi",
                                    "ephemeral-storage": "100G",
                                }
                            }
                        }
                    ],
                    "initContainers": [
                        {
                            "resources": {
                                "requests": {"cpu": "12", "memory": "32Gi", "ephemeral-storage": "200G"},
                                "limits": {"nvidia.com/gpu": "2"},
                            }
                        }
                    ],
                    "overhead": {"cpu": "1", "memory": "1Gi", "ephemeral-storage": "1G"},
                },
                "status": {"phase": "Running"},
            },
            {
                "spec": {
                    "nodeName": "gpu-a",
                    "containers": [
                        {"resources": {"requests": {"nvidia.com/gpu": "2"}}}
                    ],
                },
                "status": {"phase": "Succeeded"},
            },
            {
                "metadata": {"name": "competing-gang-rank"},
                "spec": {
                    "containers": [{"resources": {"requests": {"nvidia.com/gpu": "1"}}}]
                },
                "status": {"phase": "Pending"},
            },
        ]
    }
    seen: list[list[str]] = []

    def runner(cmd, **_kwargs):
        seen.append(cmd)
        payload = pods if "pods" in cmd else nodes
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr=""
        )

    inventory = discover_kubernetes_gpu_inventory(
        context="exact-context", runner=runner
    )

    assert seen == [
        ["kubectl", "--context", "exact-context", "get", "nodes", "-o", "json"],
        [
            "kubectl",
            "--context",
            "exact-context",
            "get",
            "pods",
            "--all-namespaces",
            "-o",
            "json",
        ],
    ]
    assert inventory.nodes[0].committed == 2
    assert inventory.nodes[0].free == 0
    assert inventory.nodes[0].committed_cpu_millis == 13_000
    assert inventory.nodes[0].free_cpu_millis == 19_000
    assert inventory.nodes[0].committed_memory_bytes == 33 * 1024**3
    assert inventory.nodes[0].free_memory_bytes == 31 * 1024**3
    assert inventory.nodes[0].committed_ephemeral_storage_bytes == 201 * 10**9
    assert inventory.nodes[0].free_ephemeral_storage_bytes == 749 * 10**9
    assert inventory.nodes[0].committed_pods == 1
    assert inventory.nodes[0].free_pod_slots == 109
    assert inventory.unbound_pending_gpu_pods == 1
    assert inventory.unbound_pending_gpu_requests == 1


def test_live_inventory_pins_explicit_kubeconfig_for_nodes_and_pods(
    tmp_path,
) -> None:  # noqa: ANN001
    kubeconfig = tmp_path / "exact-kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    seen: list[tuple[list[str], str]] = []

    def runner(cmd, **kwargs):  # noqa: ANN001 - test stub
        seen.append((cmd, kwargs["env"]["KUBECONFIG"]))
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"items": []}), stderr=""
        )

    discover_kubernetes_gpu_inventory(
        context="exact-context", kubeconfig=kubeconfig, runner=runner
    )

    prefix = [
        "kubectl",
        "--kubeconfig",
        str(kubeconfig),
        "--context",
        "exact-context",
        "get",
    ]
    assert seen == [
        (prefix + ["nodes", "-o", "json"], str(kubeconfig)),
        (
            prefix + ["pods", "--all-namespaces", "-o", "json"],
            str(kubeconfig),
        ),
    ]


def test_gang_capacity_waits_for_unbound_pending_gpu_demand() -> None:
    from npa.orchestration.skypilot.k8s_gpu_catalog import PendingGpuPlacementError
    inventory = KubernetesGpuInventory(
        context="exact-context",
        ready_nodes=2,
        eligible_gpu_nodes=2,
        capacity=2,
        allocatable=2,
        products=("RTXPRO-6000-BLACKWELL-SERVER-EDITION",),
        node_labels={},
        nodes=(_node("a"), _node("b")),
        unbound_pending_gpu_pods=1,
        unbound_pending_gpu_requests=1,
    )

    with pytest.raises(PendingGpuPlacementError, match="active unbound GPU pod"):
        preflight_kubernetes_gpu_gang(
            inventory, accelerator="RTXPRO6000:1", node_count=2
        )


def test_gang_capacity_applies_profile_node_selector_and_required_affinity() -> None:
    def labelled(name: str, zone: str, pool: str) -> KubernetesGpuNode:
        return KubernetesGpuNode(
            **{
                **_node(name).to_dict(),
                "products": ("RTXPRO-6000-BLACKWELL-SERVER-EDITION",),
                "labels": (("topology.kubernetes.io/zone", zone), ("pool", pool)),
            }
        )

    inventory = KubernetesGpuInventory(
        context="exact-context",
        ready_nodes=3,
        eligible_gpu_nodes=3,
        capacity=3,
        allocatable=3,
        products=("RTXPRO-6000-BLACKWELL-SERVER-EDITION",),
        node_labels={},
        nodes=(
            labelled("a", "central-a", "shared"),
            labelled("b", "central-b", "owned"),
            labelled("c", "central-b", "owned"),
        ),
    )

    evidence = preflight_kubernetes_gpu_gang(
        inventory,
        accelerator="RTXPRO6000:1",
        node_count=2,
        pod_spec={
            "nodeSelector": {"pool": "owned"},
            "affinity": {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {
                                        "key": "topology.kubernetes.io/zone",
                                        "operator": "In",
                                        "values": ["central-b"],
                                    }
                                ]
                            }
                        ]
                    }
                }
            },
        },
    )

    assert evidence["selected_nodes"] == ["b", "c"]


def test_gang_capacity_applies_allowed_nodes_pod_slots_and_sky_tolerations() -> None:
    inventory = KubernetesGpuInventory(
        context="exact-context",
        ready_nodes=3,
        eligible_gpu_nodes=3,
        capacity=3,
        allocatable=3,
        products=("RTXPRO-6000-BLACKWELL-SERVER-EDITION",),
        node_labels={},
        nodes=(
            _node("not-allowed"),
            KubernetesGpuNode(**{**_node("pod-full").to_dict(), "free_pod_slots": 0}),
            _node("allowed"),
        ),
    )
    with pytest.raises(UnsatisfiableAcceleratorError, match="1 distinct compatible"):
        preflight_kubernetes_gpu_gang(
            inventory,
            accelerator="RTXPRO6000:1",
            node_count=2,
            allowed_nodes=["pod-full", "allowed"],
        )


def test_nvidia_noexecute_taint_is_not_covered_by_skypilot_toleration() -> None:
    nodes = {
        "items": [
            {
                "metadata": {
                    "name": "evicting-gpu",
                    "labels": {"nvidia.com/gpu.product": "RTXPRO6000"},
                },
                "spec": {"taints": [{"key": "nvidia.com/gpu", "effect": "NoExecute"}]},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "capacity": {"nvidia.com/gpu": "1"},
                    "allocatable": {
                        "nvidia.com/gpu": "1",
                        "cpu": "32",
                        "memory": "128Gi",
                        "pods": "110",
                    },
                },
            }
        ]
    }

    def runner(cmd, **_kwargs):
        payload = {"items": []} if "pods" in cmd else nodes
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr=""
        )

    inventory = discover_kubernetes_gpu_inventory(context="exact", runner=runner)
    assert inventory.nodes[0].schedulable is False
    assert inventory.nodes[0].exclusion == "cordoned-or-unsupported-taint"


def test_idle_validation_scope_recovery_archives_only_after_two_empty_pod_probes(
    tmp_path
) -> None:
    scope = tmp_path / "cluster-validation" / ("a" * 24)
    scope.mkdir(parents=True)
    for name in ("home", "sky-runtime", "local-api"):
        (scope / name).mkdir()
    (scope / "client-config.yaml").write_text("allowed_clouds: [nebius]\n", encoding="utf-8")
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    calls: list[list[str]] = []
    stopped: list[Path] = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"items": []}), stderr=""
        )

    assert _recover_idle_validation_scope(
        scope,
        context="exact-context",
        kubeconfig_path=kubeconfig,
        user_id="npa-validation",
        runner=runner,
        stop_api=stopped.append,
    )

    assert len(calls) == 2
    assert all("skypilot-cluster-name=sky-jobs-controller-npa-validation" in call for call in calls)
    assert stopped == [scope]
    assert not any(
        (scope / name).exists()
        for name in ("home", "sky-runtime", "local-api", "client-config.yaml")
    )
    archives = [path for path in scope.iterdir() if path.name.startswith("retired-")]
    assert len(archives) == 1
    assert all((archives[0] / name).is_dir() for name in ("home", "sky-runtime", "local-api"))
    assert (archives[0] / "client-config.yaml").is_file()


def test_idle_validation_scope_recovery_refuses_a_live_controller(tmp_path) -> None:
    scope = tmp_path / "cluster-validation" / ("b" * 24)
    scope.mkdir(parents=True)
    (scope / "home").mkdir()
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    stopped: list[Path] = []

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"items": [{"metadata": {"name": "live"}}]}), stderr=""
        )

    assert not _recover_idle_validation_scope(
        scope,
        context="exact-context",
        kubeconfig_path=kubeconfig,
        user_id="npa-validation",
        runner=runner,
        stop_api=stopped.append,
    )
    assert stopped == []
    assert (scope / "home").is_dir()


@pytest.mark.parametrize("message", [
    "isolated SkyPilot API process lifetime disagrees with its ownership record",
    "isolated SkyPilot API port is held by an unowned process",
    "isolated SkyPilot API belongs to another network namespace",
])
def test_stale_validation_recovery_refuses_unproven_api_ownership(message: str) -> None:
    from npa.orchestration.skypilot import local_api

    assert not gpu_catalog._is_stale_validation_api_error(
        local_api.IsolatedApiError(message)
    )


@pytest.mark.parametrize("failure_message", [
    "isolated SkyPilot API recovery requires the original executing identity and credential configuration",
    "running isolated SkyPilot API has a different executing identity or changed credential configuration",
    "isolated SkyPilot API credential configuration changed after verification",
    "isolated SkyPilot API recovery requires the original selected NPA configuration",
    "running isolated SkyPilot API has a different verified configuration; preserve its jobs before restarting",
    "isolated SkyPilot API verified configuration changed on disk",
    "isolated SkyPilot API process environment disagrees with its ownership record",
])
def test_validation_environment_recovers_stale_identity_raised_before_api_ensure(
    failure_message: str,
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale receipt can fail while ``sky_environment`` establishes intent.

    That failure happens before the explicit ``ensure_isolated_api`` call, so
    the narrow controller-absence recovery must cover both operations and both
    stopped and running stale daemon receipts.
    """

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    isolated_root = tmp_path / "isolated"
    monkeypatch.setenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", str(isolated_root))
    monkeypatch.delenv("SKYPILOT_USER_ID", raising=False)
    calls: list[str] = []
    recovered: list[dict[str, str]] = []
    client_configs: list[dict] = []

    from npa.orchestration.skypilot import local_api
    from npa.orchestration.skypilot import cleanup

    def fake_sky_environment(scope: Path, *, environment: dict[str, str]) -> dict[str, str]:
        calls.append("environment")
        client_configs.append(
            yaml.safe_load(
                Path(environment["SKYPILOT_GLOBAL_CONFIG"]).read_text(encoding="utf-8")
            )
        )
        if calls.count("environment") == 1:
            raise local_api.IsolatedApiError(failure_message)
        return {**environment, "SKYPILOT_USER_ID": "npa-test-validation"}

    def fake_recover(scope: Path, **kwargs: object) -> bool:
        recovered.append({"scope": str(scope), "user_id": str(kwargs["user_id"])})
        return True

    monkeypatch.setattr(cleanup, "sky_environment", fake_sky_environment)
    monkeypatch.setattr(
        local_api,
        "ensure_isolated_api",
        lambda **_kwargs: calls.append("ensure"),
    )
    monkeypatch.setattr(gpu_catalog, "_recover_idle_validation_scope", fake_recover)

    env = gpu_catalog.kubernetes_sky_environment(
        context="exact-context",
        kubeconfig=kubeconfig,
        sky_executable="/opt/sky/bin/sky",
    )

    assert calls == ["environment", "environment", "ensure"]
    assert env["SKYPILOT_USER_ID"] == "npa-test-validation"
    assert all(config["allowed_clouds"] == ["kubernetes"] for config in client_configs)
    assert len(recovered) == 1
    expected_scope = isolated_root / "cluster-validation" / (
        hashlib.sha256(
            f"{kubeconfig.resolve()}\0exact-context".encode()
        ).hexdigest()[:24]
    )
    assert recovered == [{
        "scope": str(expected_scope),
        "user_id": "npa-" + hashlib.sha256(
            str(expected_scope.resolve()).encode()
        ).hexdigest()[:12],
    }]


def test_validation_environment_migrates_changed_config_only_after_safe_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A changed validation config is archived, never overwritten in place."""

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    isolated_root = tmp_path / "isolated"
    context = "exact-context"
    scope = isolated_root / "cluster-validation" / hashlib.sha256(
        f"{kubeconfig.resolve()}\0{context}".encode()
    ).hexdigest()[:24]
    scope.mkdir(parents=True)
    stale_config = scope / "client-config.yaml"
    stale_config.write_text("allowed_clouds: [nebius]\n", encoding="utf-8")
    monkeypatch.setenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", str(isolated_root))
    monkeypatch.delenv("SKYPILOT_USER_ID", raising=False)
    recovered: list[str] = []

    from npa.orchestration.skypilot import cleanup, local_api

    def fake_recover(recovery_scope: Path, **_kwargs: object) -> bool:
        recovered.append(str(recovery_scope))
        stale_config.unlink()
        return True

    def fake_sky_environment(
        _scope: Path, *, environment: dict[str, str]
    ) -> dict[str, str]:
        return dict(environment)

    monkeypatch.setattr(gpu_catalog, "_recover_idle_validation_scope", fake_recover)
    monkeypatch.setattr(cleanup, "sky_environment", fake_sky_environment)
    monkeypatch.setattr(local_api, "ensure_isolated_api", lambda **_kwargs: None)

    gpu_catalog.kubernetes_sky_environment(
        context=context,
        kubeconfig=kubeconfig,
        sky_executable="/opt/sky/bin/sky",
    )

    assert recovered == [str(scope)]
    assert yaml.safe_load(stale_config.read_text(encoding="utf-8"))[
        "allowed_clouds"
    ] == ["kubernetes"]
