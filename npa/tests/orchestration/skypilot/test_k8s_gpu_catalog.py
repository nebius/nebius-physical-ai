from __future__ import annotations

import json
import subprocess

import pytest

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


def test_a_non_offered_quantity_lists_what_is_offered() -> None:
    catalog = parse_kubernetes_gpu_catalog(LIVE_OUTPUT, context="npa-rtxpro-mk8s")

    with pytest.raises(UnsatisfiableAcceleratorError) as excinfo:
        resolve_kubernetes_accelerator("RTXPRO6000:3", catalog=catalog)

    assert "it offers 1, 2, 4, 8 per node" in str(excinfo.value)


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
                                }
                            }
                        }
                    ],
                    "initContainers": [
                        {
                            "resources": {
                                "requests": {"cpu": "12", "memory": "32Gi"},
                                "limits": {"nvidia.com/gpu": "2"},
                            }
                        }
                    ],
                    "overhead": {"cpu": "1", "memory": "1Gi"},
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


def test_gang_capacity_fails_unknown_for_unbound_pending_gpu_demand() -> None:
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

    with pytest.raises(KubernetesGpuCatalogError, match="active unbound GPU pod"):
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
