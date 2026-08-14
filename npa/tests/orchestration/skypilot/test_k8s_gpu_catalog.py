from __future__ import annotations

import json
import subprocess

import pytest

from npa.orchestration.skypilot.k8s_gpu_catalog import (
    KubernetesGpuInventory,
    KubernetesGpuCatalog,
    KubernetesGpuCatalogError,
    UnsatisfiableAcceleratorError,
    context_from_infra,
    discover_kubernetes_gpu_catalog,
    discover_kubernetes_gpu_inventory,
    label_known_kubernetes_gpus_for_skypilot,
    parse_kubernetes_gpu_catalog,
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
        return subprocess.CompletedProcess(cmd, 0, stdout="node/node-a labelled", stderr="")

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
        "npa.orchestration.skypilot.k8s_gpu_catalog."
        "discover_kubernetes_gpu_inventory",
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
