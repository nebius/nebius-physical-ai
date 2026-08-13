from __future__ import annotations

import json
import subprocess

import pytest

from npa.cluster.drain import (
    DisruptionBlocker,
    DrainInventory,
    PdbRestoreError,
    relax_system_pdbs_for_full_destroy,
)


def _blocker(name: str, *, namespace: str = "kube-system") -> DisruptionBlocker:
    app = name
    return DisruptionBlocker(
        namespace=namespace,
        name=name,
        matching_pods=(f"{namespace}/{name}-pod",),
        workloads=(f"{namespace}/Deployment/{app}",),
        nodes=("node-a",),
        reason="fixture blocker",
        uid=f"uid-{namespace}-{name}",
        resource_version="42",
        labels=(("managed-by", "fixture"),),
        annotations=(("note", "restore-me"),),
        spec={"minAvailable": 1, "selector": {"matchLabels": {"app": app}}},
    )


def _inventory(*blockers: DisruptionBlocker) -> DrainInventory:
    return DrainInventory(
        nodes=("node-a",),
        pod_count=len(blockers),
        pdb_count=len(blockers),
        blockers=blockers,
    )


def _live_pdb_get(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    if "get" not in cmd:
        return None
    name = cmd[-1].rstrip("/").rsplit("/", 1)[-1]
    blocker = _blocker(name)
    payload = {
        "metadata": {
            "name": blocker.name,
            "namespace": blocker.namespace,
            "uid": blocker.uid,
            "resourceVersion": blocker.resource_version,
            "labels": dict(blocker.labels),
            "annotations": dict(blocker.annotations),
        },
        "spec": blocker.spec,
    }
    return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")


@pytest.mark.parametrize(
    "name", ["cilium-operator", "coredns", "coredns-autoscaler", "metrics-server"]
)
def test_full_destroy_relaxes_each_exact_system_blocker(name: str) -> None:
    calls: list[list[str]] = []

    def runner(cmd, **kwargs):  # noqa: ANN001
        calls.append(cmd)
        fetched = _live_pdb_get(cmd)
        if fetched is not None:
            return fetched
        if "create" in cmd:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="Cannot evict pod: disruption budget"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="deleted", stderr="")

    with relax_system_pdbs_for_full_destroy(
        _inventory(_blocker(name)),
        context="verified-context",
        kubeconfig="",
        confirmed_full_destroy=True,
        identity_verified=True,
        runner=runner,
    ) as report:
        pass

    assert report.removed == [f"kube-system/{name}"]
    assert any("eviction" in " ".join(call) for call in calls)
    assert any("poddisruptionbudgets" in " ".join(call) for call in calls)


def test_mixed_user_and_system_blockers_never_mutate_user_workloads() -> None:
    calls: list[list[str]] = []

    def runner(cmd, **kwargs):  # noqa: ANN001
        calls.append(cmd)
        fetched = _live_pdb_get(cmd)
        if fetched is not None:
            return fetched
        return subprocess.CompletedProcess(
            cmd,
            1 if "create" in cmd else 0,
            stdout="",
            stderr="disruption budget" if "create" in cmd else "",
        )

    with relax_system_pdbs_for_full_destroy(
        _inventory(_blocker("coredns"), _blocker("orders", namespace="customer")),
        context="verified-context",
        kubeconfig="",
        confirmed_full_destroy=True,
        identity_verified=True,
        runner=runner,
    ) as report:
        pass

    serialized = "\n".join(" ".join(call) for call in calls)
    assert "customer" not in serialized
    assert report.user_or_unverified == ["customer/orders"]
    assert report.removed == ["kube-system/coredns"]


def test_allowlisted_pdb_name_with_unrelated_workload_is_never_mutated() -> None:
    calls: list[list[str]] = []
    disguised = DisruptionBlocker(
        **{
            **_blocker("coredns").__dict__,
            "workloads": ("kube-system/Deployment/customer-dns",),
        }
    )

    with relax_system_pdbs_for_full_destroy(
        _inventory(disguised),
        context="verified-context",
        kubeconfig="",
        confirmed_full_destroy=True,
        identity_verified=True,
        runner=lambda cmd, **kwargs: calls.append(cmd),
    ) as report:
        pass

    assert calls == []
    assert report.user_or_unverified == ["kube-system/coredns"]


@pytest.mark.parametrize(
    ("confirmed", "verified"), [(False, True), (True, False), (False, False)]
)
def test_shared_node_pool_or_context_mismatch_performs_no_mutation(
    confirmed: bool, verified: bool
) -> None:
    calls: list[list[str]] = []

    with relax_system_pdbs_for_full_destroy(
        _inventory(_blocker("coredns")),
        context="unverified-context",
        kubeconfig="",
        confirmed_full_destroy=confirmed,
        identity_verified=verified,
        runner=lambda cmd, **kwargs: calls.append(cmd),
    ):
        pass

    assert calls == []


def test_rbac_failure_preserves_the_pdb() -> None:
    calls: list[list[str]] = []

    def runner(cmd, **kwargs):  # noqa: ANN001
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="Forbidden: cannot create pods/eviction"
        )

    with relax_system_pdbs_for_full_destroy(
        _inventory(_blocker("metrics-server")),
        context="verified-context",
        kubeconfig="",
        confirmed_full_destroy=True,
        identity_verified=True,
        runner=runner,
    ) as report:
        pass

    assert len(calls) == 1
    assert report.removed == []
    assert report.errors == [
        "kube-system/metrics-server: normal eviction could not be verified; PDB preserved"
    ]


def test_failed_destroy_restores_the_exact_snapshot() -> None:
    calls: list[tuple[list[str], str]] = []

    def runner(cmd, **kwargs):  # noqa: ANN001
        calls.append((cmd, str(kwargs.get("input") or "")))
        fetched = _live_pdb_get(cmd)
        if fetched is not None:
            return fetched
        if "create" in cmd:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="disruption budget"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    report = None
    with pytest.raises(RuntimeError, match="terraform failed"):
        with relax_system_pdbs_for_full_destroy(
            _inventory(_blocker("coredns")),
            context="verified-context",
            kubeconfig="",
            confirmed_full_destroy=True,
            identity_verified=True,
            runner=runner,
        ) as report:
            raise RuntimeError("terraform failed")

    assert report is not None
    assert report.restored == ["kube-system/coredns"]
    apply_payload = next(payload for cmd, payload in calls if "apply" in cmd)
    assert "minAvailable: 1" in apply_payload
    assert "restore-me" in apply_payload


def test_restore_failure_is_reported_with_original_cause() -> None:
    def runner(cmd, **kwargs):  # noqa: ANN001
        fetched = _live_pdb_get(cmd)
        if fetched is not None:
            return fetched
        if "create" in cmd:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="disruption budget"
            )
        if "apply" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Forbidden")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    with pytest.raises(PdbRestoreError, match="restoration was incomplete") as error:
        with relax_system_pdbs_for_full_destroy(
            _inventory(_blocker("coredns")),
            context="verified-context",
            kubeconfig="",
            confirmed_full_destroy=True,
            identity_verified=True,
            runner=runner,
        ):
            raise RuntimeError("destroy aborted")

    assert isinstance(error.value.__cause__, RuntimeError)


def test_already_gone_pod_and_pdb_are_successful_convergence() -> None:
    calls: list[list[str]] = []

    def runner(cmd, **kwargs):  # noqa: ANN001
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="NotFound")

    with relax_system_pdbs_for_full_destroy(
        _inventory(_blocker("coredns-autoscaler")),
        context="verified-context",
        kubeconfig="",
        confirmed_full_destroy=True,
        identity_verified=True,
        runner=runner,
    ) as report:
        pass

    assert report.errors == []
    assert report.eviction_attempts == [
        "kube-system/coredns-autoscaler-pod:already-gone"
    ]
    assert not any("poddisruptionbudgets" in " ".join(call) for call in calls)
