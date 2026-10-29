"""Custody tests for the canonical Sim2Real Isaac-cache provisioner."""

from __future__ import annotations

from copy import deepcopy

import pytest
from typer.testing import CliRunner

from npa.cli.workbench.sim2real import app
from npa.workflows.sim2real import isaac_cache


class _ApiError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(str(status))
        self.status = status


def _claim(
    *,
    phase: str = "Bound",
    labels: dict[str, str] | None = None,
    access_modes: list[str] | None = None,
    storage_class: str = "csi-mounted-fs-path-sc",
    storage: str = "64Gi",
) -> dict[str, object]:
    return {
        "metadata": {
            "name": "npa-isaac-cache",
            "namespace": "default",
            "labels": labels or {},
        },
        "spec": {
            "accessModes": access_modes or ["ReadWriteMany"],
            "storageClassName": storage_class,
            "resources": {"requests": {"storage": storage}},
        },
        "status": {"phase": phase},
    }


class _Core:
    def __init__(self, reads: list[object], *, created: object | None = None) -> None:
        self.reads = list(reads)
        self.created = created
        self.read_calls: list[tuple[str, str]] = []
        self.create_calls: list[tuple[str, dict[str, object]]] = []

    def read_namespaced_persistent_volume_claim(
        self, *, name: str, namespace: str
    ) -> object:
        self.read_calls.append((name, namespace))
        item = self.reads.pop(0)
        if isinstance(item, BaseException):
            raise item
        return deepcopy(item)

    def create_namespaced_persistent_volume_claim(
        self, *, namespace: str, body: dict[str, object]
    ) -> object:
        self.create_calls.append((namespace, deepcopy(body)))
        if isinstance(self.created, BaseException):
            raise self.created
        return deepcopy(self.created)


def test_cache_claim_manifest_is_rwx_and_custody_labeled() -> None:
    manifest = isaac_cache.cache_claim_manifest(
        name="npa-isaac-cache",
        namespace="default",
        storage_class="csi-mounted-fs-path-sc",
        storage="64Gi",
    )

    assert manifest == {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": "npa-isaac-cache",
            "namespace": "default",
            "labels": {
                isaac_cache.MANAGED_BY_LABEL: isaac_cache.MANAGED_BY_VALUE,
                isaac_cache.CACHE_ROLE_LABEL: isaac_cache.CACHE_ROLE_VALUE,
            },
        },
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "storageClassName": "csi-mounted-fs-path-sc",
            "resources": {"requests": {"storage": "64Gi"}},
        },
    }


def test_cache_claim_manifest_rejects_namespace_that_is_not_a_dns_label() -> None:
    with pytest.raises(isaac_cache.IsaacCacheClaimError, match="DNS-label"):
        isaac_cache.cache_claim_manifest(
            name="npa-isaac-cache",
            namespace="not.a-namespace",
            storage_class="csi-mounted-fs-path-sc",
            storage="64Gi",
        )


def test_provision_creates_only_the_owned_claim() -> None:
    created = _claim(
        labels={
            isaac_cache.MANAGED_BY_LABEL: isaac_cache.MANAGED_BY_VALUE,
            isaac_cache.CACHE_ROLE_LABEL: isaac_cache.CACHE_ROLE_VALUE,
        }
    )
    core = _Core([_ApiError(404)], created=created)

    receipt = isaac_cache.provision_cache_claim(
        core_api=core,
        name="npa-isaac-cache",
        namespace="default",
        storage_class="csi-mounted-fs-path-sc",
        storage="64Gi",
    )

    assert receipt.created is True
    assert receipt.custody == "npa-managed"
    assert receipt.phase == "Bound"
    assert len(core.create_calls) == 1
    assert core.create_calls[0][1]["kind"] == "PersistentVolumeClaim"


def test_provision_adopts_a_complete_operator_claim_without_mutating_it() -> None:
    core = _Core([_claim(labels={"operator.example/owned": "true"})])

    receipt = isaac_cache.provision_cache_claim(
        core_api=core,
        name="npa-isaac-cache",
        namespace="default",
        storage_class="different-but-valid",
        storage="128Gi",
    )

    assert receipt.custody == "operator-existing"
    assert receipt.created is False
    assert core.create_calls == []


def test_provision_refuses_partial_npa_custody_without_replacing_claim() -> None:
    core = _Core(
        [_claim(labels={isaac_cache.MANAGED_BY_LABEL: isaac_cache.MANAGED_BY_VALUE})]
    )

    with pytest.raises(isaac_cache.IsaacCacheClaimError, match="incomplete NPA"):
        isaac_cache.provision_cache_claim(
            core_api=core,
            name="npa-isaac-cache",
            namespace="default",
            storage_class="csi-mounted-fs-path-sc",
            storage="64Gi",
        )

    assert core.create_calls == []


def test_provision_refuses_managed_claim_with_immutable_shape_drift() -> None:
    core = _Core(
        [
            _claim(
                labels={
                    isaac_cache.MANAGED_BY_LABEL: isaac_cache.MANAGED_BY_VALUE,
                    isaac_cache.CACHE_ROLE_LABEL: isaac_cache.CACHE_ROLE_VALUE,
                },
                storage="128Gi",
            )
        ]
    )

    with pytest.raises(isaac_cache.IsaacCacheClaimError, match="immutable"):
        isaac_cache.provision_cache_claim(
            core_api=core,
            name="npa-isaac-cache",
            namespace="default",
            storage_class="csi-mounted-fs-path-sc",
            storage="64Gi",
        )


def test_provision_waits_for_bound_without_creating_a_second_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _Core([_claim(phase="Pending"), _claim(phase="Bound")])
    monkeypatch.setattr(isaac_cache.time, "sleep", lambda _seconds: None)

    receipt = isaac_cache.provision_cache_claim(
        core_api=core,
        name="npa-isaac-cache",
        namespace="default",
        storage_class="csi-mounted-fs-path-sc",
        storage="64Gi",
    )

    assert receipt.phase == "Bound"
    assert core.create_calls == []


def test_cli_provision_uses_an_explicit_context_and_emits_a_claim_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workflows.sim2real.k8s_client import KubernetesJobClient

    core = _Core([_claim(labels={"operator.example/owned": "true"})])
    seen: dict[str, str] = {}

    def fake_from_environment(cls, **kwargs):
        seen.update(kwargs)
        return KubernetesJobClient(
            batch_api=object(), core_api=core, namespace="default"
        )

    monkeypatch.setattr(
        KubernetesJobClient, "from_environment", classmethod(fake_from_environment)
    )

    result = CliRunner().invoke(
        app,
        [
            "cache",
            "provision",
            "--name",
            "npa-isaac-cache",
            "--context",
            "customer-context",
            "--namespace",
            "default",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"status": "ready"' in result.output
    assert seen["context"] == "customer-context"
    assert seen["prefer_kubeconfig"] is True
    assert core.create_calls == []


def test_cli_pending_claim_directs_the_operator_to_guarded_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workflows.sim2real.k8s_client import KubernetesJobClient

    core = _Core(
        [
            _claim(
                phase="Pending",
                labels={
                    isaac_cache.MANAGED_BY_LABEL: isaac_cache.MANAGED_BY_VALUE,
                    isaac_cache.CACHE_ROLE_LABEL: isaac_cache.CACHE_ROLE_VALUE,
                },
            )
        ]
    )

    def fake_from_environment(cls, **_kwargs):
        return KubernetesJobClient(
            batch_api=object(), core_api=core, namespace="default"
        )

    monkeypatch.setattr(
        KubernetesJobClient, "from_environment", classmethod(fake_from_environment)
    )
    result = CliRunner().invoke(
        app,
        [
            "cache",
            "provision",
            "--name",
            "npa-isaac-cache",
            "--context",
            "customer-context",
            "--no-wait",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"status": "pending"' in result.output
    assert "WaitForFirstConsumer" in result.output
