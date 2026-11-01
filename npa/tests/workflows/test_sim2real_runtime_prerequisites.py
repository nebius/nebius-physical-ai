from __future__ import annotations

import base64
import re
from argparse import Namespace
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from npa.workflows.sim2real import runtime_prerequisites
from npa.workflows.sim2real.runtime_prerequisites import (
    MANAGED_BY,
    RuntimePrerequisiteError,
    _assert_contract,
    _assert_owned,
    build_manifests,
    resource_identities,
)


def _args(**overrides) -> Namespace:
    run_id = "turnkey-smoke"
    values = {
        "run_id": run_id,
        "namespace": "default",
        "isaac_image": "registry.example/npa-isaac@sha256:" + "a" * 64,
        **resource_identities(run_id),
        "storage_class": "csi-mounted-fs-path-sc",
        "storage_access_mode": "ReadWriteOnce",
        "isaac_cache_size": "40Gi",
        "policy_cache_size": "100Gi",
        "warm_timeout_seconds": 3600.0,
        "poll_interval_seconds": 0.01,
        "output_uri": "s3://bucket/runtime/prerequisites.json",
    }
    values.update(overrides)
    return Namespace(**values)


def test_manifests_bind_exact_run_owned_resources_and_hardened_warmer() -> None:
    args = _args()
    manifests = build_manifests(args)

    assert set(manifests) == {
        "isaac_cache_pvc",
        "policy_cache_pvc",
        "auth_secret",
        "base_service_account",
        "adapted_service_account",
        "base_role",
        "adapted_role",
        "base_role_binding",
        "adapted_role_binding",
        "isaac_warm_job",
    }
    assert all(
        manifest["metadata"]["annotations"]["npa.nebius.ai/cleanup-owner"]
        == args.run_id
        for manifest in manifests.values()
    )
    pod = manifests["isaac_warm_job"]["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["image"] == args.isaac_image
    assert container["securityContext"] == {
        "readOnlyRootFilesystem": True,
        "allowPrivilegeEscalation": False,
        "runAsNonRoot": True,
        "capabilities": {"drop": ["ALL"]},
    }
    mounts = {item["name"]: item["mountPath"] for item in container["volumeMounts"]}
    volumes = {item["name"]: item for item in pod["volumes"]}
    assert mounts["tmp"] == PurePosixPath("/").joinpath("tmp").as_posix()
    assert volumes["tmp"]["emptyDir"] == {"sizeLimit": "12Gi"}
    assert pod["volumes"][0]["persistentVolumeClaim"]["claimName"] == (
        args.isaac_cache_pvc
    )
    assert manifests["isaac_cache_pvc"]["spec"]["accessModes"] == ["ReadWriteOnce"]
    auth_token = base64.b64decode(
        manifests["auth_secret"]["data"]["auth_token"], validate=True
    )
    assert re.fullmatch(rb"[A-Za-z0-9_-]{32}", auth_token)


def test_manifests_support_declared_multi_writer_storage() -> None:
    manifests = build_manifests(_args(storage_access_mode="ReadWriteMany"))

    assert manifests["policy_cache_pvc"]["spec"]["accessModes"] == ["ReadWriteMany"]


def test_manifests_reject_unknown_storage_access_mode() -> None:
    with pytest.raises(RuntimePrerequisiteError, match="storage access mode"):
        build_manifests(_args(storage_access_mode="ReadOnlyMany"))


def test_controller_roles_are_limited_to_their_checkpoint_role() -> None:
    manifests = build_manifests(_args())
    base_names = {
        name
        for rule in manifests["base_role"]["rules"]
        for name in rule.get("resourceNames", [])
    }
    adapted_names = {
        name
        for rule in manifests["adapted_role"]["rules"]
        for name in rule.get("resourceNames", [])
    }
    assert base_names
    assert adapted_names
    assert base_names.isdisjoint(adapted_names)


def test_manifest_rejects_operator_substituted_identity_or_mutable_image() -> None:
    with pytest.raises(RuntimePrerequisiteError, match="deterministic run-owned"):
        build_manifests(_args(auth_secret="shared-secret"))
    with pytest.raises(RuntimePrerequisiteError, match="digest-pinned"):
        build_manifests(_args(isaac_image="registry.example/npa-isaac:latest"))


def test_long_run_ids_produce_distinct_kubernetes_label_safe_identities() -> None:
    common = "turnkey-smolvla-customer-adaptation-run-with-a-long-prefix-"
    first = resource_identities(common + "first")
    second = resource_identities(common + "second")

    assert first != second
    assert all(len(name) <= 63 for name in first.values())
    assert all(
        re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", name)
        for name in first.values()
    )

    args = _args(run_id=common + "first", **first)
    warm = build_manifests(args)["isaac_warm_job"]
    assert len(warm["metadata"]["labels"]["app"]) <= 63


def test_foreign_existing_resource_is_never_adopted() -> None:
    manifest = build_manifests(_args())["policy_cache_pvc"]
    foreign = SimpleNamespace(metadata=SimpleNamespace(labels={}, annotations={}))
    with pytest.raises(RuntimePrerequisiteError, match="refusing foreign"):
        _assert_owned(foreign, manifest, "turnkey-smoke")

    owned = SimpleNamespace(
        metadata=SimpleNamespace(
            labels={"app.kubernetes.io/managed-by": MANAGED_BY},
            annotations={"npa.nebius.ai/cleanup-owner": "turnkey-smoke"},
        )
    )
    _assert_owned(owned, manifest, "turnkey-smoke")


def test_owned_pvc_and_rbac_drift_are_rejected() -> None:
    manifests = build_manifests(_args())
    pvc = manifests["policy_cache_pvc"]
    drifted_pvc = {
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "storageClassName": pvc["spec"]["storageClassName"],
            "resources": pvc["spec"]["resources"],
        }
    }
    with pytest.raises(RuntimePrerequisiteError, match="drifted"):
        _assert_contract("policy_cache_pvc", drifted_pvc, pvc)

    role = manifests["base_role"]
    drifted_role = {
        "rules": [
            *role["rules"],
            {"apiGroups": ["*"], "resources": ["*"], "verbs": ["*"]},
        ]
    }
    with pytest.raises(RuntimePrerequisiteError, match="drifted"):
        _assert_contract("base_role", drifted_role, role)

    secret = manifests["auth_secret"]
    with pytest.raises(RuntimePrerequisiteError, match="drifted"):
        _assert_contract(
            "auth_secret",
            {"type": "Opaque", "data": {"auth_token": "dG9vLXNob3J0"}},
            secret,
        )
    with pytest.raises(RuntimePrerequisiteError, match="drifted"):
        _assert_contract(
            "auth_secret",
            {
                "type": "Opaque",
                "data": {"auth_token": base64.b64encode(bytes(range(32))).decode()},
            },
            secret,
        )


def test_apply_is_exactly_resumable_and_cleanup_is_idempotent(monkeypatch) -> None:
    args = _args()
    manifests = build_manifests(args)
    stored = {}
    deletion_bodies = {}

    def objects(*_clients):
        result = {}
        for key in manifests:

            def read(_name, _namespace, _request_timeout, *, item=key):
                return stored.get(item)

            def create(_namespace, body, _request_timeout, *, item=key):
                stored[item] = body

            def delete(
                _name,
                _namespace,
                _request_timeout,
                body=None,
                *,
                item=key,
            ):
                deletion_bodies[item] = body
                stored.pop(item, None)

            result[key] = (read, create, delete)
        return result

    core = SimpleNamespace(
        list_namespaced_pod=lambda *_args, **_kwargs: SimpleNamespace(items=[])
    )
    monkeypatch.setattr(
        runtime_prerequisites, "_clients", lambda: (core, object(), object(), object())
    )
    monkeypatch.setattr(runtime_prerequisites, "_objects", objects)
    monkeypatch.setattr(
        runtime_prerequisites,
        "_wait_for_warm",
        lambda _batch, _args: SimpleNamespace(metadata=SimpleNamespace(uid="warm-uid")),
    )
    monkeypatch.setattr(
        runtime_prerequisites, "_cleanup_smolvla_services", lambda *_args: None
    )

    first = runtime_prerequisites._apply(args, manifests)
    resumed = runtime_prerequisites._apply(args, manifests)
    cleaned = runtime_prerequisites._delete(args, manifests)
    repeated = runtime_prerequisites._delete(args, manifests)

    assert set(first["changed"]) == set(manifests)
    assert set(resumed["retained_or_absent"]) == set(manifests)
    assert set(cleaned["changed"]) == set(manifests)
    assert set(repeated["retained_or_absent"]) == set(manifests)
    assert stored == {}
    assert deletion_bodies["isaac_warm_job"] == {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "propagationPolicy": "Foreground",
    }
    assert all(
        body is None for key, body in deletion_bodies.items() if key != "isaac_warm_job"
    )
