"""Provision exact, run-owned RBAC for the OpenPI cross-pod service stage."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable, Mapping, Sequence

from npa.workflows.byof.openpi_service import (
    CONTROLLER_MANAGED_BY,
    DEFAULT_API_TIMEOUT_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    OpenPIServiceError,
    _api_call,
    _validated_positive_seconds,
    build_controller_rbac_manifests,
    controller_service_account_name,
    service_resource_names,
)

DEFAULT_RBAC_DELETE_TIMEOUT_SECONDS = 120.0


def _status(exc: Exception) -> int | None:
    return getattr(exc, "status", None)


def _metadata_value(obj: object, field: str, key: str) -> str | None:
    metadata = getattr(obj, "metadata", None)
    values = getattr(metadata, field, None) or {}
    return values.get(key)


def _assert_owned(obj: object, *, run_id: str, kind: str, name: str) -> None:
    owner = _metadata_value(obj, "annotations", "npa.nebius.ai/cleanup-owner")
    managed_by = _metadata_value(obj, "labels", "app.kubernetes.io/managed-by")
    if owner != run_id or managed_by != CONTROLLER_MANAGED_BY:
        raise OpenPIServiceError(
            f"refusing to modify pre-existing {kind} {name!r}: "
            "exact OpenPI run ownership is not proven"
        )


def _normalized_rules(rules: Sequence[object] | None) -> list[dict[str, list[str]]]:
    normalized = []
    for rule in rules or []:
        if isinstance(rule, Mapping):
            api_groups = rule.get("apiGroups", [])
            resources = rule.get("resources", [])
            resource_names = rule.get("resourceNames", [])
            verbs = rule.get("verbs", [])
        else:
            api_groups = getattr(rule, "api_groups", None) or []
            resources = getattr(rule, "resources", None) or []
            resource_names = getattr(rule, "resource_names", None) or []
            verbs = getattr(rule, "verbs", None) or []
        normalized.append(
            {
                "apiGroups": sorted(str(value) for value in api_groups),
                "resources": sorted(str(value) for value in resources),
                "resourceNames": sorted(str(value) for value in resource_names),
                "verbs": sorted(str(value) for value in verbs),
            }
        )
    return sorted(normalized, key=lambda value: json.dumps(value, sort_keys=True))


def _assert_role_contract(role: object, desired: Mapping[str, Any], name: str) -> None:
    if _normalized_rules(getattr(role, "rules", None)) != _normalized_rules(
        desired["rules"]
    ):
        raise OpenPIServiceError(
            f"refusing to reuse Role {name!r}: permissions differ from the "
            "name-scoped OpenPI controller contract"
        )


def _assert_binding_contract(
    binding: object, desired: Mapping[str, Any], name: str
) -> None:
    actual_subjects = [
        {
            "kind": str(getattr(subject, "kind", "")),
            "name": str(getattr(subject, "name", "")),
            "namespace": str(getattr(subject, "namespace", "")),
        }
        for subject in (getattr(binding, "subjects", None) or [])
    ]
    role_ref = getattr(binding, "role_ref", None)
    actual_role_ref = {
        "apiGroup": str(getattr(role_ref, "api_group", "")),
        "kind": str(getattr(role_ref, "kind", "")),
        "name": str(getattr(role_ref, "name", "")),
    }
    if actual_subjects != desired["subjects"] or actual_role_ref != desired["roleRef"]:
        raise OpenPIServiceError(
            f"refusing to reuse RoleBinding {name!r}: binding differs from the "
            "OpenPI controller contract"
        )


def _read_or_none(
    reader: Callable[[str, str], Any],
    name: str,
    namespace: str,
    *,
    request_timeout: float,
) -> object | None:
    try:
        return _api_call(
            reader,
            name,
            namespace,
            request_timeout=request_timeout,
        )
    except Exception as exc:
        if _status(exc) == 404:
            return None
        raise


def apply_controller_rbac(
    core: Any,
    rbac: Any,
    *,
    run_id: str,
    namespace: str,
    service_account: str,
    request_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
    delete_timeout: float = DEFAULT_RBAC_DELETE_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> dict[str, object]:
    """Create missing controller RBAC and fail closed on any foreign identity."""

    manifests = build_controller_rbac_manifests(
        run_id=run_id,
        namespace=namespace,
        service_account=service_account,
    )
    readers = {
        "service_account": core.read_namespaced_service_account,
        "role": rbac.read_namespaced_role,
        "role_binding": rbac.read_namespaced_role_binding,
    }
    existing = {
        key: _read_or_none(
            reader,
            service_account,
            namespace,
            request_timeout=request_timeout,
        )
        for key, reader in readers.items()
    }
    for key, obj in existing.items():
        if obj is not None:
            _assert_owned(obj, run_id=run_id, kind=key, name=service_account)
    if existing["role"] is not None:
        _assert_role_contract(existing["role"], manifests["role"], service_account)
    if existing["role_binding"] is not None:
        _assert_binding_contract(
            existing["role_binding"], manifests["role_binding"], service_account
        )

    creators = {
        "service_account": core.create_namespaced_service_account,
        "role": rbac.create_namespaced_role,
        "role_binding": rbac.create_namespaced_role_binding,
    }
    created: list[str] = []
    try:
        for key in ("service_account", "role", "role_binding"):
            if existing[key] is None:
                # The API response can be lost after the object is created. Track
                # the deterministic identity before the request so the exception
                # path performs an ownership-checked absence verification.
                created.append(key)
                _api_call(
                    creators[key],
                    namespace,
                    manifests[key],
                    request_timeout=request_timeout,
                )
    except Exception:
        _delete_controller_rbac(
            core,
            rbac,
            run_id=run_id,
            namespace=namespace,
            service_account=service_account,
            only=set(created),
            timeout=delete_timeout,
            poll_interval=poll_interval,
            request_timeout=request_timeout,
        )
        raise
    return {
        "schema": "npa.workbench.openpi.service-controller-rbac.v1",
        "action": "apply",
        "status": "passed",
        "run_id": run_id,
        "namespace": namespace,
        "service_account": service_account,
        "created": created,
        "reused_exact_owned": [key for key, obj in existing.items() if obj is not None],
        "permission_scope": {
            "classification": "name_scoped_with_kubernetes_residuals",
            "foreign_secret_contents_readable": False,
            "pod_logs_readable": False,
            "exact_resource_names": service_resource_names(run_id),
            "residual_namespace_scope": [
                {
                    "resources": ["services", "secrets", "deployments", "jobs"],
                    "verbs": ["create"],
                    "reason": "Kubernetes RBAC cannot resourceName-scope create",
                },
                {
                    "resources": ["pods"],
                    "verbs": ["list"],
                    "reason": (
                        "Deployment and Job pod names are controller-generated; "
                        "the controller applies a run selector and validates ownership"
                    ),
                },
            ],
        },
    }


def _delete_controller_rbac(
    core: Any,
    rbac: Any,
    *,
    run_id: str,
    namespace: str,
    service_account: str,
    only: set[str] | None = None,
    timeout: float = DEFAULT_RBAC_DELETE_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    request_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, bool]:
    timeout = _validated_positive_seconds(timeout, "RBAC deletion timeout")
    poll_interval = _validated_positive_seconds(poll_interval, "poll interval")
    request_timeout = _validated_positive_seconds(
        request_timeout, "Kubernetes API timeout"
    )
    deadline = clock() + timeout
    readers = {
        "service_account": core.read_namespaced_service_account,
        "role": rbac.read_namespaced_role,
        "role_binding": rbac.read_namespaced_role_binding,
    }
    existing = {
        key: _read_or_none(
            reader,
            service_account,
            namespace,
            request_timeout=request_timeout,
        )
        for key, reader in readers.items()
    }
    for key, obj in existing.items():
        if obj is not None and (only is None or key in only):
            _assert_owned(obj, run_id=run_id, kind=key, name=service_account)

    deleters = {
        "role_binding": rbac.delete_namespaced_role_binding,
        "role": rbac.delete_namespaced_role,
        "service_account": core.delete_namespaced_service_account,
    }
    issues: list[str] = []
    for key in ("role_binding", "role", "service_account"):
        if existing[key] is None or (only is not None and key not in only):
            continue
        try:
            _api_call(
                deleters[key],
                service_account,
                namespace,
                request_timeout=request_timeout,
            )
        except Exception as exc:
            if _status(exc) != 404:
                issues.append(f"{key} delete request failed: {exc}")

    verified: dict[str, bool] = {}
    for key, reader in readers.items():
        if only is not None and key not in only:
            continue
        last_state: dict[str, object] = {}
        while clock() < deadline:
            try:
                obj = _read_or_none(
                    reader,
                    service_account,
                    namespace,
                    request_timeout=request_timeout,
                )
            except Exception as exc:
                issues.append(f"{key} absence is uncertain: {exc}")
                break
            if obj is None:
                verified[key] = True
                break
            metadata = getattr(obj, "metadata", None)
            last_state = {
                "deletion_timestamp": str(
                    getattr(metadata, "deletion_timestamp", "") or ""
                ),
                "finalizers": list(getattr(metadata, "finalizers", None) or []),
            }
            sleep(min(poll_interval, max(0.0, deadline - clock())))
        if key not in verified and not any(
            issue.startswith(f"{key} absence is uncertain") for issue in issues
        ):
            issues.append(
                f"{key} deletion timed out after {timeout:g}s; "
                f"last_state={json.dumps(last_state, sort_keys=True)}"
            )
    if issues:
        raise OpenPIServiceError("exact RBAC cleanup incomplete: " + "; ".join(issues))
    return verified


def delete_controller_rbac(
    core: Any,
    rbac: Any,
    *,
    run_id: str,
    namespace: str,
    service_account: str,
    timeout: float = DEFAULT_RBAC_DELETE_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    request_timeout: float = DEFAULT_API_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Remove only the exact run-owned controller identity and verify absence."""

    verified = _delete_controller_rbac(
        core,
        rbac,
        run_id=run_id,
        namespace=namespace,
        service_account=service_account,
        timeout=timeout,
        poll_interval=poll_interval,
        request_timeout=request_timeout,
    )
    return {
        "schema": "npa.workbench.openpi.service-controller-rbac.v1",
        "action": "delete",
        "status": "passed",
        "run_id": run_id,
        "namespace": namespace,
        "service_account": service_account,
        "all_exact_resources_absent": all(verified.values()),
        "verified": verified,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "delete"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--service-account")
    parser.add_argument("--kubeconfig")
    parser.add_argument("--context")
    parser.add_argument(
        "--delete-timeout-seconds",
        type=float,
        default=DEFAULT_RBAC_DELETE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--api-timeout-seconds", type=float, default=DEFAULT_API_TIMEOUT_SECONDS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service_account = args.service_account or controller_service_account_name(
        args.run_id
    )
    from kubernetes import client, config

    config.load_kube_config(config_file=args.kubeconfig, context=args.context)
    core = client.CoreV1Api()
    rbac = client.RbacAuthorizationV1Api()
    if args.action == "apply":
        result = apply_controller_rbac(
            core,
            rbac,
            run_id=args.run_id,
            namespace=args.namespace,
            service_account=service_account,
            request_timeout=args.api_timeout_seconds,
            delete_timeout=args.delete_timeout_seconds,
            poll_interval=args.poll_interval_seconds,
        )
    else:
        result = delete_controller_rbac(
            core,
            rbac,
            run_id=args.run_id,
            namespace=args.namespace,
            service_account=service_account,
            timeout=args.delete_timeout_seconds,
            poll_interval=args.poll_interval_seconds,
            request_timeout=args.api_timeout_seconds,
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
