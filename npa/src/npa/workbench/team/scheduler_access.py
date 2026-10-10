"""Render the trusted scheduler's Kubernetes access without cluster-wide writes."""

import hashlib

from .manifests import WORKER_ACCOUNT


def scheduler_access_manifests(
    bindings, namespace: str, account: str, *, subject_kind: str = "ServiceAccount"
) -> list[dict]:
    """Bind one installed scheduler account to explicit allocations in one cluster.

    Args:
        bindings: Personal allocations belonging to one enrolled cluster.
        namespace, account: Management namespace and service account or group name.
        subject_kind: ServiceAccount, or Group for an external IAM-mapped identity.
    Returns:
        Read-only discovery and admission access plus namespaced execution grants.
    Raises:
        ValueError: Allocations span multiple clusters or the subject is invalid.
    """
    bindings = list(bindings)
    if not bindings or len({binding.cluster for binding in bindings}) != 1:
        raise ValueError("select allocations from exactly one cluster")
    subject, name = _scheduler_subject(namespace, account, subject_kind)
    documents = _grant(name, None, subject, _discovery_rules())
    for binding in bindings:
        documents.extend(
            _grant("npa-team-scheduler", binding.namespace, subject, _worker_rules())
        )
    return documents


def _scheduler_subject(namespace, account, kind):
    if kind not in ("ServiceAccount", "Group") or not account.strip():
        raise ValueError("scheduler identity must be a ServiceAccount or Group")
    if kind == "ServiceAccount":
        return {
            "kind": kind,
            "name": account,
            "namespace": namespace,
        }, f"{namespace}-{account}-discovery"
    identity = hashlib.sha256(account.encode()).hexdigest()[:24]
    return {
        "kind": "Group",
        "name": account,
        "apiGroup": "rbac.authorization.k8s.io",
    }, f"npa-scheduler-{identity}-discovery"


def _grant(name, namespace, subject, rules):
    kind = "Role" if namespace else "ClusterRole"
    metadata = {"name": name}
    if namespace:
        metadata["namespace"] = namespace
    base = {"apiVersion": "rbac.authorization.k8s.io/v1", "metadata": metadata}
    return [
        {**base, "kind": kind, "rules": rules},
        {
            **base,
            "kind": kind + "Binding",
            "subjects": [subject],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": kind,
                "name": name,
            },
        },
    ]


def _discovery_rules():
    return [
        {
            "apiGroups": [""],
            "resources": ["nodes", "namespaces"],
            "verbs": ["get", "list", "watch"],
        },
        {
            "apiGroups": ["node.k8s.io"],
            "resources": ["runtimeclasses"],
            "verbs": ["get", "list"],
        },
        {
            "apiGroups": ["admissionregistration.k8s.io"],
            "resources": [
                "validatingadmissionpolicies",
                "validatingadmissionpolicybindings",
            ],
            "resourceNames": ["npa-team-pods"],
            "verbs": ["get"],
        },
    ]


def _worker_rules():
    return [
        _execution_rule(),
        {"apiGroups": [""], "resources": ["services"], "verbs": ["deletecollection"]},
        {
            "apiGroups": [""],
            "resources": ["serviceaccounts", "resourcequotas"],
            "verbs": ["get", "list"],
        },
        {
            "apiGroups": ["networking.k8s.io"],
            "resources": ["networkpolicies"],
            "verbs": ["get"],
        },
        {
            "apiGroups": [""],
            "resources": ["serviceaccounts"],
            "resourceNames": [WORKER_ACCOUNT],
            "verbs": ["impersonate"],
        },
    ]


def _execution_rule():
    return {
        "apiGroups": [""],
        "resources": [
            "pods",
            "pods/exec",
            "pods/portforward",
            "pods/log",
            "services",
            "events",
            "configmaps",
            "secrets",
            "persistentvolumeclaims",
        ],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    }
