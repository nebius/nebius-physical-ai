"""Render the trusted scheduler's Kubernetes access without cluster-wide writes."""

from .manifests import WORKER_ACCOUNT


def scheduler_access_manifests(bindings, namespace: str, account: str) -> list[dict]:
    """Bind one installed scheduler account to explicit allocations in one cluster.

    Args:
        bindings: Personal allocations belonging to one enrolled cluster.
        namespace, account: Existing management namespace and service account.
    Returns:
        Read-only discovery and admission access plus namespaced execution grants.
    Raises:
        ValueError: Allocations are empty or span multiple clusters.
    """
    bindings = list(bindings)
    if not bindings or len({binding.cluster for binding in bindings}) != 1:
        raise ValueError("select allocations from exactly one cluster")
    subject = {"kind": "ServiceAccount", "name": account, "namespace": namespace}
    name = f"{namespace}-{account}-discovery"
    documents = _grant(name, None, subject, _discovery_rules())
    for binding in bindings:
        documents.extend(
            _grant("npa-team-scheduler", binding.namespace, subject, _worker_rules())
        )
    return documents


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
