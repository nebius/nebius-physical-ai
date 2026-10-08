"""Render personal execution boundaries with explicit quota and restricted identities."""

from __future__ import annotations

from .authorization import ExecutionBinding

WORKER_ACCOUNT = "npa-team-worker"
CONTROLLER_ACCOUNT = "npa-team-controller"
_LABEL = "npa.nebius.ai/team-execution"


def execution_manifests(binding: ExecutionBinding) -> list[dict]:
    """Render the resources an administrator applies for one personal allocation.

    Args:
        binding: Validated person, workspace, cluster, and capacity allocation.
    Returns:
        Namespaces, accounts, scoped roles, quotas, and network policy manifests.
    Raises:
        None.
    """
    worker = binding.namespace
    control = worker + "-control"
    result = [_namespace(worker, worker), _namespace(control, worker)]
    result += [
        _account(worker, WORKER_ACCOUNT, False),
        _account(control, CONTROLLER_ACCOUNT, True),
    ]
    result += [
        _quota(worker, binding.allocation.clusters[binding.cluster]),
        _quota(control, 0),
    ]
    result += _controller_roles(worker, control)
    result += _controller_roles(control, control)
    result += _discovery_roles(control)
    result += [
        _network_policy(worker, control, None),
        _network_policy(control, worker, binding),
    ]
    return result


def admission_manifests() -> list[dict]:
    """Render cluster admission rules for team-labeled execution namespaces.

    Args:
        None.
    Returns:
        A validating admission policy and its namespace-scoped binding.
    Raises:
        None.
    """
    return [_pod_policy(), _policy_binding()]


def _pod_policy():
    return {
        "apiVersion": "admissionregistration.k8s.io/v1",
        "kind": "ValidatingAdmissionPolicy",
        "metadata": {"name": "npa-team-pods"},
        "spec": {
            "failurePolicy": "Fail",
            "matchConstraints": {
                "resourceRules": [
                    {
                        "apiGroups": [""],
                        "apiVersions": ["v1"],
                        "operations": ["CREATE", "UPDATE"],
                        "resources": ["pods", "pods/ephemeralcontainers"],
                    }
                ]
            },
            "validations": _pod_validations(),
        },
    }


def _policy_binding():
    return {
        "apiVersion": "admissionregistration.k8s.io/v1",
        "kind": "ValidatingAdmissionPolicyBinding",
        "metadata": {"name": "npa-team-pods"},
        "spec": {
            "policyName": "npa-team-pods",
            "validationActions": ["Deny"],
            "matchResources": {
                "namespaceSelector": {
                    "matchExpressions": [
                        {
                            "key": _LABEL,
                            "operator": "Exists",
                        }
                    ]
                }
            },
        },
    }


def _namespace(name, execution):
    return {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": name,
            "labels": {
                _LABEL: execution,
                "pod-security.kubernetes.io/enforce": "baseline",
                "npa.nebius.ai/team-role": "controller"
                if name.endswith("-control")
                else "worker",
            },
        },
    }


def _account(namespace, name, automount):
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {"name": name, "namespace": namespace},
        "automountServiceAccountToken": automount,
    }


def _quota(namespace, gpus):
    return {
        "apiVersion": "v1",
        "kind": "ResourceQuota",
        "metadata": {"name": "npa-team-gpus", "namespace": namespace},
        "spec": {"hard": {"requests.nvidia.com/gpu": str(gpus)}},
    }


def _controller_roles(worker, control):
    metadata = {"name": "npa-team-execution", "namespace": worker}
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": metadata,
        "rules": _execution_rules(),
    }
    binding = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": metadata,
        "subjects": [
            {"kind": "ServiceAccount", "name": CONTROLLER_ACCOUNT, "namespace": control}
        ],
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": metadata["name"],
        },
    }
    return [role, binding]


def _execution_rules():
    return [
        {
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
            "verbs": [
                "get",
                "list",
                "watch",
                "create",
                "update",
                "patch",
                "delete",
            ],
        },
        {
            "apiGroups": [""],
            "resources": ["serviceaccounts"],
            "verbs": ["get", "list"],
        },
    ]


def _network_policy(namespace, peer, connection):
    own = {
        "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": namespace}}
    }
    other = {
        "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": peer}}
    }
    egress = [{"to": [own, other]}, _dns_egress(), _public_egress()]
    if connection is not None:
        cluster = connection.connection
        egress.append(
            {
                "to": [{"ipBlock": {"cidr": cluster.api_server_cidr}}],
                "ports": [
                    {
                        "protocol": "TCP",
                        "port": cluster.api_server_port,
                    }
                ],
            }
        )
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": "npa-team-boundary", "namespace": namespace},
        "spec": {
            "podSelector": {},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [{"from": [own, other]}],
            "egress": egress,
        },
    }


def _dns_egress():
    return {
        "to": [
            {
                "namespaceSelector": {
                    "matchLabels": {"kubernetes.io/metadata.name": "kube-system"}
                },
                "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
            }
        ],
        "ports": [{"protocol": protocol, "port": 53} for protocol in ("UDP", "TCP")],
    }


def _public_egress():
    excluded = [
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
    ]
    return {
        "to": [{"ipBlock": {"cidr": "0.0.0.0/0", "except": excluded}}],
        "ports": [{"protocol": "TCP", "port": port} for port in (80, 443)],
    }


def _pod_validations():
    role = "namespaceObject.metadata.labels['npa.nebius.ai/team-role']"
    expected = f"({role} == 'worker' ? '{WORKER_ACCOUNT}' : '{CONTROLLER_ACCOUNT}')"
    return [
        _whole_gpu_validation("containers"),
        _whole_gpu_validation("initContainers"),
        {
            "expression": f"object.spec.serviceAccountName == {expected}",
            "message": "team workload identity is fixed by the administrator",
        },
        {
            "expression": f"{role} != 'worker' || (has(object.spec.automountServiceAccountToken) && !object.spec.automountServiceAccountToken)",
            "message": "worker Kubernetes tokens are disabled",
        },
        {
            "expression": f"{role} != 'worker' || !has(object.spec.volumes) || object.spec.volumes.all(v, !has(v.projected) || v.projected.sources.all(s, !has(s.serviceAccountToken)))",
            "message": "workers cannot project Kubernetes tokens",
        },
        {
            "expression": "!has(object.spec.volumes) || object.spec.volumes.all(v, !has(v.hostPath))",
            "message": "team pods cannot mount host paths",
        },
        {
            "expression": "!has(object.spec.ephemeralContainers) || size(object.spec.ephemeralContainers) == 0",
            "message": "ephemeral containers require a separate administration path",
        },
        {
            "expression": "object.spec.containers.all(c, !has(c.securityContext) || !has(c.securityContext.privileged) || !c.securityContext.privileged)",
            "message": "team workloads cannot be privileged",
        },
    ]


def _whole_gpu_validation(field):
    return {
        "expression": f"!has(object.spec.{field}) || object.spec.{field}.all(c, !has(c.resources) || !has(c.resources.requests) || c.resources.requests.all(k, !k.startsWith('nvidia.com/') || k == 'nvidia.com/gpu'))",
        "message": "team quotas currently support whole NVIDIA GPUs only",
    }


def _discovery_roles(control):
    name = control + "-discovery"
    return [
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": name},
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["nodes", "namespaces"],
                    "verbs": ["get", "list", "watch"],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": name},
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": CONTROLLER_ACCOUNT,
                    "namespace": control,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": name,
            },
        },
    ]
