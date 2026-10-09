"""Render personal execution boundaries with explicit quota and restricted identities."""

from __future__ import annotations

from .authorization import ExecutionBinding

WORKER_ACCOUNT = "npa-team-worker"
_LABEL = "npa.nebius.ai/team-execution"


def execution_manifests(binding: ExecutionBinding) -> list[dict]:
    """Render the resources an administrator applies for one personal allocation.

    Args:
        binding: Validated person, workspace, cluster, and capacity allocation.
    Returns:
        Namespaces, worker accounts, quotas, and network policy manifests.
    Raises:
        None.
    """
    worker = binding.namespace
    return [
        _namespace(worker, worker),
        _account(worker, WORKER_ACCOUNT, False),
        _quota(worker, binding.allocation.clusters[binding.cluster]),
        _network_policy(worker),
    ]


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
                "npa.nebius.ai/team-role": "worker",
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


def _network_policy(namespace):
    own = {
        "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": namespace}}
    }
    egress = [{"to": [own]}, _dns_egress(), _public_egress()]
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": "npa-team-boundary", "namespace": namespace},
        "spec": {
            "podSelector": {},
            "policyTypes": ["Ingress", "Egress"],
            "ingress": [{"from": [own]}],
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
    return [
        _whole_gpu_validation("containers"),
        _whole_gpu_validation("initContainers"),
        {
            "expression": f"object.spec.serviceAccountName == '{WORKER_ACCOUNT}'",
            "message": "team workload identity is fixed by the administrator",
        },
        {
            "expression": "has(object.spec.automountServiceAccountToken) && !object.spec.automountServiceAccountToken",
            "message": "worker Kubernetes tokens are disabled",
        },
        {
            "expression": "!has(object.spec.volumes) || object.spec.volumes.all(v, !has(v.projected) || v.projected.sources.all(s, !has(s.serviceAccountToken)))",
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
    # ResourceRequirements contains Quantity-valued maps. Some Kubernetes
    # OpenAPI versions omit these fields from CEL's static type description;
    # dynamic access preserves the runtime validation of every resource key.
    requests = "dyn(c.resources).requests"
    return {
        "expression": f"!has(object.spec.{field}) || object.spec.{field}.all(c, !has(c.resources) || !has({requests}) || {requests}.all(k, !k.startsWith('nvidia.com/') || k == 'nvidia.com/gpu'))",
        "message": "team quotas currently support whole NVIDIA GPUs only",
    }
