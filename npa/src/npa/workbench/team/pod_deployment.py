"""Render a CPU-only shared gateway with a loopback scheduler and durable state."""

from __future__ import annotations


def service_manifests(*, namespace: str, image: str, secret: str, claim: str):
    """Render an installation using administrator-created secret and persistent volume.

    Args:
        namespace: Dedicated management namespace, outside every execution namespace.
        image: Administrator-built team-server image, preferably an immutable digest.
        secret: Existing Secret containing team.yaml, sky-server.yaml, and credential files.
        claim: Existing ReadWriteOnce PVC for both authoritative ledgers.
    Returns:
        Deployment, Service, and NetworkPolicy; no public scheduler service.
    Raises:
        None.
    """
    labels = {"app.kubernetes.io/name": "npa-team"}
    gateway = _container("gateway", image)
    gateway.update(
        ports=[{"name": "gateway", "containerPort": 8443}],
        readinessProbe={"httpGet": {"path": "/health", "port": "gateway"}},
    )
    pod = _pod_spec(image, secret, claim, [gateway, _scheduler(image)])
    deployment = _deployment(namespace, labels, pod)
    return [deployment, _service(namespace, labels), _network(namespace, labels)]


def _scheduler(image):
    scheduler = _container("scheduler", image)
    scheduler.update(
        command=["/bin/sh", "-c"],
        args=[
            "mkdir -p /state/sky/.sky && ln -sfn /opt/sky /state/sky/skypilot-runtime && cd /state/sky && exec /opt/sky/bin/python -m sky.server.server --host 127.0.0.1 --port 46580"
        ],
        env=[
            {"name": key, "value": value}
            for key, value in {
                "HOME": "/state/sky",
                "PATH": "/opt/sky/bin:/usr/local/bin:/usr/bin:/bin",
                "SKYPILOT_GLOBAL_CONFIG": "/etc/npa-team/sky-server.yaml",
                "KUBECONFIG": "/etc/npa-team/sky-kubeconfig.yaml",
                "SKYPILOT_API_SERVER_ENDPOINT": "http://127.0.0.1:46580",
                "IS_SKYPILOT_SERVER": "true",
                "SKYPILOT_DISABLE_USAGE_COLLECTION": "1",
            }.items()
        ],
    )
    return scheduler


def _deployment(namespace, labels, pod):
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "npa-team", "namespace": namespace},
        "spec": {
            "replicas": 1,
            "strategy": {"type": "Recreate"},
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": pod,
            },
        },
    }


def _pod_spec(image, secret, claim, containers):
    return {
        "automountServiceAccountToken": False,
        "securityContext": {"runAsUser": 1000, "runAsGroup": 1000, "fsGroup": 1000},
        "containers": containers,
        "initContainers": [_private_configuration(image)],
        "volumes": [
            {"name": "state", "persistentVolumeClaim": {"claimName": claim}},
            {"name": "team-configuration", "emptyDir": {"medium": "Memory"}},
            {"name": "source", "secret": {"secretName": secret, "defaultMode": 0o440}},
        ],
    }


def _private_configuration(image):
    return {
        "name": "private-configuration",
        "image": image,
        "command": ["/bin/sh", "-c"],
        "args": ["umask 077; cp /source/* /private/; chmod 600 /private/*"],
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        },
        "volumeMounts": [
            {"name": "source", "mountPath": "/source", "readOnly": True},
            {"name": "team-configuration", "mountPath": "/private"},
        ],
    }


def _container(name, image):
    return {
        "name": name,
        "image": image,
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        },
        "volumeMounts": [
            {"name": "state", "mountPath": "/state"},
            {
                "name": "team-configuration",
                "mountPath": "/etc/npa-team",
                "readOnly": True,
            },
        ],
    }


def _service(namespace, labels):
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": "npa-team", "namespace": namespace},
        "spec": {
            "type": "ClusterIP",
            "selector": labels,
            "ports": [
                {"name": "https-upstream", "port": 8443, "targetPort": "gateway"}
            ],
        },
    }


def _network(namespace, labels):
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {"name": "npa-team", "namespace": namespace},
        "spec": {
            "podSelector": {"matchLabels": labels},
            "policyTypes": ["Ingress"],
            "ingress": [{"ports": [{"protocol": "TCP", "port": 8443}]}],
        },
    }
