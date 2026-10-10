"""Render HTTPS-only control-plane networking while keeping SkyPilot private."""

from .pod_deployment import service_manifests

OWNER_LABEL = "npa.nebius.ai/control-plane-installation"


def endpoint_service(request, installation_id, allocation_id=""):
    """Render the single public Workbench HTTPS listener.

    Args:
        request, installation_id: Operator selection and durable setup owner.
        allocation_id: Previously retained address to reuse after Service recreation.
    Returns:
        Namespaced Kubernetes LoadBalancer Service.
    Raises:
        None.
    """
    metadata = {
        "name": request.service_name,
        "namespace": request.namespace,
        "labels": {OWNER_LABEL: installation_id},
    }
    if allocation_id:
        metadata["annotations"] = {
            "nebius.com/load-balancer-allocation-id": allocation_id
        }
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": metadata,
        "spec": {
            "type": "LoadBalancer",
            "externalTrafficPolicy": "Cluster",
            "selector": {"app.kubernetes.io/name": "npa-team"},
            "ports": [{"name": "https", "port": 443, "targetPort": 8444}],
        },
    }


def https_manifests(request, installation_id):
    """Add TLS termination to the existing CPU-only team server renderer.

    Args:
        request: Reviewed server image, private configuration, state and TLS references.
        installation_id: Durable setup owner label for created resources.
    Returns:
        Deployment, private TLS proxy configuration, and HTTPS-only NetworkPolicy.
    Raises:
        None.
    """
    deployment, _, policy = service_manifests(
        namespace=request.namespace,
        image=request.image,
        secret=request.configuration_secret,
        claim=request.state_claim,
    )
    pod = deployment["spec"]["template"]["spec"]
    pod["nodeSelector"] = request.node_selector
    if request.image_pull_secrets:
        pod["imagePullSecrets"] = [
            {"name": name} for name in request.image_pull_secrets
        ]
    pod["containers"].append(_proxy(request))
    pod["volumes"] += [
        {"name": "https-configuration", "configMap": {"name": "npa-team-https"}},
        {
            "name": "https-certificate",
            "secret": {"secretName": request.tls_secret, "defaultMode": 0o440},
        },
        {"name": "https-data", "emptyDir": {}},
    ]
    policy["spec"]["ingress"] = [{"ports": [{"protocol": "TCP", "port": 8444}]}]
    documents = [deployment, _proxy_config(request.namespace), policy]
    for document in documents:
        document["metadata"].setdefault("labels", {})[OWNER_LABEL] = installation_id
    return documents


def _proxy(request):
    return {
        "name": "https",
        "image": request.proxy_image,
        "command": [
            "caddy",
            "run",
            "--config",
            "/etc/npa-https/Caddyfile",
            "--adapter",
            "caddyfile",
        ],
        "ports": [{"name": "https", "containerPort": 8444}],
        "readinessProbe": {
            "httpGet": {"scheme": "HTTPS", "path": "/health", "port": "https"}
        },
        "securityContext": {
            "allowPrivilegeEscalation": False,
            # The upstream binary's file capability must remain in the bounding set.
            "capabilities": {"drop": ["ALL"], "add": ["NET_BIND_SERVICE"]},
        },
        "resources": {"requests": {"cpu": "100m", "memory": "64Mi"}},
        "env": [
            {"name": "XDG_CONFIG_HOME", "value": "/data/config"},
            {"name": "XDG_DATA_HOME", "value": "/data"},
        ],
        "volumeMounts": _proxy_mounts(),
    }


def _proxy_mounts():
    return [
        {
            "name": "https-configuration",
            "mountPath": "/etc/npa-https",
            "readOnly": True,
        },
        {
            "name": "https-certificate",
            "mountPath": "/etc/npa-tls",
            "readOnly": True,
        },
        {"name": "https-data", "mountPath": "/data"},
    ]


def _proxy_config(namespace):
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "npa-team-https", "namespace": namespace},
        "data": {
            "Caddyfile": "{\n admin off\n auto_https off\n}\n:8444 {\n tls /etc/npa-tls/tls.crt /etc/npa-tls/tls.key\n reverse_proxy 127.0.0.1:8443\n}\n"
        },
    }
