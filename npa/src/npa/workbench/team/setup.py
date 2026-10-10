"""Set up one persistent HTTPS Workbench control plane independently of user clients."""

import ssl
import time
from pathlib import Path

from .endpoint_allocation import (
    ALLOCATION_ANNOTATION,
    retain_allocation,
    verify_saved_allocation,
)
from .endpoint_certificate import verify_certificate
from .endpoint_health import verify_endpoint
from .endpoint_manifests import OWNER_LABEL, endpoint_service, https_manifests
from .errors import BackendError, ConflictError
from .setup_models import SetupRequest
from .setup_receipt import SetupReceipt, setup_lock
from .setup_service_selection import select_service
from .setup_transport import kubernetes, provider, resource


def setup_control_plane(request: SetupRequest, output_path: Path):
    """Apply shared server networking and publish a verified persistent endpoint.

    Args:
        request: Operator-owned provider, cluster, server and TLS configuration.
        output_path: Private durable receipt reused for retries and Service recovery.
    Returns:
        Credential-free status and receipt path; no personal client is required.
    Raises:
        BackendError, ConflictError, OSError: Preconditions, ownership, or I/O fail.
    """
    output_path = output_path.absolute()
    with setup_lock(output_path):
        receipt = SetupReceipt(output_path, request)
        _preflight(request)
        request = select_service(request, receipt)
        deployment = resource(request, "deployment", "npa-team")
        service = resource(request, "service", request.service_name)
        _validate_existing(request, receipt, deployment, service)
        _ensure_deployment(request, receipt, deployment)
        _wait_deployment(request, receipt)
        service = _ensure_service(request, receipt, service)
        service = _wait_address(request, receipt)
        retain_allocation(request, receipt, service)
        receipt.save(endpoint=request.endpoint.rstrip("/"), phase="endpoint-configured")
        return _qualify(request, receipt)


def _preflight(request):
    cluster = provider("mk8s", "cluster", "get", "--id", request.cluster_id)
    if cluster.get("metadata", {}).get("parent_id") != request.project_id:
        raise ConflictError("selected provider cluster belongs to a different project")
    config = kubernetes(request, "config", "view", "--minify", "-o", "json")
    server = config["clusters"][0]["cluster"]["server"].rstrip("/")
    endpoints = cluster.get("status", {}).get("control_plane", {}).get("endpoints", {})
    if server not in {
        value.rstrip("/") for value in endpoints.values() if isinstance(value, str)
    }:
        raise ConflictError(
            "kubeconfig server does not match the selected provider cluster"
        )
    namespace = kubernetes(request, "get", "namespace", request.namespace, "-o", "json")
    labels = namespace["metadata"].get("labels", {})
    if (
        labels.get("npa.nebius.ai/team-role") == "worker"
        or "npa.nebius.ai/team-execution" in labels
    ):
        raise ConflictError("control-plane setup cannot target an execution namespace")
    _verify_cpu_pool(request)
    _verify_prerequisites(request)


def _verify_prerequisites(request):
    for kind, name in (
        ("secret", request.configuration_secret),
        ("persistentvolumeclaim", request.state_claim),
        *(("secret", name) for name in request.image_pull_secrets),
    ):
        if resource(request, kind, name) is None:
            raise BackendError(
                "server configuration, image pull Secrets and persistent state claim must exist"
            )
    certificate = resource(request, "secret", request.tls_secret)
    if (
        not certificate
        or not {"tls.crt", "tls.key"} <= certificate.get("data", {}).keys()
    ):
        raise BackendError(
            "the operator's TLS Secret must contain a certificate and key"
        )
    verify_certificate(request, certificate)
    ssl.create_default_context(cafile=request.ca_file)


def _verify_cpu_pool(request):
    nodes = kubernetes(request, "get", "nodes", "-o", "json").get("items", [])
    selected = [
        node
        for node in nodes
        if all(
            node.get("metadata", {}).get("labels", {}).get(key) == value
            for key, value in request.node_selector.items()
        )
    ]
    if not selected or any(
        int(node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", "0"))
        for node in selected
    ):
        raise ConflictError("control-plane selector must identify a CPU-only node pool")
    if not any(
        not node.get("spec", {}).get("unschedulable")
        and any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in node.get("status", {}).get("conditions", [])
        )
        for node in selected
    ):
        raise BackendError("selected CPU node pool has no ready schedulable node")


def _validate_existing(request, receipt, deployment, service):
    if (
        request.existing_service_uid
        and service is None
        and not receipt.data.get("service_uid")
    ):
        raise ConflictError(
            "selected existing Service is absent and has no setup recovery receipt"
        )
    adopted = bool(
        service and request.existing_service_uid == service["metadata"]["uid"]
    )
    _verify_resource_ownership(request, receipt, deployment, service, adopted)
    owned_service = bool(
        service
        and service["metadata"].get("labels", {}).get(OWNER_LABEL)
        == receipt.data["installation_id"]
    )
    if request.existing_service_uid and service and not adopted and not owned_service:
        raise ConflictError("the selected existing Service UID has changed")
    if service:
        _validate_service_receipt(receipt, service, owned_service)
    if deployment:
        _verify_deployment(request, deployment)
        receipt.save(
            deployment_uid=deployment["metadata"]["uid"],
            deployment_preserved=request.existing_deployment_uid
            == deployment["metadata"]["uid"]
            or receipt.data.get("deployment_preserved", False),
        )


def _verify_resource_ownership(request, receipt, deployment, service, adopted):
    for document in (deployment, service):
        if document is None:
            continue
        saved_deployment = (
            document is deployment
            and receipt.data.get("deployment_uid") == document["metadata"]["uid"]
        )
        selected = (
            request.existing_deployment_uid == document["metadata"]["uid"]
            if document is deployment
            else adopted
        )
        if (
            not selected
            and not saved_deployment
            and document["metadata"].get("labels", {}).get(OWNER_LABEL)
            != receipt.data["installation_id"]
        ):
            raise ConflictError(
                "existing control plane needs its exact Service UID and Deployment UID selected"
            )


def _validate_service_receipt(receipt, service, owned_service):
    _verify_service(service)
    saved = receipt.data.get("service_uid")
    recovered_creation = (
        owned_service
        and receipt.data.get("phase") == "service-create-intent"
        and service["metadata"].get("annotations", {}).get(ALLOCATION_ANNOTATION)
        == receipt.data.get("allocation_id")
    )
    if saved and saved != service["metadata"]["uid"] and not recovered_creation:
        raise ConflictError("Service UID differs from the setup receipt")
    receipt.save(service_uid=service["metadata"]["uid"])


def _verify_service(service):
    spec = service["spec"]
    ports = spec.get("ports", [])
    if (
        spec.get("selector") != {"app.kubernetes.io/name": "npa-team"}
        or len(ports) != 1
        or ports[0].get("port") != 443
        or ports[0].get("targetPort") not in (8444, "https")
        or service["metadata"]
        .get("annotations", {})
        .get("nebius.com/load-balancer-type")
        == "internal"
    ):
        raise ConflictError(
            "selected Service must expose only the Workbench HTTPS gateway"
        )


def _verify_deployment(request, deployment):
    pod = deployment["spec"]["template"]["spec"]
    containers = {item["name"]: item for item in pod.get("containers", [])}
    if (
        set(containers) != {"gateway", "scheduler", "https"}
        or any(
            containers[name]["image"] != request.image
            for name in ("gateway", "scheduler")
        )
        or containers["https"]["image"] != request.proxy_image
        or pod.get("automountServiceAccountToken") is not False
        or deployment["spec"].get("replicas") != 1
    ):
        raise ConflictError(
            "existing server differs from the selected private Workbench deployment"
        )
    arguments = " ".join(containers["scheduler"].get("args", []))
    if "--host 127.0.0.1" not in arguments or "--port 46580" not in arguments:
        raise ConflictError("the native SkyPilot scheduler must remain loopback-only")
    claims = {
        volume.get("persistentVolumeClaim", {}).get("claimName")
        for volume in pod.get("volumes", [])
    }
    secrets = _secret_references(pod)
    if (
        request.state_claim not in claims
        or not {request.configuration_secret, request.tls_secret} <= secrets
    ):
        raise ConflictError(
            "existing server state or configuration differs from setup input"
        )


def _secret_references(pod):
    names = set()
    for volume in pod.get("volumes", []):
        names.add(volume.get("secret", {}).get("secretName"))
        names.update(
            source.get("secret", {}).get("name")
            for source in volume.get("projected", {}).get("sources", [])
        )
    return names


def _ensure_deployment(request, receipt, deployment):
    if deployment and receipt.data.get("deployment_preserved"):
        return
    receipt.save(phase="server-create-intent")
    documents = https_manifests(request, receipt.data["installation_id"])
    for document in sorted(documents, key=lambda item: item["kind"] == "Deployment"):
        kind, name = document["kind"], document["metadata"]["name"]
        current = resource(request, kind, name)
        if (
            current
            and current["metadata"].get("labels", {}).get(OWNER_LABEL)
            != receipt.data["installation_id"]
        ):
            raise ConflictError("refusing to overwrite unowned control-plane resources")
        if current is None:
            kubernetes(request, "create", "-f", "-", "-o", "json", document=document)
    installed = resource(request, "deployment", "npa-team")
    receipt.save(phase="server-applied", deployment_uid=installed["metadata"]["uid"])


def _wait_deployment(request, receipt):
    while True:
        deployment = resource(request, "deployment", "npa-team")
        if (
            not deployment
            or deployment["metadata"]["uid"] != receipt.data["deployment_uid"]
        ):
            raise ConflictError("Deployment identity changed during setup")
        status = deployment.get("status", {})
        if (
            deployment
            and status.get("observedGeneration", 0)
            >= deployment["metadata"].get("generation", 1)
            and status.get("availableReplicas") == 1
        ):
            return
        if any(
            condition.get("reason") == "ProgressDeadlineExceeded"
            for condition in status.get("conditions", [])
        ):
            raise BackendError("control-plane deployment failed to become ready")
        time.sleep(2)


def _ensure_service(request, receipt, service):
    if service and service["spec"].get("type") == "LoadBalancer":
        return service
    allocation_id = receipt.data.get("allocation_id", "")
    if allocation_id:
        verify_saved_allocation(request, receipt)
    desired = endpoint_service(request, receipt.data["installation_id"], allocation_id)
    if service:
        return _convert_service(request, receipt, service, desired)
    receipt.save(phase="service-create-intent")
    updated = kubernetes(request, "create", "-f", "-", "-o", "json", document=desired)
    receipt.save(service_uid=updated["metadata"]["uid"], phase="service-applied")
    return updated


def _convert_service(request, receipt, service, desired):
    patch = {
        "metadata": {
            "resourceVersion": service["metadata"]["resourceVersion"],
            "labels": desired["metadata"]["labels"],
        },
        "spec": {"type": "LoadBalancer"},
    }
    if desired["metadata"].get("annotations"):
        patch["metadata"]["annotations"] = desired["metadata"]["annotations"]
    receipt.save(phase="service-create-intent")
    updated = kubernetes(
        request,
        "patch",
        "service",
        request.service_name,
        "--namespace",
        request.namespace,
        "--type=merge",
        "--patch-file=/dev/stdin",
        "-o",
        "json",
        document=patch,
    )
    if updated["metadata"]["uid"] != service["metadata"]["uid"]:
        raise ConflictError("Service identity changed during LB conversion")
    receipt.save(phase="service-applied")
    return updated


def _wait_address(request, receipt):
    receipt.save(phase="service-awaiting-address")
    while True:
        service = resource(request, "service", request.service_name)
        if not service or service["metadata"]["uid"] != receipt.data["service_uid"]:
            raise ConflictError("Service identity changed during setup")
        _verify_service(service)
        _verify_allocation_conditions(service)
        if service.get("status", {}).get("loadBalancer", {}).get("ingress"):
            return service
        time.sleep(2)


def _verify_allocation_conditions(service):
    terminal_reasons = {
        "QuotaExceeded",
        "InvalidConfiguration",
        "PermissionDenied",
        "Unsupported",
    }
    if any(
        condition.get("status") == "False"
        and condition.get("reason") in terminal_reasons
        for condition in service.get("status", {}).get("conditions", [])
    ):
        raise BackendError(
            "provider reported a terminal LoadBalancer allocation failure"
        )


def _qualify(request, receipt):
    try:
        evidence = verify_endpoint(request, receipt.data["address"])
    except ssl.SSLError:
        receipt.save(phase="endpoint-tls-failed", external_https_verified=False)
        raise BackendError("HTTPS endpoint TLS verification failed") from None
    except OSError:
        receipt.save(
            phase="endpoint-awaiting-connectivity", external_https_verified=False
        )
    else:
        phase = "ready" if evidence["dns_points_to_lb"] else "endpoint-awaiting-dns"
        receipt.save(phase=phase, external_https_verified=True, **evidence)
    return {
        "status": receipt.data["phase"],
        "receipt": str(receipt.path),
        "personal_client_required": False,
        "external_https_verified": receipt.data["external_https_verified"],
    }
