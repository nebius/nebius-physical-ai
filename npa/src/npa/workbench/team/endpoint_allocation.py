"""Retain only the allocation belonging to the explicitly selected HTTPS Service."""

from .errors import BackendError, ConflictError
from .setup_transport import kubernetes, provider

ALLOCATION_ANNOTATION = "nebius.com/load-balancer-allocation-id"


def verify_saved_allocation(request, receipt):
    """Verify retained identity before recreating a Service with its saved address.

    Args:
        request, receipt: Exact installation and its private retained allocation.
    Returns:
        None.
    Raises:
        BackendError, ConflictError: The allocation is absent, changed or managed.
    """
    allocation = provider(
        "vpc", "allocation", "get", "--id", receipt.data["allocation_id"]
    )
    _verify_allocation(request, allocation, receipt.data["address"])
    if allocation["metadata"].get("labels", {}).get("nebius.com/managed-by"):
        raise ConflictError("saved allocation is not retained for Service recreation")


def retain_allocation(request, receipt, service):
    """Preserve the verified LB address with intent recorded before detachment.

    Args:
        request, receipt: Exact operator selection and private retry journal.
        service: Observed ready LoadBalancer Service.
    Returns:
        Verified retained provider allocation identity.
    Raises:
        BackendError, ConflictError: Provider or ownership evidence is invalid.
    """
    address = _service_address(service)
    allocation = _select_allocation(request, receipt, service, address)
    metadata = allocation["metadata"]
    _verify_allocation(request, allocation, address)
    labels = metadata.get("labels", {})
    if labels.get("nebius.com/managed-by"):
        _verify_management(request, service, labels)
    receipt.save(
        phase="allocation-retention-intent",
        allocation_id=metadata["id"],
        address=address,
    )
    if labels.get("nebius.com/managed-by"):
        _detach(allocation)
    retained = provider("vpc", "allocation", "get", "--id", metadata["id"])
    _verify_allocation(request, retained, address)
    if retained["metadata"].get("labels", {}).get("nebius.com/managed-by"):
        raise BackendError("LB allocation remains provider-managed")
    _link(request, service, metadata["id"])
    receipt.save(phase="allocation-retained")
    return metadata["id"]


def _service_address(service):
    addresses = service.get("status", {}).get("loadBalancer", {}).get("ingress", [])
    if len(addresses) != 1 or not addresses[0].get("ip"):
        raise BackendError("one allocated LB address is required")
    return addresses[0]["ip"]


def _select_allocation(request, receipt, service, address):
    linked = service["metadata"].get("annotations", {}).get(ALLOCATION_ANNOTATION)
    saved = receipt.data.get("allocation_id")
    if saved and linked and saved != linked:
        raise ConflictError("Service allocation differs from the setup receipt")
    if saved or linked:
        return provider("vpc", "allocation", "get", "--id", saved or linked)
    matches = []
    token = ""
    visited = set()
    while True:
        arguments = ["vpc", "allocation", "list", "--parent-id", request.project_id]
        if token:
            arguments += ["--page-token", token]
        response = provider(*arguments)
        matches += [
            item
            for item in response.get("items", [])
            if item.get("status", {}).get("details", {}).get("allocated_cidr")
            == address + "/32"
        ]
        token = response.get("next_page_token", "")
        if not token:
            break
        if token in visited:
            raise BackendError("provider allocation pagination repeated a page")
        visited.add(token)
    if len(matches) != 1:
        raise ConflictError("cannot identify exactly one selected Service allocation")
    _verify_management(request, service, matches[0]["metadata"].get("labels", {}))
    return matches[0]


def _verify_allocation(request, allocation, address):
    if (
        allocation.get("metadata", {}).get("parent_id") != request.project_id
        or allocation.get("status", {}).get("details", {}).get("allocated_cidr")
        != address + "/32"
    ):
        raise ConflictError(
            "allocation project or address does not match the selected Service"
        )


def _verify_management(request, service, labels):
    expected = {
        "nebius.com/cluster-id": request.cluster_id,
        "nebius.com/service-name": service["metadata"]["name"],
        "nebius.com/service-namespace": request.namespace,
        "nebius.com/service-uid": service["metadata"]["uid"],
        "nebius.com/managed-by": "mk8s",
    }
    if any(labels.get(key) != value for key, value in expected.items()):
        raise ConflictError(
            "allocation does not belong to the exact selected cluster and Service"
        )


def _detach(allocation):
    metadata = allocation["metadata"]
    arguments = [
        "vpc",
        "allocation",
        "update",
        "--id",
        metadata["id"],
        "--resource-version",
        str(metadata["resource_version"]),
    ]
    for key in (
        "managed-by",
        "cluster-id",
        "service-name",
        "service-namespace",
        "service-uid",
    ):
        arguments += ["--labels-remove", "nebius.com/" + key]
    provider(*arguments)


def _link(request, service, allocation_id):
    if (
        service["metadata"].get("annotations", {}).get(ALLOCATION_ANNOTATION)
        == allocation_id
    ):
        return
    patch = {
        "metadata": {
            "resourceVersion": service["metadata"]["resourceVersion"],
            "annotations": {ALLOCATION_ANNOTATION: allocation_id},
        }
    }
    updated = kubernetes(
        request,
        "patch",
        "service",
        service["metadata"]["name"],
        "--namespace",
        request.namespace,
        "--type=merge",
        "--patch-file=/dev/stdin",
        "-o",
        "json",
        document=patch,
    )
    if updated["metadata"]["uid"] != service["metadata"]["uid"] or _service_address(
        updated
    ) != _service_address(service):
        raise ConflictError(
            "Service identity or address changed during allocation retention"
        )
