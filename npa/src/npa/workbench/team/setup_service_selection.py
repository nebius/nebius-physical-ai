"""Reuse the selected gateway's public Service instead of creating a parallel LB."""

from .errors import ConflictError
from .setup_transport import kubernetes


def select_service(request, receipt):
    """Resolve one existing public gateway or preserve the recorded Service name.

    Args:
        request: Exact operator cluster and installation selection.
        receipt: Private stable installation journal.
    Returns:
        Request with its immutable Service name resolved.
    Raises:
        ConflictError: More than one public Service matches and none was selected.
    """
    name = receipt.data.get("service_name")
    if name is None:
        name = _existing_name(request)
        receipt.save(service_name=name)
    return request.model_copy(update={"service_name": name})


def _existing_name(request):
    if "service_name" in request.model_fields_set:
        return request.service_name
    response = kubernetes(
        request, "get", "services", "--namespace", request.namespace, "-o", "json"
    )
    candidates = [item for item in response.get("items", []) if _public_gateway(item)]
    if len(candidates) > 1:
        raise ConflictError(
            "select the exact public Workbench Service; multiple endpoints exist"
        )
    if candidates:
        if (
            request.existing_service_uid
            and candidates[0]["metadata"]["uid"] != request.existing_service_uid
        ):
            raise ConflictError(
                "selected existing UID is not the gateway's public Service UID"
            )
        return candidates[0]["metadata"]["name"]
    return request.service_name


def _public_gateway(service):
    spec = service.get("spec", {})
    ports = spec.get("ports", [])
    return (
        spec.get("type") == "LoadBalancer"
        and spec.get("selector") == {"app.kubernetes.io/name": "npa-team"}
        and len(ports) == 1
        and ports[0].get("port") == 443
        and ports[0].get("targetPort") in (8444, "https")
    )
