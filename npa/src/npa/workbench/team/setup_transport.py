"""Use explicitly selected operator transports without exposing provider diagnostics."""

import json
import subprocess

from npa.clients import nebius

from .errors import BackendError


def kubernetes(request, *arguments, document=None):
    """Execute Kubernetes against the setup request's exact context.

    Args:
        request: Validated operator selection.
        arguments: Kubernetes arguments without credentials.
        document: Optional JSON resource sent on standard input.
    Returns:
        Parsed JSON response, or None for an absent resource.
    Raises:
        BackendError: Transport, authorization, or response parsing fails.
    """
    command = [
        "kubectl",
        "--kubeconfig",
        str(request.kubeconfig),
        "--context",
        request.context,
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            input=json.dumps(document) if document else None,
        )
        if result.returncode:
            raise BackendError("control-plane Kubernetes operation failed")
        return json.loads(result.stdout) if result.stdout.strip() else None
    except (OSError, ValueError):
        raise BackendError("control-plane Kubernetes transport failed") from None


def provider(*arguments):
    """Call the selected Nebius profile with credential-free public errors.

    Args:
        arguments: Exact resource operation; never personal user credentials.
    Returns:
        Parsed provider response.
    Raises:
        BackendError: Provider authorization or response parsing fails.
    """
    try:
        return nebius._run_json(list(arguments))
    except (nebius.NebiusError, ValueError):
        raise BackendError("control-plane provider operation failed") from None


def resource(request, kind, name):
    """Read one named resource without confusing absence and permission failure.

    Args:
        request: Explicit operator selection.
        kind, name: Namespaced resource identity.
    Returns:
        Resource document, or None after verified absence.
    Raises:
        BackendError: Kubernetes cannot verify the selected resource.
    """
    return kubernetes(
        request,
        "get",
        kind,
        name,
        "--namespace",
        request.namespace,
        "--ignore-not-found",
        "-o",
        "json",
    )
