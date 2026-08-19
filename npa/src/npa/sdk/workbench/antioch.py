"""Python SDK for local or deployed Antioch Workbench operations."""

from __future__ import annotations

import os
from typing import Any

import httpx

from npa.workbench.antioch.manager import AntiochManager
from npa.workbench.antioch.openpi_bridge import render_stack
from npa.workbench.antioch.schemas import (
    CollectRequest,
    OperationRecord,
    ResumeRequest,
    SubmitRequest,
)


def _call(
    path: str, body: dict[str, Any], *, endpoint: str = "", token: str = ""
) -> OperationRecord:
    if not endpoint:
        manager = AntiochManager()
        request = (
            SubmitRequest.model_validate(body)
            if path in {"submit", "run"}
            else (
                CollectRequest.model_validate(body)
                if path == "collect"
                else ResumeRequest.model_validate(body)
            )
        )
        return getattr(manager, path if path != "status" else "reconcile")(request)
    resolved_token = token or os.environ.get("ANTIOCH_WORKBENCH_TOKEN", "")
    headers = {"Authorization": f"Bearer {resolved_token}"} if resolved_token else {}
    response = httpx.post(
        f"{endpoint.rstrip('/')}/{path}",
        json=body,
        headers=headers,
        timeout=None,
    )
    response.raise_for_status()
    return OperationRecord.model_validate(response.json())


def submit(
    request: SubmitRequest, *, endpoint: str = "", token: str = ""
) -> OperationRecord:
    return _call(
        "submit", request.model_dump(mode="json"), endpoint=endpoint, token=token
    )


def run(
    request: SubmitRequest, *, endpoint: str = "", token: str = ""
) -> OperationRecord:
    return _call("run", request.model_dump(mode="json"), endpoint=endpoint, token=token)


def status(
    request: ResumeRequest, *, endpoint: str = "", token: str = ""
) -> OperationRecord:
    return _call(
        "status", request.model_dump(mode="json"), endpoint=endpoint, token=token
    )


def resume(
    request: ResumeRequest, *, endpoint: str = "", token: str = ""
) -> OperationRecord:
    return _call(
        "resume", request.model_dump(mode="json"), endpoint=endpoint, token=token
    )


def reconcile(
    request: ResumeRequest, *, endpoint: str = "", token: str = ""
) -> OperationRecord:
    return _call(
        "reconcile", request.model_dump(mode="json"), endpoint=endpoint, token=token
    )


def cancel(
    request: ResumeRequest, *, endpoint: str = "", token: str = ""
) -> OperationRecord:
    return _call(
        "cancel", request.model_dump(mode="json"), endpoint=endpoint, token=token
    )


def collect(
    request: CollectRequest, *, endpoint: str = "", token: str = ""
) -> OperationRecord:
    return _call(
        "collect", request.model_dump(mode="json"), endpoint=endpoint, token=token
    )


def render_openpi_stack(**kwargs: Any) -> dict[str, object]:
    """Render the digest-pinned RTX bridge + B200 policy Kubernetes stack."""

    return render_stack(**kwargs)
