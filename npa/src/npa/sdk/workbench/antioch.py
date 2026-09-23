"""Python SDK for local or deployed Antioch Workbench operations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from npa.workbench.antioch.manager import AntiochManager
from npa.workbench.antioch.manager import AntiochOperationError
from npa.workbench.antioch.live import start_live, status_live, stop_live
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
    if response.is_error:
        try:
            detail = response.json().get("detail", {})
        except (ValueError, AttributeError):
            detail = {}
        if isinstance(detail, dict):
            raise AntiochOperationError(
                str(detail.get("message") or "Antioch Workbench request failed"),
                retryable=bool(detail.get("retryable")),
                error_type=str(detail.get("type") or "service_error"),
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


def live_start(
    *,
    source: Path,
    project_id: str,
    client_bundle: Path,
    scenario_timeout_seconds: int = 14_400,
) -> dict[str, Any]:
    return start_live(
        source=source,
        project_id=project_id,
        client_bundle=client_bundle,
        scenario_timeout_seconds=scenario_timeout_seconds,
    )


def live_status(*, project_id: str) -> dict[str, Any]:
    return status_live(project_id=project_id)


def live_stop(*, project_id: str, timeout_seconds: float = 120.0) -> dict[str, Any]:
    return stop_live(project_id=project_id, timeout_seconds=timeout_seconds)


def live_k8s_deploy(*, runtime_config: Path) -> dict[str, Any]:
    from npa.workbench.antioch.cluster_deploy import apply_cluster, load_private_config

    return apply_cluster(load_private_config(runtime_config))


def live_k8s_status(*, runtime_config: Path) -> dict[str, Any]:
    from npa.workbench.antioch.cluster_deploy import cluster_status, load_private_config

    return cluster_status(load_private_config(runtime_config))


def live_k8s_stop(
    *, runtime_config: Path, timeout_seconds: float = 360.0
) -> dict[str, Any]:
    from npa.workbench.antioch.cluster_deploy import load_private_config, stop_cluster

    return stop_cluster(
        load_private_config(runtime_config), timeout_seconds=timeout_seconds
    )


def live_k8s_finalize_cutover(*, runtime_config: Path) -> dict[str, Any]:
    from npa.workbench.antioch.cluster_deploy import (
        disable_public_rollback_service,
        load_private_config,
    )

    return disable_public_rollback_service(load_private_config(runtime_config))
