"""SDK for the RoboCasa workbench."""

from __future__ import annotations

import os
from typing import Any

import httpx

from npa.cli.path_contract import PathContractError, validate_read_path, validate_write_path
from npa.workbench.robocasa.schemas import (
    DEFAULT_ENV_ID,
    DEFAULT_ITERATIONS,
    DEFAULT_NUM_ENVS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOKEN_ENV,
    RoboCasaRunListResponse,
    RoboCasaRunRequest,
    RoboCasaRunResponse,
    RoboCasaStatusResponse,
    RoboCasaSystemInfo,
)


class RoboCasaServiceError(RuntimeError):
    """Raised when a RoboCasa service request fails."""


class RoboCasaValidationError(ValueError):
    """Raised when local SDK inputs are invalid."""


def run(
    *,
    capability: str,
    output_path: str = "",
    output_uri: str | None = None,
    env_id: str = DEFAULT_ENV_ID,
    iterations: int = DEFAULT_ITERATIONS,
    num_envs: int = DEFAULT_NUM_ENVS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    download_assets: bool = True,
    seed: int | None = None,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 30.0,
    checkpoint_uri: str = "",
    train_env_ids: str = "",
    heldout_env_ids: str = "",
) -> RoboCasaRunResponse:
    """Run a RoboCasa capability.

    ``output_uri`` remains as a compatibility alias for callers predating the
    canonical cross-tool ``output_path`` spelling.
    """
    if output_path and output_uri and output_path != output_uri:
        raise RoboCasaValidationError(
            "output_path and compatibility output_uri must identify the same S3 path"
        )
    try:
        resolved_output = validate_write_path(
            output_path or output_uri or "",
            tool="RoboCasa SDK run",
            option="output_path",
            required=True,
        )
        if capability == "kitchen_policy_eval":
            checkpoint_uri = validate_read_path(
                checkpoint_uri,
                tool="RoboCasa SDK run",
                option="checkpoint_uri",
                allow_hf=False,
            )
    except PathContractError as exc:
        raise RoboCasaValidationError(str(exc)) from exc
    request = RoboCasaRunRequest(
        env_id=env_id,
        capability=capability,
        output_uri=resolved_output,
        iterations=iterations,
        num_envs=num_envs,
        timeout_seconds=timeout_seconds,
        download_assets=download_assets,
        seed=seed,
        checkpoint_uri=checkpoint_uri,
        train_env_ids=train_env_ids,
        heldout_env_ids=heldout_env_ids,
    )
    if _resolve_mode(mode=mode, service=service):
        return RoboCasaRunResponse.model_validate(
            _request_json(
                "POST",
                endpoint or os.environ.get("NPA_ROBOCASA_ENDPOINT", ""),
                "/run",
                payload=request.model_dump(mode="json"),
                token_env=token_env,
                timeout=timeout,
            )
        )
    from npa.workbench.robocasa.capabilities import run_capability_with_output

    # Local execution persists and uploads output exactly like a service run:
    # a temporary output directory is created, the capability writes its
    # artifacts there, and any produced output is uploaded to ``output_uri``.
    # This keeps local runs truthful (artifacts are produced or uploaded) and
    # makes capabilities that require an output directory (policy evaluation)
    # work in local mode.
    run_capability_with_output(request)
    return RoboCasaRunResponse(
        run_id="local",
        status="completed",
        env_id=request.env_id,
        capability=request.capability,
        output_uri=request.output_uri,
        manifest_sha256="local",
    )


def status(
    *,
    run_id: str,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 30.0,
) -> RoboCasaStatusResponse:
    """Return status for a RoboCasa run."""
    if _resolve_mode(mode=mode, service=service):
        return RoboCasaStatusResponse.model_validate(
            _request_json(
                "GET",
                endpoint or os.environ.get("NPA_ROBOCASA_ENDPOINT", ""),
                "/status",
                params={"run_id": run_id},
                token_env=token_env,
                timeout=timeout,
            )
        )
    from npa.workbench.robocasa.service import status_for_run

    return status_for_run(run_id)


def system_info(
    *,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 30.0,
) -> RoboCasaSystemInfo:
    """Return RoboCasa system information."""
    if _resolve_mode(mode=mode, service=service):
        return RoboCasaSystemInfo.model_validate(
            _request_json(
                "GET",
                endpoint or os.environ.get("NPA_ROBOCASA_ENDPOINT", ""),
                "/system-info",
                token_env=token_env,
                timeout=timeout,
            )
        )
    from npa.workbench.robocasa.capabilities import system_info as _system_info

    return _system_info()


def list_runs(
    *,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 30.0,
) -> RoboCasaRunListResponse:
    """List RoboCasa runs."""
    if _resolve_mode(mode=mode, service=service):
        return RoboCasaRunListResponse.model_validate(
            _request_json(
                "GET",
                endpoint or os.environ.get("NPA_ROBOCASA_ENDPOINT", ""),
                "/runs",
                token_env=token_env,
                timeout=timeout,
            )
        )
    from npa.workbench.robocasa.service import RUNS

    return RoboCasaRunListResponse(runs=list(RUNS.values()))


def _resolve_mode(*, mode: str | None, service: bool) -> bool:
    if mode is None:
        return service
    value = mode.strip().lower()
    if value == "local":
        return False
    if value == "service":
        return True
    raise RoboCasaValidationError("mode must be either 'local' or 'service'")


def _request_json(
    method: str,
    endpoint: str,
    path: str,
    *,
    token_env: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = endpoint.strip().rstrip("/")
    if not resolved:
        raise RoboCasaValidationError("endpoint is required for service mode")
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(
            method,
            f"{resolved}{path}",
            headers=headers,
            json=payload,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise RoboCasaServiceError(
            f"RoboCasa service request failed ({exc.response.status_code}): {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RoboCasaServiceError(f"Cannot reach RoboCasa service {resolved}: {exc}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise RoboCasaServiceError("RoboCasa service returned non-JSON response") from exc
    if not isinstance(data, dict):
        raise RoboCasaServiceError("RoboCasa service returned an unexpected response")
    return data


__all__ = [
    "RoboCasaRunListResponse",
    "RoboCasaRunRequest",
    "RoboCasaRunResponse",
    "RoboCasaServiceError",
    "RoboCasaStatusResponse",
    "RoboCasaSystemInfo",
    "RoboCasaValidationError",
    "list_runs",
    "run",
    "status",
    "system_info",
]
