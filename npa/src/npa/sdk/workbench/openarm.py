"""Python SDK for real Enactic OpenArm simulator workloads."""

from __future__ import annotations

import os
from typing import Any

import httpx

from npa.cli.path_contract import (
    PathContractError,
    validate_read_path,
    validate_write_path,
)
from npa.workbench.openarm.schemas import (
    DEFAULT_ISAAC_TASK,
    DEFAULT_STEPS,
    DEFAULT_TOKEN_ENV,
    OpenArmQualificationRequest,
    OpenArmQualificationResponse,
    OpenArmRunListResponse,
    OpenArmRunRequest,
    OpenArmRunResponse,
    OpenArmStatusResponse,
    OpenArmSystemInfo,
)


class OpenArmServiceError(RuntimeError):
    """A remote OpenArm service request failed."""


class OpenArmValidationError(ValueError):
    """An SDK argument violates the public contract."""


def _request(
    method: str,
    endpoint: str,
    path: str,
    *,
    token_env: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = endpoint.rstrip("/")
    if not resolved.startswith(("http://", "https://")):
        raise OpenArmValidationError("endpoint must be an http:// or https:// URL")
    token = os.environ.get(token_env, "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = httpx.request(
            method,
            resolved + path,
            json=payload,
            params=params,
            headers=headers,
            timeout=None,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise OpenArmServiceError(f"OpenArm service request failed: {exc}") from exc
    if not isinstance(data, dict):
        raise OpenArmServiceError("OpenArm service returned a non-object response")
    return data


def _run_request(
    simulator: str,
    output_path: str,
    steps: int,
    seed: int,
    render: bool,
    task: str,
    num_envs: int,
    isaac_mode: str,
    max_iterations: int,
) -> OpenArmRunRequest:
    destination = validate_write_path(
        output_path, tool="OpenArm SDK run", required=True
    )
    return OpenArmRunRequest(
        simulator=simulator,
        output_uri=destination,
        steps=steps,
        seed=seed,
        render=render,
        task=task,
        num_envs=num_envs,
        isaac_mode=isaac_mode,
        max_iterations=max_iterations,
    )


def _local_run(request: OpenArmRunRequest) -> OpenArmRunResponse:
    from npa.workbench.openarm.runtime import manifest_sha256, run as run_local

    run_local(request)
    return OpenArmRunResponse(
        run_id="local",
        status="completed",
        simulator=request.simulator,
        output_uri=request.output_uri,
        manifest_sha256=manifest_sha256(request),
    )


def _service_run(
    request: OpenArmRunRequest, endpoint: str, token_env: str
) -> OpenArmRunResponse:
    return OpenArmRunResponse.model_validate(
        _request(
            "POST",
            endpoint or os.environ.get("NPA_OPENARM_ENDPOINT", ""),
            "/run",
            token_env=token_env,
            payload=request.model_dump(mode="json"),
        )
    )


def run(
    *,
    simulator: str,
    output_path: str,
    steps: int = DEFAULT_STEPS,
    seed: int = 17,
    render: bool = False,
    task: str = DEFAULT_ISAAC_TASK,
    num_envs: int = 64,
    isaac_mode: str = "rollout",
    max_iterations: int = 1,
    mode: str = "local",
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
) -> OpenArmRunResponse:
    """Run MuJoCo locally or submit MuJoCo/Isaac Lab to a deployed service."""
    try:
        request = _run_request(
            simulator,
            output_path,
            steps,
            seed,
            render,
            task,
            num_envs,
            isaac_mode,
            max_iterations,
        )
    except (PathContractError, ValueError) as exc:
        raise OpenArmValidationError(str(exc)) from exc
    if mode == "service":
        return _service_run(request, endpoint, token_env)
    if mode != "local":
        raise OpenArmValidationError("mode must be local or service")
    return _local_run(request)


def qualify(*, input_path: str, output_path: str) -> OpenArmQualificationResponse:
    """Validate and index the complete MuJoCo and Isaac artifact tree."""
    try:
        request = OpenArmQualificationRequest(
            input_uri=validate_read_path(
                input_path,
                tool="OpenArm SDK qualification",
                allow_hf=False,
                required=True,
            ),
            output_uri=validate_write_path(
                output_path, tool="OpenArm SDK qualification", required=True
            ),
        )
    except (PathContractError, ValueError) as exc:
        raise OpenArmValidationError(str(exc)) from exc
    from npa.workbench.openarm.runtime import qualify as qualify_local

    return qualify_local(request)


def status(
    *, run_id: str, endpoint: str = "", token_env: str = DEFAULT_TOKEN_ENV
) -> OpenArmStatusResponse:
    """Read one deployed-service run state."""
    return OpenArmStatusResponse.model_validate(
        _request(
            "GET",
            endpoint or os.environ.get("NPA_OPENARM_ENDPOINT", ""),
            "/status",
            token_env=token_env,
            params={"run_id": run_id},
        )
    )


def list_runs(
    *, endpoint: str = "", token_env: str = DEFAULT_TOKEN_ENV
) -> OpenArmRunListResponse:
    """List deployed-service runs."""
    return OpenArmRunListResponse.model_validate(
        _request(
            "GET",
            endpoint or os.environ.get("NPA_OPENARM_ENDPOINT", ""),
            "/runs",
            token_env=token_env,
        )
    )


def system_info(
    *, endpoint: str = "", token_env: str = DEFAULT_TOKEN_ENV, service: bool = False
) -> OpenArmSystemInfo:
    """Return local package or deployed service identity."""
    if service:
        return OpenArmSystemInfo.model_validate(
            _request(
                "GET",
                endpoint or os.environ.get("NPA_OPENARM_ENDPOINT", ""),
                "/system-info",
                token_env=token_env,
            )
        )
    from npa.workbench.openarm.runtime import system_info as local_info

    return local_info()
