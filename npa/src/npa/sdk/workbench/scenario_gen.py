"""Compatibility SDK for the adversarial scenario-generation workbench."""

from __future__ import annotations

import os
from typing import Any

import httpx

from npa.workbench.scenario_gen.schemas import (
    DEFAULT_ADVERSARY_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIVERSITY_WEIGHT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_SCENARIOS,
    DEFAULT_SEED,
    DEFAULT_SEVERITY_WEIGHT,
    DEFAULT_TASK,
    DEFAULT_TOKEN_ENV,
    DEFAULT_TOP_K,
    GenerateRequest,
    GenerateResponse,
    RankRequest,
    RankResponse,
    StatusResponse,
)

ENDPOINT_ENV = "NPA_SCENARIO_GEN_ENDPOINT"


class ScenarioGenServiceError(RuntimeError):
    """Raised when a scenario-generation service request fails."""


class ScenarioGenValidationError(ValueError):
    """Raised when local SDK inputs are invalid."""


def generate(
    *,
    policy_uri: str,
    base_config_uri: str,
    output_uri: str,
    task_name: str = DEFAULT_TASK,
    num_scenarios: int = DEFAULT_NUM_SCENARIOS,
    adversary_steps: int = DEFAULT_ADVERSARY_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    workflow_run: str = "",
    visualize: bool = True,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 120.0,
) -> GenerateResponse:
    """Train an adversary and mine ranked adversarial scenarios."""
    request = GenerateRequest(
        policy_uri=policy_uri,
        base_config_uri=base_config_uri,
        output_uri=output_uri,
        task_name=task_name,
        num_scenarios=num_scenarios,
        adversary_steps=adversary_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        workflow_run=workflow_run,
        visualize=visualize,
    )
    if _resolve_mode(mode=mode, service=service):
        return GenerateResponse.model_validate(
            _request_json(
                "POST",
                endpoint or os.environ.get(ENDPOINT_ENV, ""),
                "/generate",
                payload=request.model_dump(mode="json"),
                token_env=token_env,
                timeout=timeout,
            )
        )
    from npa.workbench.scenario_gen.generation import generate_scenarios

    return generate_scenarios(request)


def rank(
    *,
    input_uri: str,
    output_uri: str,
    top_k: int = DEFAULT_TOP_K,
    severity_weight: float = DEFAULT_SEVERITY_WEIGHT,
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
    workflow_run: str = "",
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 60.0,
) -> RankResponse:
    """Rank a generated adversarial scenario set."""
    request = RankRequest(
        input_uri=input_uri,
        output_uri=output_uri,
        top_k=top_k,
        severity_weight=severity_weight,
        diversity_weight=diversity_weight,
        workflow_run=workflow_run,
    )
    if _resolve_mode(mode=mode, service=service):
        return RankResponse.model_validate(
            _request_json(
                "POST",
                endpoint or os.environ.get(ENDPOINT_ENV, ""),
                "/rank",
                payload=request.model_dump(mode="json"),
                token_env=token_env,
                timeout=timeout,
            )
        )
    from npa.workbench.scenario_gen.ranking import rank_scenarios

    return rank_scenarios(request)


def status(
    *,
    run_id: str,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 30.0,
) -> StatusResponse:
    """Return status for a scenario-generation run."""
    if _resolve_mode(mode=mode, service=service):
        return StatusResponse.model_validate(
            _request_json(
                "GET",
                endpoint or os.environ.get(ENDPOINT_ENV, ""),
                "/status",
                params={"run_id": run_id},
                token_env=token_env,
                timeout=timeout,
            )
        )
    from npa.workbench.scenario_gen.service import status_for_run

    return status_for_run(run_id)


def _resolve_mode(*, mode: str | None, service: bool) -> bool:
    if mode is None:
        return service
    value = mode.strip().lower()
    if value == "local":
        return False
    if value == "service":
        return True
    raise ScenarioGenValidationError("mode must be either 'local' or 'service'")


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
        raise ScenarioGenValidationError("endpoint is required for service mode")
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(method, f"{resolved}{path}", headers=headers, json=payload, params=params, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise ScenarioGenServiceError(
            f"Scenario-gen service request failed ({exc.response.status_code}): {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ScenarioGenServiceError(f"Cannot reach scenario-gen service {resolved}: {exc}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise ScenarioGenServiceError("Scenario-gen service returned non-JSON response") from exc
    if not isinstance(data, dict):
        raise ScenarioGenServiceError("Scenario-gen service returned an unexpected response")
    return data


__all__ = [
    "ScenarioGenServiceError",
    "ScenarioGenValidationError",
    "GenerateResponse",
    "RankResponse",
    "StatusResponse",
    "generate",
    "rank",
    "status",
]
