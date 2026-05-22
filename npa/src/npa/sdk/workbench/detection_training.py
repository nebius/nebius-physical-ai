"""Compatibility SDK for the detection-training workbench."""

from __future__ import annotations

import os
from typing import Any

import httpx

from npa.solutions.workbench.detection_training.schemas import (
    DEFAULT_LANCE_URI,
    DEFAULT_TOKEN_ENV,
    EvalRequest,
    EvalResponse,
    StatusResponse,
    TrainRequest,
    TrainResponse,
)


class DetectionTrainingServiceError(RuntimeError):
    """Raised when a detection-training service request fails."""


class DetectionTrainingValidationError(ValueError):
    """Raised when local SDK inputs are invalid."""


def train(
    *,
    view: str,
    output_uri: str,
    lance_uri: str = DEFAULT_LANCE_URI,
    num_classes: int = 10,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.005,
    validation_filter_sql: str | None = None,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 30.0,
) -> TrainResponse:
    """Start a Faster R-CNN detection-training run."""
    request = TrainRequest(
        view=view,
        lance_uri=lance_uri,
        output_uri=output_uri,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_filter_sql=validation_filter_sql,
    )
    if _resolve_mode(mode=mode, service=service):
        return TrainResponse.model_validate(
            _request_json(
                "POST",
                endpoint or os.environ.get("NPA_DETECTION_TRAINING_ENDPOINT", ""),
                "/train",
                payload=request.model_dump(mode="json"),
                token_env=token_env,
                timeout=timeout,
            )
        )
    from npa.solutions.workbench.detection_training.training import train_detector

    return train_detector(request)


def eval(
    *,
    checkpoint_uri: str,
    eval_view: str,
    output_uri: str,
    lance_uri: str = DEFAULT_LANCE_URI,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 600.0,
) -> EvalResponse:
    """Evaluate a detection-training checkpoint."""
    request = EvalRequest(
        checkpoint_uri=checkpoint_uri,
        eval_view=eval_view,
        lance_uri=lance_uri,
        output_uri=output_uri,
    )
    if _resolve_mode(mode=mode, service=service):
        return EvalResponse.model_validate(
            _request_json(
                "POST",
                endpoint or os.environ.get("NPA_DETECTION_TRAINING_ENDPOINT", ""),
                "/eval",
                payload=request.model_dump(mode="json"),
                token_env=token_env,
                timeout=timeout,
            )
        )
    from npa.solutions.workbench.detection_training.evaluation import evaluate_detector

    return evaluate_detector(request)


def status(
    *,
    run_id: str,
    mode: str | None = None,
    service: bool = False,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 30.0,
) -> StatusResponse:
    """Return status for a detection-training run."""
    if _resolve_mode(mode=mode, service=service):
        return StatusResponse.model_validate(
            _request_json(
                "GET",
                endpoint or os.environ.get("NPA_DETECTION_TRAINING_ENDPOINT", ""),
                "/status",
                params={"run_id": run_id},
                token_env=token_env,
                timeout=timeout,
            )
        )
    from npa.solutions.workbench.detection_training.service import status_for_run

    return status_for_run(run_id)


def _resolve_mode(*, mode: str | None, service: bool) -> bool:
    if mode is None:
        return service
    value = mode.strip().lower()
    if value == "local":
        return False
    if value == "service":
        return True
    raise DetectionTrainingValidationError("mode must be either 'local' or 'service'")


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
        raise DetectionTrainingValidationError("endpoint is required for service mode")
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
        raise DetectionTrainingServiceError(
            f"Detection-training service request failed ({exc.response.status_code}): {detail}"
        ) from exc
    except httpx.HTTPError as exc:
        raise DetectionTrainingServiceError(
            f"Cannot reach detection-training service {resolved}: {exc}"
        ) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise DetectionTrainingServiceError(
            "Detection-training service returned non-JSON response"
        ) from exc
    if not isinstance(data, dict):
        raise DetectionTrainingServiceError(
            "Detection-training service returned an unexpected response"
        )
    return data


__all__ = [
    "DetectionTrainingServiceError",
    "DetectionTrainingValidationError",
    "EvalRequest",
    "EvalResponse",
    "StatusResponse",
    "TrainRequest",
    "TrainResponse",
    "eval",
    "status",
    "train",
]
