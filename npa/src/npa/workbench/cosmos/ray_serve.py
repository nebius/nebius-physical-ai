"""Client and durable artifact contract for native Cosmos 3 Ray Serve.

The GPU service itself lives in :mod:`npa.workbench.cosmos.ray_server` and binds
NVIDIA cosmos-framework's ``OmniModelDeployment``.  This module deliberately
stays import-light so the CLI and SDK can submit batches from an ordinary NPA
workflow pod without importing Ray, Torch, or the Cosmos framework.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from npa.clients.storage import StorageClient

RAY_BATCH_SCHEMA = "npa.cosmos3.ray-serve.batch.v1"
RAY_PROVENANCE_SCHEMA = "npa.cosmos3.ray-serve.provenance.v1"
DEFAULT_ENDPOINT_ENV = "NPA_COSMOS3_RAY_ENDPOINT"
DEFAULT_TOKEN_ENV = "NPA_COSMOS3_RAY_TOKEN"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Cosmos3RayServeError(RuntimeError):
    """Raised when the native Ray Serve contract cannot complete safely."""


class RayBatchRequest(BaseModel):
    """One client batch passed to the native Cosmos router."""

    samples: list[dict[str, Any]] = Field(min_length=1)
    model: str = "Cosmos3-Nano"
    request_id: str = Field(
        default="", pattern=r"^(?:[A-Za-z0-9][A-Za-z0-9._-]{0,127})?$"
    )

    @field_validator("samples")
    @classmethod
    def require_named_samples(
        cls, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        names: set[str] = set()
        for index, sample in enumerate(samples):
            name = str(sample.get("name", "")).strip()
            if not name:
                raise ValueError(f"samples[{index}].name is required")
            if not SAFE_NAME.fullmatch(name):
                raise ValueError(
                    f"samples[{index}].name must use only letters, numbers, '.', '_', or '-'"
                )
            if name in names:
                raise ValueError(f"duplicate sample name: {name}")
            names.add(name)
        return samples


class RayArtifact(BaseModel):
    sample: str
    path: str
    bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def require_relative_safe_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or not value or ".." in path.parts:
            raise ValueError(
                "artifact path must be non-empty, relative, and traversal-free"
            )
        return value


class RayBatchResponse(BaseModel):
    schema_version: str = RAY_BATCH_SCHEMA
    request_id: str
    model: str
    batch_size: int = Field(ge=1)
    outputs: list[dict[str, Any]]
    artifacts: list[RayArtifact]
    guardrails: bool
    max_batch_size: int = Field(ge=1)
    framework_revision: str
    server_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")


def load_batch_request(
    input_path: str, *, storage_client: Any = None
) -> RayBatchRequest:
    """Load a batch JSON document from a local path or exact ``s3://`` object."""

    value = str(input_path or "").strip()
    if not value:
        raise Cosmos3RayServeError("input_path is required")
    if value.startswith("s3://"):
        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-cosmos3-ray-input-") as tmp:
            local = Path(tmp) / "batch.json"
            client.download_file(value, str(local))
            payload = json.loads(local.read_text(encoding="utf-8"))
    else:
        path = Path(value)
        if not path.is_file():
            raise Cosmos3RayServeError(f"batch input does not exist: {value}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"samples": payload}
    try:
        return RayBatchRequest.model_validate(payload)
    except ValueError as exc:
        raise Cosmos3RayServeError(f"invalid Cosmos3 Ray batch: {exc}") from exc


def service_health(
    *, endpoint: str = "", token_env: str = DEFAULT_TOKEN_ENV, timeout: float = 30.0
) -> dict[str, Any]:
    """Return the authenticated readiness record for a native Ray service."""

    return _request_json(
        "GET",
        endpoint or os.environ.get(DEFAULT_ENDPOINT_ENV, ""),
        "/ready",
        token_env=token_env,
        timeout=timeout,
    )


def submit_batch(
    *,
    input_path: str,
    output_path: str,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 1800.0,
    run_id: str = "",
    dry_run: bool = False,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Submit one durable SDG batch and publish outputs plus provenance.

    The client issues one ``/v1/batches`` request.  The service fans its samples
    into concurrent deployment-handle calls, which are coalesced by NVIDIA's
    ``@ray.serve.batch`` implementation before ``OmniInference.generate_batch``.
    """

    request = load_batch_request(input_path, storage_client=storage_client)
    resolved_endpoint = (
        (endpoint or os.environ.get(DEFAULT_ENDPOINT_ENV, "")).strip().rstrip("/")
    )
    if not resolved_endpoint:
        raise Cosmos3RayServeError(
            f"endpoint is required (pass --endpoint or set {DEFAULT_ENDPOINT_ENV})"
        )
    if not output_path.startswith("s3://") and not output_path.startswith("/"):
        raise Cosmos3RayServeError(
            "output_path must be an s3:// URI or absolute local path"
        )

    plan = {
        "schema_version": RAY_PROVENANCE_SCHEMA,
        "status": "planned" if dry_run else "submitting",
        "run_id": run_id,
        "endpoint": resolved_endpoint,
        "input_path": input_path,
        "output_path": output_path,
        "model": request.model,
        "sample_names": [str(sample["name"]) for sample in request.samples],
        "batch_size": len(request.samples),
        "backend": "cosmos-framework-native-ray-serve",
        "weights_baked": False,
    }
    if dry_run:
        return plan

    response_payload = _request_json(
        "POST",
        resolved_endpoint,
        "/v1/batches",
        token_env=token_env,
        timeout=timeout,
        payload=request.model_dump(mode="json"),
    )
    response = RayBatchResponse.model_validate(response_payload)
    if response.batch_size != len(request.samples):
        raise Cosmos3RayServeError(
            f"service returned batch_size={response.batch_size}, expected {len(request.samples)}"
        )
    if len(response.outputs) != len(request.samples):
        raise Cosmos3RayServeError(
            f"service returned {len(response.outputs)} structured outputs, "
            f"expected {len(request.samples)}"
        )

    with tempfile.TemporaryDirectory(prefix="npa-cosmos3-ray-output-") as tmp:
        root = Path(tmp)
        (root / "request.json").write_text(
            request.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (root / "response.json").write_text(
            response.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        for artifact in response.artifacts:
            target = root / "artifacts" / artifact.path
            target.parent.mkdir(parents=True, exist_ok=True)
            body = _request_bytes(
                resolved_endpoint,
                f"/v1/artifacts/{artifact.path}",
                token_env=token_env,
                timeout=timeout,
            )
            digest = hashlib.sha256(body).hexdigest()
            if len(body) != artifact.bytes or digest != artifact.sha256:
                raise Cosmos3RayServeError(
                    f"artifact integrity mismatch for {artifact.path}: "
                    f"expected {artifact.bytes}/{artifact.sha256}, got {len(body)}/{digest}"
                )
            target.write_bytes(body)

        manifest = {
            **plan,
            "status": "completed",
            "request_id": response.request_id,
            "guardrails": response.guardrails,
            "max_batch_size": response.max_batch_size,
            "framework_revision": response.framework_revision,
            "server_source_revision": response.server_source_revision,
            "artifacts": [
                artifact.model_dump(mode="json") for artifact in response.artifacts
            ],
            "structured_outputs": response.outputs,
            "runtime_image": os.environ.get("NPA_TASK_IMAGE", ""),
        }
        if output_path.startswith("s3://"):
            manifest["published_uri"] = output_path.rstrip("/") + "/"
            manifest["provenance_uri"] = manifest["published_uri"] + "provenance.json"
        else:
            destination = Path(output_path)
            manifest["published_uri"] = str(destination)
            manifest["provenance_uri"] = str(destination / "provenance.json")
        (root / "provenance.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_path.startswith("s3://"):
            client = storage_client or StorageClient.from_environment()
            published = client.upload_directory(str(root), output_path)
            if published != manifest["published_uri"]:
                raise Cosmos3RayServeError(
                    f"storage published an unexpected destination: {published}"
                )
        else:
            destination.mkdir(parents=True, exist_ok=True)
            for path in root.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(root)
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
    return manifest


def _headers(token_env: str) -> dict[str, str]:
    token = os.environ.get(token_env, "")
    if not token:
        raise Cosmos3RayServeError(f"{token_env} is required for the Ray Serve API")
    return {"Authorization": f"Bearer {token}"}


def _request_json(
    method: str,
    endpoint: str,
    path: str,
    *,
    token_env: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = str(endpoint or "").strip().rstrip("/")
    if not resolved:
        raise Cosmos3RayServeError("service endpoint is required")
    try:
        response = httpx.request(
            method,
            f"{resolved}{path}",
            headers=_headers(token_env),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise Cosmos3RayServeError(
            f"Cosmos3 Ray Serve request failed ({exc.response.status_code}): "
            f"{exc.response.text[:1000]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise Cosmos3RayServeError(
            f"cannot reach Cosmos3 Ray Serve at {resolved}: {exc}"
        ) from exc
    try:
        value = response.json()
    except ValueError as exc:
        raise Cosmos3RayServeError(
            "Cosmos3 Ray Serve returned non-JSON output"
        ) from exc
    if not isinstance(value, dict):
        raise Cosmos3RayServeError(
            "Cosmos3 Ray Serve returned an unexpected JSON value"
        )
    return value


def _request_bytes(
    endpoint: str, path: str, *, token_env: str, timeout: float
) -> bytes:
    try:
        response = httpx.get(
            f"{endpoint.rstrip('/')}{path}",
            headers=_headers(token_env),
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise Cosmos3RayServeError(
            f"failed to download Cosmos artifact {path}: {exc}"
        ) from exc
    return response.content


__all__ = [
    "Cosmos3RayServeError",
    "DEFAULT_ENDPOINT_ENV",
    "DEFAULT_TOKEN_ENV",
    "RAY_BATCH_SCHEMA",
    "RAY_PROVENANCE_SCHEMA",
    "RayArtifact",
    "RayBatchRequest",
    "RayBatchResponse",
    "load_batch_request",
    "service_health",
    "submit_batch",
]
