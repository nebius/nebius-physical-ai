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
import uuid
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, PositiveInt, field_validator

from npa.clients.storage import StorageClient
from npa.workbench.storage_scope import StorageAuthorizationError, authorize_uri
from npa.workbench.cosmos.ray_inputs import validate_sample_s3_keys

RAY_BATCH_SCHEMA = "npa.cosmos3.ray-serve.batch.v1"
RAY_PROVENANCE_SCHEMA = "npa.cosmos3.ray-serve.provenance.v1"
RAY_FRAMEWORK_REVISION = "5e67049cd94acb667786f1e6dd0dab821cb90c97"
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


class _SampleBinding(BaseModel):
    """The numeric fields used for binding, with native Pydantic coercion."""

    seed: int | None = None
    num_frames: PositiveInt | None = None
    num_outputs: PositiveInt | None = None


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
    request = _prepare_client_request(request)
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

    # Choose the identity before submitting so even requests without an explicit
    # ID can reject a stale or foreign response. The server already honors it.
    if not request.request_id:
        request = request.model_copy(update={"request_id": uuid.uuid4().hex})
    response_payload = _request_json(
        "POST",
        resolved_endpoint,
        "/v1/batches",
        token_env=token_env,
        timeout=timeout,
        payload=request.model_dump(mode="json"),
    )
    response = _validate_batch_response(request, response_payload)

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


def _prepare_client_request(request: RayBatchRequest) -> RayBatchRequest:
    """Reject identities the pinned native single-result Serve path cannot bind."""

    if request.model != "Cosmos3-Nano":
        raise Cosmos3RayServeError("unsupported Cosmos3 Ray model")
    samples = []
    for sample in request.samples:
        name = sample["name"]
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise Cosmos3RayServeError("sample name must be a canonical safe string")
        try:
            binding = _SampleBinding.model_validate(sample)
        except ValueError as exc:
            raise Cosmos3RayServeError(
                "invalid numeric Cosmos3 sample overrides"
            ) from exc
        if binding.num_outputs not in (None, 1):
            raise Cosmos3RayServeError(
                "native Ray batches require num_outputs=1 per sample"
            )
        normalized = {**sample, **binding.model_dump(exclude_unset=True)}
        try:
            validate_sample_s3_keys(normalized)
        except StorageAuthorizationError as exc:
            raise Cosmos3RayServeError("invalid conditioning S3 URI") from exc
        if sample.get("defaults_file") is not None:
            raise Cosmos3RayServeError(
                "custom defaults_file cannot be bound by this client; inline sample overrides"
            )
        # Reject an unresolvable implicit mode before spending inference work.
        _requested_mode(normalized)
        samples.append(normalized)
    return request.model_copy(update={"samples": samples})


def _validate_batch_response(
    request: RayBatchRequest, payload: dict[str, Any]
) -> RayBatchResponse:
    """Bind native SampleOutputs and the complete file manifest before any I/O.

    cosmos-framework 5e67049's inference.py saves vision.jpg/vision.mp4 for
    ordinary samples and reasoner_text.txt for reasoner samples. Additional
    control/debug files are legal; all files are named by SampleOutputs.outputs.
    Keep this client-only check separate from the server's wire models.
    """

    if payload.get("schema_version") != RAY_BATCH_SCHEMA:
        raise Cosmos3RayServeError("unsupported Cosmos3 Ray response schema")
    try:
        response = RayBatchResponse.model_validate(payload, strict=True)
    except ValueError as exc:
        raise Cosmos3RayServeError("invalid Cosmos3 Ray response structure") from exc
    if response.model != request.model:
        raise Cosmos3RayServeError("service returned a different model")
    if response.request_id != request.request_id:
        raise Cosmos3RayServeError("service returned a different request identity")
    if response.framework_revision != RAY_FRAMEWORK_REVISION:
        raise Cosmos3RayServeError("unsupported Cosmos3 Ray framework revision")
    if response.batch_size != len(request.samples):
        raise Cosmos3RayServeError(
            f"service returned batch_size={response.batch_size}, expected {len(request.samples)}"
        )
    if len(response.outputs) != len(request.samples):
        raise Cosmos3RayServeError(
            f"service returned {len(response.outputs)} structured outputs, "
            f"expected {len(request.samples)}"
        )

    requested = {sample["name"]: sample for sample in request.samples}
    seen_samples: set[str] = set()
    expected: dict[str, str] = {}
    for result in response.outputs:
        args = result.get("args")
        name = args.get("name") if isinstance(args, dict) else None
        if not isinstance(name, str) or name not in requested or name in seen_samples:
            raise Cosmos3RayServeError(
                "service returned duplicate, foreign, or missing sample identity"
            )
        seen_samples.add(name)
        if result.get("status") != "success":
            raise Cosmos3RayServeError(f"sample {name} did not succeed")
        sample = requested[name]
        for key, value in (
            ("model_mode", _requested_mode(sample)),
            ("seed", sample.get("seed")),
        ):
            if value is not None and (
                args.get(key) != value or type(args.get(key)) is not type(value)
            ):
                raise Cosmos3RayServeError(f"sample {name} returned a different {key}")
        mode = args.get("model_mode")
        if not isinstance(mode, str) or mode not in {
            "text2image",
            "text2video",
            "image2image",
            "image2video",
            "video2video",
            "audio_image2video",
            "forward_dynamics",
            "inverse_dynamics",
            "wam",
            "reasoner",
        }:
            raise Cosmos3RayServeError(
                f"sample {name} returned an unsupported model_mode"
            )
        frames = args.get("num_frames")
        if type(frames) is not int or frames < 1:
            raise Cosmos3RayServeError(f"sample {name} returned invalid num_frames")
        # Native temporal compression can round a requested video length up.
        # Bind the category even when the caller uses native frame defaults.
        hints = _active_hints(sample)
        if _active_hints(args) != hints:
            raise Cosmos3RayServeError(
                f"sample {name} returned different transfer hints"
            )
        # Native generation dispatches transfer before reasoner when hints exist.
        text_output = mode == "reasoner" and not hints
        if not text_output and _requested_image(sample, mode) != (frames == 1):
            raise Cosmos3RayServeError(
                f"sample {name} returned a different frame category"
            )
        outputs = result.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise Cosmos3RayServeError(f"sample {name} has no structured output files")
        sample_files: set[str] = set()
        for output in outputs:
            if not isinstance(output, dict) or not isinstance(
                output.get("content"), dict
            ):
                raise Cosmos3RayServeError(
                    f"sample {name} returned invalid output content"
                )
            files = output.get("files")
            if not isinstance(files, list) or not files:
                raise Cosmos3RayServeError(
                    f"sample {name} has no structured output files"
                )
            for path in files:
                _validate_output_path(path, request.request_id, name)
                if path in expected:
                    raise Cosmos3RayServeError(
                        "service declared a duplicate output file"
                    )
                expected[path] = name
                sample_files.add(path)
        primary = (
            "reasoner_text.txt"
            if text_output
            else ("vision.jpg" if frames == 1 else "vision.mp4")
        )
        if f"{request.request_id}/{name}/{primary}" not in sample_files:
            raise Cosmos3RayServeError(
                f"sample {name} is missing its primary output file"
            )
        extension = ".jpg" if frames == 1 else ".mp4"
        for hint in hints:
            if (
                f"{request.request_id}/{name}/control_{hint}{extension}"
                not in sample_files
            ):
                raise Cosmos3RayServeError(
                    f"sample {name} is missing a requested control output file"
                )

    actual: dict[str, str] = {}
    for artifact in response.artifacts:
        _validate_output_path(artifact.path, request.request_id, artifact.sample)
        if artifact.path in actual:
            raise Cosmos3RayServeError("service returned a duplicate artifact path")
        actual[artifact.path] = artifact.sample
    if actual != expected:
        raise Cosmos3RayServeError(
            "artifact manifest does not exactly cover the requested sample outputs"
        )
    return response


def _active_hints(sample: dict[str, Any]) -> set[str]:
    return {
        key
        for key in ("edge", "blur", "depth", "seg", "wsm")
        if sample.get(key) is not None
    }


def _requested_image(sample: dict[str, Any], mode: str) -> bool:
    """Resolve the pinned defaults' frame category without trusting the response."""

    if sample.get("num_frames") is not None:
        return sample["num_frames"] == 1
    # args.py applies the single-WSM transfer default (101 frames) after mode
    # defaults, unless num_frames was explicitly supplied. Null hints are inactive.
    if _active_hints(sample) == {"wsm"}:
        return False
    # Image defaults resolve to one frame; all ordinary video/action defaults
    # resolve to multiple frames. Reasoner starts at one before transfer defaults.
    return mode in {"text2image", "image2image", "reasoner"}


def _requested_mode(sample: dict[str, Any]) -> Any:
    """Match pinned OmniSampleOverrides.resolved_model_mode before defaults."""

    if sample.get("model_mode") is not None:
        return sample["model_mode"]
    vision = sample.get("vision_path")
    if vision is None:
        source = "text"
    elif isinstance(vision, str):
        # The service stages S3 inputs using StorageScope's decoded object key.
        # Use that same parser so an encoded suffix cannot change the inferred
        # mode between the client and the upstream validator's local file.
        if vision.startswith("s3://"):
            try:
                vision = authorize_uri(vision, operation="read").key
            except StorageAuthorizationError as exc:
                raise Cosmos3RayServeError("invalid conditioning S3 URI") from exc
        suffix = Path(vision).suffix.lower()
        source = {
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
            ".webp": "image",
            ".mp4": "video",
        }.get(suffix)
        if source is None:
            raise Cosmos3RayServeError("unsupported conditioning file extension")
    else:
        raise Cosmos3RayServeError("invalid conditioning file path")
    target = "image" if sample.get("num_frames") == 1 else "video"
    return f"{source}2{target}"


def _validate_output_path(path: Any, request_id: str, sample: str) -> None:
    # A file path is also used as an HTTP URL suffix. Reject URL metacharacters,
    # backslashes and noncanonical spellings rather than letting either client
    # normalize an alias, traversal, query, or fragment into a different file.
    parts = path.split("/") if isinstance(path, str) else []
    if (
        len(parts) < 3
        or parts[:2] != [request_id, sample]
        or any(not SAFE_NAME.fullmatch(part) or part in {".", ".."} for part in parts)
    ):
        raise Cosmos3RayServeError(
            "artifact path must be canonical and inside its request/sample namespace"
        )


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
