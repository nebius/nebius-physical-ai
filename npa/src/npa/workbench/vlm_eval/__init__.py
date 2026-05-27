"""Stub VLM evaluation helpers for sim-to-real pipeline validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient


class VlmEvalError(ValueError):
    """Raised when a VLM evaluation request is invalid."""


@dataclass(frozen=True)
class VlmEvalResult:
    status: str
    backend: str
    input_path: str
    output_path: str
    result_uri: str
    task: str
    model: str
    score: float
    success_threshold: float
    passed: bool
    generated_at: str


__all__ = [
    "VlmEvalResult",
    "evaluate_stub",
    "result_uri_for",
    "write_result",
]


def evaluate_stub(
    *,
    input_path: str,
    output_path: str,
    task: str = "sereact-sim-to-real",
    model: str = "vlm-eval-stub",
    success_threshold: float = 0.8,
    score: float | None = None,
) -> VlmEvalResult:
    """Return deterministic stub metrics without calling a VLM backend."""
    if not input_path:
        raise VlmEvalError("input_path is required")
    if not output_path:
        raise VlmEvalError("output_path is required")
    if not 0.0 <= success_threshold <= 1.0:
        raise VlmEvalError("--success-threshold must be between 0 and 1")
    effective_score = _deterministic_score(input_path, task, model) if score is None else score
    if not 0.0 <= effective_score <= 1.0:
        raise VlmEvalError("--score must be between 0 and 1")
    passed = effective_score >= success_threshold
    return VlmEvalResult(
        status="passed" if passed else "needs_iteration",
        backend="stub",
        input_path=input_path,
        output_path=output_path,
        result_uri=result_uri_for(output_path),
        task=task,
        model=model,
        score=round(effective_score, 4),
        success_threshold=success_threshold,
        passed=passed,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def result_uri_for(output_path: str) -> str:
    """Return the JSON artifact URI for an output path."""
    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + "/vlm_eval_stub.json"


def write_result(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    """Write a VLM stub result to local disk or S3."""
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if result_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-vlm-eval-") as tmp:
            local_path = Path(tmp) / "vlm_eval_stub.json"
            local_path.write_text(body, encoding="utf-8")
            return client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    if path.suffix != ".json":
        path = path / "vlm_eval_stub.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _deterministic_score(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF
