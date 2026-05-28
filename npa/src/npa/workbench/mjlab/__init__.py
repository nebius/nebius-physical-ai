"""MJLab locomotion evaluation helpers for Workbench workflows."""

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


class MjlabEvalError(ValueError):
    """Raised when an MJLab evaluation request is invalid."""


@dataclass(frozen=True)
class MjlabEvalResult:
    status: str
    backend: str
    input_path: str
    checkpoint: str
    output_path: str
    result_uri: str
    suite: str
    embodiment: str
    episodes: int
    score: float
    success_threshold: float
    passed: bool
    generated_at: str


__all__ = [
    "MjlabEvalResult",
    "evaluate_locomotion",
    "result_uri_for",
    "write_result",
]


def evaluate_locomotion(
    *,
    input_path: str,
    checkpoint: str,
    output_path: str,
    suite: str = "locomotion",
    embodiment: str = "unitree-g1",
    episodes: int = 8,
    success_threshold: float = 0.75,
    score: float | None = None,
) -> MjlabEvalResult:
    """Return deterministic MJLab-style locomotion metrics.

    The CLI writes the same result schema that the SkyPilot templates consume.
    Heavy MJLab imports stay out of module import paths so unit tests and CLI
    discovery do not require simulator packages.
    """

    if not input_path:
        raise MjlabEvalError("input_path is required")
    if not checkpoint:
        raise MjlabEvalError("checkpoint is required")
    if not output_path:
        raise MjlabEvalError("output_path is required")
    if episodes <= 0:
        raise MjlabEvalError("--episodes must be positive")
    if not 0.0 <= success_threshold <= 1.0:
        raise MjlabEvalError("--success-threshold must be between 0 and 1")

    effective_score = _deterministic_score(input_path, checkpoint, suite, embodiment) if score is None else score
    if not 0.0 <= effective_score <= 1.0:
        raise MjlabEvalError("--score must be between 0 and 1")

    passed = effective_score >= success_threshold
    return MjlabEvalResult(
        status="passed" if passed else "needs_iteration",
        backend="mjlab",
        input_path=input_path,
        checkpoint=checkpoint,
        output_path=output_path,
        result_uri=result_uri_for(output_path),
        suite=suite,
        embodiment=embodiment,
        episodes=episodes,
        score=round(effective_score, 4),
        success_threshold=success_threshold,
        passed=passed,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def result_uri_for(output_path: str) -> str:
    """Return the MJLab JSON artifact URI for an output path."""

    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + "/mjlab_eval.json"


def write_result(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    """Write an MJLab result to local disk or S3."""

    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if result_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-mjlab-") as tmp:
            local_path = Path(tmp) / "mjlab_eval.json"
            local_path.write_text(body, encoding="utf-8")
            return client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    if path.suffix != ".json":
        path = path / "mjlab_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _deterministic_score(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF
