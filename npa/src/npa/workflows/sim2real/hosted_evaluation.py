"""Evaluate rollout inputs concurrently with content-bound durable receipts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from npa.clients.token_factory import resolve_config
from npa.workbench.cosmos import reason
from npa.workbench.cosmos.visual_grounding import validate_stored_visual_grounding


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rollout_input(path: Path, model: str, threshold: float, max_frames: int) -> dict:
    manifest = json.loads(path.read_text())
    frames = [path.parent / name for name in manifest.get("camera_observations", [])]
    if any(
        not frame.resolve().is_relative_to(path.parent.resolve()) for frame in frames
    ):
        raise ValueError("Stage 8 frame escapes its declared rollout directory")
    if not frames or any(not frame.is_file() for frame in frames):
        raise RuntimeError("Stage 8 requires every declared primary rollout frame")
    return {
        "model_id": model,
        "image_paths": frames,
        "frame_metadata": (manifest.get("camera_frame_metadata") or {}).get("primary"),
        "actions": list(manifest.get("actions") or []),
        "task_description": reason.task_description_from_manifest(manifest),
        "rollout_id": str(manifest.get("rollout_id") or path.parent.name),
        "threshold": threshold,
        "max_frames": max_frames or len(frames),
    }


def _input_identity(request: dict, source_sha: str) -> dict:
    frames = request["image_paths"]
    return {
        "schema": "npa.sim2real.hosted_evaluation_receipt.v1",
        "source_sha": source_sha,
        "endpoint_sha256": _digest(resolve_config(require_api_key=False).base_url),
        "request": copy.deepcopy(
            {key: value for key, value in request.items() if key != "image_paths"}
        ),
        "frames": [
            {
                "name": frame.name,
                "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
            }
            for frame in frames
        ],
    }


def _cached_result(store: Any, uri: str, path: Path, identity: dict) -> dict | None:
    try:
        store.download_file(uri, str(path))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {
            "NoSuchKey",
            "404",
            "NotFound",
        }:
            return None
        raise
    receipt = json.loads(path.read_text())
    result = receipt["evaluation"]
    if receipt["identity"] != identity or receipt["evaluation_sha256"] != _digest(
        result
    ):
        raise RuntimeError("Stage 8 cached evaluation identity or digest differs")
    request = identity["request"]
    if (
        result["model"] != request["model_id"]
        or result["rollout_id"] != request["rollout_id"]
    ):
        raise RuntimeError("Stage 8 cached evaluation model or rollout differs")
    validate_stored_visual_grounding(result)
    return result


def _evaluate_one(
    path: Path, *, store: Any, prefix: str, settings: dict
) -> tuple[dict, bool]:
    request = _rollout_input(
        path, settings["model"], settings["threshold"], settings["max_frames"]
    )
    identity = _input_identity(request, settings["source_sha"])
    receipt_path = path.parent / "hosted-evaluation-receipt.json"
    uri = f"{prefix}/receipts/{_digest(identity)}.json"
    cached = _cached_result(store, uri, receipt_path, identity)
    if cached is not None:
        return cached, True
    result = reason.run_token_factory_rollout_vlm(**request)
    receipt = {
        "identity": identity,
        "evaluation": result,
        "evaluation_sha256": _digest(result),
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    store.upload_file(str(receipt_path), uri)
    verified = _cached_result(store, uri, receipt_path, identity)
    if verified is None:
        raise RuntimeError("Stage 8 evaluation receipt disappeared after publication")
    return verified, False


def _evaluate_all(
    paths: list[Path], store: Any, prefix: str, settings: dict, concurrency: int
) -> list:
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _evaluate_one, path, store=store, prefix=prefix, settings=settings
            )
            for path in paths
        ]
        return [future.result() for future in futures]


def _validate_rollouts(paths, model, threshold, concurrency, max_frames) -> None:
    if concurrency < 1 or max_frames < 0:
        raise ValueError(
            "evaluation_concurrency must be positive; evaluation_max_frames must be nonnegative"
        )
    requests = [_rollout_input(path, model, threshold, max_frames) for path in paths]
    identities = [request["rollout_id"] for request in requests]
    if len(set(identities)) != len(identities):
        raise ValueError("Stage 8 found duplicate Stage 7 rollout identities")


def evaluate_rollouts(
    paths: list[Path],
    *,
    store: Any,
    prefix: str,
    model: str,
    threshold: float,
    source_sha: str,
    concurrency: int,
    max_frames: int,
) -> tuple[list[dict], int]:
    """Score rollouts in deterministic order and reuse exact durable evaluations.

    Args:
        paths: Ordered Stage 7 manifest paths.
        store: Run-scoped object storage client.
        prefix: Iteration-scoped evaluator S3 prefix.
        model: Exact hosted evaluator model ID.
        threshold: Evaluation threshold.
        source_sha: Source attested by the executing image.
        concurrency: Positive number of simultaneous inference requests.
        max_frames: Frame sample size; zero sends all declared primary frames.
    Returns:
        Evaluations in input order and the number reused from verified receipts.
    Raises:
        ValueError: Invalid settings or duplicate rollout identities.
        RuntimeError: Missing frames or inconsistent durable evidence.
        ClientError: Storage access or publication failed.
    """
    _validate_rollouts(paths, model, threshold, concurrency, max_frames)
    settings = {
        "model": model,
        "threshold": threshold,
        "source_sha": source_sha,
        "max_frames": max_frames,
    }
    completed = _evaluate_all(paths, store, prefix, settings, concurrency)
    return [result for result, _ in completed], sum(reused for _, reused in completed)
