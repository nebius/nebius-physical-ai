"""Evaluation orchestration for detection-training checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
import math
from typing import Any

from .dataloader import make_dataloader
from .models import build_fasterrcnn_resnet50_fpn_v2
from .schemas import EvalRequest, EvalResponse
from .storage import describe_artifact, read_bytes_uri, uri_join, write_json_uri


class DetectionEvaluationError(RuntimeError):
    """Raised when detector evaluation fails."""


def evaluate_detector(request: EvalRequest) -> EvalResponse:
    """Evaluate a saved detector checkpoint against a Lance materialized view."""
    eval_run_id, manifest = evaluation_identity(request)
    metrics = _evaluate_with_model(request)
    result = EvalResponse(
        mAP=metrics["mAP"],
        mAP_50=metrics["mAP_50"],
        mAP_75=metrics["mAP_75"],
        per_category_AP=metrics.get("per_category_AP", {}),
        eval_run_id=eval_run_id,
        manifest_sha256=manifest,
    )
    output_uri = uri_join(request.output_uri, eval_run_id, "eval_metrics.json")
    write_json_uri(output_uri, result.model_dump(mode="json"))
    artifact = describe_artifact(output_uri, role="evaluation_metrics", media_type="application/json", schema_version="npa.detection.evaluation-metrics.v1")
    return result.model_copy(update={"artifacts": [artifact]})


def _evaluate_with_model(request: EvalRequest) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise DetectionEvaluationError("torch is required for detection evaluation") from exc

    checkpoint = _load_checkpoint(request.checkpoint_uri)
    label_map, numeric_map = checkpoint_labels(checkpoint, requested=request.label_map)
    num_classes = int(checkpoint.get("num_classes") or 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_fasterrcnn_resnet50_fpn_v2(num_classes=num_classes, weights=None)
    state = checkpoint.get("model_state_dict") or {}
    if not state:
        raise DetectionEvaluationError("checkpoint contains no model weights")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise DetectionEvaluationError("checkpoint weights do not match the detector architecture") from exc
    model.to(device)
    model.eval()
    dataloader = make_dataloader(
        lance_uri=request.lance_uri,
        view=request.eval_view,
        batch_size=1,
        shuffle=False,
        label_map=label_map,
        category_id_map=numeric_map,
    )
    result = _compute_map_metrics(model, dataloader, device=device)
    if label_map:
        names = {f"class_{value}": name for name, value in label_map.items()}
        result["per_category_AP"] = {names.get(key, key): value for key, value in result.get("per_category_AP", {}).items()}
    return result


def checkpoint_labels(checkpoint: dict[str, Any], *, requested: dict[str, int] | None):
    """New checkpoints are authoritative; old checkpoints keep their original IDs."""
    original = checkpoint.get("label_map", (checkpoint.get("request") or {}).get("label_map"))
    if original is not None and requested is not None and requested != original:
        raise DetectionEvaluationError("evaluation label_map differs from checkpoint category identity")
    source = original if original is not None else requested
    if "detector_label_map" in checkpoint:
        model_labels = checkpoint["detector_label_map"]
        if original is None and model_labels is None:
            # Numeric-only checkpoints have no recorded category identity. An
            # explicit eval map names those existing detector IDs; do not shift it.
            return requested, None
        numeric = {category_id: model_labels[name] for name, category_id in source.items()} if source and model_labels else None
        return model_labels, numeric
    return source, None


def _compute_map_metrics(model: Any, dataloader: Any, *, device: Any) -> dict[str, Any]:
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as exc:
        raise DetectionEvaluationError("torchmetrics and its COCO backend are required for real detection evaluation") from exc

    metric = MeanAveragePrecision(class_metrics=True)
    try:
        import torch

        with torch.no_grad():
            for images, targets in dataloader:
                moved_images = [_to_device(image, device) for image in images]
                moved_targets = [
                    {key: _to_device(value, device) for key, value in target.items()}
                    for target in targets
                ]
                predictions = model(moved_images)
                metric.update(predictions, moved_targets)
    except Exception as exc:
        raise DetectionEvaluationError(f"mAP evaluation failed: {exc}") from exc
    values = metric.compute()
    per_category: dict[str, float] = {}
    classes = values.get("classes")
    map_per_class = values.get("map_per_class")
    if classes is not None and map_per_class is not None:
        for cls, ap in zip(_to_list(classes), _to_list(map_per_class)):
            per_category[f"class_{int(cls)}"] = _finite_float(ap)
    return {
        "mAP": _finite_float(values.get("map", 0.0)),
        "mAP_50": _finite_float(values.get("map_50", 0.0)),
        "mAP_75": _finite_float(values.get("map_75", 0.0)),
        "per_category_AP": per_category,
    }


def _load_checkpoint(uri: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise DetectionEvaluationError("torch is required to load checkpoints") from exc
    data = read_bytes_uri(uri)
    loaded = torch.load(io.BytesIO(data), map_location="cpu")
    if not isinstance(loaded, dict):
        raise DetectionEvaluationError("checkpoint did not contain a dictionary")
    return loaded


def _eval_manifest(request: EvalRequest) -> str:
    digest = hashlib.sha256()
    digest.update(b"eval\n")
    digest.update(json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
    return digest.hexdigest()


def evaluation_identity(request: EvalRequest) -> tuple[str, str]:
    manifest = _eval_manifest(request)
    return f"eval-{manifest[:12]}", manifest


def _to_device(value: Any, device: Any) -> Any:
    return value.to(device) if hasattr(value, "to") else value


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _finite_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    number = float(value)
    if not math.isfinite(number):
        raise DetectionEvaluationError("evaluation returned a non-finite metric")
    return number
