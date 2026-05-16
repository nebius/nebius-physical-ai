"""Model construction helpers for Faster R-CNN detection training."""

from __future__ import annotations


class DetectionTrainingDependencyError(RuntimeError):
    """Raised when a required ML dependency is unavailable."""


def build_fasterrcnn_resnet50_fpn_v2(
    *,
    num_classes: int,
    weights: str | None = "COCO_V1",
):
    """Build Faster R-CNN ResNet50 FPN v2 with a request-sized predictor head."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as exc:  # pragma: no cover - covered with dependency-free tests.
        raise DetectionTrainingDependencyError("torchvision is required for detector construction") from exc

    resolved_weights = None
    if weights:
        try:
            resolved_weights = getattr(FasterRCNN_ResNet50_FPN_V2_Weights, weights)
        except AttributeError as exc:
            raise ValueError(f"unknown Faster R-CNN v2 weights: {weights}") from exc
    model = fasterrcnn_resnet50_fpn_v2(weights=resolved_weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

