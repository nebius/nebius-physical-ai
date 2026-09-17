"""Build and score the native TorchVision ResNet-18 warehouse classifier."""

import hashlib

import torch
from torchvision.models import ResNet18_Weights, resnet18

from .dataset import CLASSES


def build_model(*, pretrained: bool):
    """Construct ResNet-18 with a warehouse operation classification head.

    Args:
        pretrained: Fetch official ImageNet-1K V1 weights for post-training.
    Returns:
        Native TorchVision model with five output logits.
    Raises:
        OSError: Official pretrained weights cannot be fetched.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def backbone_hash(model) -> str:
    """Identify backbone parameters independently of the replacement head.

    Args:
        model: Warehouse classifier.
    Returns:
        SHA-256 of ordered named backbone parameter bytes.
    Raises:
        RuntimeError: Parameters cannot be copied to CPU.
    """
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def normalize(images, *, augment: bool = False):
    """Normalize RGB frames while preserving spatial operation labels.

    Args:
        images: NCHW uint8 tensor.
        augment: Apply training-only photometric variation.
    Returns:
        ImageNet-normalized float tensor.
    Raises:
        RuntimeError: Tensor dimensions are incompatible.
    """
    values = images.float() / 255
    if augment:
        gain = torch.empty((len(values), 1, 1, 1), device=values.device).uniform_(0.85, 1.15)
        values = (values * gain).clamp(0, 1)
    mean = values.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = values.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (values - mean) / std


def score(model, loader, device) -> dict:
    """Measure exact class confusion, accuracy, and macro F1.

    Args:
        model: Classifier being evaluated.
        loader: Ordered image and label batches.
        device: Torch execution device.
    Returns:
        Metrics and predictions in loader order.
    Raises:
        ValueError: Evaluation contains no samples.
    """
    model.eval()
    truth, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(normalize(images.to(device)))
            predicted.extend(logits.argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    if not truth:
        raise ValueError("Evaluation split is empty")
    matrix = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)
    for actual, prediction in zip(truth, predicted, strict=True):
        matrix[actual, prediction] += 1
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = 2 * matrix.diag().float() / denominator.clamp_min(1)
    return {"samples": len(truth), "accuracy": float(matrix.diag().sum() / len(truth)),
            "macro_f1": float(f1.mean()), "per_class_f1": dict(zip(CLASSES, f1.tolist(), strict=True)),
            "confusion_matrix": matrix.tolist(), "predictions": predicted, "labels": truth}
