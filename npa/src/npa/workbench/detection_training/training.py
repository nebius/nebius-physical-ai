"""Training orchestration for the detection-training workbench."""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
import uuid
from typing import Any, Callable

from .dataloader import make_dataloader
from .models import build_fasterrcnn_resnet50_fpn_v2
from .labels import category_id_map, detector_label_map
from .schemas import DEFAULT_NUM_CLASSES, TrainRequest, TrainResponse
from .storage import ArtifactWriteReceipt, describe_artifact, storage_settings, uri_join, write_bytes_uri, write_json_uri

StatusCallback = Callable[[str, int, dict[str, Any], str | None], None]


class DetectionTrainingError(RuntimeError):
    """Raised when a training run fails."""


def compute_manifest_sha256(kind: str, payload: dict[str, Any]) -> str:
    """Compute a deterministic manifest hash for inputs and hyperparameters."""
    digest = hashlib.sha256()
    if kind == "train" and payload.get("num_classes") is None:
        mapped = detector_label_map(payload.get("label_map"))
        payload = {**payload, "num_classes": max(mapped.values()) + 1 if mapped else DEFAULT_NUM_CLASSES}
    digest.update(kind.encode("utf-8"))
    digest.update(b"\n")
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    digest.update(b"\n")
    return digest.hexdigest()


def make_run_id(prefix: str, manifest_sha256: str) -> str:
    """Create a reproducible-looking but unique run id."""
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{manifest_sha256[:12]}-{uuid.uuid4().hex[:12]}"


def checkpoint_uri_pattern(output_uri: str, run_id: str) -> str:
    return uri_join(output_uri, run_id, "checkpoints", "epoch_{epoch}.pt")


def metrics_uri(output_uri: str, run_id: str) -> str:
    return uri_join(output_uri, run_id, "metrics.json")


def train_detector(
    request: TrainRequest,
    *,
    run_id: str | None = None,
    status_callback: StatusCallback | None = None,
    artifact_callback: Callable[[list[Any]], None] | None = None,
) -> TrainResponse:
    """Run Faster R-CNN training and persist checkpoints plus metrics."""
    # Direct callers can supply a mutated/model_copy request. Validate the final
    # override values before callbacks, runtime initialization, or artifact IO.
    request = TrainRequest.model_validate(request.model_dump())
    manifest = compute_manifest_sha256("train", request.model_dump(mode="json"))
    resolved_run_id = run_id or make_run_id("train", manifest)
    effective_output_uri = request.checkpoint_s3.uri or request.output_uri
    pattern = checkpoint_uri_pattern(effective_output_uri, resolved_run_id)
    metrics_path = metrics_uri(effective_output_uri, resolved_run_id)
    artifacts = []
    if status_callback:
        status_callback("running", 0, {}, None)

    with storage_settings(request.checkpoint_s3):
        try:
            import torch
        except ImportError as exc:
            raise DetectionTrainingError("torch is required for detection training") from exc

        wandb_run = _start_wandb(request)
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_classes = resolve_num_classes(request)
            dataloader = make_dataloader(
                lance_uri=request.data_path or request.lance_uri,
                view=request.view,
                batch_size=request.batch_size,
                shuffle=True,
                label_map=detector_label_map(request.label_map),
                category_id_map=category_id_map(request.label_map),
            )
            model = build_fasterrcnn_resnet50_fpn_v2(num_classes=num_classes)
            model.to(device)
            optimizer = torch.optim.SGD(
                [param for param in model.parameters() if getattr(param, "requires_grad", True)],
                lr=request.learning_rate,
                momentum=0.9,
                weight_decay=0.0005,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
            history: list[dict[str, Any]] = []

            for epoch in range(1, request.epochs + 1):
                loss = train_one_epoch(model, dataloader, optimizer, device=device)
                scheduler.step()
                snapshot = {
                    "epoch": epoch,
                    "train_loss": loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                history.append(snapshot)
                if wandb_run is not None:
                    wandb_run.log(snapshot, step=epoch)
                checkpoint_uri = pattern.format(epoch=epoch)
                checkpoint_receipt = save_checkpoint(
                    checkpoint_uri,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    manifest_sha256=manifest,
                    num_classes=num_classes,
                    request=_checkpoint_request(request),
                )
                artifacts.append(describe_artifact(
                    checkpoint_uri, role="checkpoint", media_type="application/x-pytorch",
                    schema_version="npa.detection.checkpoint.v1", epoch=epoch,
                    write_receipt=checkpoint_receipt,
                ))
                metrics_receipt = write_json_uri(
                    metrics_path,
                    {
                        "run_id": resolved_run_id,
                        "manifest_sha256": manifest,
                        "status": "running" if epoch < request.epochs else "completed",
                        "epochs": history,
                    },
                )
                metrics_artifact = describe_artifact(
                    metrics_path, role="training_metrics", media_type="application/json",
                    schema_version="npa.detection.training-metrics.v1",
                    write_receipt=metrics_receipt,
                )
                if artifact_callback:
                    artifact_callback([*artifacts, metrics_artifact])
                if status_callback:
                    status_callback("running" if epoch < request.epochs else "completed", epoch, snapshot, None)
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    return TrainResponse(
        run_id=resolved_run_id,
        status="completed",
        checkpoint_uri_pattern=pattern,
        metrics_uri=metrics_path,
        total_epochs=request.epochs,
        manifest_sha256=manifest,
        data_path=request.data_path or request.lance_uri,
        training_config=_training_config_public_dict(request),
        artifacts=[*artifacts, metrics_artifact],
    )


def _checkpoint_request(request: TrainRequest) -> dict[str, Any]:
    payload = request.model_dump(mode="json")
    payload["checkpoint_s3"].pop("aws_access_key_id", None)
    payload["checkpoint_s3"].pop("aws_secret_access_key", None)
    return payload


def _start_wandb(request: TrainRequest) -> Any | None:
    if not request.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise DetectionTrainingError("wandb is required when wandb.enabled=true") from exc
    return wandb.init(
        project=request.wandb.project or None,
        name=request.wandb.run_name or None,
        mode=request.wandb.mode or "offline",
        config=_checkpoint_request(request),
    )


def _training_config_public_dict(request: TrainRequest) -> dict[str, Any]:
    return {
        "data_path": request.data_path or request.lance_uri,
        "overrides": list(request.overrides),
        "wandb": request.wandb.model_dump(mode="json"),
        "checkpoint_s3": {
            "uri": request.checkpoint_s3.uri,
            "endpoint_url": request.checkpoint_s3.endpoint_url,
            "aws_access_key_id": "set" if request.checkpoint_s3.aws_access_key_id else "",
            "aws_secret_access_key": "set" if request.checkpoint_s3.aws_secret_access_key else "",
        },
    }


def resolve_num_classes(request: TrainRequest) -> int:
    """Resolve detector classes, adding the background class for mapped labels."""
    if request.num_classes is not None:
        return request.num_classes
    if request.label_map is not None:
        return max(detector_label_map(request.label_map).values()) + 1
    return DEFAULT_NUM_CLASSES


def train_one_epoch(model: Any, dataloader: Any, optimizer: Any | None, *, device: Any) -> float:
    """Run one training epoch and return the mean finite loss."""
    if hasattr(model, "train"):
        model.train()
    total = 0.0
    steps = 0
    for images, targets in dataloader:
        moved_images = [_to_device(image, device) for image in images]
        moved_targets = [
            {key: _to_device(value, device) for key, value in target.items()}
            for target in targets
        ]
        losses = model(moved_images, moved_targets)
        loss = sum(losses.values()) if isinstance(losses, dict) else losses
        value = _loss_value(loss)
        if not math.isfinite(value):
            raise DetectionTrainingError("training loss became non-finite")
        if optimizer is not None:
            if hasattr(optimizer, "zero_grad"):
                optimizer.zero_grad()
            if hasattr(loss, "backward"):
                loss.backward()
            if hasattr(optimizer, "step"):
                optimizer.step()
        total += value
        steps += 1
    if steps == 0:
        raise DetectionTrainingError("training dataloader produced no batches")
    return total / steps


def save_checkpoint(
    uri: str,
    *,
    model: Any,
    optimizer: Any | None,
    epoch: int,
    manifest_sha256: str,
    num_classes: int,
    request: dict[str, Any],
) -> ArtifactWriteReceipt:
    """Serialize and write a checkpoint, returning its exact uploaded byte metadata."""
    try:
        import torch
    except ImportError as exc:
        raise DetectionTrainingError("torch is required to save checkpoints") from exc
    payload = {
        "epoch": epoch,
        "manifest_sha256": manifest_sha256,
        "num_classes": num_classes,
        "request": request,
        "schema_version": "npa.detection.checkpoint.v1",
        "label_map": request.get("label_map"),
        "detector_label_map": detector_label_map(request.get("label_map")),
        "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else {},
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None and hasattr(optimizer, "state_dict") else {},
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return write_bytes_uri(uri, buffer.getvalue())


def _to_device(value: Any, device: Any) -> Any:
    return value.to(device) if hasattr(value, "to") else value


def _loss_value(loss: Any) -> float:
    value = loss
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)
