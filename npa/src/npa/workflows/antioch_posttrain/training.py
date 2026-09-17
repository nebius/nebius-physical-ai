"""Fine-tune official ImageNet weights and choose a checkpoint using validation only."""

import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import file_hash, write_json
from .dataset import CLASSES, SPLIT_CYCLES
from .model import backbone_hash, build_model, normalize, score


def split_loader(root: Path, split: str, batch_size: int, *, shuffle: bool = False):
    """Load only an explicit sealed data split with no pickle deserialization.

    Args:
        root: Verified prepared dataset.
        split: Train, validation, or test split.
        batch_size: Positive batch size.
        shuffle: Shuffle training samples.
    Returns:
        Torch data loader.
    Raises:
        ValueError: Split, labels, or image dimensions are invalid.
    """
    records = json.loads((root / "records.json").read_text())
    indices = [index for index, row in enumerate(records) if row["split"] == split]
    if not indices or split not in SPLIT_CYCLES or batch_size < 1:
        raise ValueError("Invalid or empty data split")
    if any(records[index]["cycle"] not in SPLIT_CYCLES[split] for index in indices):
        raise ValueError("Carton cycles leak across splits")
    with np.load(root / "frames.npz", allow_pickle=False) as archive:
        images = archive["images"]
    if images.dtype != np.uint8 or images.shape != (len(records), 224, 224, 3):
        raise ValueError("Invalid prepared image array")
    labels = torch.tensor([records[index]["label"] for index in indices])
    if set(labels.tolist()) != set(range(len(CLASSES))):
        raise ValueError("Split does not contain every operation class")
    tensors = torch.from_numpy(images[indices].copy()).permute(0, 3, 1, 2).contiguous()
    return DataLoader(TensorDataset(tensors, labels), batch_size=batch_size, shuffle=shuffle)


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(normalize(images.to(device), augment=True))
        loss = criterion(logits, labels.to(device))
        if not torch.isfinite(loss):
            raise ValueError("Training loss is nonfinite")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


def _save_weights(model, path: Path):
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, path)


def _optimize(model, loaders, output, epochs, device):
    backbone = [value for name, value in model.named_parameters() if not name.startswith("fc.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": 0.0001}, {"params": model.fc.parameters(), "lr": 0.001},
    ], weight_decay=0.0001)
    labels = loaders["train"].dataset.tensors[1]
    counts = torch.bincount(labels, minlength=len(CLASSES)).float()
    criterion = torch.nn.CrossEntropyLoss(weight=(counts.sum() / counts).to(device))
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, loaders["train"], optimizer, criterion, device)
        validation = score(model, loaders["validation"], device)
        row = {"epoch": epoch, "training_loss": loss,
               "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["macro_f1"] > best:
            best = validation["macro_f1"]
            _save_weights(model, output / "model.pt")
            write_json(output / "selection.json", row)
    return history


def train_model(source: Path, output: Path, epochs: int, batch_size: int, seed: int) -> None:
    """Post-train the backbone on CUDA without inspecting held-out test labels.

    Args:
        source: Verified dataset directory.
        output: New checkpoint directory.
        epochs: Positive training epochs.
        batch_size: Positive examples per batch.
        seed: Nonnegative reproducibility seed.
    Returns:
        None.
    Raises:
        ValueError: Training settings or data are invalid.
        RuntimeError: CUDA is unavailable or backbone weights do not change.
    """
    if epochs < 1 or batch_size < 1 or seed < 0:
        raise ValueError("Positive epochs/batch size and nonnegative seed are required")
    if not torch.cuda.is_available():
        raise RuntimeError("Post-training requires the selected Nebius CUDA worker")
    torch.manual_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    model = build_model(pretrained=True).to(device)
    initial_hash = backbone_hash(model)
    _save_weights(model, output / "baseline.pt")
    loaders = {split: split_loader(source, split, batch_size, shuffle=split == "train")
               for split in ("train", "validation")}
    history = _optimize(model, loaders, output, epochs, device)
    model.load_state_dict(torch.load(output / "model.pt", map_location=device, weights_only=True))
    final_hash = backbone_hash(model)
    if initial_hash == final_hash:
        raise RuntimeError("Backbone was not post-trained")
    _training_report(source, output, history, initial_hash, final_hash, seed, batch_size)


def _training_report(source, output, history, initial_hash, final_hash, seed, batch_size):
    write_json(output / "training.json", {
        "schema": "npa.antioch-posttrain.training.v1", "model": "torchvision.resnet18",
        "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1", "classes": CLASSES,
        "pretrained_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "baseline_kind": "ImageNet backbone with a seeded, untrained five-class head",
        "initial_backbone_sha256": initial_hash, "trained_backbone_sha256": final_hash,
        "dataset_checksums_sha256": file_hash(source / "checksums.json"),
        "checkpoint_sha256": file_hash(output / "model.pt"),
        "baseline_sha256": file_hash(output / "baseline.pt"),
        "seed": seed, "batch_size": batch_size, "epochs": len(history), "history": history,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "gpu": torch.cuda.get_device_name(), "checkpoint_selection": "validation_macro_f1",
        "test_used_for_selection": False,
    })
