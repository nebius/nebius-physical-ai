"""Compare sealed baseline and fine-tuned weights on the untouched final carton cycle."""

import json
from pathlib import Path

import torch

from .artifacts import file_hash, write_json
from .dataset import CLASSES
from .model import build_model, score
from .training import split_loader


def evaluate_model(dataset: Path, checkpoint: Path, output: Path) -> bool:
    """Evaluate exact checkpoints, retaining reports even when quality fails.

    Args:
        dataset: Verified prepared data.
        checkpoint: Verified trained weights and provenance.
        output: New report directory.
    Returns:
        Whether fine-tuning beats baseline and majority-class accuracy.
    Raises:
        ValueError: Checkpoint provenance does not match the dataset.
        RuntimeError: Checkpoint loading or inference fails.
    """
    training = json.loads((checkpoint / "training.json").read_text())
    if training["dataset_checksums_sha256"] != file_hash(dataset / "checksums.json"):
        raise ValueError("Checkpoint was trained on a different dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = split_loader(dataset, "test", 64)
    model = build_model(pretrained=False).to(device)
    reports = {}
    for name in ("baseline", "model"):
        weights = torch.load(checkpoint / f"{name}.pt", map_location=device, weights_only=True)
        model.load_state_dict(weights, strict=True)
        reports[name] = score(model, loader, device)
    labels = loader.dataset.tensors[1]
    majority = float(torch.bincount(labels).max() / len(labels))
    trained, baseline = reports["model"], reports["baseline"]
    passed = (trained["accuracy"] > majority and trained["macro_f1"] > baseline["macro_f1"])
    output.mkdir(parents=True, exist_ok=False)
    _save_report(dataset, checkpoint, output, reports, majority, passed)
    return passed


def _save_report(dataset, checkpoint, output, reports, majority, passed):
    write_json(output / "evaluation.json", {
        "schema": "npa.antioch-posttrain.evaluation.v1", "all_passed": passed,
        "classes": CLASSES, "split": "test", "carton_cycles": [5],
        "baseline": reports["baseline"], "finetuned": reports["model"],
        "majority_class_accuracy": majority,
        "dataset_checksums_sha256": file_hash(dataset / "checksums.json"),
        "training_checksums_sha256": file_hash(checkpoint / "checksums.json"),
        "checkpoint_sha256": file_hash(checkpoint / "model.pt"),
        "scope": "One unseen carton cycle in the same recorded scene; no cross-scene or control-policy validation.",
    })
