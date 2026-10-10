"""Package a trained policy with portable tokenizer dependencies and verified checkpoint provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from .public_vla_data import PINS, write_json


def file_sha256(path: Path) -> str:
    """Hash a large artifact without loading it into memory.

    Args:
        path: Existing artifact file.
    Returns:
        Lowercase SHA256 digest.
    Raises:
        OSError: The artifact cannot be read.
    """
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def export_policy(root: Path, policy: Path) -> dict:
    """Export a portable inference bundle and prove that training changed its weights.

    Args:
        root: Complete pipeline run directory.
        policy: Selected native LeRobot checkpoint directory.
    Returns:
        Portable artifact manifest with exact weight hashes and relative filenames.
    Raises:
        ValueError: The selected checkpoint is unchanged from the downloaded policy.
        OSError: Checkpoint or dependency files cannot be copied.
    """
    checksum = file_sha256(policy / "model.safetensors")
    initial = file_sha256(root / "inputs/model/model.safetensors")
    if checksum == initial:
        raise ValueError("training did not change the downloaded model weights")
    output = root / "exported-policy"
    _copy_policy(policy, output / "policy")
    _copy_backbone(root / "inputs/backbone", output / "backbone")
    (output / "README.txt").write_text(_instructions())
    _attribution(root, output)
    manifest = {
        "schema": "npa.public-vla.policy.v1",
        "inputs": PINS,
        "initial_weights_sha256": initial,
        "trained_weights_sha256": checksum,
        "files": {
            str(path.relative_to(output)): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _attribution(root, output):
    shutil.copy2(
        Path(__file__).with_name("apache-2.0.txt"), output / "license-apache-2.0.txt"
    )
    card = root / "inputs/model/README.md"
    if card.is_file():
        shutil.copy2(card, output / "source-model-card.md")
    (output / "attribution.txt").write_text(
        "Derived from HuggingFaceVLA/smolvla_libero and HuggingFaceTB/SmolVLM2-500M-Instruct.\n"
        "Both upstream models declare Apache-2.0; immutable source revisions are in manifest.json.\n"
        "Modifications: continued LeRobot training, task adaptation and portable inference paths.\n"
        "Public training demonstrations: lerobot/libero (Apache-2.0).\n"
        "No raw demonstrations or simulator assets are included in this inference bundle.\n"
    )


def _copy_policy(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if (
            path.is_file()
            and path.suffix in {".json", ".safetensors"}
            and path.name != "train_config.json"
        ):
            shutil.copy2(path, target / path.name)
    config = json.loads((target / "config.json").read_text())
    config.update(
        vlm_model_name="./backbone", load_vlm_weights=False, push_to_hub=False
    )
    config.pop("pretrained_path", None)
    write_json(target / "config.json", config)
    path = target / "policy_preprocessor.json"
    value = json.loads(path.read_text())
    _portable_tokenizer(value)
    write_json(path, value)


def _portable_tokenizer(value):
    if isinstance(value, dict):
        if "tokenizer_name" in value:
            value["tokenizer_name"] = "./backbone"
        for child in value.values():
            _portable_tokenizer(child)
    elif isinstance(value, list):
        for child in value:
            _portable_tokenizer(child)


def _copy_backbone(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix in {".json", ".txt", ".model", ".md"}:
            shutil.copy2(path, target / path.name)


def _instructions():
    return (
        "SmolVLA LIBERO inference bundle\n\n"
        "Use the pinned LeRobot runtime documented with this pipeline.\n"
        "Run commands from this directory: the policy and tokenizer use relative paths.\n"
        "Load SmolVLAPolicy.from_pretrained('policy') and make_pre_post_processors with "
        "pretrained_path='policy'; pass real observations through the saved processors.\n"
        "The full reference runner prepares LIBERO assets and noninteractive simulator "
        "configuration before evaluation. The inference bundle contains model dependencies, "
        "not simulator assets. Offline CUDA inference is verified before report export.\n\n"
        "Training resume state remains in the original run's checkpoints directory.\n"
        "This bundle is for simulator evaluation; physical robot use needs its own validation.\n"
        "Model and dataset sources and immutable revisions are recorded in manifest.json.\n"
    )
