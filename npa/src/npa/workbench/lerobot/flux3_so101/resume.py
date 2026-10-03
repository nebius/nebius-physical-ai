"""Restore a verified SO-101 checkpoint through LeRobot's native resume path."""

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from .calibration import ID as CALIBRATION_ID


def _verify_adapters(checkpoint: Path, record: dict, base_repo: str) -> None:
    for variant, key in (
        ("pretrained_model", "adapter_sha256"),
        ("pretrained_model_ema", "ema_adapter_sha256"),
    ):
        adapter = checkpoint / variant / "adapter_model.safetensors"
        with adapter.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record[key]:
            raise ValueError(f"resume {variant} SHA-256 mismatch")
        config = json.loads((checkpoint / variant / "adapter_config.json").read_text())
        if config.get("base_model_name_or_path") != base_repo:
            raise ValueError("resume adapter references a different base policy")


def _verify_state(checkpoint: Path, record: dict, recipe: dict, steps: int) -> None:
    metadata = json.loads(
        (checkpoint / "training_state/training_step.json").read_text()
    )
    expected = {
        "step": record["step"],
        "dp_world_size": 1,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "mixed_precision": "bf16",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("resume training topology or step is incompatible")
    if record.get("calibration_id") != CALIBRATION_ID:
        raise ValueError("resume calibration differs from this recipe")
    if not 0 < record["step"] < steps or record["step"] % 4:
        raise ValueError("resume step must precede the requested total microsteps")
    config = json.loads((checkpoint / "pretrained_model/train_config.json").read_text())
    dataset = config["dataset"]
    if (dataset["repo_id"], dataset["revision"]) != (
        recipe["dataset"]["repo_id"],
        recipe["dataset"]["revision"],
    ):
        raise ValueError("resume dataset differs from the pinned recipe")
    if not config["policy"].get("use_peft") or not config["ema"]["enable"]:
        raise ValueError("resume requires the original PEFT and EMA training state")
    for name in (
        "optimizer_state.safetensors",
        "optimizer_param_groups.json",
        "rng_state.safetensors",
        "scheduler_state.json",
        "ema_state.pt",
    ):
        if not (checkpoint / "training_state" / name).is_file():
            raise FileNotFoundError(f"resume checkpoint lacks {name}")


def _verify_finite(checkpoint: Path) -> None:
    import torch
    from safetensors import safe_open

    paths = [
        checkpoint / variant / "adapter_model.safetensors"
        for variant in ("pretrained_model", "pretrained_model_ema")
    ]
    paths.append(checkpoint / "training_state/optimizer_state.safetensors")
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if not torch.isfinite(tensors.get_tensor(key)).all():
                    raise ValueError(
                        f"resume checkpoint contains non-finite tensors: {path.name}"
                    )


def restore(
    storage, input_path: str, checkpoint: Path, policy: Path, recipe: dict, steps: int
) -> dict:
    """Download and validate a complete checkpoint without modifying its source.

    Args:
        storage: Configured Object Storage client.
        input_path: S3 prefix of one numbered checkpoint with COMPLETE.json.
        checkpoint: Fresh local checkpoint directory.
        policy: Locally downloaded pinned base policy.
        recipe: Pinned dataset and model contract.
        steps: Requested final total microsteps, including restored progress.
    Returns:
        Non-identifying checkpoint evidence for the final training record.
    Raises:
        ValueError: Source, hashes, calibration, topology, or tensors are incompatible.
        FileNotFoundError: Required native training state is missing.
    """
    parsed = urlparse(input_path)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("resume input_path must be an S3 checkpoint prefix")
    storage.download_directory(input_path, str(checkpoint))
    record = json.loads((checkpoint / "COMPLETE.json").read_text())
    _verify_adapters(checkpoint, record, recipe["policy"]["repo_id"])
    _verify_state(checkpoint, record, recipe, steps)
    _verify_finite(checkpoint)
    path = checkpoint / "pretrained_model/adapter_config.json"
    config = json.loads(path.read_text())
    config["base_model_name_or_path"] = str(policy)
    config["revision"] = None
    path.write_text(json.dumps(config, indent=2) + "\n")
    return {
        "microsteps": record["step"],
        "optimizer_updates": record["step"] // 4,
        "adapter_sha256": record["adapter_sha256"],
        "ema_adapter_sha256": record["ema_adapter_sha256"],
        "checkpoint_tensors_finite": True,
    }
