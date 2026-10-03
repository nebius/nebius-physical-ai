"""Materialize sealed robot episodes and train-only statistics for upstream XR1."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil

from .assets import MODEL_SHA256, PROCESSOR_REVISION, SOURCE_REVISION
from .dataset import native_annotation, validate_splits, verify_video
from .storage import _Store, _relative


def _episode(store: _Store, entry: dict, split: str, root: Path) -> dict:
    name = entry["episode_id"]
    if name != f"{split}-{int(entry['seed'])}":
        raise ValueError("Episode paths must follow the sealed split/seed identity")
    receipt = store.read_json(f"receipts/{name}.json")
    if (receipt["seed"], receipt["split"], receipt["episode_id"]) != (
        entry["seed"],
        split,
        name,
    ):
        raise ValueError("Dataset receipt differs from the sealed episode split")
    output = root / "episodes" / name
    for filename, identity in receipt["transfer"]["files"].items():
        store.download(
            f"episodes/{name}/{filename}",
            output / _relative(filename),
            identity["sha256"],
        )
    episode = json.loads((output / "episode.json").read_text())
    if (episode["seed"], episode["split"], episode["episode_id"]) != (
        entry["seed"],
        split,
        name,
    ):
        raise ValueError("Recorded trajectory differs from its split assignment")
    for filename in episode["videos"].values():
        verify_video(output / filename, episode["num_frames"], episode["control_hz"])
    result = {
        **entry,
        "success": episode["success"],
        "num_frames": episode["num_frames"],
    }
    if episode["success"]:
        annotation = native_annotation(episode, output)
        destination = root / "annotations" / split / f"{name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(annotation, allow_nan=False))
        result["annotation"] = str(destination)
    return result


def _dataset(uri: str, root: Path) -> dict:
    store = _Store(uri)
    splits = store.read_json("split-manifest.json")
    validate_splits(splits)
    report = {"test_seeds_reserved": [row["seed"] for row in splits["test"]]}
    for split in ("train", "validation"):
        report[split] = [_episode(store, entry, split, root) for entry in splits[split]]
        if not any(row["success"] for row in report[split]):
            raise ValueError(f"No physically qualified {split} demonstrations")
    (root / "dataset-report.json").write_text(json.dumps(report, indent=2))
    return report


def _assets(uri: str, root: Path) -> None:
    store = _Store(uri)
    hashes = store.read_json("checksums.json")
    if hashes.get("base/model_states.pt") != MODEL_SHA256:
        raise ValueError("The base model is not the pinned publisher checkpoint")
    for filename, digest in hashes.items():
        store.download(filename, root / _relative(filename), digest)
    metadata = json.loads((root / "assets.json").read_text())
    if metadata["source_revision"] != SOURCE_REVISION:
        raise ValueError(
            "The asset source revision differs from the reviewed XR1 source"
        )
    if metadata["processor"]["revision"] != PROCESSOR_REVISION:
        raise ValueError(
            "The processor revision differs from the pinned upstream config"
        )
    cache = root / "hf-cache" / "hub" / "models--Qwen--Qwen3-VL-4B-Instruct"
    snapshot = cache / "snapshots" / PROCESSOR_REVISION
    shutil.copytree(root / "processor", snapshot)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text(PROCESSOR_REVISION)
    os.environ.update(
        HF_HOME=str(root / "hf-cache"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )


def _normalization(source: Path, root: Path) -> dict:
    script = source / "tools" / "compute_normalize.py"
    specification = importlib.util.spec_from_file_location(
        "xr1_native_normalization", script
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    result = module.compute([str(root / "annotations" / "train")], 30)
    from mibot.utils.io import validate_quantiles, validate_stats

    validate_stats(result["mean"], result["std"], 30)
    validate_quantiles(result["q01"], result["q99"])
    (root / "normalization.json").write_text(json.dumps(result, allow_nan=False))
    return result


def _configuration(source: Path, root: Path, statistics: dict) -> dict:
    import yaml

    model = yaml.safe_load((source / "configs/model/posttrain.yaml").read_text())[
        "model"
    ]
    trainer = yaml.safe_load((source / "configs/trainer/deepspeed.yaml").read_text())[
        "trainer"
    ]
    # YAML 1.1 reads upstream's bare scientific notation as strings. Preserve
    # the numeric types that its Hydra loader supplies to DeepSpeed/scheduling.
    for name in ("allgather_bucket_size", "reduce_bucket_size"):
        trainer["strategy"]["params"][name] = int(
            float(trainer["strategy"]["params"][name])
        )
    for name in ("warmup_lr_start", "max_lr", "min_lr"):
        trainer["scheduler"]["params"][name] = float(
            trainer["scheduler"]["params"][name]
        )
    model["params"]["pretrained"] = str(root / "assets/base/model_states.pt")
    model["params"]["model"]["async_train"] = False
    trainer.update(
        num_nodes=1,
        devices=8,
        default_root_dir=str(root / "checkpoints"),
        project="xr1-antioch",
        exp_name="dual-franka",
        save_interval=1000,
    )
    trainer["scheduler"]["params"]["num_training_steps"] = trainer["max_steps"]
    training = {
        "batch_size": 2,
        "action_length": 30,
        "paths": [str(root / "annotations/train")],
        **statistics,
    }
    data = {
        "type": "BaseDataModule",
        "params": {
            "type": "json",
            "max_steps": trainer["max_steps"],
            "train_datasets": training,
        },
    }
    return {
        "model": model,
        "trainer": trainer,
        "data": data,
        "validation_paths": [str(root / "annotations/validation")],
    }
