"""Run one arm of the matched native RLC stage-conditioning pair."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from action_partition import EXPECTED_TRAINABLE_PATHS, freeze_filter
from panel_data import (
    balanced_prefix_counts,
    digest,
    install_balanced_loader,
    load_split,
)
from stage_conditioning import load_trace

EXPECTED_REVISIONS = {
    ".": "ca556f74a455cef7987a2be4537b5ac85cc56dd7",
    "openpi": "01177e0242a1c7e8fad2547caa0e987def614cda",
    "BEHAVIOR-1K": "684a83050ddd398de231e6aa7fc605bc34458d4b",
}
LEROBOT_REVISION = "c43f58116b975ae79af62714e1417b38facd4e37"
PARENT_ARCHIVE_SHA256 = (
    "9e7e078a721e5a0db60ca180e8ed6ace57d66da03d9884923b5d88304b5f98ea"
)
PARENT_CHECKPOINT_PREFIX = "checkpoint_2/"
NORMALIZATION = "assets/IliaLarchenko/behavior_224_rgb/norm_stats.json"
COMMON_CONFIG = {
    "schema": "npa.behavior.rlc-matched-stage-training.v1",
    "steps": 3600,
    "batch_size": 16,
    "warmup_steps": 600,
    "peak_learning_rate": 5e-6,
    "decay_learning_rate": 1e-6,
    "save_interval": 600,
    "seed": 0,
    "num_flow_samples": 4,
    "num_workers": 12,
    "subtask_loss_weight": 0.0,
    "fast_loss_weight": 0.0,
    "training_examples_per_task": 19200,
    "training_data": "released_2026_tasks_0_1_22_stride20_training_split_only",
    "selection_rule": "minimum_equal_task_mean_action_loss_then_earliest_step",
}


def _git_identity(source: Path) -> tuple[str, str]:
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    return revision, dirty


def _verify_lerobot() -> None:
    metadata = importlib.metadata.distribution("lerobot")
    lerobot = json.loads(metadata.read_text("direct_url.json"))
    if lerobot.get("vcs_info", {}).get("commit_id") == LEROBOT_REVISION:
        return
    location = urlparse(lerobot.get("url", ""))
    if location.scheme != "file":
        raise ValueError("Training requires the verified BEHAVIOR LeRobot fork")
    actual, dirty = _git_identity(Path(unquote(location.path)))
    if actual != LEROBOT_REVISION or dirty:
        raise ValueError("Training requires the clean verified BEHAVIOR LeRobot fork")


def verify_source(root: Path) -> dict:
    """Verify the native RLC, OpenPI, BEHAVIOR, and LeRobot revisions.

    Args:
        root: Native RLC source checkout.

    Returns:
        Verified revision mapping.

    Raises:
        OSError: Git metadata cannot be read.
        ValueError: A source revision or worktree differs.
    """

    for relative, revision in EXPECTED_REVISIONS.items():
        actual, dirty = _git_identity(root / relative)
        if actual != revision or dirty:
            raise ValueError(f"Training requires clean pinned source at {relative}")
    _verify_lerobot()
    return {**EXPECTED_REVISIONS, "lerobot": LEROBOT_REVISION}


def load_config(path: Path) -> dict:
    """Load one frozen matched-arm configuration.

    Args:
        path: Candidate JSON configuration.

    Returns:
        Validated configuration payload.

    Raises:
        OSError: The file cannot be read.
        ValueError: Fields or frozen settings differ.
    """
    config = json.loads(path.read_text())
    expected = {
        "schema",
        "arm",
        "candidate",
        "steps",
        "batch_size",
        "warmup_steps",
        "peak_learning_rate",
        "decay_learning_rate",
        "save_interval",
        "seed",
        "num_flow_samples",
        "num_workers",
        "subtask_loss_weight",
        "fast_loss_weight",
        "training_examples_per_task",
        "training_data",
        "selection_rule",
    }
    if set(config) != expected:
        raise ValueError("Candidate configuration fields differ")
    for key, value in COMMON_CONFIG.items():
        if config.get(key) != value:
            raise ValueError(f"candidate setting differs: {key}")
    expected_names = {
        "teacher": "stage_teacher_action_only_3600",
        "replay": "stage_parent_replay_action_only_3600",
    }
    if config.get("candidate") != expected_names.get(config.get("arm")):
        raise ValueError("candidate name and arm differ")
    return config


def verify_inputs(args, config: dict) -> dict:
    """Verify the frozen split, dataset, trace, and adapter inputs.

    Args:
        args: Runtime paths supplied by the workflow.
        config: Validated candidate configuration.

    Returns:
        Validated input manifest.

    Raises:
        OSError: An input cannot be read.
        ValueError: Any identity or dataset contract differs.
    """

    manifest = json.loads(args.input_manifest.read_text())
    if manifest.get("schema") != "npa.behavior.rlc-matched-stage-inputs.v1":
        raise ValueError("Training input manifest schema differs")
    if manifest.get("parent_checkpoint_archive_sha256") != PARENT_ARCHIVE_SHA256:
        raise ValueError("Parent checkpoint archive identity differs")
    local = {
        "config.json": args.config,
        "episode-split.json": args.episode_split,
        "source-manifest.json": args.source_manifest,
        "dataset-validation.json": args.dataset_validation,
        "training-trace.jsonl": args.training_trace,
    }
    for name, path in local.items():
        if digest(path) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Frozen input differs: {name}")
    if digest(args.config) != manifest["candidate_config_sha256"]:
        raise ValueError("Candidate configuration identity differs")
    split = load_split(args.episode_split)
    if split["training_episodes"] != 540 or split["holdout_episodes"] != 60:
        raise ValueError("Panel split totals differ")
    validation = json.loads(args.dataset_validation.read_text())
    if validation.get("split") != {"training": 540, "holdout": 60}:
        raise ValueError("Dataset validation split differs")
    if (
        validation.get("depth_included") is not False
        or validation.get("annotations_included") is not False
    ):
        raise ValueError("Privileged dataset features are not excluded")
    _verify_dataset_files(args, manifest)
    _verify_adapter_files(args.adapter_root, manifest["adapter_files"])
    return manifest


def _verify_dataset_files(args, manifest: dict) -> None:
    source = args.dataset_root.parent / "source"
    source_manifest = json.loads(args.source_manifest.read_text())
    if source_manifest.get("revision") != manifest["dataset_revision"]:
        raise ValueError("Dataset source revision differs")
    for row in source_manifest["files"]:
        path = source / row["path"]
        if not path.is_file() or path.stat().st_size != row["size"]:
            raise ValueError(f"Staged dataset file size differs: {row['path']}")
        if digest(path) != row["verified_sha256"]:
            raise ValueError(f"Staged dataset file hash differs: {row['path']}")


def _verify_adapter_files(root: Path, expected: dict[str, str]) -> None:
    for relative, expected_digest in expected.items():
        if digest(root / relative) != expected_digest:
            raise ValueError(f"Policy adapter source differs: {relative}")


def verify_parent_checkpoint(
    archive: Path,
    checkpoint: Path,
    expected_archive_sha256: str = PARENT_ARCHIVE_SHA256,
) -> dict[str, str]:
    """Match every loaded parent-checkpoint byte to the pinned release archive."""
    if digest(archive) != expected_archive_sha256:
        raise ValueError("Parent checkpoint archive differs from the pinned release")
    checkpoint = checkpoint.resolve()
    files = {}
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            relative, member_digest = _verify_checkpoint_member(
                source, member, checkpoint, files
            )
            files[relative] = member_digest
    actual = {
        str(path.relative_to(checkpoint))
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    if actual != set(files) or NORMALIZATION not in files:
        raise ValueError("Parent checkpoint file set or normalization asset differs")
    return files


def _verify_checkpoint_member(source, member, checkpoint: Path, files: dict):
    relative = member.filename.removeprefix(PARENT_CHECKPOINT_PREFIX)
    target = (checkpoint / relative).resolve()
    if (
        relative == member.filename
        or not target.is_relative_to(checkpoint)
        or relative in files
    ):
        raise ValueError("Unexpected parent checkpoint archive layout")
    with source.open(member) as stream:
        member_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if not target.is_file() or digest(target) != member_digest:
        raise ValueError("Loaded parent checkpoint bytes differ from release archive")
    return relative, member_digest


def _data_configuration(base, args, b1k_config):
    return dataclasses.replace(
        base.data,
        assets=b1k_config.AssetsConfig(
            assets_dir=str(args.checkpoint / "assets"),
            asset_id="IliaLarchenko/behavior_224_rgb",
        ),
        base_config=dataclasses.replace(
            base.data.base_config,
            behavior_dataset_root=str(args.dataset_root),
            episodes_index=None,
        ),
        use_fast_tokenization=False,
    )


def training_configuration(args, config: dict):
    """Build the pinned native action-only training configuration.

    Args:
        args: Checkpoint, dataset, and output paths.
        config: Validated arm hyperparameters.

    Returns:
        Native RLC training configuration.
    """

    from b1k.training import config as b1k_config
    from b1k.training import weight_loaders

    base = b1k_config.get_config("pi_behavior_b1k_fast")
    return dataclasses.replace(
        base,
        exp_name=config["candidate"],
        model=dataclasses.replace(
            base.model,
            fast_loss_weight=config["fast_loss_weight"],
            subtask_loss_weight=config["subtask_loss_weight"],
            use_knowledge_insulation=True,
        ),
        data=_data_configuration(base, args, b1k_config),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(args.checkpoint / "params")
        ),
        freeze_filter=freeze_filter(),
        lr_schedule=_learning_rate_schedule(config),
        batch_size=config["batch_size"],
        num_train_steps=config["steps"],
        num_flow_samples=config["num_flow_samples"],
        num_workers=config["num_workers"],
        seed=config["seed"],
        wandb_enabled=False,
        checkpoint_base_dir=str(args.output_root / "checkpoints"),
        assets_base_dir=str(args.output_root / "assets"),
        save_interval=config["save_interval"],
        keep_period=config["save_interval"],
    )


def _learning_rate_schedule(config: dict):
    from openpi.training.optimizer import CosineDecaySchedule

    return CosineDecaySchedule(
        warmup_steps=config["warmup_steps"],
        peak_lr=config["peak_learning_rate"],
        decay_steps=config["steps"],
        decay_lr=config["decay_learning_rate"],
    )


def _preserve_frozen_ema(previous, current, traverse_util):
    """Keep non-action EMA leaves at their exact pre-update values."""
    if previous.ema_params is None or current.ema_params is None:
        return current
    old = traverse_util.flatten_dict(previous.ema_params.to_pure_dict())
    new = traverse_util.flatten_dict(current.ema_params.to_pure_dict())
    if set(old) != set(new):
        raise ValueError("EMA parameter paths changed during update")
    for path in new:
        normalized = tuple(str(part) for part in path)
        if normalized not in EXPECTED_TRAINABLE_PATHS:
            new[path] = old[path]
    ema_params = current.ema_params
    ema_params.replace_by_pure_dict(traverse_util.unflatten_dict(new))
    return dataclasses.replace(current, ema_params=ema_params)


def guarded_update(update):
    """Wrap a native update with finite loss, gradient, and parameter checks.

    Args:
        update: Native train-step function.

    Returns:
        Train-step function with a device-to-host finite-value assertion.

    Raises:
        FloatingPointError: An update produces a non-finite metric.
    """

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import traverse_util

    def require_finite(*values):
        if not np.isfinite(np.asarray(values)).all():
            raise FloatingPointError(
                "Non-finite RLC training loss, gradient, or parameters"
            )

    def checked(config, rng, state, batch):
        updated, metrics = update(config, rng, state, batch)
        updated = _preserve_frozen_ema(state, updated, traverse_util)
        jax.debug.callback(
            require_finite, metrics["loss"], metrics["grad_norm"], metrics["param_norm"]
        )
        return updated, _logger_compatible_metrics(metrics, jnp)

    return checked


def _logger_compatible_metrics(metrics: dict, array_module) -> dict:
    """Return only scalar numeric metrics in the native logger's dtype."""
    compatible = {}
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        array = array_module.asarray(value)
        if array.ndim != 0:
            raise ValueError(f"Training metric {name!r} is not scalar")
        compatible[name] = array.astype(array_module.float32)
    return compatible


def _load_trainer(source_root: Path):
    path = source_root / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("rlc_native_panel_train", path)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    trainer.train_step = guarded_update(trainer.train_step)
    return trainer


def _changed_paths(jax, jnp, before: dict, after: dict) -> set[tuple[str, ...]]:
    if set(before) != set(after):
        raise ValueError("Parameter paths changed during the real update")
    changed = set()
    for path, value in before.items():
        equal = bool(jax.device_get(jnp.array_equal(value, after[path])))
        normalized = tuple(str(part) for part in path)
        if not equal:
            changed.add(normalized)
        if normalized not in EXPECTED_TRAINABLE_PATHS and not equal:
            raise ValueError(
                f"Frozen parameter changed during real update: {normalized}"
            )
    return changed


def _write_update_receipt(
    output: Path,
    changed: set,
    changed_ema: set,
    metrics: dict,
    formatted: str,
) -> None:
    checked = {
        name: float(metrics[name]) for name in ("loss", "grad_norm", "param_norm")
    }
    if not np.isfinite(np.asarray(list(checked.values()))).all():
        raise FloatingPointError("Real update metrics are non-finite")
    receipt = {
        "schema": "npa.behavior.matched-stage-real-update.v1",
        "changed_trainable_paths": sorted("/".join(path) for path in changed),
        "frozen_paths_byte_equal": True,
        "changed_ema_trainable_paths": sorted(
            "/".join(path) for path in changed_ema
        ),
        "frozen_ema_paths_byte_equal": True,
        "native_formatter_line": formatted,
        "native_formatter_updates": 2,
        "metrics": checked,
        "status": "passed",
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _validate_action_and_ema_update(state, updated) -> tuple[set, set]:
    """Check the real parameter and export-EMA changes against the action partition."""
    import jax
    import jax.numpy as jnp
    from flax import traverse_util

    before = traverse_util.flatten_dict(state.params.to_pure_dict())
    after = traverse_util.flatten_dict(updated.params.to_pure_dict())
    changed = _changed_paths(jax, jnp, before, after)
    if not changed or not changed.issubset(EXPECTED_TRAINABLE_PATHS):
        raise ValueError("Real update did not remain inside the exact action partition")
    if state.ema_params is None or updated.ema_params is None:
        raise ValueError("Real update lacks the EMA used for export")
    ema_before = traverse_util.flatten_dict(state.ema_params.to_pure_dict())
    ema_after = traverse_util.flatten_dict(updated.ema_params.to_pure_dict())
    changed_ema = _changed_paths(jax, jnp, ema_before, ema_after)
    if not changed_ema.issubset(EXPECTED_TRAINABLE_PATHS):
        raise ValueError("EMA update did not remain inside the exact action partition")
    return changed, changed_ema


def validate_real_update(training, trainer, output: Path) -> None:
    """Prove action-only updates and the native two-update logging contract.

    Args:
        training: Pinned native training configuration.
        trainer: Native trainer with guarded updates installed.
        output: Destination for the successful GPU validation receipt.
    Returns:
        None.
    Raises:
        ValueError: The update changes frozen parameters or lacks export EMA.
        FloatingPointError: The update produces non-finite metrics.
    """
    import jax
    from b1k.training import data_loader
    from openpi.training import sharding

    mesh = sharding.make_mesh(training.fsdp_devices)
    loader = data_loader.create_behavior_data_loader(training, shuffle=True)
    iterator = iter(loader)
    batch = next(iterator)
    state, _ = trainer.init_train_state(
        training, jax.random.key(training.seed), mesh,
        resume=False, norm_stats=loader.data_config().norm_stats,
    )
    update = jax.jit(functools.partial(trainer.train_step, training))
    updated, metrics = update(jax.random.key(training.seed), state, batch)
    changed, changed_ema = _validate_action_and_ema_update(state, updated)
    second, second_metrics = update(
        jax.random.key(training.seed), updated, next(iterator)
    )
    del second
    device_metrics = jax.device_get(metrics)
    second_device_metrics = jax.device_get(second_metrics)
    formatted = _format_native_metrics([device_metrics, second_device_metrics])
    _write_update_receipt(
        output, changed, changed_ema, device_metrics, formatted
    )


def _format_native_metrics(metrics: list[dict]) -> str:
    """Exercise the pinned trainer's metric reduction and format contract."""
    import jax
    import jax.numpy as jnp
    from flax.training import common_utils

    stacked = common_utils.stack_forest(metrics)
    reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
    return _format_reduced_metrics(reduced)


def _format_reduced_metrics(reduced: dict) -> str:
    """Format the same reduced metric subset as the pinned native trainer."""
    main = {
        key: value
        for key, value in reduced.items()
        if "loss" in key
        or "accuracy" in key
        or key
        in {
            "grad_norm",
            "param_norm",
            "grad_norm_vlm",
            "grad_norm_action_expert",
        }
    }
    return ", ".join(f"{key}={value:.4f}" for key, value in main.items())


def _training_provenance(args, config, manifest, revisions, checkpoint_files) -> dict:
    planned_samples = config["steps"] * config["batch_size"]
    return {
        "schema": "npa.behavior.rlc-panel-training-run.v1",
        "candidate": config,
        "source_revisions": revisions,
        "input_manifest_sha256": digest(args.input_manifest),
        "parent_checkpoint_archive_sha256": digest(args.checkpoint_archive),
        "parent_checkpoint_files": checkpoint_files,
        "dataset_revision": manifest["dataset_revision"],
        "training_episodes": 540,
        "holdout_episodes": 60,
        "holdout_selection": config["selection_rule"],
        "development_or_reporting_data_used": False,
        "losses": "flow matching only; stage and FAST auxiliary weights are zero",
        "adaptation": "exact 23-leaf action path; task/stage/classifier modules frozen",
        "normalization": "unchanged published checkpoint statistics",
        "sampler": {
            "algorithm": "seeded task-balanced stride-20 replan frames",
            "seed": config["seed"],
            "planned_samples": planned_samples,
            "planned_task_counts": balanced_prefix_counts(planned_samples),
        },
    }


def _load_training_trace(args, manifest: dict) -> dict:
    trace = load_trace(
        args.training_trace,
        split="training",
        identities=manifest["trace_identities"],
    )
    if len(trace) != 19388 + 47820 + 69882:
        raise ValueError("training replay cardinality differs")
    return trace


def run(args: argparse.Namespace) -> None:
    """Validate inputs, run the real-update gate, and start native training.

    Args:
        args: Parsed training paths and candidate configuration.

    Raises:
        FileExistsError: The output directory already exists.
        OSError: A required source or input cannot be read.
        ValueError: A frozen contract or real-update gate fails.
    """

    config = load_config(args.config)
    manifest = verify_inputs(args, config)
    revisions = verify_source(args.source_root)
    checkpoint_files = verify_parent_checkpoint(
        args.checkpoint_archive, args.checkpoint
    )
    trace = _load_training_trace(args, manifest)
    args.output_root.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled"
    )
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(args.adapter_root))
    from rlc_server import _policy_source

    with tempfile.TemporaryDirectory(prefix="npa-rlc-panel-") as temporary:
        _policy_source(args.source_root, Path(temporary))
        _run_native_training(args, config, trace, manifest, revisions, checkpoint_files)


def _run_native_training(args, config, trace, manifest, revisions, checkpoint_files):
    from b1k.training import data_loader

    install_balanced_loader(data_loader, arm=config["arm"], trace=trace)
    training = training_configuration(args, config)
    provenance = _training_provenance(
        args, config, manifest, revisions, checkpoint_files
    )
    (args.output_root / "training-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    trainer = _load_trainer(args.source_root)
    validate_real_update(
        training, trainer, args.output_root / "real-update-preflight.json"
    )
    logging.basicConfig(level=logging.INFO)
    trainer.main(training)


def arguments() -> argparse.Namespace:
    """Parse matched-training command-line arguments.

    Returns:
        Parsed path arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "adapter-root",
        "checkpoint",
        "checkpoint-archive",
        "dataset-root",
        "output-root",
        "config",
        "input-manifest",
        "episode-split",
        "source-manifest",
        "dataset-validation",
        "training-trace",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
