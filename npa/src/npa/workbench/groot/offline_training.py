"""Run feedback-weighted flow matching in an isolated native GR00T N1.5 runtime."""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from npa.workbench.groot.offline_feedback import (
    OfflineManifest,
    dataset_digest,
    file_digest,
)

GROOT_N15_REF = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"


class FeedbackDataset:
    """Select manifest frames and mask actions beyond each recorded episode.

    Args:
        native: Native N1.5 LeRobotSingleDataset instance.
        manifest: Validated weights and split membership.
        split: Requested split; evaluation weights are always one.
    Returns:
        None.
    Raises:
        ValueError: Native frame identities differ from the prepared manifest.
    """

    def __init__(self, native, manifest: OfflineManifest, split: str = "train"):
        self.native = native
        self.horizon = manifest.action_horizon
        episodes = {episode.episode_index: episode for episode in manifest.episodes}
        expected = {
            (e.episode_index, frame)
            for e in manifest.episodes
            for frame in range(e.length)
        }
        steps = [(int(episode), int(frame)) for episode, frame in native.all_steps]
        if len(steps) != len(expected) or set(steps) != expected:
            raise ValueError("native dataset frames differ from prepared manifest")
        self.samples = [
            (index, episodes[episode], frame)
            for index, (episode, frame) in enumerate(steps)
            if episodes[episode].split == split
        ]
        if not self.samples:
            raise ValueError("requested split contains no frames")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        native_index, episode, frame = self.samples[index]
        sample = dict(self.native[native_index])
        mask = np.asarray(sample["action_mask"]).copy()
        if mask.ndim != 2 or mask.shape[0] != self.horizon:
            raise ValueError("native action_mask differs from manifest action horizon")
        valid = np.arange(self.horizon) + frame < episode.length
        sample["action_mask"] = mask * valid[:, None]
        sample["offline_weight"] = episode.weights[frame]
        return sample

    def set_epoch(self, epoch):
        """Forward epoch changes to native transforms and sampling state.

        Args:
            epoch: Native sampler epoch.
        Returns:
            None.
        Raises:
            None.
        """
        self.native.set_epoch(epoch)


class FeedbackCollator:
    """Keep per-chunk weights outside native image and language preprocessing.

    Args:
        native: Native DefaultDataCollator instance.
    Returns:
        None.
    Raises:
        None.
    """

    def __init__(self, native):
        self.native = native

    def __call__(self, features):
        import torch

        weights = torch.tensor(
            [sample["offline_weight"] for sample in features], dtype=torch.float32
        )
        batch = self.native(
            [
                {k: v for k, v in sample.items() if k != "offline_weight"}
                for sample in features
            ]
        )
        batch["offline_weight"] = weights
        return batch


def weighted_flow_loss(model, inputs):
    """Weight native flow losses without changing actions or checkpoint structure.

    Args:
        model: Callable native GR00T N1.5 model.
        inputs: Collated action masks and one offline_weight per sample.
    Returns:
        Scalar loss and native output mapping. Weights retain their effect at batch size one.
    Raises:
        ValueError: Weights or masks are malformed or have no valid actions.
    """

    batch = dict(inputs)
    weights = batch.pop("offline_weight").detach().float()
    mask = batch["action_mask"].float()
    _validate_loss_inputs(weights, mask, batch["action"].shape)
    weights = weights.to(mask.device).reshape(-1, 1, 1)
    weighted_mask = mask * weights
    mass = weighted_mask.sum()
    if mass.item() == 0:
        outputs = model(batch)
        return outputs["loss"] * 0, outputs
    batch["action_mask"] = weighted_mask
    outputs = model(batch)
    # Native reduction divides by weighted mask mass; restore the unweighted
    # denominator so batch size one and gradient accumulation preserve weighting.
    loss = outputs["loss"] * (mass / mask.sum())
    return loss, outputs


def _validate_loss_inputs(weights, mask, action_shape):
    import torch

    if mask.ndim != 3 or mask.shape != action_shape:
        raise ValueError("action_mask must match [batch, horizon, action_dim]")
    if weights.ndim != 1 or len(weights) != mask.shape[0]:
        raise ValueError("offline_weight must contain one weight per sample")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("offline_weight must be finite and nonnegative")
    if not ((mask == 0) | (mask == 1)).all() or mask.sum().item() <= 0:
        raise ValueError("action_mask must be binary with at least one valid action")


def train_offline(
    *,
    dataset_root: Path,
    manifest_path: Path,
    groot_source: Path,
    base_model: Path,
    output_dir: Path,
    data_config: str,
    embodiment_tag: str,
    max_steps: int,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Fine-tune a staged N1.5 checkpoint using only prepared training frames.

    Args:
        dataset_root: Immutable, locally staged dataset.
        manifest_path: Prepared content-bound feedback manifest.
        groot_source: Clean checkout at GROOT_N15_REF, with its dependencies installed.
        base_model: Locally staged N1.5 checkpoint.
        output_dir: New directory for checkpoints, metadata, and training receipt.
        data_config: Native built-in or module:ClassName modality configuration.
        embodiment_tag: Native embodiment tag.
        max_steps: Operator-selected optimizer step count.
        batch_size: Per-device batch size.
        learning_rate: Positive optimizer learning rate.
        seed: Training seed.
    Returns:
        Receipt identifying the trained checkpoint, without claiming robot evaluation.
    Raises:
        ValueError: Inputs, source pin, checkpoint, or CUDA runtime are incompatible.
        OSError: Dataset, checkpoint, or output cannot be accessed.
    """
    manifest = OfflineManifest.model_validate_json(manifest_path.read_text())
    _validate_training_paths(dataset_root, manifest_path, base_model, output_dir)
    if dataset_digest(dataset_root) != manifest.dataset_sha256:
        raise ValueError("dataset bytes changed after feedback preparation")
    _validate_schedule(max_steps, batch_size, learning_rate)
    runtime = _load_native_runtime(groot_source)
    base_digest = checkpoint_digest(base_model)
    manifest_digest = file_digest(manifest_path)
    runtime["set_seed"](seed)
    train_dataset, eval_dataset, config = _load_training_dataset(
        runtime, dataset_root, manifest, data_config, embodiment_tag
    )
    terminal_mask = _terminal_mask_evidence(train_dataset)
    model = _load_model(runtime, base_model, config, manifest.action_horizon)
    trainer = _create_trainer(
        runtime,
        model,
        train_dataset,
        eval_dataset,
        output_dir,
        max_steps,
        batch_size,
        learning_rate,
        seed,
    )
    result = trainer.train()
    training_trace = _training_trace(trainer)
    evaluation = trainer.evaluate()
    trainer.save_model(str(output_dir), _internal_call=False)
    trainer.save_state()
    reload_evidence = _verify_candidate_reload(
        runtime, output_dir, config, manifest.action_horizon
    )
    training_config = {
        "data_config": data_config,
        "embodiment_tag": embodiment_tag,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    return _write_training_receipt(
        output_dir,
        base_digest,
        manifest_digest,
        manifest,
        result,
        training_trace,
        evaluation,
        training_config,
        runtime["visible_cuda_devices"],
        reload_evidence,
        terminal_mask,
    )


def _validate_training_paths(dataset_root, manifest_path, base_model, output_dir):
    for source in (dataset_root, base_model):
        if output_dir.resolve().is_relative_to(
            source.resolve()
        ) or source.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(
                "output directory must be separate from dataset and base checkpoint"
            )
    if manifest_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("manifest must be outside the new output directory")
    if output_dir.exists():
        raise ValueError(
            "output directory must be new; use a new directory for each iteration"
        )
    config = json.loads((base_model / "config.json").read_text())
    if config.get("model_type") != "gr00t_n1_5":
        raise ValueError("base checkpoint must be GR00T N1.5")


def _validate_schedule(max_steps, batch_size, learning_rate):
    if (
        type(max_steps) is not int
        or max_steps <= 0
        or type(batch_size) is not int
        or batch_size <= 0
    ):
        raise ValueError("max_steps and batch_size must be positive integers")
    if (
        isinstance(learning_rate, bool)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be positive and finite")


def _load_native_runtime(source):
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("experimental N1.5 offline training supports one GPU per run")
    reference = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--", "gr00t"], text=True
    )
    if reference != GROOT_N15_REF or status.strip():
        raise ValueError("groot_source must contain clean pinned N1.5 source")
    sys.path.insert(0, str(source.resolve()))
    import torch
    from transformers import TrainingArguments, set_seed

    visible_cuda_devices = _validate_cuda_runtime(torch)
    modules = {}
    for name in (
        "data.dataset",
        "data.schema",
        "experiment.data_config",
        "experiment.trainer",
        "model.gr00t_n1",
        "model.transforms",
        "utils.experiment",
    ):
        module = importlib.import_module(f"gr00t.{name}")
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise ValueError("installed GR00T resolves outside the pinned checkout")
        modules.update(vars(module))
    modules.update(
        TrainingArguments=TrainingArguments,
        set_seed=set_seed,
        visible_cuda_devices=visible_cuda_devices,
    )
    return modules


def _validate_cuda_runtime(torch):
    """Require exactly one visible BF16-capable CUDA device for this objective."""

    if not torch.cuda.is_available():
        raise ValueError("N1.5 policy training requires a configured CUDA GPU runtime")
    visible_cuda_devices = torch.cuda.device_count()
    if visible_cuda_devices != 1:
        raise ValueError(
            "experimental N1.5 offline training requires exactly one visible CUDA GPU"
        )
    if not torch.cuda.is_bf16_supported():
        raise ValueError("N1.5 offline training requires BF16 CUDA support")
    return visible_cuda_devices


def _load_training_dataset(runtime, root, manifest, data_config, embodiment_tag):
    statistics = json.loads((root / "meta/stats.json").read_text())
    if not isinstance(statistics, dict) or not statistics:
        raise ValueError("precomputed frozen meta/stats.json is required")
    for values in statistics.values():
        runtime["DatasetStatisticalValues"].model_validate(values)
    config = runtime["load_data_config"](data_config)
    if list(config.action_indices) != list(range(manifest.action_horizon)):
        raise ValueError(
            "data_config action indices must match contiguous manifest horizon"
        )
    native = runtime["LeRobotSingleDataset"](
        dataset_path=str(root),
        modality_configs=config.modality_config(),
        transforms=config.transform(),
        embodiment_tag=runtime["EmbodimentTag"](embodiment_tag),
    )
    return (
        FeedbackDataset(native, manifest, split="train"),
        FeedbackDataset(native, manifest, split="eval"),
        config,
    )


def _load_model(runtime, base_model, config, horizon):
    model = runtime["GR00T_N1_5"].from_pretrained(
        str(base_model.resolve()),
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
    )
    transform = config.transform().transforms[-1]
    if model.action_dim != transform.max_action_dim or model.action_horizon != horizon:
        raise ValueError(
            "checkpoint dimensions and action horizon must match data_config and manifest"
        )
    if model.action_head.config.expand_batch is not None:
        raise ValueError("N1.5 expand_batch is unsupported for offline weighting")
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    return model


def _terminal_mask_evidence(dataset):
    """Decode one terminal observation and prove padding actions are excluded."""

    for position, (_, episode, frame) in enumerate(dataset.samples):
        if frame != episode.length - 1:
            continue
        sample = dataset[position]
        mask = np.asarray(sample["action_mask"])
        row_is_valid = np.any(mask != 0, axis=1)
        expected = np.arange(dataset.horizon) + frame < episode.length
        if mask.ndim != 2 or not np.array_equal(row_is_valid, expected):
            raise ValueError(
                "terminal action mask differs from recorded episode boundary"
            )
        decoded = {
            key: list(value.shape)
            for key, value in sample.items()
            if (key == "video" or key.startswith("video.")) and hasattr(value, "shape")
        }
        if not decoded:
            raise ValueError("native reader did not decode an observation video")
        return {
            "episode_index": episode.episode_index,
            "frame_index": frame,
            "valid_action_horizon_steps": int(row_is_valid.sum()),
            "masked_action_horizon_steps": int((~row_is_valid).sum()),
            "decoded_observation_shapes": decoded,
        }
    raise ValueError("training split has no terminal frame")


def _create_trainer(
    runtime,
    model,
    train_dataset,
    eval_dataset,
    output,
    steps,
    batch_size,
    learning_rate,
    seed,
):
    class WeightedTrainer(runtime["DualBrainTrainer"]):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss, outputs = weighted_flow_loss(model, inputs)
            return (loss, outputs) if return_outputs else loss

    output.mkdir(parents=True)
    metadata_dir = output / "experiment_cfg"
    metadata_dir.mkdir()
    metadata = {
        train_dataset.native.tag: train_dataset.native.metadata.model_dump(mode="json")
    }
    (metadata_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    args = runtime["TrainingArguments"](
        output_dir=str(output),
        remove_unused_columns=False,
        report_to=[],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        bf16=True,
        dataloader_num_workers=0,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        ddp_find_unused_parameters=False,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=FeedbackCollator(runtime["DefaultDataCollator"]()),
        compute_dtype=model.dtype,
    )
    trainer.add_callback(
        runtime["CheckpointFormatCallback"](
            run_name=output.name, exp_cfg_dir=metadata_dir
        )
    )
    return trainer


def _training_trace(trainer):
    """Extract the native Trainer's recorded finite optimizer-loss trace."""

    trace = []
    for record in trainer.state.log_history:
        loss = record.get("loss")
        step = record.get("step")
        if isinstance(loss, (float, int)) and isinstance(step, (float, int)):
            if not math.isfinite(loss) or not math.isfinite(step) or step < 1:
                raise ValueError("trainer logged a nonfinite optimizer trace value")
            trace.append({"step": int(step), "loss": float(loss)})
    if not trace:
        raise ValueError("trainer did not record an optimizer-loss trace")
    if [point["step"] for point in trace] != sorted(point["step"] for point in trace):
        raise ValueError("trainer optimizer-loss trace is not step ordered")
    return trace


def _verify_candidate_reload(runtime, output, config, horizon):
    """Reload the emitted native checkpoint with the inference model contract."""

    reloaded = _load_model(runtime, output, config, horizon)
    try:
        return {
            "verified": True,
            "model_class": type(reloaded).__name__,
            "action_dim": int(reloaded.action_dim),
            "action_horizon": int(reloaded.action_horizon),
            "checkpoint_sha256": checkpoint_digest(output),
        }
    finally:
        del reloaded


def _write_training_receipt(
    output,
    base_digest,
    manifest_digest,
    manifest,
    result,
    training_trace,
    evaluation,
    training_config,
    visible_cuda_devices,
    reload_evidence,
    terminal_mask,
):
    if not math.isfinite(result.training_loss):
        raise ValueError("training produced a nonfinite loss")
    if result.global_step != training_config["max_steps"]:
        raise ValueError("training did not complete the requested optimizer steps")
    if training_trace[-1]["step"] != result.global_step:
        raise ValueError(
            "optimizer-loss trace does not reach the completed training step"
        )
    eval_loss = evaluation.get("eval_loss")
    if not isinstance(eval_loss, (float, int)) or not math.isfinite(eval_loss):
        raise ValueError("held-out evaluation produced a nonfinite loss")
    train_weights = [
        weight
        for episode in manifest.episodes
        if episode.split == "train"
        for weight in episode.weights
    ]
    receipt = {
        "schema_version": "npa.groot-offline-training/v1",
        "training_config": training_config,
        "objective": (
            "uniform-behavior-cloning"
            if manifest.mode == "uniform"
            else f"{manifest.mode}-weighted-flow-matching"
        ),
        "upstream_ref": GROOT_N15_REF,
        "dataset_sha256": manifest.dataset_sha256,
        "manifest_sha256": manifest_digest,
        "base_checkpoint_sha256": base_digest,
        "candidate_checkpoint_sha256": checkpoint_digest(output),
        "optimizer_steps": result.global_step,
        "training_loss": result.training_loss,
        "optimizer_loss_trace": training_trace,
        "train_episodes": sum(e.split == "train" for e in manifest.episodes),
        "eval_episodes": sum(e.split == "eval" for e in manifest.episodes),
        "train_frames": sum(e.length for e in manifest.episodes if e.split == "train"),
        "eval_frames": sum(e.length for e in manifest.episodes if e.split == "eval"),
        "held_out_eval_loss": eval_loss,
        "visible_cuda_devices": visible_cuda_devices,
        "checkpoint_reload": reload_evidence,
        "training_weight_statistics": {
            "minimum": min(train_weights),
            "maximum": max(train_weights),
            "mean": sum(train_weights) / len(train_weights),
        },
        "terminal_mask_evidence": terminal_mask,
        "robot_evaluation_performed": False,
    }
    (output / "offline-training.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def checkpoint_digest(checkpoint: Path) -> str:
    """Bind checkpoint identity to config and all serialized model weight files.

    Args:
        checkpoint: Native checkpoint directory.
    Returns:
        SHA-256 identifying configuration and model weights.
    Raises:
        ValueError: No supported weight files exist.
        OSError: Checkpoint files cannot be read.
    """
    import hashlib

    files = sorted(checkpoint.glob("*.safetensors")) + sorted(
        checkpoint.glob("pytorch_model*.bin")
    )
    if not files:
        raise ValueError("checkpoint has no native model weights")
    digest = hashlib.sha256()
    metadata = checkpoint / "experiment_cfg/metadata.json"
    if metadata.exists():
        files.append(metadata)
    files.extend(sorted(checkpoint.glob("*.safetensors.index.json")))
    files.extend(sorted(checkpoint.glob("pytorch_model*.bin.index.json")))
    for path in [checkpoint / "config.json", *files]:
        relative = path.relative_to(checkpoint).as_posix().encode()
        digest.update(relative + b"\0" + bytes.fromhex(file_digest(path)))
    return digest.hexdigest()
