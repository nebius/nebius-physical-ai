"""Execute the native LeRobot FastWAM qualification stages.

FastWAM trains a video world/action model but selects actions directly at
inference.  This module deliberately calls LeRobot's ``lerobot-train`` and
``lerobot-eval`` entry points instead of reimplementing either behaviour.  It
also keeps checkpoints, Wan/UMT5 dependencies, and datasets out of image
layers: all of those bytes are resolved under the operator's configured runtime
cache from immutable Hugging Face revisions.

The workflow is an offline/simulator qualification.  It does not command a
physical robot; physical deployment needs a separately reviewed robot adapter
and measured safety controls.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from npa.clients.storage import StorageClient
from npa.workflows.lerobot_dataset import (
    LeRobotDatasetError,
    materialize_lerobot_dataset,
    seeded_episode_split,
    summarize_lerobot_dataset,
)

LEROBOT_VERSION = "0.6.1"
LEROBOT_RELEASE_COMMIT = "7e241bd630a3719a56157a497ce5d08f244784f1"
FASTWAM_BASE_REPOSITORY = "lerobot/fastwam_base"
WAN_REPOSITORY = "Wan-AI/Wan2.2-TI2V-5B"
WAN_DIFFUSERS_REPOSITORY = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
UMT5_REPOSITORY = "google/umt5-xxl"
# This is LeRobot 0.6.1's upstream default.  Making it explicit keeps the
# native protocol unchanged while giving the recovery publisher a concrete,
# auditable cadence to protect.
UPSTREAM_CHECKPOINT_SAVE_FREQ = 20_000
RECOVERY_MANIFEST_SCHEMA = "npa.fastwam.recovery.v1"
_RECOVERY_MIRROR_POLL_SECONDS = 10.0


class FastWAMPolicyError(RuntimeError):
    """A FastWAM stage cannot meet its native runtime contract."""


def build_parser() -> argparse.ArgumentParser:
    """Return the parser used by the workflow catalog tool references."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)

    prepare = commands.add_parser("prepare", help="Seal a real LeRobot dataset split.")
    prepare.add_argument("--input-path", required=True)
    prepare.add_argument("--output-path", required=True)
    prepare.add_argument("--dataset-repo-id", required=True)
    prepare.add_argument("--dataset-revision", required=True)
    prepare.add_argument("--dataset-license", required=True)
    prepare.add_argument("--train-fraction", type=float, required=True)
    prepare.add_argument("--seed", type=int, required=True)

    train = commands.add_parser("train", help="Fine-tune upstream FastWAM.")
    train.add_argument("--input-path", required=True)
    train.add_argument("--output-path", required=True)
    _add_runtime_model_args(train)
    train.add_argument("--train-steps", type=int, required=True)
    train.add_argument("--batch-size", type=int, required=True)
    train.add_argument(
        "--checkpoint-save-freq",
        type=int,
        default=UPSTREAM_CHECKPOINT_SAVE_FREQ,
        help="Native LeRobot checkpoint frequency; defaults to the pinned upstream default.",
    )
    train.add_argument("--device", default="cuda")

    rollout = commands.add_parser("rollout", help="Run direct-action native rollouts.")
    rollout.add_argument("--prepared-path", required=True)
    rollout.add_argument("--checkpoint-path", required=True)
    rollout.add_argument("--output-path", required=True)
    _add_runtime_model_args(rollout)
    rollout.add_argument("--environment", required=True)
    rollout.add_argument("--environment-task", required=True)
    rollout.add_argument("--episode-length", type=int, required=True)
    rollout.add_argument("--observation-height", type=int, required=True)
    rollout.add_argument("--observation-width", type=int, required=True)
    rollout.add_argument("--eval-batch-size", type=int, required=True)
    rollout.add_argument("--policy-dtype", required=True)
    rollout.add_argument("--n-action-steps", type=int, required=True)
    rollout.add_argument("--episodes", type=int, required=True)
    rollout.add_argument("--seed", type=int, required=True)
    rollout.add_argument("--device", default="cuda")
    rollout.add_argument("--compile-action-infer", action="store_true")

    evaluate = commands.add_parser(
        "evaluate", help="Measure native success and action latency."
    )
    evaluate.add_argument("--prepared-path", required=True)
    evaluate.add_argument("--training-path", required=True)
    evaluate.add_argument("--rollouts-path", required=True)
    evaluate.add_argument("--output-path", required=True)
    _add_runtime_model_args(evaluate)
    evaluate.add_argument("--warmup-actions", type=int, required=True)
    evaluate.add_argument("--measured-actions", type=int, required=True)
    evaluate.add_argument("--device", default="cuda")

    report = commands.add_parser(
        "report", help="Emit actual RRD and rollout MP4 evidence."
    )
    report.add_argument("--input-path", required=True)
    report.add_argument("--rollouts-path", required=True)
    report.add_argument("--output-path", required=True)
    report.add_argument("--run-id", required=True)
    return parser


def _add_runtime_model_args(parser: argparse.ArgumentParser) -> None:
    """Add immutable runtime model sources shared by model-consuming stages."""

    parser.add_argument("--fastwam-base-revision", required=True)
    parser.add_argument("--wan-revision", required=True)
    parser.add_argument("--wan-diffusers-revision", required=True)
    parser.add_argument("--umt5-revision", required=True)


def main(argv: list[str] | None = None) -> int:
    """Execute one stage, publishing only after its local artifacts validate."""

    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="npa-fastwam-") as temp_dir:
        work = Path(temp_dir)
        output = work / "output"
        if args.stage == "prepare":
            _prepare(args, work, output)
        elif args.stage == "train":
            _train(args, work, output)
        elif args.stage == "rollout":
            _rollout(args, work, output)
        elif args.stage == "evaluate":
            _evaluate(args, work, output)
        else:
            _report(args, work, output)
        _publish_directory(output, args.output_path)
    return 0


def _prepare(args: argparse.Namespace, work: Path, output: Path) -> None:
    """Validate input data and seal a deterministic episode-disjoint recipe."""

    _require_positive_fraction(args.train_fraction)
    source = _materialize_dataset(args.input_path, work / "dataset", args)
    try:
        summary = summarize_lerobot_dataset(
            source,
            source_uri=args.input_path,
            repo_id=args.dataset_repo_id,
            revision=args.dataset_revision,
            license=args.dataset_license,
        )
    except LeRobotDatasetError as exc:
        raise FastWAMPolicyError(f"FastWAM data qualification failed: {exc}") from exc
    train, heldout = seeded_episode_split(
        summary.episode_indices, train_fraction=args.train_fraction, seed=args.seed
    )
    if not summary.loaded_with_lerobot_dataset:
        raise FastWAMPolicyError(
            "LeRobot could not load the submitted dataset: "
            + summary.lerobot_dataset_error
        )
    native_contract = _validate_native_fastwam_dataset_contract(source, summary)
    output.mkdir(parents=True, exist_ok=True)
    info = source / "meta" / "info.json"
    _write_json(
        output / "recipe.json",
        {
            "schema": "npa.fastwam.recipe.v1",
            "policy": "fastwam",
            "inference_contract": "direct-action-prediction; no test-time future-video generation",
            "dataset": summary.to_dict(),
            "dataset_info_sha256": _sha256(info),
            "native_fastwam_feature_contract": native_contract,
            "train_episode_indices": train,
            "heldout_episode_indices": heldout,
            "train_fraction": args.train_fraction,
            "seed": args.seed,
            "training_normalization": "LeRobot statistics over selected training episodes only",
            "upstream": _upstream_provenance(),
            "physical_robot_tested": False,
        },
    )


def _validate_native_fastwam_dataset_contract(
    source: Path, summary: Any
) -> dict[str, Any]:
    """Validate the actual data through LeRobot's FastWAM feature contract.

    FastWAM's config reconstructs image feature names from the dataset metadata,
    so this deliberately validates the native implementation rather than
    hard-coding a particular camera name or raw camera resolution.  Reading one
    sample is also the only reliable way to prove that the supplied dataset
    carries a language task (or already prepared text context) for training.
    """

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    except ImportError as exc:  # pragma: no cover - executed in the LeRobot image
        raise FastWAMPolicyError(
            "FastWAM preparation requires the LeRobot 0.6.1 fastwam runtime"
        ) from exc
    if importlib.metadata.version("lerobot") != LEROBOT_VERSION:
        raise FastWAMPolicyError(
            f"expected LeRobot {LEROBOT_VERSION} for FastWAM preparation"
        )
    dataset = LeRobotDataset(
        repo_id=summary.repo_id,
        root=source,
        episodes=summary.episode_indices[:1],
        revision=summary.revision,
    )
    config = FastWAMConfig()
    config.set_dataset_feature_metadata(dataset.meta.features)
    sample = dataset[0]
    has_context = "context" in sample and "context_mask" in sample
    if not has_context and not sample.get("task"):
        raise FastWAMPolicyError(
            "FastWAM needs a LeRobot task/prompt or precomputed context and context_mask"
        )
    return {
        "action_dim": config.action_dim,
        "proprio_dim": config.proprio_dim,
        "action_horizon": config.action_horizon,
        "image_size": list(config.image_size),
        "image_features": {
            name: list(feature.shape) for name, feature in config.image_features.items()
        },
        "has_task_or_precomputed_context": True,
    }


def _train(args: argparse.Namespace, work: Path, output: Path) -> None:
    """Run the documented upstream ``lerobot-train --policy.type=fastwam`` path."""

    _require_positive(args.train_steps, "train steps")
    _require_positive(args.batch_size, "batch size")
    _require_positive(args.checkpoint_save_freq, "checkpoint save frequency")
    prepared = _materialize_directory(args.input_path, work / "prepared")
    recipe = _read_recipe(prepared)
    dataset = _materialize_dataset(
        recipe["dataset"]["source_uri"], work / "dataset", args, recipe
    )
    _assert_dataset_receipt(dataset, recipe)
    models = _fetch_runtime_models(args)
    training = output / "training"
    recovery = _restore_recoverable_checkpoint(args.output_path, work / "recovery")
    command = _fastwam_train_command(
        args,
        recipe,
        dataset,
        training,
        models,
        resume_checkpoint=recovery,
    )
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    recovery_checkpoints = _run_training_with_checkpoint_mirror(
        command,
        output / "train.log",
        training,
        args.output_path,
    )
    checkpoint = _final_checkpoint(training)
    _assert_checkpoint(checkpoint)
    shutil.copytree(checkpoint, output / "checkpoint")
    _write_json(
        output / "training.json",
        {
            "schema": "npa.fastwam.training.v1",
            "recipe_sha256": _sha256(prepared / "recipe.json"),
            "checkpoint_sha256": _tree_digest(output / "checkpoint"),
            "duration_seconds": time.monotonic() - started,
            "command": command,
            "checkpoint_recovery": {
                "schema": RECOVERY_MANIFEST_SCHEMA,
                "save_freq": args.checkpoint_save_freq,
                "resumed_from": recovery.parent.name if recovery is not None else None,
                "mirrored_checkpoints": recovery_checkpoints,
            },
            "runtime": _runtime_provenance(),
            "runtime_models": _runtime_model_receipt(models, args),
            "upstream": _upstream_provenance(),
        },
    )


def _rollout(args: argparse.Namespace, work: Path, output: Path) -> None:
    """Call native LeRobot evaluation, whose FastWAM policy selects actions directly."""

    _require_positive(args.episodes, "rollout episodes")
    _require_positive(args.episode_length, "rollout episode length")
    _require_positive(args.observation_height, "rollout observation height")
    _require_positive(args.observation_width, "rollout observation width")
    _require_positive(args.eval_batch_size, "rollout evaluation batch size")
    _require_positive(args.n_action_steps, "FastWAM action steps")
    prepared = _materialize_directory(args.prepared_path, work / "prepared")
    recipe = _read_recipe(prepared)
    trained = _materialize_directory(args.checkpoint_path, work / "training")
    checkpoint = trained / "checkpoint"
    _assert_checkpoint(checkpoint)
    models = _fetch_runtime_models(args)
    output.mkdir(parents=True, exist_ok=True)
    native = output / "native-eval"
    command = _fastwam_eval_command(args, checkpoint, native, models)
    _run_command(command, output / "rollout.log")
    info = native / "eval_info.json"
    if not info.is_file():
        raise FastWAMPolicyError("native lerobot-eval did not write eval_info.json")
    evaluation = json.loads(info.read_text(encoding="utf-8"))
    overall = evaluation.get("overall")
    if not isinstance(overall, dict) or int(overall.get("n_episodes", 0)) <= 0:
        raise FastWAMPolicyError("native lerobot-eval produced no completed episodes")
    videos = _video_receipt(native)
    _write_json(
        output / "rollout.json",
        {
            "schema": "npa.fastwam.rollout.v1",
            "recipe_sha256": _sha256(prepared / "recipe.json"),
            "training_checkpoint_sha256": _tree_digest(checkpoint),
            "environment": args.environment,
            "environment_task": args.environment_task,
            "episode_length": args.episode_length,
            "observation_height": args.observation_height,
            "observation_width": args.observation_width,
            "eval_batch_size": args.eval_batch_size,
            "policy_dtype": args.policy_dtype,
            "n_action_steps": args.n_action_steps,
            "evaluation": evaluation,
            "videos": videos,
            "direct_action_inference": True,
            "compile_action_infer": bool(args.compile_action_infer),
            "runtime_models": _runtime_model_receipt(models, args),
            "physical_robot_tested": False,
        },
    )
    # The held-out split is not consumed by the simulator. Keep it in the receipt
    # so the numerical latency stage can prove it read the same sealed data.
    _write_json(
        output / "heldout-contract.json",
        {"heldout_episode_indices": recipe["heldout_episode_indices"]},
    )


def _evaluate(args: argparse.Namespace, work: Path, output: Path) -> None:
    """Numerically inspect success and time real ``select_action`` calls on CUDA."""

    _require_positive(args.warmup_actions, "warmup action count")
    _require_positive(args.measured_actions, "measured action count")
    prepared = _materialize_directory(args.prepared_path, work / "prepared")
    trained = _materialize_directory(args.training_path, work / "training")
    rolled = _materialize_directory(args.rollouts_path, work / "rollouts")
    recipe = _read_recipe(prepared)
    checkpoint = trained / "checkpoint"
    _assert_checkpoint(checkpoint)
    rollout = _read_json(rolled / "rollout.json")
    models = _fetch_runtime_models(args)
    dataset = _materialize_dataset(
        recipe["dataset"]["source_uri"], work / "heldout", args, recipe
    )
    _assert_dataset_receipt(dataset, recipe)
    latency = _measure_direct_action_latency(
        dataset=dataset,
        recipe=recipe,
        checkpoint=checkpoint,
        models=models,
        warmup_actions=args.warmup_actions,
        measured_actions=args.measured_actions,
    )
    overall = rollout["evaluation"]["overall"]
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "evaluation.json",
        {
            "schema": "npa.fastwam.evaluation.v1",
            "recipe_sha256": _sha256(prepared / "recipe.json"),
            "training_sha256": _sha256(trained / "training.json"),
            "rollout_sha256": _sha256(rolled / "rollout.json"),
            "native_task_success": {
                "success_percent": overall.get("pc_success"),
                "successful_episodes": overall.get("n_success"),
                "episodes": overall.get("n_episodes"),
                "success_ci95": overall.get("pc_success_ci95"),
                "environment": rollout["environment"],
            },
            "direct_action_latency": latency,
            "runtime_models": _runtime_model_receipt(models, args),
            "physical_robot_tested": False,
            "limitations": [
                "Simulation task success is not a physical-robot success claim.",
                "Latency is measured after the configured warmup and excludes initial model download and graph compilation.",
            ],
        },
    )


def _report(args: argparse.Namespace, work: Path, output: Path) -> None:
    """Produce a factual Rerun recording from evaluation and decoded native videos."""

    evaluated = _materialize_directory(args.input_path, work / "evaluation")
    rolled = _materialize_directory(args.rollouts_path, work / "rollouts")
    evaluation = _read_json(evaluated / "evaluation.json")
    _ = _read_json(rolled / "rollout.json")
    videos = _video_receipt(rolled / "native-eval")
    if not videos:
        raise FastWAMPolicyError(
            "report requires at least one decodable native rollout MP4"
        )
    output.mkdir(parents=True, exist_ok=True)
    rrd = output / "fastwam.rrd"
    _write_rrd(rrd, args.run_id, evaluation, videos)
    _inspect_rrd(rrd, args.run_id)
    _write_json(
        output / "report.json",
        {
            "schema": "npa.fastwam.report.v1",
            "evaluation_sha256": _sha256(evaluated / "evaluation.json"),
            "rollout_sha256": _sha256(rolled / "rollout.json"),
            "rrd_sha256": _sha256(rrd),
            "videos": videos,
            "native_task_success": evaluation["native_task_success"],
            "direct_action_latency": evaluation["direct_action_latency"],
            "upstream": _upstream_provenance(),
            "credit": "Hugging Face LeRobot FastWAM; see NOTICE-FASTWAM and the run provenance.",
            "physical_robot_tested": False,
        },
    )


def _fastwam_train_command(
    args: argparse.Namespace,
    recipe: dict[str, Any],
    dataset: Path,
    output: Path,
    models: dict[str, Path],
    *,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        "lerobot-train",
        f"--dataset.repo_id={recipe['dataset']['repo_id']}",
        f"--dataset.root={dataset}",
        f"--dataset.revision={recipe['dataset']['revision']}",
        f"--dataset.episodes={json.dumps(recipe['train_episode_indices'])}",
        "--policy.type=fastwam",
        f"--policy.base_model_id={models['fastwam_base']}",
        f"--policy.model_id={models['wan']}",
        f"--policy.text_encoder_model_id={models['wan_diffusers']}",
        f"--policy.tokenizer_model_id={models['umt5']}",
        f"--policy.device={args.device}",
        f"--output_dir={output}",
        f"--steps={args.train_steps}",
        f"--batch_size={args.batch_size}",
        "--save_checkpoint=true",
        f"--save_freq={args.checkpoint_save_freq}",
        "--env_eval_freq=0",
        # This qualification keeps checkpoints in run-scoped object storage.
        # LeRobot otherwise attempts an implicit Hub publication and requires a
        # repository id even though this workflow never publishes one.
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
    ]
    if resume_checkpoint is not None:
        # LeRobot 0.6.1 resumes optimizer, scheduler, RNG, and sampler state
        # from the checkpoint directory selected by this documented CLI pair.
        command.extend(["--resume=true", f"--config_path={resume_checkpoint}"])
    return command


def _fastwam_eval_command(
    args: argparse.Namespace, checkpoint: Path, output: Path, models: dict[str, Path]
) -> list[str]:
    command = [
        "lerobot-eval",
        f"--policy.path={checkpoint}",
        f"--policy.model_id={models['wan']}",
        f"--policy.text_encoder_model_id={models['wan_diffusers']}",
        f"--policy.tokenizer_model_id={models['umt5']}",
        f"--policy.base_model_id={models['fastwam_base']}",
        f"--policy.device={args.device}",
        f"--policy.dtype={args.policy_dtype}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--env.type={args.environment}",
        f"--env.task={args.environment_task}",
        f"--env.episode_length={args.episode_length}",
        f"--env.observation_height={args.observation_height}",
        f"--env.observation_width={args.observation_width}",
        f"--eval.batch_size={args.eval_batch_size}",
        f"--eval.n_episodes={args.episodes}",
        f"--seed={args.seed}",
        f"--output_dir={output}",
    ]
    if args.compile_action_infer:
        command.append("--policy.compile_action_infer=true")
    return command


def _measure_direct_action_latency(
    *,
    dataset: Path,
    recipe: dict[str, Any],
    checkpoint: Path,
    models: dict[str, Path],
    warmup_actions: int,
    measured_actions: int,
) -> dict[str, Any]:
    """Time FastWAM's native direct ``select_action`` method with CUDA events."""

    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    except ImportError as exc:  # pragma: no cover - image contract exercises this
        raise FastWAMPolicyError(
            "FastWAM evaluation needs the LeRobot 0.6 runtime"
        ) from exc
    if importlib.metadata.version("lerobot") != LEROBOT_VERSION:
        raise FastWAMPolicyError(f"expected LeRobot {LEROBOT_VERSION}")
    if not torch.cuda.is_available():
        raise FastWAMPolicyError(
            "direct-action latency requires a working CUDA runtime"
        )
    heldout = list(recipe["heldout_episode_indices"])
    ds = LeRobotDataset(
        repo_id=recipe["dataset"]["repo_id"],
        root=dataset,
        episodes=heldout,
        revision=recipe["dataset"]["revision"],
    )
    if len(ds) <= 0:
        raise FastWAMPolicyError("sealed held-out dataset has no frames")
    config = PreTrainedConfig.from_pretrained(
        checkpoint,
        cli_overrides=[
            f"--model_id={models['wan']}",
            f"--text_encoder_model_id={models['wan_diffusers']}",
            f"--tokenizer_model_id={models['umt5']}",
            f"--base_model_id={models['fastwam_base']}",
        ],
    )
    # ``make_policy`` loads a checkpoint only when the parsed config's
    # ``pretrained_path`` survives serialization. FastWAM deliberately clears
    # that field while saving a checkpoint, so load the exact checkpoint
    # explicitly rather than accidentally constructing a fresh base model.
    config.set_dataset_feature_metadata(ds.meta.features)
    policy = FastWAMPolicy.from_pretrained(
        checkpoint,
        config=config,
        dataset_stats=ds.meta.stats,
        dataset_meta=ds.meta,
    )
    preprocessor, _ = make_pre_post_processors(
        config, pretrained_path=checkpoint, dataset_meta=ds.meta
    )
    item = _add_batch_dimension(ds[0])
    batch = preprocessor(item)
    policy.reset()
    with torch.inference_mode():
        for _ in range(warmup_actions):
            policy.select_action(batch)
        torch.cuda.synchronize()
        samples_ms: list[float] = []
        for _ in range(measured_actions):
            policy.reset()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            action = policy.select_action(batch)
            end.record()
            torch.cuda.synchronize()
            elapsed = float(start.elapsed_time(end))
            if not torch.isfinite(action).all() or elapsed <= 0:
                raise FastWAMPolicyError(
                    "FastWAM direct-action measurement returned invalid output"
                )
            samples_ms.append(elapsed)
    ordered = sorted(samples_ms)
    return {
        "method": "torch.cuda.Event around FastWAMPolicy.select_action",
        "warmup_actions": warmup_actions,
        "measured_actions": measured_actions,
        "samples_ms": samples_ms,
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "p50_ms": _quantile(ordered, 0.5),
        "p95_ms": _quantile(ordered, 0.95),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
    }


def _add_batch_dimension(item: dict[str, Any]) -> dict[str, Any]:
    """Give a single LeRobot sample the batch axis expected by policy processors."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise FastWAMPolicyError("torch is required for direct-action latency") from exc
    return {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
        for key, value in item.items()
    }


def _fetch_runtime_models(args: argparse.Namespace) -> dict[str, Path]:
    """Fetch exact public model revisions into the configured Hub cache.

    ``snapshot_download`` uses Hugging Face's per-revision locks and immutable
    snapshot layout.  The returned directories are passed as local paths to
    FastWAM, preventing a moving ``main`` revision from being silently fetched
    during a later training or evaluation call.
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - toolRef dependency checks it
        raise FastWAMPolicyError(
            "huggingface_hub is required for FastWAM runtime fetch"
        ) from exc
    cache_dir = os.environ.get("HF_HUB_CACHE") or None
    requested = {
        "fastwam_base": (FASTWAM_BASE_REPOSITORY, args.fastwam_base_revision),
        "wan": (WAN_REPOSITORY, args.wan_revision),
        "wan_diffusers": (WAN_DIFFUSERS_REPOSITORY, args.wan_diffusers_revision),
        "umt5": (UMT5_REPOSITORY, args.umt5_revision),
    }
    resolved: dict[str, Path] = {}
    for key, (repo_id, revision) in requested.items():
        if len(str(revision)) < 7:
            raise FastWAMPolicyError(
                f"{key} revision must be an immutable revision, not {revision!r}"
            )
        resolved[key] = Path(
            snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)
        )
    return resolved


def _materialize_dataset(
    source: str,
    destination: Path,
    args: argparse.Namespace,
    recipe: dict[str, Any] | None = None,
) -> Path:
    dataset = recipe.get("dataset", {}) if recipe else {}
    try:
        return materialize_lerobot_dataset(
            source,
            destination,
            repo_id=str(dataset.get("repo_id") or args.dataset_repo_id),
            revision=str(dataset.get("revision") or args.dataset_revision),
        )
    except LeRobotDatasetError as exc:
        raise FastWAMPolicyError(
            f"could not materialize LeRobot dataset {source!r}: {exc}"
        ) from exc


def _materialize_directory(source: str, destination: Path) -> Path:
    source_path = Path(source)
    if source_path.is_dir():
        return source_path
    if not source.startswith("s3://"):
        raise FastWAMPolicyError(
            f"stage artifact must be a local directory or s3:// prefix, got {source!r}"
        )
    StorageClient.from_environment().download_directory(source, str(destination))
    if not destination.is_dir():
        raise FastWAMPolicyError(
            f"artifact materialization did not create {destination}"
        )
    return destination


def _publish_directory(source: Path, destination: str) -> None:
    if destination.startswith("s3://"):
        StorageClient.from_environment().upload_directory(str(source), destination)
        return
    target = Path(destination)
    if target.exists():
        raise FastWAMPolicyError(
            f"refusing to overwrite existing local artifact {target}"
        )
    shutil.copytree(source, target)


def _recovery_prefix(destination: str) -> str | None:
    """Return the task-owned recovery prefix for an object-storage output."""

    if not destination.startswith("s3://"):
        return None
    return destination.rstrip("/") + "/recovery"


def _checkpoint_step(checkpoint: Path) -> int:
    """Return an upstream checkpoint's numeric step, rejecting ambiguous names."""

    if not checkpoint.name.isdecimal() or int(checkpoint.name) <= 0:
        raise FastWAMPolicyError(
            f"LeRobot checkpoint directory does not name a positive step: {checkpoint}"
        )
    return int(checkpoint.name)


def _missing_recovery_state(checkpoint: Path, pretrained: Path) -> list[str]:
    """List pinned-LeRobot resume files absent from an otherwise valid checkpoint."""

    train_config = pretrained / "train_config.json"
    required = (
        pretrained / "config.json",
        train_config,
        checkpoint / "training_state" / "training_step.json",
        checkpoint / "training_state" / "rng_state.safetensors",
        checkpoint / "training_state" / "optimizer_state.safetensors",
    )
    missing = [
        path.relative_to(checkpoint).as_posix() for path in required if not path.is_file()
    ]
    if missing:
        return missing
    train_settings = _read_json(train_config)
    if train_settings.get("scheduler") is not None:
        scheduler = checkpoint / "training_state" / "scheduler_state.json"
        if not scheduler.is_file():
            missing.append(scheduler.relative_to(checkpoint).as_posix())
    return missing


def _recoverable_checkpoint_record(checkpoint: Path) -> dict[str, Any]:
    """Validate the exact state LeRobot 0.6.1 needs for a real resume."""

    step = _checkpoint_step(checkpoint)
    pretrained = checkpoint / "pretrained_model"
    _assert_checkpoint(pretrained)
    missing = _missing_recovery_state(checkpoint, pretrained)
    if missing:
        raise FastWAMPolicyError(
            "LeRobot checkpoint is not yet resumable; missing " + ", ".join(missing)
        )
    saved_step = _read_json(checkpoint / "training_state" / "training_step.json").get(
        "step"
    )
    if not isinstance(saved_step, int) or saved_step != step:
        raise FastWAMPolicyError(
            "LeRobot checkpoint training state does not match its directory step"
        )
    return {"step": step, "checkpoint_sha256": _tree_digest(checkpoint)}


def _upstream_last_checkpoint(training: Path) -> Path | None:
    """Resolve LeRobot's atomic ``checkpoints/last`` pointer, if present."""

    root = training / "checkpoints"
    pointer = root / "last"
    if not pointer.is_symlink():
        return None
    checkpoint = pointer.resolve(strict=False)
    if not checkpoint.is_relative_to(root.resolve()) or not checkpoint.is_dir():
        raise FastWAMPolicyError("LeRobot last checkpoint pointer is unsafe or missing")
    return checkpoint


def _complete_recoverable_checkpoints(training: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return LeRobot's one complete checkpoint selected by its last pointer."""

    checkpoint = _upstream_last_checkpoint(training)
    if checkpoint is None:
        return []
    try:
        return [(checkpoint, _recoverable_checkpoint_record(checkpoint))]
    except FastWAMPolicyError:
        # ``last`` is only updated after LeRobot saves a checkpoint, but retain
        # the failure-closed check if its on-disk contract ever changes.
        return []


def _read_recovery_manifest(
    storage: StorageClient, manifest_uri: str
) -> tuple[dict[str, Any] | None, str]:
    """Read an optional recovery manifest and its conditional-write token."""

    stored = storage.read_bytes_with_etag(manifest_uri)
    if stored is None:
        return None, ""
    payload, etag = stored
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FastWAMPolicyError("recovery manifest is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != RECOVERY_MANIFEST_SCHEMA:
        raise FastWAMPolicyError("recovery manifest has an unexpected schema")
    return value, etag


def _recovery_manifest_state(
    storage: StorageClient, recovery: str
) -> tuple[str, str, int]:
    """Return the recovery manifest URI, ETag, and currently published step."""

    manifest_uri = recovery + "/latest.json"
    latest, etag = _read_recovery_manifest(storage, manifest_uri)
    if latest is None:
        return manifest_uri, etag, -1
    step = latest.get("step")
    if not isinstance(step, int) or step <= 0:
        raise FastWAMPolicyError("recovery manifest has an invalid checkpoint step")
    return manifest_uri, etag, step


def _manifest_for_checkpoint(record: dict[str, Any], relative_path: str) -> bytes:
    """Encode the durable equivalent of LeRobot's local ``last`` pointer."""

    manifest = {
        "schema": RECOVERY_MANIFEST_SCHEMA,
        "step": record["step"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "checkpoint_relative_path": relative_path,
        "upstream_last_pointer": f"checkpoints/{record['step']:06d}",
    }
    return (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")


def _verify_uploaded_checkpoint(
    storage: StorageClient,
    checkpoint_uri: str,
    source: Path,
    expected: dict[str, Any],
) -> None:
    """Read back an uploaded checkpoint and prove its complete tree digest."""

    with tempfile.TemporaryDirectory(
        prefix="npa-fastwam-checkpoint-verify-", dir=str(source.parent)
    ) as temporary:
        downloaded = Path(temporary) / f"{expected['step']:06d}"
        storage.download_directory(checkpoint_uri, str(downloaded))
        observed = _recoverable_checkpoint_record(downloaded)
    if observed != expected:
        raise FastWAMPolicyError(
            "uploaded recovery checkpoint does not match local LeRobot state"
        )


def _publish_recoverable_checkpoint(
    storage: StorageClient,
    checkpoint: Path,
    recovery: str,
    manifest_uri: str,
    etag: str,
    record: dict[str, Any],
) -> str:
    """Upload, verify, then conditionally publish one complete checkpoint."""

    relative_path = (
        f"checkpoints/{record['step']:06d}-{record['checkpoint_sha256']}/"
        f"{uuid.uuid4().hex}/"
    )
    checkpoint_uri = recovery + "/" + relative_path
    storage.upload_directory(str(checkpoint), checkpoint_uri, require_empty=True)
    _verify_uploaded_checkpoint(storage, checkpoint_uri, checkpoint, record)
    payload = _manifest_for_checkpoint(record, relative_path)
    kwargs: dict[str, Any] = {"content_type": "application/json"}
    if etag:
        kwargs["if_match"] = etag
    else:
        kwargs["if_none_match"] = True
    return storage.put_bytes_conditional(payload, manifest_uri, **kwargs)


def _sync_recoverable_checkpoints(
    storage: StorageClient,
    training: Path,
    destination: str,
    published: dict[int, dict[str, Any]],
) -> None:
    """Mirror complete native checkpoints before advancing a recovery manifest.

    Each checkpoint is content-addressed under the task-owned output prefix.
    The mutable manifest is conditional and written last, so a resume never
    selects a partial directory or lets an older worker move the pointer back.
    """

    recovery = _recovery_prefix(destination)
    if recovery is None:
        return
    manifest_uri, etag, latest_step = _recovery_manifest_state(storage, recovery)
    for checkpoint, record in _complete_recoverable_checkpoints(training):
        step = record["step"]
        if step in published or step <= latest_step:
            continue
        etag = _publish_recoverable_checkpoint(
            storage, checkpoint, recovery, manifest_uri, etag, record
        )
        latest_step = step
        published[step] = record


def _restore_recoverable_checkpoint(destination: str, local_root: Path) -> Path | None:
    """Restore and revalidate the latest durable checkpoint before a resume."""

    recovery = _recovery_prefix(destination)
    if recovery is None:
        return None
    storage = StorageClient.from_environment()
    manifest, _etag = _read_recovery_manifest(storage, recovery + "/latest.json")
    if manifest is None:
        return None
    step = manifest.get("step")
    digest = manifest.get("checkpoint_sha256")
    relative_path = manifest.get("checkpoint_relative_path")
    if (
        not isinstance(step, int)
        or step <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or not isinstance(relative_path, str)
        or not relative_path.startswith("checkpoints/")
        or not relative_path.endswith("/")
        or ".." in Path(relative_path).parts
    ):
        raise FastWAMPolicyError("recovery manifest has an invalid checkpoint reference")
    if manifest.get("upstream_last_pointer") != f"checkpoints/{step:06d}":
        raise FastWAMPolicyError("recovery manifest does not bind LeRobot last pointer")
    checkpoint = local_root / "checkpoints" / f"{step:06d}"
    storage.download_directory(recovery + "/" + relative_path, str(checkpoint))
    record = _recoverable_checkpoint_record(checkpoint)
    if record["step"] != step or record["checkpoint_sha256"] != digest:
        raise FastWAMPolicyError(
            "downloaded recovery checkpoint does not match its durable manifest"
        )
    (checkpoint.parent / "last").symlink_to(checkpoint.name)
    return checkpoint / "pretrained_model"


def _checkpoint_mirror_monitor(
    stopped: threading.Event,
    mirror_once: Callable[[], None],
    failures: list[Exception],
) -> None:
    """Continue periodic recovery publication until the owning process stops."""

    while not stopped.wait(_RECOVERY_MIRROR_POLL_SECONDS):
        try:
            mirror_once()
        except Exception as exc:
            failures.append(exc)
            return


def _raise_training_mirror_failure(
    native_failure: Exception | None, mirror_failures: list[Exception]
) -> None:
    """Raise the most useful failure after native training and mirroring finish."""

    if mirror_failures:
        detail = str(mirror_failures[0])
        if native_failure is not None:
            raise FastWAMPolicyError(
                "native training failed and durable checkpoint mirroring also failed: "
                + detail
            ) from native_failure
        raise FastWAMPolicyError("durable checkpoint mirroring failed: " + detail)
    if native_failure is not None:
        raise native_failure


def _start_checkpoint_mirror(
    stopped: threading.Event,
    mirror_once: Callable[[], None],
    mirror_failures: list[Exception],
) -> threading.Thread:
    """Start the background mirror after its task-owned prefix is verified."""

    mirror_once()
    worker = threading.Thread(
        target=_checkpoint_mirror_monitor,
        args=(stopped, mirror_once, mirror_failures),
        name="fastwam-checkpoint-mirror",
        daemon=True,
    )
    worker.start()
    return worker


def _finish_checkpoint_mirror(
    stopped: threading.Event,
    worker: threading.Thread,
    mirror_once: Callable[[], None],
    mirror_failures: list[Exception],
) -> None:
    """Stop the monitor and perform one final complete-checkpoint mirror."""

    stopped.set()
    worker.join()
    if mirror_failures:
        return
    try:
        mirror_once()
    except Exception as exc:
        mirror_failures.append(exc)


def _run_training_with_checkpoint_mirror(
    command: list[str],
    log_path: Path,
    training: Path,
    destination: str,
) -> list[dict[str, Any]]:
    """Run native training while durably mirroring complete periodic checkpoints."""

    recovery = _recovery_prefix(destination)
    if recovery is None:
        _run_command(command, log_path)
        return []

    storage = StorageClient.from_environment()
    published: dict[int, dict[str, Any]] = {}
    mirror_failures: list[Exception] = []
    stopped = threading.Event()

    def mirror_once() -> None:
        _sync_recoverable_checkpoints(storage, training, destination, published)

    worker = _start_checkpoint_mirror(stopped, mirror_once, mirror_failures)
    native_failure: Exception | None = None
    try:
        _run_command(command, log_path)
    except Exception as exc:
        native_failure = exc
    finally:
        _finish_checkpoint_mirror(stopped, worker, mirror_once, mirror_failures)

    _raise_training_mirror_failure(native_failure, mirror_failures)
    return [
        {"step": step, "checkpoint_sha256": record["checkpoint_sha256"]}
        for step, record in sorted(published.items())
    ]


def _read_recipe(prepared: Path) -> dict[str, Any]:
    recipe = _read_json(prepared / "recipe.json")
    if recipe.get("schema") != "npa.fastwam.recipe.v1":
        raise FastWAMPolicyError("prepared input is not a FastWAM recipe")
    if recipe.get("policy") != "fastwam":
        raise FastWAMPolicyError("prepared recipe does not name FastWAM")
    if not recipe.get("train_episode_indices") or not recipe.get(
        "heldout_episode_indices"
    ):
        raise FastWAMPolicyError(
            "prepared recipe does not contain a nonempty episode-disjoint split"
        )
    return recipe


def _assert_dataset_receipt(dataset: Path, recipe: dict[str, Any]) -> None:
    actual = _sha256(dataset / "meta" / "info.json")
    if actual != recipe.get("dataset_info_sha256"):
        raise FastWAMPolicyError(
            "materialized dataset metadata differs from the sealed preparation receipt"
        )


def _final_checkpoint(training: Path) -> Path:
    checkpoint = _upstream_last_checkpoint(training)
    if checkpoint is None:
        raise FastWAMPolicyError("lerobot-train wrote no last checkpoint pointer")
    pretrained = checkpoint / "pretrained_model"
    _assert_checkpoint(pretrained)
    return pretrained


def _assert_checkpoint(checkpoint: Path) -> None:
    if not checkpoint.is_dir() or not any(checkpoint.glob("*.safetensors")):
        raise FastWAMPolicyError(
            f"FastWAM checkpoint is missing safetensors weights: {checkpoint}"
        )


def _video_receipt(root: Path) -> dict[str, dict[str, Any]]:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - report toolRef declares av
        raise FastWAMPolicyError(
            "PyAV is required to validate native rollout videos"
        ) from exc
    videos: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.mp4")):
        with av.open(str(path)) as container:
            frames = sum(1 for _ in container.decode(video=0))
        if frames <= 0:
            raise FastWAMPolicyError(
                f"native rollout video has no decodable frames: {path}"
            )
        videos[path.relative_to(root).as_posix()] = {
            "frames": frames,
            "sha256": _sha256(path),
        }
    return videos


def _write_rrd(
    path: Path, run_id: str, evaluation: dict[str, Any], videos: dict[str, Any]
) -> None:
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - report toolRef declares rerun
        raise FastWAMPolicyError(
            "rerun-sdk is required to emit FastWAM evidence"
        ) from exc
    recording = rr.RecordingStream("npa_fastwam", recording_id=run_id)
    recording.save(str(path))
    recording.log(
        "provenance",
        rr.TextDocument(
            json.dumps({"evaluation": evaluation, "videos": videos}, sort_keys=True)
        ),
        static=True,
    )
    success = evaluation["native_task_success"]
    latency = evaluation["direct_action_latency"]
    recording.log(
        "evaluation/success_percent", rr.Scalars(float(success["success_percent"]))
    )
    recording.log("evaluation/action_latency_ms", rr.Scalars(float(latency["mean_ms"])))
    recording.log(
        "evaluation/action_latency_p95_ms", rr.Scalars(float(latency["p95_ms"]))
    )
    recording.flush()
    recording.disconnect()


def _inspect_rrd(path: Path, run_id: str) -> None:
    rerun = str(Path(sys.executable).parent / "rerun")
    verified = subprocess.run(
        [rerun, "rrd", "verify", str(path)], capture_output=True, text=True, check=True
    )
    decoded = subprocess.run(
        [rerun, "rrd", "print", "-vv", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    for expected in ("npa_fastwam", run_id, "evaluation/action_latency_ms"):
        if expected not in decoded.stdout:
            raise FastWAMPolicyError(f"Rerun recording is missing {expected!r}")
    path.with_suffix(".inspection.txt").write_text(
        verified.stdout + decoded.stdout, encoding="utf-8"
    )


def _run_command(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if process.wait() != 0:
            raise FastWAMPolicyError(
                f"native command failed ({process.returncode}): {' '.join(command)}"
            )


def _runtime_provenance() -> dict[str, Any]:
    try:
        import torch

        cuda: dict[str, Any] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            cuda |= {
                "device": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
            }
    except ImportError:
        cuda = {"torch": "unavailable"}
    return {"lerobot": importlib.metadata.version("lerobot"), **cuda}


def _runtime_model_receipt(
    models: dict[str, Path], args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "fastwam_base": {
            "repository": FASTWAM_BASE_REPOSITORY,
            "revision": args.fastwam_base_revision,
            "path_sha256": _tree_digest(models["fastwam_base"]),
        },
        "wan": {
            "repository": WAN_REPOSITORY,
            "revision": args.wan_revision,
            "path_sha256": _tree_digest(models["wan"]),
        },
        "wan_diffusers": {
            "repository": WAN_DIFFUSERS_REPOSITORY,
            "revision": args.wan_diffusers_revision,
            "path_sha256": _tree_digest(models["wan_diffusers"]),
        },
        "umt5": {
            "repository": UMT5_REPOSITORY,
            "revision": args.umt5_revision,
            "path_sha256": _tree_digest(models["umt5"]),
        },
    }


def _upstream_provenance() -> dict[str, Any]:
    return {
        "lerobot": {
            "repository": "https://github.com/huggingface/lerobot",
            "version": LEROBOT_VERSION,
            "revision": LEROBOT_RELEASE_COMMIT,
            "license": "Apache-2.0",
            "copyright": "Hugging Face Inc. team",
        },
        "fastwam": {
            "repository": "https://huggingface.co/lerobot/fastwam_base",
            "license": "Apache-2.0",
            "citation": "Yuan et al., Fast-WAM: Do World Action Models Need Test-time Future Imagination?, arXiv:2603.16666 (2026)",
        },
        "distinction": "This LeRobot FastWAM workflow is not the Cosmos3 FastWAM-K2 experiment.",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FastWAMPolicyError(f"required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FastWAMPolicyError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise FastWAMPolicyError(
            "cannot compute latency percentile from no measurements"
        )
    index = min(len(values) - 1, max(0, round((len(values) - 1) * quantile)))
    return values[index]


def _require_positive(value: int, label: str) -> None:
    if value <= 0:
        raise FastWAMPolicyError(f"{label} must be positive")


def _require_positive_fraction(value: float) -> None:
    if not 0 < value < 1:
        raise FastWAMPolicyError("train fraction must be in (0, 1)")


if __name__ == "__main__":
    raise SystemExit(main())
