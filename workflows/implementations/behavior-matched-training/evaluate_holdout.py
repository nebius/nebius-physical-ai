"""Score every retained checkpoint on the frozen training-distribution holdout."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from checkpoint_selection import LossRow, write_selection
from matched_train import (
    load_config,
    training_configuration,
    verify_parent_checkpoint,
    verify_source,
)
from native_metrics import deterministic_batch_metrics
from ordered_data import OrderedBatches
from panel_data import PanelDataset, ReplanDataset, transform_conditioned_dataset
from stage_conditioning import load_trace


def _stack(samples: list[dict]):
    import jax
    from b1k.models.observation import Observation

    batch = jax.tree.map(lambda *items: np.stack(items), *samples)
    teacher = np.asarray(batch.pop("teacher_stage"), dtype=np.int32)
    batch.pop("replay_stage")
    return Observation.from_dict(batch), batch["actions"], teacher


def _load_native_trainer(source_root: Path):
    import importlib.util

    path = source_root / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("rlc_native_holdout_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _holdout_data(args, training, config_values, trace, data_loader):
    data_config = training.data.create(training.assets_dirs, training.model)
    panel = PanelDataset(args.dataset_root, args.episode_split, split_name="holdout")
    conditioned = ReplanDataset(panel, trace)
    transformed = transform_conditioned_dataset(
        data_loader, conditioned, data_config, arm=config_values["arm"]
    )
    return data_config, conditioned, transformed


def _checkpoint_manager(
    args, training, config_values, data_config, jax, sharding, checkpoints
):
    mesh = sharding.make_mesh(training.fsdp_devices)
    native = _load_native_trainer(args.source_root)
    initial, _ = native.init_train_state(
        training,
        jax.random.key(config_values["seed"]),
        mesh,
        resume=False,
        norm_stats=data_config.norm_stats,
    )
    manager, _ = checkpoints.initialize_checkpoint_dir(
        args.checkpoint_root,
        keep_period=config_values["save_interval"],
        overwrite=False,
        resume=True,
    )
    expected = [600, 1200, 1800, 2400, 3000, 3599]
    if sorted(manager.all_steps()) != expected:
        raise ValueError(
            "retained checkpoint set differs from the frozen selection plan"
        )
    return initial, manager, expected


def _serving_model(state, jax, nnx, nnx_utils):
    if state.ema_params is None:
        raise ValueError("retained checkpoint lacks required EMA parameters")
    serving_params = nnx_utils.state_map(
        state.ema_params,
        nnx.Param,
        lambda parameter: parameter.replace(parameter.value.astype(jax.numpy.bfloat16)),
    )
    model = nnx.merge(state.model_def, serving_params)
    model.eval()
    return model


def _loss_rows(step, indices, conditioned, metrics) -> list[LossRow]:
    rows = []
    for row_index, index in enumerate(indices):
        key = conditioned.key_for_index(index)
        rows.append(
            LossRow(
                step=step,
                task_id=key.task_id,
                episode_index=key.episode_index,
                episode_relative_frame=key.episode_relative_frame,
                action_loss=float(metrics["action_loss"][row_index]),
                stage_cross_entropy=float(metrics["stage_cross_entropy"][row_index]),
                action_dimension_losses=tuple(
                    float(value)
                    for value in metrics["action_dimension_losses"][row_index]
                ),
            )
        )
    return rows


def _score_checkpoint(args, step, model, conditioned, batches, jax, nnx):
    metric_fn = nnx.jit(
        deterministic_batch_metrics,
        static_argnames=("seed", "flow_draws"),
    )
    rows = []
    for ordinal, (indices, samples) in enumerate(batches):
        observation, actions, teacher = _stack(samples)
        metrics = jax.device_get(
            metric_fn(
                model,
                observation,
                actions,
                teacher,
                seed=args.loss_seed,
                batch_ordinal=ordinal,
                flow_draws=args.flow_draws,
            )
        )
        rows.extend(_loss_rows(step, indices, conditioned, metrics))
    return rows


def _write_results(args, rows: list[LossRow]) -> None:
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row.__dict__, sort_keys=True) + "\n")
    write_selection(args.selection, rows)


def run(args: argparse.Namespace) -> None:
    """Score every retained checkpoint on the frozen holdout split.

    Args:
        args: Parsed checkpoint, dataset, trace, and output paths.

    Raises:
        FileExistsError: A holdout output already exists.
        OSError: A required input cannot be read.
        ValueError: A frozen identity or checkpoint set differs.
    """

    config_values = load_config(args.config)
    verify_source(args.source_root)
    verify_parent_checkpoint(args.checkpoint_archive, args.checkpoint)
    manifest = json.loads(args.input_manifest.read_text())
    trace = load_trace(
        args.holdout_trace,
        split="holdout",
        identities=manifest["trace_identities"],
    )
    if args.output.exists() or args.selection.exists():
        raise FileExistsError("holdout outputs must not already exist")
    args.output_root = args.output.parent / ".holdout-runtime-unused"
    with tempfile.TemporaryDirectory(prefix="npa-rlc-holdout-") as temporary:
        sys.path.insert(0, str(args.adapter_root))
        from rlc_server import _policy_source

        _policy_source(args.source_root, Path(temporary))
        _evaluate_checkpoints(args, config_values, trace)


def _evaluate_checkpoints(args, config_values, trace) -> None:
    import jax
    from b1k.training import checkpoints, data_loader
    from flax import nnx
    from openpi.shared import nnx_utils
    from openpi.training import sharding

    training = training_configuration(args, config_values)
    data_config, conditioned, transformed = _holdout_data(
        args, training, config_values, trace, data_loader
    )
    initial, manager, expected_steps = _checkpoint_manager(
        args, training, config_values, data_config, jax, sharding, checkpoints
    )
    indices = [
        conditioned.flat_indices[start : start + args.batch_size]
        for start in range(0, len(conditioned.flat_indices), args.batch_size)
    ]
    all_rows = []
    with OrderedBatches(
        transformed, indices, workers=config_values["num_workers"]
    ) as batches:
        for step in expected_steps:
            state = checkpoints.restore_state(manager, initial, None, step=step)
            model = _serving_model(state, jax, nnx, nnx_utils)
            all_rows.extend(
                _score_checkpoint(args, step, model, conditioned, batches, jax, nnx)
            )
    _write_results(args, all_rows)


def arguments() -> argparse.Namespace:
    """Parse deterministic holdout-scoring arguments.

    Returns:
        Parsed paths, batch size, flow draws, and RNG seed.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "adapter-root",
        "checkpoint",
        "checkpoint-archive",
        "checkpoint-root",
        "dataset-root",
        "episode-split",
        "config",
        "input-manifest",
        "holdout-trace",
        "output",
        "selection",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--flow-draws", type=int, default=8)
    parser.add_argument("--loss-seed", type=int, default=1701)
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
