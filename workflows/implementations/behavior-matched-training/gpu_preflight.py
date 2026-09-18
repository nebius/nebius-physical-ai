"""Run real restored-parent GPU gates before prefix generation or training."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from action_partition import assert_partition
from generate_prefix_records import _stack
from matched_train import (
    load_config,
    training_configuration,
    verify_parent_checkpoint,
    verify_source,
)
from native_metrics import predict_stage_logits
from panel_data import PanelDataset


def _paths(value, prefix=()):
    if hasattr(value, "items"):
        for name, child in value.items():
            yield from _paths(child, (*prefix, str(name)))
    else:
        yield prefix


def _logits_equal(direct: np.ndarray, native: np.ndarray) -> tuple[bool, int]:
    """Compare finite stage logits in float32 across JAX storage dtypes."""
    finite = np.isfinite(native)
    if not finite.any():
        return False, 0
    direct_values = np.asarray(direct[finite], dtype=np.float32)
    native_values = np.asarray(native[finite], dtype=np.float32)
    return bool(np.allclose(direct_values, native_values, rtol=1e-5, atol=1e-5)), int(
        finite.sum()
    )


def _verify_logits(jax, nnx, model, observation) -> int:
    direct = np.asarray(nnx.jit(predict_stage_logits)(model, observation))
    _, native = nnx.jit(model.sample_actions, static_argnames=("num_steps",))(
        jax.random.key(0), observation, num_steps=1
    )
    native = np.asarray(native)
    equal, finite_logits = _logits_equal(direct, native)
    if not equal:
        raise ValueError("Direct prefix logits differ from native inference logits")
    return finite_logits


def _result(jax, panel, finite_logits: int) -> dict:
    key = panel.key_for_index(0)
    return {
        "schema": "npa.behavior.matched-stage-gpu-preflight.v1",
        "device": jax.devices()[0].device_kind,
        "source_revisions_verified": True,
        "parent_checkpoint_tree_verified": True,
        "normalization_loaded_from_parent": True,
        "real_training_sample": {
            "task_id": key.task_id,
            "episode_index": key.episode_index,
            "episode_relative_frame": key.episode_relative_frame,
        },
        "native_prefix_finite_logits": finite_logits,
        "native_prefix_logits_equal": True,
        "exact_trainable_paths_present": 23,
        "finite_update_verified": False,
        "status": "prefix_equivalence_passed_update_gate_pending",
    }


def run(args: argparse.Namespace) -> None:
    """Restore the parent policy and run the prefix-equivalence GPU gate.

    Args:
        args: Parsed source, checkpoint, dataset, and output paths.

    Raises:
        OSError: A required input cannot be read or the receipt cannot be written.
        ValueError: Source, checkpoint, device, logits, or partition checks fail.
    """

    values = load_config(args.config)
    verify_source(args.source_root)
    verify_parent_checkpoint(args.checkpoint_archive, args.checkpoint)
    args.output_root = args.output.parent / ".preflight-unused"
    with tempfile.TemporaryDirectory(
        prefix="npa-matched-stage-preflight-"
    ) as temporary:
        sys.path.insert(0, str(args.adapter_root))
        from rlc_server import _policy_source

        _policy_source(args.source_root, Path(temporary))
        _run_native_preflight(args, values)


def _run_native_preflight(args: argparse.Namespace, values: dict) -> None:
    import jax
    from b1k.policies import policy_config
    from b1k.training import data_loader
    from flax import nnx

    if len(jax.devices()) != 1 or "B200" not in jax.devices()[0].device_kind:
        raise ValueError("Matched-stage preflight requires exactly one B200")
    training = training_configuration(args, values)
    data_config = training.data.create(training.assets_dirs, training.model)
    panel = PanelDataset(args.dataset_root, args.episode_split, split_name="training")
    transformed = data_loader.transform_dataset(panel, data_config)
    observation, _ = _stack([transformed[0]])
    policy = policy_config.create_trained_policy(
        training, args.checkpoint, norm_stats=data_config.norm_stats
    )
    model = policy._model
    model.eval()
    finite_logits = _verify_logits(jax, nnx, model, observation)
    state = nnx.state(model, nnx.Param).to_pure_dict()
    assert_partition(list(_paths(state)))
    receipt = _result(jax, panel, finite_logits)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Parse arguments and run the GPU preflight."""

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "adapter-root",
        "checkpoint",
        "checkpoint-archive",
        "dataset-root",
        "episode-split",
        "config",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
