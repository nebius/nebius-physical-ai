"""Run real restored-parent GPU gates before prefix generation or training."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from action_partition import assert_partition
from generate_prefix_records import _stack, _transform_raw_sample
from matched_train import (
    _load_trainer,
    load_config,
    training_configuration,
    validate_real_update,
    verify_parent_checkpoint,
    verify_source,
)
from panel_data import PanelDataset, transform_conditioned_dataset
from stage_conditioning import equal_time_stage


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


def _verify_canonical_logits(jax, nnx, model, panel, transformed) -> list[dict]:
    predictor = nnx.jit(model.sample_actions, static_argnames=("num_steps",))
    checks = []
    for slot, task_id in enumerate(panel.task_ids):
        index = panel.offsets[slot]
        raw = panel[index]
        sample = _transform_raw_sample(transformed, panel, raw)
        observation, _ = _stack([sample])
        _, step1 = predictor(jax.random.key(0), observation, num_steps=1)
        _, step20 = predictor(jax.random.key(0), observation, num_steps=20)
        step1, step20 = np.asarray(step1), np.asarray(step20)
        equal, finite = _logits_equal(step1, step20)
        if not equal or not np.array_equal(np.isfinite(step1), np.isfinite(step20)):
            raise ValueError(f"canonical step-one logits differ for task {task_id}")
        key = panel.key_for_index(index, sample=raw)
        checks.append(
            {
                "task_id": task_id,
                "episode_index": key.episode_index,
                "episode_relative_frame": key.episode_relative_frame,
                "finite_logits": finite,
                "argmax": int(np.argmax(step1)),
            }
        )
    return checks


class _TeacherSmokeDataset:
    """Attach native equal-time stage labels to real released panel frames."""

    def __init__(self, panel: PanelDataset):
        self.panel = panel

    def __len__(self) -> int:
        return len(self.panel)

    def __getitem__(self, index: int) -> dict:
        sample = self.panel[index]
        key = self.panel.key_for_index(index, sample=sample)
        slot = self.panel.task_ids.index(key.task_id)
        dataset = self.panel.datasets[slot]
        episode_slot = list(dataset.episodes).index(key.episode_index)
        start = int(dataset.episode_data_index["from"][episode_slot])
        end = int(dataset.episode_data_index["to"][episode_slot])
        stage = equal_time_stage(key.episode_relative_frame, end - start, key.task_id)
        return {
            **sample,
            "teacher_stage": np.asarray(stage, dtype=np.int32),
            "replay_stage": np.asarray(stage, dtype=np.int32),
        }


def _install_teacher_smoke_loader(module, panel: PanelDataset) -> None:
    """Use a real panel sample with teacher conditioning for one discarded update."""
    native_transform = module.transform_dataset
    smoke = _TeacherSmokeDataset(panel)

    def create_dataset(data_config, action_horizon: int, seed=None):
        del data_config, seed
        if action_horizon != 30:
            raise ValueError("smoke action horizon differs")
        return smoke

    def transform_dataset(dataset, data_config, **kwargs):
        if dataset is not smoke:
            raise ValueError("smoke loader received an unexpected dataset")
        return transform_conditioned_dataset(
            module,
            dataset,
            data_config,
            arm="teacher",
            transform=native_transform,
            **kwargs,
        )

    module.create_behavior_dataset = create_dataset
    module.transform_dataset = transform_dataset


def _result(jax, checks: list[dict]) -> dict:
    return {
        "schema": "npa.behavior.matched-stage-gpu-preflight.v1",
        "device": jax.devices()[0].device_kind,
        "source_revisions_verified": True,
        "parent_checkpoint_tree_verified": True,
        "normalization_loaded_from_parent": True,
        "canonical_batch_one_checks": checks,
        "canonical_step1_equals_serving_step20": True,
        "prefix_decision_batch_size": 1,
        "exact_trainable_paths_present": 23,
        "finite_update_verified": False,
        "status": "prefix_equivalence_passed_update_gate_pending",
    }


def _restore_policy(args: argparse.Namespace, values: dict):
    """Restore the parent policy and its released training dataset."""
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
    policy = policy_config.create_trained_policy(
        training, args.checkpoint, norm_stats=data_config.norm_stats
    )
    return jax, nnx, data_loader, training, panel, transformed, policy


def _verify_policy(jax, nnx, panel, transformed, policy) -> list[dict]:
    """Check canonical logits and the exact trainable parameter partition."""
    model = policy._model
    model.eval()
    checks = _verify_canonical_logits(jax, nnx, model, panel, transformed)
    state = nnx.state(model, nnx.Param).to_pure_dict()
    assert_partition(list(_paths(state)))
    return checks


def _verify_real_update(args, training, trainer, data_loader, panel) -> dict:
    """Run one discarded native update and return its validated receipt."""
    _install_teacher_smoke_loader(data_loader, panel)
    update_receipt = args.output.with_name("gpu-prefix-update-preflight.json")
    validate_real_update(training, trainer, update_receipt)
    update = json.loads(update_receipt.read_text())
    if update.get("status") != "passed":
        raise ValueError("Real update preflight did not pass")
    return update


def _run_native_preflight(args: argparse.Namespace, values: dict) -> None:
    """Execute the restored-policy checks after installing the source overlay."""
    jax, nnx, data_loader, training, panel, transformed, policy = _restore_policy(
        args, values
    )
    checks = _verify_policy(jax, nnx, panel, transformed, policy)
    del transformed, policy
    gc.collect()
    jax.clear_caches()
    update = _verify_real_update(
        args, training, _load_trainer(args.source_root), data_loader, panel
    )
    result = _result(jax, checks)
    result.update(
        finite_update_verified=True,
        real_update=update,
        status="prefix_equivalence_and_update_passed",
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> None:
    """Run restored-policy equivalence, partition, and optimizer checks.

    Args:
        args: Parsed source, checkpoint, dataset, configuration, and output paths.

    Raises:
        OSError: A required input cannot be read or a receipt cannot be written.
        ValueError: Source, checkpoint, device, logits, partition, or update checks fail.
    """
    values = load_config(args.config)
    verify_source(args.source_root)
    verify_parent_checkpoint(args.checkpoint_archive, args.checkpoint)
    args.output_root = args.output.parent / ".preflight-unused"
    with tempfile.TemporaryDirectory(prefix="npa-matched-stage-preflight-") as temporary:
        sys.path.insert(0, str(args.adapter_root))
        from rlc_server import _policy_source

        _policy_source(args.source_root, Path(temporary))
        _run_native_preflight(args, values)


def main() -> None:
    """Parse command-line paths and run the GPU preflight."""
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
