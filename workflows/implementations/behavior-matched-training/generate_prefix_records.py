"""Run frozen-parent prefix inference over released replan frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from matched_train import (
    load_config,
    training_configuration,
    verify_parent_checkpoint,
    verify_source,
)
from native_metrics import predict_stage_logits
from ordered_data import OrderedBatches
from panel_data import PanelDataset
from stage_conditioning import TASK_NUM_STAGES


def _array_digest(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _stack(samples: list[dict]):
    import jax
    from b1k.models.observation import Observation

    batch = jax.tree.map(lambda *items: np.stack(items), *samples)
    return Observation.from_dict(batch), batch["actions"]


def _transform_raw_sample(transformed, panel: PanelDataset, raw: dict) -> dict:
    if getattr(transformed, "_dataset", None) is not panel:
        raise ValueError("native transformed dataset does not wrap the panel directly")
    transform = getattr(transformed, "_transform", None)
    if not callable(transform):
        raise TypeError("native transformed dataset lacks its pinned transform")
    return transform(raw)


def _sample_identities(samples: list[dict]) -> list[dict[str, str]]:
    identities = []
    for sample in samples:
        observations = {
            key: value
            for key, value in sample.items()
            if key.startswith("observation.")
        }
        identities.append(
            {
                "observation_sha256": _array_digest(observations),
                "action_sha256": _array_digest({"action": sample["action"]}),
            }
        )
    return identities


def _predict_batches(model, predictor, samples: list[dict], batch_size: int) -> list:
    logits = []
    for start in range(0, len(samples), batch_size):
        selected = samples[start : start + batch_size]
        padded = selected + [selected[-1]] * (batch_size - len(selected))
        observation, _ = _stack(padded)
        result = predictor(model, observation)
        logits.extend(np.asarray(result, dtype=np.float32)[: len(selected)].tolist())
    return logits


def _predict_served_logits(predictor, samples: list[dict], stages: int) -> list:
    """Produce logits in the exact batch-one serving graph; discard sampled actions."""
    import jax

    logits = []
    for sample in samples:
        observation, _ = _stack([sample])
        _, result = predictor(jax.random.key(0), observation, num_steps=1)
        values = np.asarray(result, dtype=np.float32)[0, :stages]
        if values.shape != (stages,) or not np.isfinite(values).all():
            raise ValueError("canonical serving logits are invalid")
        logits.append(values.tolist())
    return logits


def _episode_plan(panel, slot: int, episode_slot: int) -> dict:
    source = panel.datasets[slot]
    start = int(source.episode_data_index["from"][episode_slot])
    end = int(source.episode_data_index["to"][episode_slot])
    frames = list(range(0, end - start, 20))
    return {
        "task_id": panel.task_ids[slot],
        "episode_index": int(source.episodes[episode_slot]),
        "episode_length": end - start,
        "frames": frames,
        "indices": [panel.offsets[slot] + start + frame for frame in frames],
    }


class PrefixSamples:
    """Decode, transform, and hash each original sample in CPU workers."""

    def __init__(self, panel, transformed):
        self.panel = panel
        self.transformed = transformed

    def __len__(self):
        return len(self.panel)

    def __getitem__(self, index):
        raw = self.panel[index]
        sample = _transform_raw_sample(self.transformed, self.panel, raw)
        return {"sample": sample, "identity": _sample_identities([raw])[0]}


def _append_predictions(record, samples, model, raw_predictor, served_predictor, size):
    transformed = [sample["sample"] for sample in samples]
    record["auxiliary_raw_logits"].extend(
        _predict_batches(model, raw_predictor, transformed, size)
    )
    record["served_valid_logits"].extend(
        _predict_served_logits(
            served_predictor, transformed, TASK_NUM_STAGES[record["task_id"]]
        )
    )
    record["sample_identities"].extend(sample["identity"] for sample in samples)


def _stream_records(args, plans, batches, model, raw_predictor, served_predictor):
    with args.output.open("x") as stream:
        for plan in plans:
            expected = plan["indices"]
            record = {key: value for key, value in plan.items() if key != "indices"}
            record.update(
                schema="npa.behavior.rlc-prefix-record.v2",
                auxiliary_raw_logits=[],
                served_valid_logits=[],
                sample_identities=[],
            )
            for start in range(0, len(expected), args.batch_size):
                indices, samples = next(batches)
                if tuple(expected[start : start + args.batch_size]) != indices:
                    raise ValueError("prefix decoder crossed an episode boundary")
                _append_predictions(
                    record,
                    samples,
                    model,
                    raw_predictor,
                    served_predictor,
                    args.batch_size,
                )
            stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            stream.flush()
        if next(batches, None) is not None:
            raise ValueError("prefix decoder emitted extra samples")


def _write_records(
    args, panel, transformed, model, raw_predictor, served_predictor
) -> None:
    plans = [
        _episode_plan(panel, slot, episode_slot)
        for slot in range(len(panel.task_ids))
        for episode_slot in range(len(panel.datasets[slot].episodes))
    ]
    plans = [
        plan
        for ordinal, plan in enumerate(plans)
        if ordinal % args.shard_count == args.shard_index
    ]
    indices = [
        plan["indices"][start : start + args.batch_size]
        for plan in plans
        for start in range(0, len(plan["indices"]), args.batch_size)
    ]
    dataset = PrefixSamples(panel, transformed)
    workers = load_config(args.config)["num_workers"]
    with OrderedBatches(dataset, indices, workers=workers) as batches:
        _stream_records(
            args, plans, iter(batches), model, raw_predictor, served_predictor
        )


def run(args: argparse.Namespace) -> None:
    """Generate training and holdout prefix records from the restored parent.

    Args:
        args: Parsed source, dataset, split, and output paths.

    Raises:
        FileExistsError: An output already exists.
        OSError: A required input cannot be read or output cannot be written.
        ValueError: A frozen identity or dataset contract differs.
    """

    config_values = load_config(args.config)
    if args.batch_size < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("batch size or episode shard selection is invalid")
    verify_source(args.source_root)
    verify_parent_checkpoint(args.checkpoint_archive, args.checkpoint)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output_root = args.output.parent / ".prefix-runtime-unused"
    with tempfile.TemporaryDirectory(prefix="npa-rlc-prefix-") as temporary:
        sys.path.insert(0, str(args.adapter_root))
        from rlc_server import _policy_source

        _policy_source(args.source_root, Path(temporary))
        from b1k.policies import policy_config
        from b1k.training import data_loader
        from flax import nnx

        training = training_configuration(args, config_values)
        data_config = training.data.create(training.assets_dirs, training.model)
        panel = PanelDataset(
            args.dataset_root, args.episode_split, split_name=args.split
        )
        transformed = data_loader.transform_dataset(panel, data_config)
        policy = policy_config.create_trained_policy(
            training,
            args.checkpoint,
            norm_stats=data_config.norm_stats,
        )
        model = policy._model
        model.eval()
        raw_predictor = nnx.jit(predict_stage_logits)
        served_predictor = nnx.jit(model.sample_actions, static_argnames=("num_steps",))
        _write_records(args, panel, transformed, model, raw_predictor, served_predictor)


def arguments() -> argparse.Namespace:
    """Parse parent-prefix generation arguments.

    Returns:
        Parsed paths and inference settings.
    """

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
    parser.add_argument("--split", choices=("training", "holdout"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
