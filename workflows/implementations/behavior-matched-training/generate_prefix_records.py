"""Run frozen-parent prefix inference over released replan frames without action denoising."""

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
from panel_data import PanelDataset


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


def _episode_samples(dataset: PanelDataset, slot: int, episode_slot: int):
    source = dataset.datasets[slot]
    start = int(source.episode_data_index["from"][episode_slot])
    end = int(source.episode_data_index["to"][episode_slot])
    for frame in range(0, end - start, 20):
        global_index = dataset.offsets[slot] + start + frame
        yield frame, global_index, dataset[global_index]


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
        observation, _ = _stack(samples[start : start + batch_size])
        result = predictor(model, observation)
        logits.extend(np.asarray(result, dtype=np.float32).tolist())
    return logits


def _episode_record(
    panel, transformed, model, predictor, slot, episode_slot, batch_size
):
    source = panel.datasets[slot]
    frames, raw_samples, transformed_samples = [], [], []
    for frame, index, raw in _episode_samples(panel, slot, episode_slot):
        frames.append(frame)
        raw_samples.append(raw)
        transformed_samples.append(_transform_raw_sample(transformed, panel, raw))
    return {
        "task_id": panel.task_ids[slot],
        "episode_index": int(source.episodes[episode_slot]),
        "episode_length": int(
            source.episode_data_index["to"][episode_slot]
            - source.episode_data_index["from"][episode_slot]
        ),
        "frames": frames,
        "raw_logits": _predict_batches(
            model, predictor, transformed_samples, batch_size
        ),
        "sample_identities": _sample_identities(raw_samples),
    }


def _write_records(args, panel, transformed, model, predictor) -> None:
    with args.output.open("x") as stream:
        for slot in range(len(panel.task_ids)):
            for episode_slot in range(len(panel.datasets[slot].episodes)):
                record = _episode_record(
                    panel,
                    transformed,
                    model,
                    predictor,
                    slot,
                    episode_slot,
                    args.batch_size,
                )
                stream.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )
                stream.flush()


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
        predictor = nnx.jit(predict_stage_logits)
        _write_records(args, panel, transformed, model, predictor)


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
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
