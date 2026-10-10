"""Run upstream LeRobot training while recording real optimizer metrics and checkpoint events."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from .checkpoints import publish_checkpoint
from .public_vla_data import write_json


def main() -> None:
    """Attach evidence writers to the pinned upstream trainer, then run its native CLI.

    Args:
        None; arguments are parsed by LeRobot's training configuration.
    Returns:
        None.
    Raises:
        RuntimeError: Native training fails.
        OSError: Durable metrics or checkpoint evidence cannot be written.
    """
    import lerobot.scripts.lerobot_train as native

    evidence = Path(os.environ["NPA_VLA_EVIDENCE"])
    evidence.mkdir(parents=True, exist_ok=True)
    update = native.update_policy
    save = native.save_checkpoint
    native.update_policy = _observe_updates(update, evidence)
    native.save_checkpoint = _observe_checkpoints(save, evidence)
    try:
        native.main()
    finally:
        native.update_policy = update
        native.save_checkpoint = save


def _observe_updates(update, evidence):
    rank = int(os.environ.get("RANK", "0"))
    path = evidence / f"metrics-rank-{rank}.jsonl"

    def recorded(*args, **kwargs):
        metrics, outputs = update(*args, **kwargs)
        if args and not (evidence / f"runtime-rank-{rank}.json").exists():
            _runtime_evidence(args[1], evidence, rank)
        row = {"step": int(metrics.steps) + 1, "rank": rank, "unix_time": time.time()}
        for name in (
            "loss",
            "grad_norm",
            "lr",
            "update_s",
            "dataloading_s",
            "gpu_mem_gb",
        ):
            meter = metrics.metrics.get(name)
            if meter is not None:
                row[name] = float(meter.val)
        with path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return metrics, outputs

    return recorded


def _runtime_evidence(policy, evidence, rank):
    import torch

    write_json(
        evidence / f"runtime-rank-{rank}.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "parameters": sum(parameter.numel() for parameter in policy.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in policy.parameters()
                if parameter.requires_grad
            ),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": rank,
        },
    )


def _observe_checkpoints(save, evidence):
    def recorded(*args, **kwargs):
        result = save(*args, **kwargs)
        root = Path(kwargs["checkpoint_dir"])
        states = {
            "model": ["pretrained_model/model.safetensors"],
            "optimizer": [
                "training_state/optimizer_state.safetensors",
                "training_state/optimizer_param_groups.json",
            ],
            "scheduler": ["training_state/scheduler_state.json"],
            "rng": ["training_state/rng_state.safetensors"],
            "sampler": [
                "training_state/training_step.json",
                "pretrained_model/train_config.json",
            ],
        }
        publish_checkpoint(root, kwargs["step"], states, evidence / "checkpoints")
        recovery_uri = os.environ.get("NPA_VLA_RECOVERY_URI")
        if recovery_uri:
            from .turnkey_runtime import publish_recovery

            publish_recovery(root, kwargs["step"], recovery_uri)
        return result

    return recorded


if __name__ == "__main__":
    main()
