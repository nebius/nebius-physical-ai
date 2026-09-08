"""Native Ray Train V2: synthetic regression, CUDA DDP, recoverable checkpoints.

Submit this application with the upstream Ray Jobs CLI. SkyPilot owns its host.
No infrastructure, job submission or credentials are managed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time


def digest(path: Path) -> str:
    """Use the shared checksum implementation on every supported Python version.

    Args:
        path: Artifact whose bytes must be hashed.
    Returns:
        Hexadecimal SHA-256 digest.
    Raises:
        OSError: The artifact cannot be read.
    """
    from artifacts import file_sha256

    return file_sha256(path)


def validate_recipe(recipe: dict) -> None:
    """Reject invalid or non-distributed recipes before connecting to Ray.

    Args:
        recipe: Complete training settings from the application arguments.
    Returns:
        None.
    Raises:
        ValueError: Training settings cannot prove distributed progress and recovery.
    """
    for key in ("workers", "steps", "samples_per_rank", "checkpoint_interval"):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if recipe["workers"] < 2:
        raise ValueError("This distributed reference requires at least two workers")
    if recipe["steps"] < 2:
        raise ValueError("At least two steps are required to verify loss improvement")
    if not math.isfinite(recipe["learning_rate"]) or not 0 < recipe["learning_rate"] < 1:
        raise ValueError("learning_rate must be finite and between zero and one")
    if not 0 <= recipe["fail_after_step"] < recipe["steps"]:
        raise ValueError("fail_after_step must precede the final step")
    if recipe["fail_after_step"] % recipe["checkpoint_interval"]:
        raise ValueError("fail_after_step must be a checkpoint boundary")


def _validate_ranks(ranks, recipe):
    """Reject missing devices or divergent parameters before accepting DDP evidence."""
    if sorted(rank["rank"] for rank in ranks) != list(range(recipe["workers"])):
        raise ValueError("Missing or duplicate distributed ranks")
    if any(rank["device_type"] != "cuda" or rank["world_size"] != recipe["workers"] for rank in ranks):
        raise ValueError("CUDA world-size evidence mismatch")
    if len({rank["device_fingerprint"] for rank in ranks}) != recipe["workers"]:
        raise ValueError("Ranks did not use distinct CUDA devices")
    if len({rank["parameter_sha256"] for rank in ranks}) != 1:
        raise ValueError("DDP parameters diverged")


def validate_journal(journal: list[dict], recipe: dict) -> None:
    """Require complete, finite optimizer and actual CUDA-rank evidence.

    Args:
        journal: Observed metrics and per-rank facts for every optimizer step.
        recipe: Expected distributed training and recovery settings.
    Returns:
        None.
    Raises:
        ValueError: Progress, timing, rank identity, or checkpoint recovery is invalid.
    """
    if [row["optimizer_step"] for row in journal] != list(range(1, recipe["steps"] + 1)):
        raise ValueError("Incomplete or duplicate optimizer steps")
    for row in journal:
        for key in ("loss", "gradient_norm", "parameter_delta", "samples_per_second", "learning_rate"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid {key}")
        # SGD momentum can move parameters even when this step's gradient is zero.
        if row["parameter_delta"] <= 0:
            raise ValueError("No optimizer progress")
        if row["samples_per_second"] <= 0 or row["learning_rate"] != recipe["learning_rate"]:
            raise ValueError("Throughput or applied learning rate mismatch")
        _validate_ranks(row["ranks"], recipe)
    if journal[-1]["loss"] >= journal[0]["loss"]:
        raise ValueError("Synthetic training loss did not improve")
    failure_step = recipe["fail_after_step"]
    if failure_step and any(rank["restored_from_step"] != failure_step for rank in journal[failure_step]["ranks"]):
        raise ValueError("Every worker must restore the committed failure checkpoint")


def validate_optimizer_checkpoint(optimizer, model) -> None:
    """Require complete SGD momentum and the expected constructor settings.

    Args:
        optimizer: Reloaded SGD optimizer to verify against its original defaults.
        model: Model whose ordered parameters must own every momentum buffer.
    Returns:
        None.
    Raises:
        ValueError: Parameter groups, hyperparameters, or momentum buffers differ.
    """
    import torch

    parameters = list(model.parameters())
    groups = optimizer.param_groups
    if len(groups) != 1 or len(groups[0]["params"]) != len(parameters):
        raise ValueError("Checkpoint optimizer parameter groups differ")
    if any(actual is not expected for actual, expected in zip(groups[0]["params"], parameters, strict=True)):
        raise ValueError("Checkpoint optimizer parameter order differs")
    if any(groups[0].get(key) != value for key, value in optimizer.defaults.items()):
        raise ValueError("Checkpoint optimizer hyperparameters differ from the recipe")
    if set(optimizer.state) != set(parameters):
        raise ValueError("Checkpoint optimizer lacks per-parameter momentum state")
    for parameter in parameters:
        buffer = optimizer.state[parameter].get("momentum_buffer")
        if (not isinstance(buffer, torch.Tensor) or buffer.shape != parameter.shape
                or buffer.dtype != parameter.dtype or buffer.device != parameter.device
                or not torch.isfinite(buffer).all().item()):
            raise ValueError("Checkpoint optimizer momentum must be finite and match every parameter")


def validate_torch_runtime() -> None:
    """Reject environment drift in the driver and every newly started worker.

    Args:
        None.
    Returns:
        None.
    Raises:
        RuntimeError: The imported Torch version is not the required CUDA build.
        ImportError: Torch is unavailable in the application environment.
    """
    import torch

    if torch.__version__ != "2.13.0+cu130":
        raise RuntimeError("This training runtime requires Torch 2.13.0+cu130")


def runtime_versions() -> dict[str, str]:
    """Keep runtime provenance compatible with safe weights-only checkpoint loading.

    Args:
        None.
    Returns:
        Torch, CUDA, and Ray versions as plain strings.
    Raises:
        ImportError: Required application runtime packages are unavailable.
    """
    import ray
    import torch

    return {"torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
            "ray_version": str(ray.__version__)}


def _training_shard(recipe, rank, device):
    """Give each rank a distinct deterministic shard without external data."""
    import torch

    generator = torch.Generator().manual_seed(recipe["seed"] + rank + 1)
    inputs = torch.randn(recipe["samples_per_rank"], 8, generator=generator).to(device)
    targets = inputs @ torch.arange(1, 9, device=device, dtype=torch.float32).reshape(8, 1) / 8 + 0.25
    return inputs, targets


def _restore_worker_checkpoint(recipe, model, optimizer, device):
    """Resume only a complete checkpoint from the same training recipe."""
    import torch
    from ray import train

    checkpoint = train.get_checkpoint()
    if checkpoint is None:
        return [], 0
    with checkpoint.as_directory() as directory:
        state = torch.load(Path(directory) / "state.pt", map_location=device, weights_only=True)
    if state["recipe"] != recipe:
        raise ValueError("Checkpoint recipe differs; choose a fresh run name")
    model.module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    validate_optimizer_checkpoint(optimizer, model.module)
    if len(state["journal"]) != state["step"]:
        raise ValueError("Checkpoint lacks a complete journal")
    return state["journal"], state["step"]


class _Worker:
    """Keep one rank's model, shard, and recovery evidence together."""

    def __init__(self, recipe):
        import torch
        from ray import train
        from ray.train.torch import get_device, prepare_model

        self.versions = runtime_versions()
        self.context = train.get_context()
        self.rank = self.context.get_world_rank()
        self.world_size = self.context.get_world_size()
        self.device = get_device()
        if self.device.type != "cuda" or self.world_size != recipe["workers"]:
            raise RuntimeError("CUDA workers and the requested world size are required")
        torch.manual_seed(recipe["seed"])
        self.model = prepare_model(torch.nn.Linear(8, 1))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=recipe["learning_rate"], momentum=0.8)
        self.inputs, self.targets = _training_shard(recipe, self.rank, self.device)
        self.journal, self.restored_step = _restore_worker_checkpoint(
            recipe, self.model, self.optimizer, self.device,
        )
        self.properties = torch.cuda.get_device_properties(self.device)
        self.fingerprint = hashlib.sha256(str(self.properties.uuid).encode()).hexdigest()


def _optimizer_step(worker, recipe):
    """Measure the synchronized optimizer interval before collecting rank hashes."""
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize(worker.device)
    started = time.perf_counter()
    before = torch.cat([parameter.detach().flatten() for parameter in worker.model.parameters()]).clone()
    worker.optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(worker.model(worker.inputs), worker.targets)
    loss.backward()
    gradient = torch.linalg.vector_norm(torch.cat([parameter.grad.flatten() for parameter in worker.model.parameters()]))
    worker.optimizer.step()
    parameters = torch.cat([parameter.detach().flatten() for parameter in worker.model.parameters()])
    delta = torch.linalg.vector_norm(parameters - before)
    aggregate = loss.detach().clone()
    dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
    aggregate /= worker.world_size
    torch.cuda.synchronize(worker.device)
    seconds = torch.tensor(time.perf_counter() - started, device=worker.device)
    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    return {
        "loss": aggregate.item(), "gradient_norm": gradient.item(), "parameter_delta": delta.item(),
        "learning_rate": worker.optimizer.param_groups[0]["lr"],
        "samples_per_second": recipe["samples_per_rank"] * worker.world_size / seconds.item(),
    }, parameters


def _gather_rank_evidence(worker, parameters):
    """Bind each observed rank to its actual device, node, and updated parameters."""
    import ray
    import torch.distributed as dist

    evidence = {
        "rank": worker.rank, "local_rank": worker.context.get_local_rank(), "world_size": worker.world_size,
        "device_type": worker.device.type, "device_name": worker.properties.name,
        **worker.versions,
        "node_fingerprint": hashlib.sha256(ray.get_runtime_context().get_node_id().encode()).hexdigest(),
        "device_fingerprint": worker.fingerprint,
        "parameter_sha256": hashlib.sha256(parameters.cpu().numpy().tobytes()).hexdigest(),
        "restored_from_step": worker.restored_step,
    }
    ranks = [None] * worker.world_size
    dist.all_gather_object(ranks, evidence)
    return ranks


def _fail_after_committed_checkpoint(worker, recipe, step):
    """Exercise native recovery only after every worker observes a durable checkpoint."""
    if step != recipe["fail_after_step"] or worker.restored_step != 0:
        return
    from ray import train
    import torch.distributed as dist

    committed = train.get_all_reported_checkpoints(
        consistency_mode=train.CheckpointConsistencyMode.COMMITTED,
    )
    if not any(item.metrics["optimizer_step"] == step for item in committed):
        raise RuntimeError("Failure checkpoint was not committed by Ray Train")
    dist.barrier()
    if worker.rank == 1:
        raise RuntimeError("Intentional worker failure after committed checkpoint")


def _report_checkpoint(worker, recipe, step, row):
    """Report on every worker while only rank zero uploads checkpoint bytes."""
    import torch
    from ray import train

    save = step % recipe["checkpoint_interval"] == 0 or step == recipe["steps"]
    with tempfile.TemporaryDirectory() as directory:
        saved = None
        if save and worker.rank == 0:
            torch.save({
                "model": worker.model.module.state_dict(), "optimizer": worker.optimizer.state_dict(),
                "step": step, "recipe": recipe, "journal": worker.journal,
            }, Path(directory) / "state.pt")
            saved = train.Checkpoint.from_directory(directory)
        # Train V2 report is the synchronization barrier for all workers.
        train.report({"loss": row["loss"], "optimizer_step": step}, checkpoint=saved)
    if save and worker.rank == 0:
        print(f"optimizer_step={step} checkpoint_uploaded", flush=True)
    _fail_after_committed_checkpoint(worker, recipe, step)


def train_loop(recipe: dict) -> None:
    """Train each CUDA rank with synchronized, recoverable Ray checkpoints.

    Args:
        recipe: Validated distributed training and failure-injection settings.
    Returns:
        None.
    Raises:
        ValueError: A checkpoint cannot restore the exact recipe and optimizer.
        RuntimeError: Runtime drift, distributed execution, or intentional failure occurs.
    """
    validate_torch_runtime()
    worker = _Worker(recipe)
    for step in range(worker.restored_step + 1, recipe["steps"] + 1):
        row, parameters = _optimizer_step(worker, recipe)
        row["optimizer_step"] = step
        row["ranks"] = _gather_rank_evidence(worker, parameters)
        worker.journal.append(row)
        _report_checkpoint(worker, recipe, step, row)


def _checkpoint_evaluation(state, recipe):
    """Require the safely reloaded model and momentum to retain actual CUDA progress."""
    import torch

    model = torch.nn.Linear(8, 1)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.SGD(model.parameters(), lr=recipe["learning_rate"], momentum=0.8)
    optimizer.load_state_dict(state["optimizer"])
    validate_optimizer_checkpoint(optimizer, model)
    inputs = torch.eye(8)
    target = torch.arange(1, 9).reshape(8, 1) / 8 + 0.25
    held_out_loss = torch.nn.functional.mse_loss(model(inputs), target).item()
    torch.manual_seed(recipe["seed"])
    baseline_loss = torch.nn.functional.mse_loss(torch.nn.Linear(8, 1)(inputs), target).item()
    if not math.isfinite(held_out_loss) or held_out_loss >= baseline_loss:
        raise ValueError("Reloaded checkpoint fails held-out improvement")
    parameters = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    parameter_hash = hashlib.sha256(parameters.numpy().tobytes()).hexdigest()
    if parameter_hash != state["journal"][-1]["ranks"][0]["parameter_sha256"]:
        raise ValueError("Reloaded model differs from final CUDA parameters")
    return {"held_out_loss": held_out_loss, "baseline_held_out_loss": baseline_loss,
            "parameter_sha256": parameter_hash}


def _validated_export_state(checkpoint, output, recipe):
    """Keep the native checkpoint open until validated bytes reach a fresh export."""
    import torch

    with checkpoint.as_directory() as directory:
        state_path = Path(directory) / "state.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["recipe"] != recipe or state["step"] != recipe["steps"]:
            raise ValueError("Final checkpoint identity or completed step differs")
        validate_journal(state["journal"], recipe)
        evaluation = _checkpoint_evaluation(state, recipe)
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        shutil.copy2(state_path, output / "state.pt")
    return state, evaluation


def _result_report(output, recipe, run_name, journal, evaluation):
    """Bind the observed journal, checkpoint, and evaluation to their application source."""
    return {
        "schema": "npa.ray-train-synthetic.v1", "run_name": run_name, "recipe": recipe,
        "source_sha256": digest(Path(__file__)), "checkpoint_sha256": digest(output / "state.pt"),
        "application_sources": {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "journal_sha256": digest(output / "metrics.json"), "checkpoint_reloaded": True,
        "optimizer_momentum_reloaded": True, **evaluation,
        "initial_loss": journal[0]["loss"], "final_loss": journal[-1]["loss"],
        "recovery_steps": sorted({rank["restored_from_step"] for row in journal for rank in row["ranks"]}),
        "throughput_scope": "Synchronized optimizer interval including global-loss reduction; excludes checkpoint/report and rank-hash collection.",
        "limitations": "Synthetic linear regression; CUDA ranks across Ray hosts; native S3 checkpoints and verified local export.",
    }


def export_result(checkpoint, output: Path, recipe: dict, run_name: str) -> dict:
    """Validate the final model and momentum before exporting factual artifacts.

    Args:
        checkpoint: Native Ray checkpoint containing the completed training state.
        output: Fresh local export directory.
        recipe: Exact recipe expected in the checkpoint.
        run_name: Application identifier used for the factual recording.
    Returns:
        Report binding source, checkpoint, journal, and Rerun artifact hashes.
    Raises:
        ValueError: Checkpoint identity, optimizer, rank evidence, or progress differs.
        OSError: Checkpoint reading or private artifact writing fails.
    """
    state, evaluation = _validated_export_state(checkpoint, output, recipe)
    journal = state["journal"]
    (output / "metrics.json").write_text(json.dumps(journal, indent=2) + "\n")
    report = _result_report(output, recipe, run_name, journal, evaluation)
    write_recording(output / "metrics.rrd", journal, report)
    report["rrd_sha256"] = digest(output / "metrics.rrd")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "SHA256SUMS").write_text("".join(
        f"{digest(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path.is_file()
    ))
    return report


def write_recording(path: Path, journal: list[dict], report: dict) -> None:
    """Write observed optimizer metrics and checkpoint events to Rerun.

    Args:
        path: Destination recording file.
        journal: Actual per-step observations to record.
        report: Run identity, recipe, and artifact provenance to embed.
    Returns:
        None.
    Raises:
        RuntimeError: Rerun cannot initialize or write the recording.
    """
    import rerun as rr

    recording = rr.RecordingStream("npa-ray-train-synthetic", recording_id=report["run_name"])
    recording.save(str(path))
    recording.log("provenance/run", rr.TextDocument(json.dumps(report, sort_keys=True)), static=True)
    for row in journal:
        recording.set_time("optimizer_step", sequence=row["optimizer_step"])
        for key in ("loss", "gradient_norm", "parameter_delta", "learning_rate", "samples_per_second"):
            recording.log(f"metrics/{key}", rr.Scalars(row[key]))
        recording.log("health/cuda_ranks", rr.Scalars(len(row["ranks"])))
        if row["optimizer_step"] % report["recipe"]["checkpoint_interval"] == 0 or row["optimizer_step"] == report["recipe"]["steps"]:
            recording.log("checkpoint/materialized", rr.Scalars(1))
    recording.flush()
    recording.disconnect()


def _arguments(argv):
    """Expose the reference recipe without changing infrastructure ownership."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-path", required=True, help="Run-scoped s3:// bucket/prefix for native Train checkpoints")
    parser.add_argument("--output-dir", required=True, type=Path, help="Fresh export directory on the Ray host")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--samples-per-rank", type=int, default=1024)
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fail-after-step", type=int, default=0)
    return parser.parse_args(argv)


def _submission_destination(args, recipe):
    """Reject unsafe destinations and discovery overrides before contacting Ray."""
    validate_recipe(recipe)
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.run_name):
        raise ValueError("run-name must be a short filesystem-safe application identifier")
    from artifacts import storage

    storage_fs, storage_path = storage(args.storage_path)
    if not args.output_dir.is_absolute():
        raise ValueError("output-dir must be absolute")
    if args.output_dir.exists():
        raise ValueError("output-dir already exists; preserve earlier evidence")
    if os.environ.get("RAY_TRAIN_V2_ENABLED", "1") != "1":
        raise RuntimeError("RAY_TRAIN_V2_ENABLED must be 1")
    address = os.environ.get("RAY_ADDRESS", "")
    if not address or not address.endswith(":6381"):
        raise RuntimeError("Submit through Ray Jobs on the isolated application Ray service (port 6381)")
    return storage_fs, storage_path, address


def _connect_application_ray(address):
    """Bind descendants' State API discovery to the selected application GCS."""
    import ray

    if ray.__version__ != "2.58.0":
        raise RuntimeError("This reference requires Ray 2.58.0 and Train V2")
    validate_torch_runtime()
    # Train's detached cleanup actor uses the State API without an address.
    # Propagate the selected application GCS so it cannot discover management Ray.
    ray.init(address=address, runtime_env={"env_vars": {
        "RAY_ADDRESS": address, "RAY_API_SERVER_ADDRESS": "",
    }})
    return ray


def _fit_training(args, recipe, storage_fs, storage_path):
    """Delegate distributed execution and failure recovery to native TorchTrainer."""
    import ray
    from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    live = [node for node in ray.nodes() if node["Alive"]]
    if sum(node["Resources"].get("GPU", 0) for node in live) < args.workers:
        raise RuntimeError("Reference requires enough distinct GPUs across its Ray hosts")
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop, train_loop_config=recipe,
        scaling_config=ScalingConfig(num_workers=args.workers, use_gpu=True, placement_strategy="SPREAD"),
        torch_config=TorchConfig(backend="nccl"),
        run_config=RunConfig(
            name=args.run_name, storage_path=storage_path, storage_filesystem=storage_fs,
            checkpoint_config=CheckpointConfig(),
            failure_config=FailureConfig(max_failures=1 if args.fail_after_step else 0),
        ),
    )
    result = trainer.fit()
    if result.error is not None or result.checkpoint is None:
        raise RuntimeError("Ray Train did not produce a successful checkpoint")
    return result.checkpoint


def main(argv: list[str] | None = None) -> None:
    """Run native training while leaving service and hosting ownership with the operator.

    Args:
        argv: Application arguments, or None to read the process arguments.
    Returns:
        None.
    Raises:
        ValueError: A recipe, destination, checkpoint, or artifact is invalid.
        RuntimeError: The selected runtime or distributed training fails.
        OSError: Artifact reading, writing, or publication fails.
    """
    from artifacts import publish

    args = _arguments(argv)
    recipe = {key: getattr(args, key) for key in (
        "workers", "steps", "samples_per_rank", "checkpoint_interval", "learning_rate", "seed", "fail_after_step",
    )}
    storage_fs, storage_path, address = _submission_destination(args, recipe)
    ray = _connect_application_ray(address)
    try:
        checkpoint = _fit_training(args, recipe, storage_fs, storage_path)
        report = export_result(checkpoint, args.output_dir, recipe, args.run_name)
        from inspect_results import inspect

        inspect(args.output_dir)
        publish(storage_fs, storage_path + "/" + args.run_name + "/exports", args.output_dir)
        print(json.dumps(report, sort_keys=True))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
