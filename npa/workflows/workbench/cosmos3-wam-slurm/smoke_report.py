"""Verify the two-update RTX smoke against native training receipts."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import uuid

from recipe import RTX_PRO_6000
from report import _check_log_errors, _checkpoint

_UPDATE = re.compile(
    r"Iteration (\d+): Hit counter: (\d+)/(\d+) \| Loss: (\S+) \| Time: (\S+)s"
)


def _updates(log):
    _check_log_errors(log)
    updates = []
    for match in _UPDATE.finditer(log):
        step, hit, threshold = (int(match[index]) for index in (1, 2, 3))
        loss, seconds = float(match[4]), float(match[5])
        if step != hit or threshold != 50:
            raise ValueError("iteration and native warmup counter disagree")
        if not math.isfinite(loss) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("need finite native loss and positive warmup time")
        updates.append({"step": step, "loss": loss, "warmup_seconds": seconds})
    if [update["step"] for update in updates] != [1, 2]:
        raise ValueError("need exactly two native completed optimizer updates")
    return updates


def _trainer_ranks(run, count, run_sha256):
    directory = run / "ranks"
    if {p.name for p in directory.glob("*.json")} != {
        f"{rank}.json" for rank in range(count)
    }:
        raise ValueError("trainer receipt set differs from the planned ranks")
    records = []
    for rank in range(count):
        record = json.loads((directory / f"{rank}.json").read_text())
        for key, expected in (("rank", rank), ("local_rank", rank), ("returncode", 0)):
            if type(record[key]) is not int or record[key] != expected:
                raise ValueError(f"rank {rank} has invalid {key}")
        if record["run_sha256"] != run_sha256:
            raise ValueError(f"rank {rank} belongs to a different run")
        device = record["device_uuid"]
        if (
            not isinstance(device, str)
            or str(uuid.UUID(device)) != device
            or not uuid.UUID(device).int
        ):
            raise ValueError(f"rank {rank} has an invalid device UUID")
        if not isinstance(record["host"], str) or not record["host"].strip():
            raise ValueError(f"rank {rank} has no host")
        peak = record["peak_allocated_bytes"]
        if type(peak) is not int or peak <= 0:
            raise ValueError(f"rank {rank} has invalid peak CUDA memory")
        records.append(record)
    if (
        len({r["host"] for r in records}) != 1
        or len({r["device_uuid"] for r in records}) != count
    ):
        raise ValueError("expected one host with distinct trainer devices")
    return records


def _fixed_batch(settings, job):
    import yaml

    planned = {
        "nodes": 1,
        "gpus": 8,
        "gpus_per_node": 8,
        "steps": 2,
        "samples_per_rank": 64,
        "gradient_accumulation": 4,
        "global_batch": 2048,
    }
    for key, expected in planned.items():
        if type(settings.get(key)) is not int or settings[key] != expected:
            raise ValueError(f"qualification requires {key}={expected}")
    config = yaml.safe_load((job / "config.yaml").read_text())
    fields = {
        "dataloader_train.max_samples_per_batch": 64,
        "trainer.grad_accum_iter": 4,
        "trainer.max_iter": 2,
        "model.config.parallelism.data_parallel_shard_degree": 8,
        "model.config.parallelism.data_parallel_replicate_degree": 1,
        "model.config.parallelism.context_parallel_shard_degree": 1,
    }
    for path, expected in fields.items():
        value = config
        for key in path.split("."):
            value = value[key]
        if type(value) is not int or value != expected:
            raise ValueError(f"native resolved config requires {path}={expected}")
    return fields


def _local_timing(log, node):
    # The pinned rank-zero logger emits UTC month/day and whole seconds.
    matches = re.findall(
        r"^\[(\d\d-\d\d) (\d\d:\d\d:\d\d)\|[^\n]*\] Starting training\.\.\.\s*$",
        log,
        re.MULTILINE,
    )
    saves = re.findall(r"Checkpoint save completed: Time taken: ([\d.]+) seconds", log)
    if len(matches) != 1 or len(saves) != 1:
        raise ValueError(
            "expected one native training-start marker and one synchronous final save"
        )
    day, clock = matches[0]
    launched = datetime.fromtimestamp(
        node["trainer_started_unix"], timezone.utc
    ).replace(microsecond=0)
    # Accept midnight and year boundaries within this single 45-minute run.
    start_day = (
        launched if launched.strftime("%m-%d") == day else launched + timedelta(days=1)
    )
    if start_day.strftime("%m-%d") != day:
        raise ValueError("native startup timestamp is outside the launch interval")
    marker = datetime.combine(
        start_day.date(), datetime.strptime(clock, "%H:%M:%S").time(), timezone.utc
    )
    startup = (marker - launched).total_seconds()
    save_seconds = float(saves[0])
    if (
        not 0 <= startup <= node["train_process_seconds"] + 1
        or not math.isfinite(save_seconds)
        or save_seconds <= 0
    ):
        raise ValueError("invalid native startup or save time")
    return {
        "trainer_startup_seconds": startup,
        "startup_scope": "trainer launch to native Starting training marker, UTC whole seconds",
        "worker_preflight_seconds": node["preflight_seconds"],
        "save_seconds": save_seconds,
    }


def _check_empty_shards(checkpoint, files):
    empty = {name for name, entry in files.items() if entry["bytes"] == 0}
    if not empty:
        return
    if any(
        not re.fullmatch(r"scheduler/__[0-9]+_[0-9]+\.distcp", name) for name in empty
    ):
        raise ValueError(
            "terminal checkpoint contains an empty file outside scheduler writer placeholders"
        )
    # DCP can create a writer file on every rank for replicated scheduler state.
    # The metadata identifies which shards actually hold scheduler records.
    from torch.distributed.checkpoint import FileSystemReader

    storage = FileSystemReader(checkpoint / "scheduler").read_metadata().storage_data
    if not storage:
        raise ValueError("empty scheduler shards require a nonempty DCP storage map")
    for extent in storage.values():
        name = extent.relative_path
        if not re.fullmatch(r"__[0-9]+_[0-9]+\.distcp", name):
            raise ValueError("unexpected scheduler shard name in DCP metadata")
        entry = files.get(f"scheduler/{name}")
        if (
            entry is None
            or extent.offset < 0
            or extent.length <= 0
            or extent.offset + extent.length > entry["bytes"]
        ):
            raise ValueError("scheduler metadata references missing or truncated bytes")


def summarize(run):
    settings_bytes = (run / "run.json").read_bytes()
    settings = json.loads(settings_bytes)
    if settings.get("hardware", {}).get("label") != RTX_PRO_6000["label"] or (
        settings["nodes"],
        settings["gpus"],
        settings["steps"],
    ) != (1, 8, 2):
        raise ValueError("expected a one-node eight-RTX run with two optimizer updates")
    proof = json.loads((run / "distributed-preflight.json").read_text())
    if tuple(
        proof[key]
        for key in ("status", "backend", "world_size", "hosts", "all_reduce_sum")
    ) != ("passed", "nccl", 8, 1, 36):
        raise ValueError("missing matching eight-rank NCCL proof")
    node = json.loads((run / "node-0.finished.json").read_text())
    if (node["rank"], node["returncode"]) != (0, 0):
        raise ValueError("the worker must finish successfully")
    if node["run_sha256"] != hashlib.sha256(settings_bytes).hexdigest():
        raise ValueError("run settings changed after execution")
    if node["hardware"]["gpu_names"] != [RTX_PRO_6000["names"][0]] * 8:
        raise ValueError("worker GPU inventory differs from the reserved RTX SKU")
    train_seconds = node["train_process_seconds"]
    if not math.isfinite(train_seconds) or train_seconds <= 0:
        raise ValueError("invalid training process duration")
    log = (run / "node-0.log").read_text()
    updates = _updates(log)
    rank_info = {}
    if settings.get("launcher") == "torchrun":
        rank_info["resolved_batch_fields"] = _fixed_batch(
            settings, run / "output/cosmos3_wam/libero_10" / settings["name"]
        )
        rank_info.update(_local_timing(log, node))
        rank_info["trainer_ranks"] = _trainer_ranks(
            run, settings["gpus"], node["run_sha256"]
        )
    hash_started = time.monotonic()
    checkpoint_sha256, checkpoint_files = _checkpoint(run, settings)
    checkpoint = (
        run
        / "output/cosmos3_wam/libero_10"
        / settings["name"]
        / "checkpoints"
        / "iter_000000002"
    )
    _check_empty_shards(checkpoint, checkpoint_files)
    return {
        "schema": "npa.cosmos3.wam-rtx-smoke.v1",
        "run_sha256": node["run_sha256"],
        "completed_optimizer_updates": len(updates),
        "updates_finite": True,
        "updates": updates,
        **rank_info,
        "warmup_mean_seconds": math.fsum(update["warmup_seconds"] for update in updates)
        / len(updates),
        "train_process_seconds": train_seconds,
        "terminal_step": settings["steps"],
        "checkpoint_hash_seconds": time.monotonic() - hash_started,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_files": checkpoint_files,
        "timing_scope": "native warmup updates; process duration includes startup and final save",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.run_dir), indent=2))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Smoke verification failed: {error}\n")
