"""Render the real distributed and optimizer evidence probes used by GR00T."""

from __future__ import annotations

import ast
import inspect
import json
import math
from typing import Any, Mapping


def parse_training_loss_evidence(
    training_log: str,
    *,
    training_step: int,
    logging_steps: int | None,
    trainer_log_history: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Separate real optimizer-step losses from the end-of-run aggregate."""

    if int(training_step) < 0:
        raise ValueError("training_step must be nonnegative")
    if logging_steps is not None and int(logging_steps) <= 0:
        raise ValueError("logging_steps must be positive when supplied")

    decoded_lines: list[Mapping[str, Any]] = []
    for line in training_log.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        for loader in (json.loads, ast.literal_eval):
            try:
                value = loader(candidate)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, Mapping):
                decoded_lines.append(value)
                break

    aggregate_losses = [
        float(item["train_loss"])
        for item in decoded_lines
        if isinstance(item.get("train_loss"), (int, float))
        and math.isfinite(float(item["train_loss"]))
    ]
    state_records = list(trainer_log_history or [])
    source_records = state_records or decoded_lines
    loss_history: list[dict[str, Any]] = []
    missing_step: list[float] = []
    for item in source_records:
        value = item.get("loss")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        optimizer_step = next(
            (
                int(item[key])
                for key in ("optimizer_step", "global_step", "step")
                if isinstance(item.get(key), (int, float)) and int(item[key]) > 0
            ),
            None,
        )
        if optimizer_step is None:
            missing_step.append(float(value))
        else:
            loss_history.append(
                {"optimizer_step": optimizer_step, "loss": float(value)}
            )

    if missing_step:
        raise ValueError(
            "per-step loss records omit actual optimizer/global steps; refusing "
            "to synthesize steps from logging_steps"
        )

    loss_history.sort(key=lambda item: int(item["optimizer_step"]))
    steps = [int(item["optimizer_step"]) for item in loss_history]
    if len(steps) != len(set(steps)):
        raise ValueError("per-step loss evidence contains duplicate optimizer steps")
    if steps and max(steps) > int(training_step):
        raise ValueError("loss evidence exceeds the trainer's completed global step")
    if loss_history:
        loss_step_source = (
            "trainer_state.log_history.explicit_global_step"
            if state_records
            else "training_log.explicit_optimizer_or_global_step"
        )
    elif aggregate_losses:
        loss_step_source = "aggregate_train_loss_only"
    else:
        loss_step_source = "no_loss_records"
    cadence_matches = None
    if logging_steps is not None and steps:
        cadence = int(logging_steps)
        cadence_matches = all(
            step % cadence == 0 or step == int(training_step) for step in steps
        )
    return {
        "loss_history": loss_history,
        "aggregate_train_loss": aggregate_losses[-1] if aggregate_losses else None,
        "final_step_loss": loss_history[-1]["loss"] if loss_history else None,
        "loss_step_source": loss_step_source,
        "loss_step_inference": None,
        "declared_logging_steps": logging_steps,
        "loss_logging_cadence_matches": cadence_matches,
    }


def render_distributed_probe(output_dir: str) -> str:
    """Return a torchrun program that proves ranks, GPUs, and an NCCL collective."""

    return f"""import json
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
if world_size > 1:
    dist.init_process_group(backend="nccl")
value = torch.tensor(float(rank + 1), device=f"cuda:{{local_rank}}")
if world_size > 1:
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
rank_dir = Path({output_dir!r}) / "distributed-ranks"
rank_dir.mkdir(parents=True, exist_ok=True)
rank_payload = {{
    "rank": rank,
    "local_rank": local_rank,
    "world_size": world_size,
    "cuda_device_name": torch.cuda.get_device_name(local_rank),
    "collective_sum": float(value.item()),
}}
(rank_dir / f"rank-{{rank:04d}}.json").write_text(
    json.dumps(rank_payload, sort_keys=True) + "\\n", encoding="utf-8"
)
if world_size > 1:
    dist.barrier()
if rank == 0:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpu_uuids = [line.strip() for line in raw.splitlines() if line.strip()]
    ranks = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(rank_dir.glob("rank-*.json"))]
    expected = float(world_size * (world_size + 1) // 2)
    evidence = {{
        "world_size": world_size,
        "rank_count": len(ranks),
        "ranks": ranks,
        "gpu_uuids": gpu_uuids,
        "distinct_gpu_count": len(set(gpu_uuids)),
        "collective_sum": float(value.item()),
        "collective_expected": expected,
        "collective_ok": float(value.item()) == expected,
    }}
    (Path({output_dir!r}) / "npa_groot_distributed_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )
    print("NPA_GROOT_DISTRIBUTED_EVIDENCE", json.dumps(evidence, sort_keys=True))
if world_size > 1:
    dist.barrier()
    dist.destroy_process_group()
"""


def render_training_rank_wrapper(output_dir: str) -> str:
    """Return a rank-local wrapper around the real vendor trainer.

    A completion marker is written only after ``launch_finetune.py`` returns
    successfully on that rank.  This distinguishes an NCCL preflight from
    evidence that every rank actually stayed in the vendor training process.
    """

    return f"""import json
import os
import runpy
from pathlib import Path

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
try:
    runpy.run_path("gr00t/experiment/launch_finetune.py", run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
rank_dir = Path({output_dir!r}) / "training-ranks"
rank_dir.mkdir(parents=True, exist_ok=True)
payload = {{
    "rank": rank,
    "local_rank": local_rank,
    "world_size": world_size,
    "status": "completed_vendor_training",
}}
(rank_dir / f"rank-{{rank:04d}}.json").write_text(
    json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8"
)
print("NPA_GROOT_TRAINING_RANK_COMPLETE", json.dumps(payload, sort_keys=True))
"""


def render_training_manifest_script(
    *,
    output_dir: str,
    manifest_path: str,
    manifest_fields: Mapping[str, Any],
) -> str:
    """Return a post-train program that extracts finite loss and checkpoint facts."""

    # The post-train script runs inside the vendor GR00T interpreter.  Embedding
    # this dependency-free parser is intentional: importing it through
    # ``npa.cli`` executes that package's Typer bootstrap and can make a fully
    # completed multi-hour training job fail solely because the vendor image's
    # pre-existing Typer/Click pair is incompatible.  The generated evidence
    # program needs no CLI framework at all.
    parser_source = inspect.getsource(parse_training_loss_evidence)
    manifest_literal = (
        "{\n"
        + "".join(
            f'    "{key}": {value!r},\n' for key, value in manifest_fields.items()
        )
        + "}"
    )
    return f"""import ast
import json
import math
from pathlib import Path
from typing import Any, Mapping

{parser_source}

output_dir = Path({output_dir!r})
evidence_path = output_dir / "npa_groot_distributed_evidence.json"
evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
training_log = (output_dir / "training.log").read_text(encoding="utf-8", errors="replace")
checkpoint_steps = []
for path in output_dir.rglob("checkpoint-*"):
    if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit():
        checkpoint_steps.append(int(path.name.removeprefix("checkpoint-")))
checkpoint_files = [path for path in output_dir.rglob("*") if path.is_file()]
checkpoint_bytes = sum(path.stat().st_size for path in checkpoint_files)
checkpoint_objects = len(checkpoint_files)
manifest = {manifest_literal}
trainer_histories = []
trainer_global_steps = []
for state_path in output_dir.rglob("trainer_state.json"):
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        continue
    history = state.get("log_history") if isinstance(state, dict) else None
    if isinstance(history, list):
        trainer_histories.append(history)
    global_step = state.get("global_step") if isinstance(state, dict) else None
    if isinstance(global_step, (int, float)) and int(global_step) >= 0:
        trainer_global_steps.append(int(global_step))
training_step = max(trainer_global_steps + checkpoint_steps, default=0)
trainer_history = max(trainer_histories, key=len, default=[])
loss_evidence = parse_training_loss_evidence(
    training_log,
    training_step=training_step,
    logging_steps=manifest.get("logging_steps"),
    trainer_log_history=trainer_history,
)
loss_history = loss_evidence["loss_history"]
loss = loss_evidence["final_step_loss"]
loss_finite = loss is not None and math.isfinite(loss)
optimizer_step_ok = training_step >= 1 and loss_finite
loss_steps = [int(item["optimizer_step"]) for item in loss_history]
loss_steps_real = (
    len(loss_steps) >= 2
    and loss_steps == sorted(loss_steps)
    and len(loss_steps) == len(set(loss_steps))
    and max(loss_steps) <= training_step
    and loss_evidence["loss_step_inference"] is None
)
window = min(5, len(loss_history) // 2)
if window >= 2:
    early_values = sorted(float(item["loss"]) for item in loss_history[:window])
    late_values = sorted(float(item["loss"]) for item in loss_history[-window:])
    middle = window // 2
    if window % 2:
        robust_early_loss = early_values[middle]
        robust_late_loss = late_values[middle]
    else:
        robust_early_loss = (early_values[middle - 1] + early_values[middle]) / 2
        robust_late_loss = (late_values[middle - 1] + late_values[middle]) / 2
else:
    robust_early_loss = None
    robust_late_loss = None
loss_decreased = (
    robust_early_loss is not None
    and robust_late_loss is not None
    and robust_late_loss < robust_early_loss * 0.99
)
rank_ids = sorted(int(item.get("rank", -1)) for item in evidence.get("ranks") or [])
expected_rank_ids = list(range(int(evidence.get("world_size") or 0)))
training_rank_evidence = []
for rank_path in sorted((output_dir / "training-ranks").glob("rank-*.json")):
    try:
        row = json.loads(rank_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        continue
    if isinstance(row, dict):
        training_rank_evidence.append(row)
training_rank_ids = sorted(
    int(item.get("rank", -1))
    for item in training_rank_evidence
    if item.get("status") == "completed_vendor_training"
)
rank_scoped_checkpoint_files = [
    path
    for path in checkpoint_files
    if "checkpoint-" in str(path.relative_to(output_dir))
    and any(part.startswith("rank-") for part in path.relative_to(output_dir).parts)
]
rank_zero_checkpoint_only = (
    bool(checkpoint_steps)
    and rank_ids == expected_rank_ids
    and not rank_scoped_checkpoint_files
)
manifest.update({{
    "world_size": int(evidence.get("world_size") or 0),
    "distinct_gpu_count": int(evidence.get("distinct_gpu_count") or 0),
    "gpu_uuids": list(evidence.get("gpu_uuids") or []),
    "rank_evidence": list(evidence.get("ranks") or []),
    "observed_ranks": rank_ids,
    "training_rank_evidence": training_rank_evidence,
    "training_observed_ranks": training_rank_ids,
    "both_ranks_trained": (
        int(evidence.get("world_size") or 0) == 2
        and rank_ids == [0, 1]
        and training_rank_ids == [0, 1]
        and training_step >= 2
    ),
    "collective_sum": evidence.get("collective_sum"),
    "collective_expected": evidence.get("collective_expected"),
    "collective_ok": evidence.get("collective_ok") is True,
    "training_step": training_step,
    "checkpoint_steps": sorted(set(checkpoint_steps)),
    "checkpoint_publication_process": "single launcher after torchrun completed",
    "checkpoint_upload_invocations": 1,
    "rank_zero_checkpoint_only": rank_zero_checkpoint_only,
    "optimizer_step_ok": optimizer_step_ok,
    "loss": loss,
    "loss_finite": loss_finite,
    "loss_history": loss_history,
    "aggregate_train_loss": loss_evidence["aggregate_train_loss"],
    "final_step_loss": loss_evidence["final_step_loss"],
    "loss_step_source": loss_evidence["loss_step_source"],
    "loss_step_inference": loss_evidence["loss_step_inference"],
    "loss_steps_real": loss_steps_real,
    "robust_early_loss": robust_early_loss,
    "robust_late_loss": robust_late_loss,
    "loss_decreased": loss_decreased,
    "initial_loss": loss_history[0]["loss"] if loss_history else None,
    "final_loss": loss_history[-1]["loss"] if loss_history else None,
    "training_examples": training_step * int(manifest.get("global_batch_size") or 0),
    "gpu_model": (evidence.get("ranks") or [{{}}])[0].get("cuda_device_name", ""),
    "checkpoint_object_count": checkpoint_objects,
    "checkpoint_bytes": checkpoint_bytes,
}})
target = Path({manifest_path!r})
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(f"NPA_GROOT_FINETUNE_MANIFEST {{target}}")
"""


__all__ = [
    "parse_training_loss_evidence",
    "render_distributed_probe",
    "render_training_rank_wrapper",
    "render_training_manifest_script",
]
