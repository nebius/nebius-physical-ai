"""Render the real distributed and optimizer evidence probes used by GR00T."""

from __future__ import annotations

from typing import Any, Mapping


def render_distributed_probe(output_dir: str) -> str:
    """Return a torchrun program that proves ranks, GPUs, and an NCCL collective."""

    return f'''import json
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
'''


def render_training_manifest_script(
    *,
    output_dir: str,
    manifest_path: str,
    manifest_fields: Mapping[str, Any],
) -> str:
    """Return a post-train program that extracts finite loss and checkpoint facts."""

    manifest_literal = "{\n" + "".join(
        f'    "{key}": {value!r},\n' for key, value in manifest_fields.items()
    ) + "}"
    return f'''import json
import math
import re
from pathlib import Path

output_dir = Path({output_dir!r})
evidence_path = output_dir / "npa_groot_distributed_evidence.json"
evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
training_log = (output_dir / "training.log").read_text(encoding="utf-8", errors="replace")
loss_values = []
for pattern in (
    r"[\\x27\\x22]loss[\\x27\\x22]\\s*:\\s*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)",
    r"[\\x27\\x22]train_loss[\\x27\\x22]\\s*:\\s*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)",
):
    loss_values.extend(float(value) for value in re.findall(pattern, training_log))
if not loss_values:
    fallback = r"(?i)(?:train[_ ]?)?loss[^0-9+.-]{{0,6}}([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)"
    loss_values.extend(float(value) for value in re.findall(fallback, training_log))
finite_losses = [value for value in loss_values if math.isfinite(value)]
checkpoint_steps = []
for path in output_dir.rglob("checkpoint-*"):
    if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit():
        checkpoint_steps.append(int(path.name.removeprefix("checkpoint-")))
training_step = max(checkpoint_steps, default=0)
checkpoint_files = [path for path in output_dir.rglob("*") if path.is_file()]
checkpoint_bytes = sum(path.stat().st_size for path in checkpoint_files)
checkpoint_objects = len(checkpoint_files)
loss = finite_losses[-1] if finite_losses else None
loss_finite = loss is not None and math.isfinite(loss)
optimizer_step_ok = training_step >= 1 and loss_finite
manifest = {manifest_literal}
logging_steps = 10
loss_history = [
    {{
        "optimizer_step": min((index + 1) * logging_steps, training_step),
        "loss": value,
    }}
    for index, value in enumerate(finite_losses)
]
loss_history = [
    item
    for index, item in enumerate(loss_history)
    if index == 0 or item["loss"] != loss_history[index - 1]["loss"]
]
if loss_history and training_step > 0:
    loss_history[-1]["optimizer_step"] = training_step
manifest.update({{
    "world_size": int(evidence.get("world_size") or 0),
    "distinct_gpu_count": int(evidence.get("distinct_gpu_count") or 0),
    "gpu_uuids": list(evidence.get("gpu_uuids") or []),
    "rank_evidence": list(evidence.get("ranks") or []),
    "collective_sum": evidence.get("collective_sum"),
    "collective_expected": evidence.get("collective_expected"),
    "collective_ok": evidence.get("collective_ok") is True,
    "training_step": training_step,
    "optimizer_step_ok": optimizer_step_ok,
    "loss": loss,
    "loss_finite": loss_finite,
    "loss_history": loss_history,
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
'''


__all__ = ["render_distributed_probe", "render_training_manifest_script"]
