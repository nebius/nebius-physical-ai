"""Capture one node's real GPU readings with native-training process attribution."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _query(fields, section="gpu"):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-{section}=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        dict(zip(fields, [value.strip() for value in row], strict=True))
        for row in csv.reader(output.splitlines())
    ]


def _process(app, run, job):
    root = Path("/proc") / str(int(app["pid"]))
    command = (root / "cmdline").read_bytes().split(b"\0")
    if b"cosmos_framework.scripts.train" not in command:
        raise ValueError("GPU process is not the native training command")
    if str(run / "train.toml").encode() not in command:
        raise ValueError("GPU process uses a different training configuration")
    if not re.search(rf"/job_{job}(?:/|$)", (root / "cgroup").read_text()):
        raise ValueError("GPU process is outside the selected Slurm job")
    environment = dict(
        value.split(b"=", 1)
        for value in (root / "environ").read_bytes().split(b"\0")
        if b"=" in value
    )
    return {
        "rank": int(environment[b"RANK"]),
        "node_rank": int(environment[b"SLURM_NODEID"]),
        "local_rank": int(environment[b"LOCAL_RANK"]),
    }


def _assignments(run, job):
    apps = _query(["gpu_uuid", "pid"], "compute-apps")
    if len(apps) != 8 or len({app["gpu_uuid"] for app in apps}) != 8:
        raise ValueError("requires exactly one training process per local GPU")
    assigned = {app["gpu_uuid"]: _process(app, run, job) for app in apps}
    nodes = {row["node_rank"] for row in assigned.values()}
    if len(nodes) != 1:
        raise ValueError("processes disagree on their Slurm node rank")
    node = nodes.pop()
    if {row["local_rank"] for row in assigned.values()} != set(range(8)):
        raise ValueError("local ranks do not cover all eight GPUs")
    if {row["rank"] for row in assigned.values()} != set(range(node * 8, node * 8 + 8)):
        raise ValueError("global ranks do not match the Slurm node assignment")
    return assigned, node


def _devices(assigned):
    rows = _query(
        [
            "uuid",
            "timestamp",
            "index",
            "name",
            "driver_version",
            "memory.total",
            "memory.used",
            "utilization.gpu",
            "power.draw",
        ]
    )
    if len(rows) != 8 or {row["uuid"] for row in rows} != set(assigned):
        raise ValueError("GPU readings differ from attributed process devices")
    result = []
    for row in rows:
        if row["name"] != "NVIDIA B200":
            raise ValueError("expected NVIDIA B200")
        result.append(
            {
                **assigned[row["uuid"]],
                "gpu_index": int(row["index"]),
                "name": row["name"],
                "driver": row["driver_version"],
                "sample_utc": row["timestamp"],
                "memory_total_mib": float(row["memory.total"]),
                "memory_used_mib": float(row["memory.used"]),
                "utilization_percent": float(row["utilization.gpu"]),
                "power_watts": float(row["power.draw"]),
                "process_matches_training_command_and_slurm_cgroup": True,
            }
        )
    return result


def _progress(run):
    with (run / "node-0.log").open("rb") as stream:
        stream.seek(max(0, (run / "node-0.log").stat().st_size - 262144))
        text = stream.read().decode(errors="replace")
    updates = re.findall(r"(\d+) : iter_speed \S+ seconds per iteration", text)
    if not updates:
        raise ValueError("no completed timed optimizer update in the log tail")
    return int(updates[-1])


def _main(args):
    run = args.run_dir.resolve()
    assigned, node = _assignments(run, args.job_id)
    preflight_raw = (run / "distributed-preflight.json").read_bytes()
    preflight = json.loads(preflight_raw)
    settings_raw = (run / "run.json").read_bytes()
    settings = json.loads(settings_raw)
    if preflight["status"] != "passed" or preflight["world_size"] != settings["gpus"]:
        raise ValueError("missing matching distributed preflight")
    result = {
        "schema": "npa.cosmos3.wam-node-training-snapshot.v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "node_rank": node,
        "training_gpus": settings["gpus"],
        "last_monitored_optimizer_update": _progress(run),
        "training_processes_matched_on_this_host": len(assigned),
        "distributed_preflight": preflight,
        "distributed_preflight_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "run_settings_sha256": hashlib.sha256(settings_raw).hexdigest(),
        "capture_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "gpus": _devices(assigned),
        "scope": "One live sample per device; not training throughput or policy quality.",
    }
    with args.output_path.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"node_rank": node, "processes_attributed": len(assigned)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    _main(parser.parse_args())
