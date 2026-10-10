"""Freeze NCCL transport evidence matched to active native WAM trainer ranks."""

import argparse
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def _attribution(source):
    spec = importlib.util.spec_from_file_location("wam_attribution", source)
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    return capture, hashlib.sha256(source.read_bytes()).hexdigest()


def _log(run, pid):
    matches = list(run.glob(f"nccl.*.{int(pid)}.log"))
    if len(matches) != 1:
        raise ValueError("requires exactly one NCCL stream for the active process")
    return matches[0].read_bytes()


def _freeze_process(capture, app, args, world):
    identity = capture._process(app, args.run_dir, args.job_id)
    node, rank = identity["node_rank"], identity["rank"]
    raw = _log(args.run_dir, app["pid"])
    text = raw.decode(errors="replace")
    initialized = re.search(rf"rank {rank} nranks {world} .*Init COMPLETE", text)
    if "Using network IB" not in text or not initialized:
        raise ValueError("active trainer lacks completed world-size IB initialization")
    if "Using network Socket" in text:
        raise ValueError("active trainer selected socket transport")
    folder = args.output / f"node-{node}"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (folder / f"rank-{rank}.log").open("xb") as stream:
        stream.write(raw)
    return dict(
        identity,
        nccl_log_sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
        infiniband_selected=True,
        socket_selected=False,
        world_size=world,
        initialized_communicators=text.count("Init COMPLETE"),
        gpudirect_rdma_markers=text.count("GDRDMA"),
        nccl_warning_lines=sum("NCCL WARN" in line for line in text.splitlines()),
    )


def _main(args):
    args.run_dir = args.run_dir.resolve(strict=True)
    settings_raw = (args.run_dir / "run.json").read_bytes()
    settings = json.loads(settings_raw)
    world = settings["gpus"]
    if settings["nodes"] not in (2, 4) or world != 8 * settings["nodes"]:
        raise ValueError("requires two or four eight-GPU training workers")
    capture, source_hash = _attribution(args.capture_source)
    _, node = capture._assignments(args.run_dir, args.job_id)
    rows = [
        _freeze_process(capture, app, args, world)
        for app in capture._query(["gpu_uuid", "pid"], "compute-apps")
    ]
    if len(rows) != 8 or {row["rank"] for row in rows} != set(
        range(node * 8, node * 8 + 8)
    ):
        raise ValueError("requires eight distinct native trainer ranks per worker")
    receipt = {
        "schema": "npa.cosmos3.wam-native-transport-snapshot.v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "node_rank": node,
        "processes": sorted(rows, key=lambda row: row["rank"]),
        "run_settings_sha256": hashlib.sha256(settings_raw).hexdigest(),
        "capture_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "attribution_source_sha256": source_hash,
        "scope": "Frozen NCCL log prefixes matched to active native trainer processes, job and configuration; no link-rate or collective-bandwidth claim",
    }
    with (args.output / f"node-{node}" / "transport.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"node_rank": node, "infiniband_training_processes": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--capture-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    _main(parser.parse_args())
