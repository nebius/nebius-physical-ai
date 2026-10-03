"""Freeze InfiniBand diagnostics matched to one worker's active native WAM ranks."""

import argparse
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def _attribution():
    source = (
        Path(__file__).resolve().parent.parent
        / "cosmos3-wam-live-training-16/capture.py"
    )
    spec = importlib.util.spec_from_file_location("wam_attribution", source)
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    return capture, hashlib.sha256(source.read_bytes()).hexdigest()


def _freeze_process(capture, app, args):
    identity = capture._process(app, args.run_dir, args.job_id)
    node, rank = identity["node_rank"], identity["rank"]
    source = args.run_dir / f"nccl-node-{node}-{int(app['pid'])}.log"
    raw = source.read_bytes()
    text = raw.decode(errors="replace")
    world = re.search(rf"rank {rank} nranks 32 .*Init COMPLETE", text)
    if "NET/IB : Using" not in text or not world:
        raise ValueError("active trainer lacks completed 32-rank IB initialization")
    folder = args.output / f"node-{node}"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (folder / f"rank-{rank}.log").open("xb") as stream:
        stream.write(raw)
    return dict(
        identity,
        nccl_log_sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
        infiniband_selected=True,
        world_size=32,
        initialized_communicators=text.count("Init COMPLETE"),
        nccl_warning_lines=sum("NCCL WARN" in line for line in text.splitlines()),
    )


def _main(args):
    capture, source_hash = _attribution()
    capture._assignments(args.run_dir, args.job_id)
    rows = [
        _freeze_process(capture, app, args)
        for app in capture._query(["gpu_uuid", "pid"], "compute-apps")
    ]
    nodes = {row["node_rank"] for row in rows}
    if len(rows) != 8 or len(nodes) != 1 or len({row["rank"] for row in rows}) != 8:
        raise ValueError("requires eight distinct native trainer ranks per worker")
    node = nodes.pop()
    receipt = {
        "schema": "npa.cosmos3.wam-native-transport-snapshot.v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "node_rank": node,
        "processes": sorted(rows, key=lambda row: row["rank"]),
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
    parser.add_argument("--output", type=Path, required=True)
    _main(parser.parse_args())
