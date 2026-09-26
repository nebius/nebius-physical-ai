"""Record Linux block-device counters for checkpoint and input-I/O analysis."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time


def _sample(path, device):
    values = [int(value) for value in path.read_text().split()]
    if len(values) < 11 or any(value < 0 for value in values):
        raise ValueError("invalid Linux block-device counters")
    return {
        "unix_seconds": time.time(),
        "monotonic_seconds": time.monotonic(),
        "device": device,
        "reads_completed": values[0],
        "sectors_read": values[2],
        "writes_completed": values[4],
        "sectors_written": values[6],
        "io_milliseconds": values[9],
        "weighted_io_milliseconds": values[10],
        "sector_bytes": 512,
    }


def _main(args):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.device):
        raise ValueError("device must be a Linux block-device name")
    if not math.isfinite(args.interval) or args.interval <= 0:
        raise ValueError("interval must be positive and finite")
    path = Path("/sys/class/block") / args.device / "stat"
    _sample(path, args.device)
    stopped = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: stopped.set())
    with args.output_path.open("x") as stream:
        while not stopped.is_set():
            stream.write(json.dumps(_sample(path, args.device)) + "\n")
            stream.flush()
            stopped.wait(args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    os.umask(0o077)
    _main(parser.parse_args())
