"""Run one native Cosmos trainer rank and record its physical device and outcome."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import uuid


def _run_native():
    try:
        runpy.run_module(
            "cosmos_framework.scripts.train", run_name="__main__", alter_sys=True
        )
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    return 0


def _train_and_measure(cuda, local_rank, record):
    # Cosmos resets PyTorch's allocator peak in its monitoring callbacks.
    # Keep the maximum across those windows while preserving each native reset.
    native_reset = cuda.reset_peak_memory_stats
    peak = 0

    def reset_peak(device=None):
        nonlocal peak
        peak = max(peak, cuda.max_memory_allocated(local_rank))
        return native_reset(device)

    cuda.reset_peak_memory_stats = reset_peak
    try:
        return _run_native()
    finally:
        cuda.reset_peak_memory_stats = native_reset
        record["peak_allocated_bytes"] = max(
            peak, cuda.max_memory_allocated(local_rank)
        )


def _rank_main():
    import torch

    run = Path(os.environ["NPA_WAM_RUN_DIR"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank or not 0 <= rank < 8 or os.environ["WORLD_SIZE"] != "8":
        raise ValueError("expected eight local trainer ranks on one RTX host")
    torch.cuda.set_device(local_rank)
    device = uuid.UUID(
        str(torch.cuda.get_device_properties(local_rank).uuid).removeprefix("GPU-")
    )
    if device.int == 0:
        raise ValueError("CUDA device UUID is empty")
    directory = run / "ranks"
    directory.mkdir(mode=0o700, exist_ok=True)
    record = {
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "device_uuid": str(device),
        "run_sha256": hashlib.sha256((run / "run.json").read_bytes()).hexdigest(),
    }
    # The native optional fast exit bypasses Python finally blocks. Defer it
    # until the success receipt is closed, retaining the native shutdown choice.
    fast_exit = os.environ.pop("COSMOS_EXIT_WITHOUT_FINALIZE", "").lower() in (
        "1",
        "true",
    )
    with (directory / f"{rank}.json").open("x") as stream:
        code = 1
        try:
            code = _train_and_measure(torch.cuda, local_rank, record)
        finally:
            record["returncode"] = code
            json.dump(record, stream)
    if fast_exit and code == 0:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return code


if __name__ == "__main__":
    sys.exit(_rank_main())
