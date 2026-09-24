"""Verify B200 rank placement and an actual cross-node NCCL collective before WAM training."""

import json
import os
from pathlib import Path
import socket


def _check():
    import torch
    import torch.distributed as distributed

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init_process_group("nccl")
    rank, world = distributed.get_rank(), distributed.get_world_size()
    value = torch.tensor(float(rank + 1), device="cuda")
    distributed.all_reduce(value)
    if value.item() != world * (world + 1) / 2:
        raise RuntimeError("NCCL all-reduce result is incorrect")
    identity = (socket.gethostname(), int(os.environ["LOCAL_RANK"]))
    identities = [None] * world
    distributed.all_gather_object(identities, identity)
    hosts = {host for host, _ in identities}
    if len(hosts) != world // 8 or len(set(identities)) != world:
        raise RuntimeError("ranks do not cover eight distinct devices per host")
    if rank == 0:
        report = {
            "world_size": world,
            "hosts": len(hosts),
            "backend": "nccl",
            "all_reduce_sum": value.item(),
            "status": "passed",
        }
        destination = Path(os.environ["NPA_WAM_RUN_DIR"]) / "distributed-preflight.json"
        destination.write_text(json.dumps(report, indent=2) + "\n")
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    _check()
