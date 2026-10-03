"""Verify B200 rank placement and an actual cross-node NCCL collective before WAM training."""

import json
import os
from pathlib import Path
import socket


def _verify_placement(identities, world, devices_per_host):
    if devices_per_host not in (4, 8):
        raise RuntimeError("expected four or eight devices per host")
    hosts = {host for host, _ in identities}
    expected = set(range(devices_per_host))
    if world % devices_per_host or len(hosts) != world // devices_per_host:
        raise RuntimeError("rank host count differs from the planned topology")
    if len(identities) != world or len(set(identities)) != world:
        raise RuntimeError("ranks do not cover distinct devices per host")
    for host in hosts:
        if {device for name, device in identities if name == host} != expected:
            raise RuntimeError("local ranks differ from the planned GPU count")
    return len(hosts)


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
    hosts = _verify_placement(
        identities, world, int(os.environ["NPA_WAM_GPUS_PER_NODE"])
    )
    if rank == 0:
        report = {
            "world_size": world,
            "hosts": hosts,
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
