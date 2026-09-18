"""Measure real NCCL all-reduce without overriding transport selection.

Launch with torchrun --standalone --nproc_per_node=N gpu_collective_probe.py.
One rank uses each allocated GPU. JSON records and NCCL logs go to stdout.
"""

import json
import os
import platform
import statistics
import subprocess
import time

import torch
import torch.distributed as dist


def emit(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def measure(nbytes, rank, world, repeat):
    tensor = torch.empty(nbytes // 4, device=rank, dtype=torch.float32)
    tensor.fill_(rank + 1)
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    expected = world * (world + 1) / 2
    assert torch.all(tensor == expected).item(), "all-reduce correctness failure"
    tensor.zero_()
    for _ in range(20):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(100):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = torch.tensor([time.perf_counter() - start], device=rank)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    seconds = elapsed.item() / 100
    assert torch.count_nonzero(tensor).item() == 0, "timed reduction corrupted data"
    if rank == 0:
        emit("measurement", world_size=world, repeat=repeat, bytes=nbytes,
             latency_us=seconds * 1e6, algbw_gbps=nbytes / seconds / 1e9,
             busbw_gbps=nbytes / seconds / 1e9 * 2 * (world - 1) / world,
             correctness="pass")
    return seconds


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert torch.cuda.device_count() == world, "visible GPU count differs from request"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    props = torch.cuda.get_device_properties(rank)
    emit("rank", rank=rank, world_size=world, hostname=platform.node(),
         device_name=props.name, device_uuid=str(props.uuid),
         peer_access=[torch.cuda.can_device_access_peer(rank, peer)
                      for peer in range(world)],
         visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
         nccl_env={key: value for key, value in os.environ.items()
                   if key.startswith("NCCL_")},
         pytorch=torch.__version__, cuda=torch.version.cuda,
         nccl=torch.cuda.nccl.version())
    if rank == 0:
        for args in [["nvidia-smi", "topo", "-m"],
                     ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
                      "--format=csv"]]:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            emit("topology", command=args, output=result.stdout)
    for nbytes in [1024, 1024**2, 16 * 1024**2, 64 * 1024**2, 256 * 1024**2]:
        times = [measure(nbytes, rank, world, repeat) for repeat in range(3)]
        if rank == 0:
            emit("summary", world_size=world, bytes=nbytes,
                 median_latency_us=statistics.median(times) * 1e6)
    dist.barrier()
    dist.destroy_process_group()
    emit("complete", rank=rank, world_size=world)


if __name__ == "__main__":
    main()
