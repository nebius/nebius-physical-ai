#!/usr/bin/env python3
"""All-reduce bus bandwidth (nccl-tests convention: busbw = algbw * 2(n-1)/n), BF16, in-place."""
import json, os, time
import torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
rows = []
for mib in (1, 4, 16, 64, 256, 1024):
    n = mib * 1024 * 1024 // 2
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(5): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    iters = 20 if mib <= 64 else 10
    t0 = time.perf_counter()
    for _ in range(iters): dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    algbw = mib * 1024 * 1024 / dt / 1e9
    rows.append({"size_mib": mib, "time_us": round(dt * 1e6, 1), "algbw_GBps": round(algbw, 2),
                 "busbw_GBps": round(algbw * 2 * (world - 1) / world, 2)})
if rank == 0:
    print(json.dumps({"world": world, "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "nccl": ".".join(map(str, torch.cuda.nccl.version())), "rows": rows}))
dist.destroy_process_group()
