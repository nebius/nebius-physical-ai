"""Measure real NCCL all-reduce correctness and synchronized process duration."""

import argparse
import hashlib
import json
import os
import socket
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist


def initialize():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    identities = [None] * world
    identity = (socket.gethostname(), local_rank, torch.cuda.get_device_name())
    dist.all_gather_object(identities, identity)
    hosts = Counter(item[0] for item in identities)
    if world != 16 or len(hosts) != 2 or set(hosts.values()) != {8}:
        raise ValueError("Expected sixteen real ranks, eight on each of two hosts")
    if len({(row[0], row[1]) for row in identities}) != world:
        raise ValueError("Duplicate host/device assignment")
    if any("B200" not in row[2] for row in identities):
        raise ValueError("Every rank must use an NVIDIA B200")
    return world


def measure_size(size_bytes, world, warmup, repeats):
    tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device="cuda")
    samples = []
    expected = world * (world + 1) / 2
    for index in range(warmup + repeats):
        tensor.fill_(dist.get_rank() + 1)
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if not torch.all(tensor == expected).item():
            raise ValueError("All-reduce output differs from the expected sum")
        if index >= warmup:
            samples.append(seconds)
    all_samples = [None] * world
    dist.all_gather_object(all_samples, samples)
    slowest = [max(values) for values in zip(*all_samples, strict=True)]
    average = statistics.mean(slowest)
    return {
        "size_bytes_per_rank": size_bytes,
        "dtype": "float32",
        "validated_sum_every_element": expected,
        "slowest_rank_seconds_each_iteration": slowest,
        "mean_seconds": average,
        "median_seconds": statistics.median(slowest),
        "algorithm_bandwidth_decimal_gb_s": size_bytes / average / 1e9,
        "normalized_bus_bandwidth_decimal_gb_s": (
            size_bytes / average / 1e9 * 2 * (world - 1) / world
        ),
    }


def main(output):
    world = initialize()
    rows = [measure_size(2**power, world, 5, 20) for power in (20, 24, 28, 30)]
    if dist.get_rank() == 0:
        result = {
            "schema": "npa.cosmos3.collective-bandwidth.v1",
            "world_size": world,
            "hosts": 2,
            "gpus_per_host": 8,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": list(torch.cuda.nccl.version()),
            "warmup_iterations_per_size": 5,
            "measured_iterations_per_size": 20,
            "timing": "max across ranks of synchronized host duration per all-reduce",
            "scope": "includes Python dispatch and CUDA synchronization; not an NCCL-tests result or link-speed measurement",
            "requires_separate_nccl_transport_log_verification": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "measurements": rows,
        }
        with output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": "passed", "world_size": world, "hosts": 2}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, required=True)
    main(parser.parse_args().output_path)
