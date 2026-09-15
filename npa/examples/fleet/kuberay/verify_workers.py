"""Execute deterministic CPU work on every Ray worker through native Ray Core."""

import hashlib
import json
import os

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=1)
def compute_shard(index):
    """Compute one deterministic shard on the scheduled Ray worker.

    Args:
        index: Zero-based shard index, selecting 10,000 consecutive integers.
    Returns:
        Shard index, worker identity, process ID, square sum and SHA-256 digest.
    Raises:
        RuntimeError: Ray cannot supply the current worker context.
    """
    start = index * 10000
    values = [value * value for value in range(start, start + 10000)]
    return {
        "index": index,
        "node_id": ray.get_runtime_context().get_node_id(),
        "pid": os.getpid(),
        "sum": sum(values),
        "sha256": hashlib.sha256(json.dumps(values).encode()).hexdigest(),
    }


def _verify_results(head, workers, results):
    for index, (worker, result) in enumerate(zip(workers, results, strict=True)):
        values = [value * value for value in range(index * 10000, (index + 1) * 10000)]
        assert result["node_id"] == worker != head
        assert result["sum"] == sum(values)
        assert result["sha256"] == hashlib.sha256(json.dumps(values).encode()).hexdigest()


def main():
    """Execute and verify a distinct deterministic shard on every live worker.

    Args:
        None.
    Returns:
        None. Prints a KUBERAY_RESULT JSON record after successful verification.
    Raises:
        AssertionError: Workers are absent or results violate placement or arithmetic.
        Exception: Ray connection, scheduling or remote execution fails.
    """
    ray.init(address="auto")
    head = ray.get_runtime_context().get_node_id()
    workers = sorted(node["NodeID"] for node in ray.nodes()
                     if node["Alive"] and node["NodeID"] != head)
    assert workers, "No live Ray workers"
    pending_shards = []
    for index, node in enumerate(workers):
        strategy = NodeAffinitySchedulingStrategy(node_id=node, soft=False)
        pending_shards.append(compute_shard.options(scheduling_strategy=strategy).remote(index))
    results = ray.get(pending_shards)
    _verify_results(head, workers, results)
    print("KUBERAY_RESULT=" + json.dumps({
        "ray_version": ray.__version__, "head_node_id": head,
        "worker_count": len(workers), "results": results,
    }), flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
