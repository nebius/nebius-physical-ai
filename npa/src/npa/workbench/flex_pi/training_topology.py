"""Validate the two supported four-rank layouts and build their torchrun launch."""

import ipaddress
import os
import sys

from npa.workbench.flex_pi.runtime import FlexPiError


def training_topology(environ=None):
    """Cross-check the renderer's node count against SkyPilot's allocation.

    Args:
        environ: Optional environment mapping for an isolated caller.
    Returns:
        Node count, node rank, local GPU count and rendezvous address.
    Raises:
        FlexPiError: The allocation is incomplete or is not one-by-four/four-by-one.
    """
    env = os.environ if environ is None else environ
    try:
        nodes = int(env.get("NPA_FLEX_PI_NODE_COUNT", "1"))
        rank = int(env.get("SKYPILOT_NODE_RANK", "0"))
        actual = int(env.get("SKYPILOT_NUM_NODES", str(nodes)))
    except ValueError as error:
        raise FlexPiError("invalid Flex-Pi node allocation") from error
    addresses = env.get("SKYPILOT_NODE_IPS", "").split()
    if nodes not in {1, 4} or actual != nodes or not 0 <= rank < nodes:
        raise FlexPiError("Flex-Pi requires one four-GPU or four one-GPU nodes")
    if addresses and len(addresses) != nodes:
        raise FlexPiError("SkyPilot addresses disagree with the training allocation")
    if nodes == 4:
        _validate_addresses(addresses)
    return {
        "nodes": nodes,
        "node_rank": rank,
        "gpus_per_node": 4 // nodes,
        "master_address": addresses[0] if addresses else "127.0.0.1",
    }


def _validate_addresses(addresses):
    if len(set(addresses)) != 4:
        raise FlexPiError("four-node training requires four distinct SkyPilot peers")
    try:
        for address in addresses:
            ipaddress.ip_address(address)
    except ValueError as error:
        raise FlexPiError("SkyPilot peer addresses must be IP literals") from error


def rank_command(*arguments):
    """Launch exactly four global ranks using the validated allocation.

    Args:
        arguments: Worker-mode arguments passed without shell interpretation.
    Returns:
        Complete torchrun argument vector.
    Raises:
        FlexPiError: Allocation validation fails.
    """
    topology = training_topology()
    launch = ["--standalone", "--nproc-per-node=4"]
    if topology["nodes"] == 4:
        launch = [
            "--nnodes=4",
            "--nproc-per-node=1",
            f"--node-rank={topology['node_rank']}",
            f"--master-addr={topology['master_address']}",
            "--master-port=29500",
        ]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *launch,
        "--max-restarts=0",
        "--module",
        "npa.workbench.flex_pi.training_worker",
        *map(str, arguments),
    ]
