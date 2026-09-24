"""Write a private Soperator spec with an explicit STRICT B200 reservation selector."""

import argparse
import json
import os
from pathlib import Path


def _worker(nodes, reservation):
    return {
        "name": "b200",
        "platform": "gpu-b200-sxm",
        "preset": "8gpu-160vcpu-1792gb",
        "size": nodes,
        "fabric": "us-central1-b",
        "preemptible": False,
        "capacity_block_group": reservation,
        "docker_cache": True,
        "docker_cache_gib": 930,
    }


def _render(args):
    if args.nodes < 1:
        raise ValueError("nodes must be positive")
    values = {
        key: os.environ[key]
        for key in ("NPA_PROJECT_ID", "NPA_TENANT_ID", "NPA_CAPACITY_BLOCK_GROUP")
    }
    if any(not value.strip() for value in values.values()):
        raise ValueError("project, tenant and reservation selector must be nonempty")
    spec = {
        "apiVersion": "npa.soperator/v0.0.1",
        "name": args.name,
        "region": "us-central1",
        "project_id": values["NPA_PROJECT_ID"],
        "tenant_id": values["NPA_TENANT_ID"],
        "control_plane": {
            "system": {"min_size": 3},
            "login": {"preset": "16vcpu-64gb"},
        },
        "workers": [_worker(args.nodes, values["NPA_CAPACITY_BLOCK_GROUP"])],
        "accounting": True,
        "slurm_rest_enabled": True,
        "jail_size_gib": 2048,
        "slurm_operator_version": "4.1.6",
        "k8s_version": "1.34",
        "node_group_version": "72",
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(spec, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    _render(parser.parse_args())
