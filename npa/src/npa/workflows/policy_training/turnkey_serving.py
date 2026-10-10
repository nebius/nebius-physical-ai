"""Run independent inference and rendering workers inside one managed SkyPilot gang."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
import secrets

from .turnkey_runtime import native_runtime
from .turnkey_store import inherit, materialize, publish, read, record


def serve(args, workspace: Path, output: Path) -> None:
    """Serve and benchmark the exact promoted checkpoint on two managed GPU workers.

    Args:
        args: Promoted model input and private serving prefix.
        workspace: Private rank-local workspace.
        output: Rank-local artifact directory.
    Returns:
        None.
    Raises:
        ValueError: The gang, checkpoint or benchmark session is invalid.
        RuntimeError: Native CUDA serving or rendering fails.
    """
    rank, addresses = gang_identity(dict(os.environ))
    parent = materialize(args.input_uri, workspace / "promoted")
    inherit(parent, output)
    native_runtime(workspace)
    if rank == 0:
        _server(args, parent, output, addresses)
        role = "server"
    else:
        _client(args, workspace, parent, output, addresses)
        role = "client"
    record(
        output,
        "serve-" + role,
        {
            "engine": "smolvla-http" if rank == 0 else "native-libero-nvidia-egl",
            "rank": rank,
            "workers": 2,
            "checkpoint_sha256": read(parent, "promotion.json")["checkpoint_sha256"],
        },
    )
    publish(output, args.output_uri.rstrip("/") + "/" + role + "/")


def gang_identity(environment: dict) -> tuple[int, list[str]]:
    """Require the two distinct GPU worker addresses supplied by SkyPilot.

    Args:
        environment: SkyPilot task environment.
    Returns:
        Worker rank and validated private peer addresses.
    Raises:
        ValueError: The two-worker gang is incomplete or inconsistent.
    """
    rank = int(environment.get("SKYPILOT_NODE_RANK", "-1"))
    addresses = environment.get("SKYPILOT_NODE_IPS", "").split()
    count = int(environment.get("SKYPILOT_NUM_NODES", "0"))
    gpus = int(environment.get("SKYPILOT_NUM_GPUS_PER_NODE", "0"))
    if rank not in {0, 1} or count != 2 or len(set(addresses)) != 2 or gpus != 1:
        raise ValueError("serving requires a two-worker gang with one GPU per worker")
    for address in addresses:
        ipaddress.ip_address(address)
    return rank, addresses


def _server(args, parent, output, addresses):
    import uvicorn
    from npa.workbench.dataset.storage import write_json_uri
    from .turnkey_server import create_app

    token = secrets.token_urlsafe(32)
    config = uvicorn.Config(
        app=None, host=addresses[0], port=8080, access_log=False, log_level="error"
    )
    server = uvicorn.Server(config)
    expected = read(parent, "promotion.json")["checkpoint_sha256"]
    config.app = create_app(parent / "exported-policy", output, expected, token, server)
    write_json_uri(
        args.output_uri.rstrip("/") + "/session.json",
        {"token": token, "checkpoint_sha256": expected},
    )
    server.run()
    if not config.app:
        raise RuntimeError("policy server did not start")


def _client(args, workspace, parent, output, addresses):
    from huggingface_hub import snapshot_download
    from .public_vla_data import PINS
    from .turnkey_client import benchmark_client

    snapshot_download(
        PINS["assets"]["repo"],
        repo_type="dataset",
        token=False,
        revision=PINS["assets"]["revision"],
        local_dir=workspace / "inputs/assets",
    )
    benchmark_client(
        "http://" + addresses[0] + ":8080",
        args.output_uri.rstrip("/") + "/session.json",
        output,
        read(parent, "recipe.json"),
        read(parent, "promotion.json"),
    )
