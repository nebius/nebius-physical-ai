"""Serve the pinned RLC model with 2026 proprioception and the official wire protocol."""

import argparse
import asyncio
import http
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile

from rlc_observations import policy_observation


def _policy_source(root: Path, overlay: Path) -> None:
    shutil.copytree(
        root / "src/b1k",
        overlay / "b1k",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    old = "from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES"
    new = "from rlc_observations import PROPRIOCEPTION_INDICES"
    for name in ("policies/b1k_policy.py", "shared/eval_b1k_wrapper.py"):
        path = overlay / "b1k" / name
        source = path.read_text()
        if source.count(old) != 1:
            raise ValueError("Unexpected pinned RLC proprioception import")
        path.write_text(source.replace(old, new))
    sys.path[:0] = [
        str(overlay),
        str(root / "openpi/src"),
        str(root / "openpi/packages/openpi-client/src"),
    ]


def _load_policy(args):
    from b1k.policies.policy_config import create_trained_policy
    from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
    from b1k.training.config import get_config

    policy = create_trained_policy(
        get_config("pi_behavior_b1k_fast"),
        args.checkpoint,
        sample_kwargs={"num_steps": 20},
    )
    return B1KPolicyWrapper(policy, task_id=args.task_id, config=B1KWrapperConfig())


def _health(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _connection(websocket, policy):
    import numpy as np
    from openpi_client import msgpack_numpy

    packer = msgpack_numpy.Packer()
    policy.reset()
    await websocket.send(packer.pack({"policy": "rlc-2026-transfer"}))
    try:
        async for payload in websocket:
            observation = msgpack_numpy.unpackb(payload)
            if set(observation) == {"reset"} and observation["reset"] is True:
                policy.reset()
                continue
            action = policy.act(policy_observation(observation)).cpu().numpy()
            if action.shape != (23,) or not np.isfinite(action).all():
                raise ValueError("Policy returned an invalid R1Pro action")
            await websocket.send(packer.pack({"action": action}))
    finally:
        policy.reset()


async def _serve(policy, port):
    from websockets.asyncio.server import serve

    lock = asyncio.Lock()

    async def handle(websocket):
        if lock.locked():
            await websocket.close(code=1013, reason="Evaluator already connected")
            return
        async with lock:
            await _connection(websocket, policy)

    async with serve(
        handle,
        "127.0.0.1",
        port,
        compression=None,
        max_size=16 * 1024 * 1024,
        process_request=_health,
    ) as server:
        await server.serve_forever()


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-id", type=int, choices=range(50), required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory(prefix="npa-rlc-source-") as temporary:
        _policy_source(args.source_root, Path(temporary))
        asyncio.run(_serve(_load_policy(args), args.port))


if __name__ == "__main__":
    _main()
