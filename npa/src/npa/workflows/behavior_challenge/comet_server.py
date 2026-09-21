"""Serve a pinned Comet checkpoint through the 2026 evaluator protocol."""

import argparse
import asyncio
from contextlib import contextmanager
import http
import logging
import os
from pathlib import Path
import sys
import tempfile

from comet_policy import (
    COMET_PROFILES,
    CONFIG_NAME,
    build_source_overlay,
    get_profile,
    policy_observation,
    task_identity,
    validate_action,
    verify_checkpoint,
)


@contextmanager
def _source_working_directory(root: Path):
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_policy(args, overlay: Path):
    sys.path[:0] = [
        str(overlay),
        str(args.source_root / "packages/openpi-client/src"),
    ]
    from openpi.policies.policy_config import create_trained_policy
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from openpi.training.config import get_config

    profile = getattr(args, "profile", get_profile("comet12"))
    task_identity(args.source_root, args.task_id, args.task_name, profile)
    policy = create_trained_policy(get_config(CONFIG_NAME), args.checkpoint)
    with _source_working_directory(args.source_root):
        return B1KPolicyWrapper(
            policy,
            task_name=args.task_name,
            control_mode="receeding_horizon",
            max_len=32,
            fine_grained_level=0,
        )


def _health(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _connection(websocket, policy, profile=None):
    from openpi_client import msgpack_numpy

    packer = msgpack_numpy.Packer()
    policy.reset()
    profile = profile or get_profile("comet12")
    await websocket.send(packer.pack({"policy": profile.handshake}))
    try:
        async for payload in websocket:
            observation = msgpack_numpy.unpackb(payload)
            if set(observation) == {"reset"} and observation["reset"] is True:
                policy.reset()
                continue
            action = validate_action(policy.act(policy_observation(observation)))
            await websocket.send(packer.pack({"action": action}))
    finally:
        policy.reset()


async def _serve(policy, port: int, profile=None) -> None:
    from websockets.asyncio.server import serve

    lock = asyncio.Lock()

    async def handle(websocket):
        if lock.locked():
            await websocket.close(code=1013, reason="Evaluator already connected")
            return
        async with lock:
            await _connection(websocket, policy, profile)

    async with serve(
        handle,
        "127.0.0.1",
        port,
        compression=None,
        max_size=16 * 1024 * 1024,
        process_request=_health,
    ) as server:
        await server.serve_forever()


class _ProfileParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        result = super().parse_args(args, namespace)
        profile = get_profile(result.profile)
        if result.task_id not in profile.task_ids:
            self.error(f"task ID is not declared by {profile.kind}")
        return result


def parser() -> argparse.ArgumentParser:
    """Build the Comet family policy server argument parser.

    Args:
        None.
    Returns:
        Parser for pinned source, checkpoint, task, and protocol settings.
    Raises:
        None.
    """
    value = _ProfileParser(description=__doc__)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--task-id", type=int, required=True)
    value.add_argument("--task-name", required=True)
    value.add_argument("--port", type=int, required=True)
    value.add_argument("--profile", choices=tuple(COMET_PROFILES), default="comet12")
    return value


def _main() -> None:
    args = parser().parse_args()
    args.profile = get_profile(args.profile)
    logging.basicConfig(level=logging.INFO)
    verify_checkpoint(args.checkpoint, args.profile)
    task_identity(args.source_root, args.task_id, args.task_name, args.profile)
    with tempfile.TemporaryDirectory(
        prefix=f"npa-{args.profile.kind}-source-"
    ) as temporary:
        overlay = build_source_overlay(args.source_root, Path(temporary))
        policy = _load_policy(args, overlay)
        asyncio.run(_serve(policy, args.port, args.profile))


if __name__ == "__main__":
    _main()
